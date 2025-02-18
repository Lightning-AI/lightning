# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from abc import ABC, abstractmethod
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from typing import Any, Callable, Optional, TypeVar, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

import lightning.pytorch as pl
from lightning.fabric.plugins import CheckpointIO
from lightning.fabric.strategies import _StrategyRegistry
from lightning.fabric.utilities import move_data_to_device
from lightning.fabric.utilities.distributed import ReduceOp
from lightning.fabric.utilities.init import _EmptyInit
from lightning.fabric.utilities.optimizer import _optimizer_to_device, _optimizers_to_device
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.core.optimizer import LightningOptimizer, _init_optimizers_and_lr_schedulers
from lightning.pytorch.plugins import TorchCheckpointIO
from lightning.pytorch.plugins.io.wrapper import _WrappingCheckpointIO
from lightning.pytorch.plugins.precision import Precision
from lightning.pytorch.strategies.launchers.launcher import _Launcher
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.types import STEP_OUTPUT, LRSchedulerConfig

TBroadcast = TypeVar("TBroadcast")
TReduce = TypeVar("TReduce")

log = logging.getLogger(__name__)


class Strategy(ABC):
    """Base class for all strategies that change the behaviour of the training, validation and test- loop."""

    def __init__(
        self,
        accelerator: Optional["pl.accelerators.Accelerator"] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[Precision] = None,
    ) -> None:
        self._accelerator: Optional[pl.accelerators.Accelerator] = accelerator
        self._checkpoint_io: Optional[CheckpointIO] = checkpoint_io
        self._precision_plugin: Optional[Precision] = None
        # Call the precision setter for input validation
        self.precision_plugin = precision_plugin  # type: ignore[assignment]
        self._lightning_module: Optional[pl.LightningModule] = None
        self._model: Optional[Module] = None
        self._launcher: Optional[_Launcher] = None
        self._forward_redirection: _ForwardRedirection = _ForwardRedirection()
        self._optimizers: list[Optimizer] = []
        self._lightning_optimizers: list[LightningOptimizer] = []
        self.lr_scheduler_configs: list[LRSchedulerConfig] = []

    @property
    def launcher(self) -> Optional[_Launcher]:
        return self._launcher

    @property
    def accelerator(self) -> Optional["pl.accelerators.Accelerator"]:
        return self._accelerator

    @accelerator.setter
    def accelerator(self, accelerator: "pl.accelerators.Accelerator") -> None:
        self._accelerator = accelerator

    @property
    def checkpoint_io(self) -> CheckpointIO:
        if self._checkpoint_io is None:
            self._checkpoint_io = TorchCheckpointIO()
        elif isinstance(self._checkpoint_io, _WrappingCheckpointIO):
            self._checkpoint_io.checkpoint_io = TorchCheckpointIO()

        return self._checkpoint_io

    @checkpoint_io.setter
    def checkpoint_io(self, io: CheckpointIO) -> None:
        self._checkpoint_io = io

    @property
    def precision_plugin(self) -> Precision:
        return self._precision_plugin if self._precision_plugin is not None else Precision()

    @precision_plugin.setter
    def precision_plugin(self, precision_plugin: Optional[Precision]) -> None:
        self._precision_plugin = precision_plugin

    @property
    def optimizers(self) -> list[Optimizer]:
        return self._optimizers

    @optimizers.setter
    def optimizers(self, optimizers: list[Optimizer]) -> None:
        self._optimizers = optimizers
        self._lightning_optimizers = [LightningOptimizer._to_lightning_optimizer(opt, self) for opt in optimizers]

    def connect(self, model: "pl.LightningModule") -> None:
        """Called by the Trainer to connect the strategy with the model."""
        # model conversions cannot be applied at this point because `LightningModule.{setup,configure_model}` haven't
        # run yet
        self._lightning_module = model
        self.model = model

    def _configure_launcher(self) -> None:
        """Attach the launcher based on Strategy."""

    def setup_environment(self) -> None:
        """Setup any processes or distributed connections.

        This is called before the LightningModule/DataModule setup hook which allows the user to access the accelerator
        environment before setup is complete.

        """
        assert self.accelerator is not None
        self.accelerator.setup_device(self.root_device)

    def setup_optimizers(self, trainer: "pl.Trainer") -> None:
        """Creates optimizers and schedulers.

        Args:
            trainer: the Trainer, these optimizers should be connected to

        """
        assert self.lightning_module is not None
        self.optimizers, self.lr_scheduler_configs = _init_optimizers_and_lr_schedulers(self.lightning_module)

    def setup(self, trainer: "pl.Trainer") -> None:
        """Sets up the accelerator, plugins and initializes the optimizers (if needed).

        Args:
            trainer: the trainer instance

        """
        assert self.accelerator is not None
        self.accelerator.setup(trainer)

        assert self.model is not None
        # let the precision plugin convert the module here so that this strategy hook can decide the order
        # of operations
        self.model = self.precision_plugin.convert_module(self.model)
        self.model_to_device()
        self.model = self._setup_model(self.model)

        if trainer.state.fn == TrainerFn.FITTING:
            self.setup_optimizers(trainer)
        self.setup_precision_plugin()
        if trainer.state.fn == TrainerFn.FITTING:
            _optimizers_to_device(self.optimizers, self.root_device)

    def setup_precision_plugin(self) -> None:
        """Attaches the precision plugin to the strategy."""
        assert self.model is not None
        model, optimizers, lr_scheduler_configs = self.precision_plugin.connect(
            self.model, self.optimizers, self.lr_scheduler_configs
        )
        self.model = model
        self.optimizers = optimizers
        self.lr_scheduler_configs = lr_scheduler_configs

    def optimizer_state(self, optimizer: Optimizer) -> dict[str, Tensor]:
        """Returns state of an optimizer.

        Allows for syncing/collating optimizer state from processes in custom strategies.

        """
        if isinstance(optimizer, LightningOptimizer):
            optimizer = optimizer._optimizer

        if hasattr(optimizer, "consolidate_state_dict"):
            # there are optimizers like PyTorch's ZeroRedundancyOptimizer that shard their
            # states, and to avoid OOM we consolidate the full state on rank 0 only
            optimizer.consolidate_state_dict()
            return optimizer.state_dict() if self.is_global_zero else {}

        # for optimizers that are not sharded, we return the state dict on all ranks
        return optimizer.state_dict()

    def backward(
        self,
        closure_loss: Tensor,
        optimizer: Optional[Optimizer],
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        r"""Forwards backward-calls to the precision plugin.

        Args:
            closure_loss: a tensor holding the loss value to backpropagate
            optimizer: An optional optimizer that gets passed down to the precision plugin's backward
            \*args: Positional arguments that get passed down to the precision plugin's backward, intended as arguments
                for the actual function that performs the backward, like :meth:`~torch.Tensor.backward`.
            \**kwargs: Keyword arguments for the same purpose as ``*args``.

        """
        self.pre_backward(closure_loss)
        assert self.lightning_module is not None
        closure_loss = self.precision_plugin.pre_backward(closure_loss, self.lightning_module)

        self.precision_plugin.backward(closure_loss, self.lightning_module, optimizer, *args, **kwargs)

        closure_loss = self.precision_plugin.post_backward(closure_loss, self.lightning_module)
        self.post_backward(closure_loss)

        return closure_loss

    def optimizer_step(
        self,
        optimizer: Optimizer,
        closure: Callable[[], Any],
        model: Optional[Union["pl.LightningModule", Module]] = None,
        **kwargs: Any,
    ) -> Any:
        r"""Performs the actual optimizer step.

        Args:
            optimizer: the optimizer performing the step
            closure: closure calculating the loss value
            model: reference to the model, optionally defining optimizer step related hooks
            \**kwargs: Keyword arguments to ``optimizer.step``

        """
        model = model or self.lightning_module
        # TODO(fabric): remove assertion once strategy's optimizer_step typing is fixed
        assert isinstance(model, pl.LightningModule)
        return self.precision_plugin.optimizer_step(optimizer, model=model, closure=closure, **kwargs)

    def _setup_model_and_optimizers(self, model: Module, optimizers: list[Optimizer]) -> tuple[Module, list[Optimizer]]:
        """Setup a model and multiple optimizers together.

        The returned objects are expected to be in the same order they were passed in. The default implementation will
        call :meth:`_setup_model` and :meth:`_setup_optimizer` on the inputs.

        """
        # TODO: standardize this across all plugins in Lightning and Fabric. Related refactor: #7324
        model = self._setup_model(model)
        optimizers = [self._setup_optimizer(optimizer) for optimizer in optimizers]
        return model, optimizers

    def _setup_model(self, model: Module) -> Module:
        """Performs setup for the model, e.g., by wrapping it by another class."""
        # TODO: standardize this across all plugins in Lightning and Fabric. Related refactor: #7324
        return model

    def _setup_optimizer(self, optimizer: Optimizer) -> Optimizer:
        """Performs setup for the optimizer, e.g., by wrapping it by another class."""
        # TODO: standardize this across all plugins in Lightning and Fabric. Related refactor: #7324
        return optimizer

    def batch_to_device(self, batch: Any, device: Optional[torch.device] = None, dataloader_idx: int = 0) -> Any:
        """Moves the batch to the correct device.

        The returned batch is of the same type as the input batch, just
        having all tensors on the correct device.

        Args:
            batch: The batch of samples to move to the correct device
            device: The target device
            dataloader_idx: The index of the dataloader to which the batch belongs.

        """
        model = self.lightning_module
        device = device or self.root_device
        if model is not None:
            return model._apply_batch_transfer_handler(batch, device=device, dataloader_idx=dataloader_idx)
        return move_data_to_device(batch, device)

    @property
    @abstractmethod
    def root_device(self) -> torch.device:
        """Returns the root device."""

    @abstractmethod
    def model_to_device(self) -> None:
        """Moves the model to the correct device."""

    @property
    @abstractmethod
    def is_global_zero(self) -> bool:
        """Whether the current process is the rank zero process not only on the local node, but for all nodes."""

    @abstractmethod
    def reduce(
        self,
        tensor: Union[Tensor, Any],
        group: Optional[Any] = None,
        reduce_op: Optional[Union[ReduceOp, str]] = "mean",
    ) -> Union[Tensor, Any]:
        """Reduces the given tensor (e.g. across GPUs/processes).

        Args:
            tensor: the tensor to sync and reduce
            group: the process group to reduce
            reduce_op: the reduction operation. Defaults to 'mean'.
                Can also be a string 'sum' or ReduceOp.

        """

    @abstractmethod
    def barrier(self, name: Optional[str] = None) -> None:
        """Synchronizes all processes which blocks processes until the whole group enters this function.

        Args:
            name: an optional name to pass into barrier.

        """

    @abstractmethod
    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        """Broadcasts an object to all processes.

        Args:
            obj: the object to broadcast
            src: source rank

        """

    @abstractmethod
    def all_gather(self, tensor: Tensor, group: Optional[Any] = None, sync_grads: bool = False) -> Tensor:
        """Perform an all_gather on all processes.

        Args:
            tensor: the tensor to all_gather
            group: the process group to gather results from
            sync_grads: flag that allows users to synchronize gradients for all_gather op

        """

    def reduce_boolean_decision(self, decision: bool, all: bool = True) -> bool:
        """Reduce a boolean decision across all processes."""
        return decision

    def pre_backward(self, closure_loss: Tensor) -> None:
        """Run before precision plugin executes backward."""

    def post_backward(self, closure_loss: Tensor) -> None:
        """Run after precision plugin executes backward."""

    @property
    def model(self) -> Optional[Module]:
        """Returns the potentially wrapped LightningModule."""
        return self._model if self._model is not None else self._lightning_module

    @model.setter
    def model(self, new_model: Optional[Module]) -> None:
        self._model = new_model

    @property
    def lightning_module(self) -> Optional["pl.LightningModule"]:
        """Returns the pure LightningModule without potential wrappers."""
        return self._lightning_module

    def load_checkpoint(self, checkpoint_path: _PATH) -> dict[str, Any]:
        if isinstance(self.accelerator, pl.accelerators.Accelerator) and self.accelerator.get_device_type() != "cpu":
            getattr(torch, self.root_device.type.split(":")[0]).empty_cache()
        else:
            torch.cuda.empty_cache()
        return self.checkpoint_io.load_checkpoint(checkpoint_path)

    def load_model_state_dict(self, checkpoint: Mapping[str, Any], strict: bool = True) -> None:
        assert self.lightning_module is not None
        self.lightning_module.load_state_dict(checkpoint["state_dict"], strict=strict)

    def load_optimizer_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        optimizer_states = checkpoint["optimizer_states"]
        for optimizer, opt_state in zip(self.optimizers, optimizer_states):
            optimizer.load_state_dict(opt_state)
            _optimizer_to_device(optimizer, self.root_device)

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        """The actual training step.

        See :meth:`~lightning.pytorch.core.LightningModule.training_step` for more details

        """
        assert self.lightning_module is not None
        assert self.model is not None
        with self.precision_plugin.train_step_context():
            if self.model != self.lightning_module:
                return self._forward_redirection(self.model, self.lightning_module, "training_step", *args, **kwargs)
            return self.lightning_module.training_step(*args, **kwargs)

    def post_training_step(self) -> None:
        """This hook is deprecated.

        Override :meth:`training_step` instead.

        """
        pass

    def validation_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        """The actual validation step.

        See :meth:`~lightning.pytorch.core.LightningModule.validation_step` for more details

        """
        assert self.lightning_module is not None
        assert self.model is not None
        with self.precision_plugin.val_step_context():
            if self.model != self.lightning_module:
                return self._forward_redirection(self.model, self.lightning_module, "validation_step", *args, **kwargs)
            return self.lightning_module.validation_step(*args, **kwargs)

    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        """The actual test step.

        See :meth:`~lightning.pytorch.core.LightningModule.test_step` for more details

        """
        assert self.lightning_module is not None
        assert self.model is not None
        with self.precision_plugin.test_step_context():
            if self.model != self.lightning_module:
                return self._forward_redirection(self.model, self.lightning_module, "test_step", *args, **kwargs)
            return self.lightning_module.test_step(*args, **kwargs)

    def predict_step(self, *args: Any, **kwargs: Any) -> Any:
        """The actual predict step.

        See :meth:`~lightning.pytorch.core.LightningModule.predict_step` for more details

        """
        assert self.lightning_module is not None
        assert self.model is not None
        with self.precision_plugin.predict_step_context():
            if self.model != self.lightning_module:
                return self._forward_redirection(self.model, self.lightning_module, "predict_step", *args, **kwargs)
            return self.lightning_module.predict_step(*args, **kwargs)

    def process_dataloader(self, dataloader: object) -> object:
        """Wraps the dataloader if necessary.

        Args:
            dataloader: iterable. Ideally of type: :class:`torch.utils.data.DataLoader`

        """
        return dataloader

    @property
    def restore_checkpoint_after_setup(self) -> bool:
        """Override to delay restoring from checkpoint till after the setup phase has completed. This is useful when
        the strategy requires all the setup hooks to run before loading checkpoint.

        Returns:
            If ``True``, restore checkpoint after strategy setup.

        """
        return False

    @property
    def lightning_restore_optimizer(self) -> bool:
        """Override to disable Lightning restoring optimizers/schedulers.

        This is useful for strategies which manage restoring optimizers/schedulers.

        """
        return True

    @property
    def handles_gradient_accumulation(self) -> bool:
        """Whether the strategy handles gradient accumulation internally."""
        return False

    def lightning_module_state_dict(self) -> dict[str, Any]:
        """Returns model state."""
        assert self.lightning_module is not None
        return self.lightning_module.state_dict()

    def save_checkpoint(
        self, checkpoint: dict[str, Any], filepath: _PATH, storage_options: Optional[Any] = None
    ) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            checkpoint: dict containing model and trainer state
            filepath: write-target file's path
            storage_options: parameter for how to save to storage, passed to ``CheckpointIO`` plugin

        """
        if self.is_global_zero:
            self.checkpoint_io.save_checkpoint(checkpoint, filepath, storage_options=storage_options)

    def remove_checkpoint(self, filepath: _PATH) -> None:
        """Remove checkpoint filepath from the filesystem.

        Args:
            filepath: Path to checkpoint

        """
        if self.is_global_zero:
            self.checkpoint_io.remove_checkpoint(filepath)

    @contextmanager
    def tensor_init_context(self, empty_init: Optional[bool] = None) -> Generator[None, None, None]:
        """Controls how tensors get created (device, dtype).

        Args:
            empty_init: Whether to initialize the model with empty weights (uninitialized memory).
                If ``None``, the strategy will decide. Some strategies may not support all options.

        """
        empty_init_context = _EmptyInit(enabled=bool(empty_init))
        with empty_init_context, self.root_device, self.precision_plugin.tensor_init_context():
            yield

    @contextmanager
    def model_sharded_context(self) -> Generator[None, None, None]:
        """Provide hook to create modules in a distributed aware context. This is useful for when we'd like to shard
        the model instantly, which is useful for extremely large models which can save memory and initialization time.

        Returns: Model parallel context.

        """
        yield

    def teardown(self) -> None:
        """This method is called to teardown the training process.

        It is the right place to release memory and free other resources.

        """
        _optimizers_to_device(self.optimizers, torch.device("cpu"))

        if self.lightning_module is not None:
            log.debug(f"{self.__class__.__name__}: moving model to CPU")
            self.lightning_module.cpu()
        self.precision_plugin.teardown()
        assert self.accelerator is not None
        self.accelerator.teardown()
        self.checkpoint_io.teardown()

    @classmethod
    def register_strategies(cls, strategy_registry: _StrategyRegistry) -> None:
        pass

    def on_train_start(self) -> None:
        """Called when train begins."""
        pass

    def on_validation_start(self) -> None:
        """Called when validation begins."""
        pass

    def on_test_start(self) -> None:
        """Called when test begins."""
        pass

    def on_predict_start(self) -> None:
        """Called when predict begins."""
        pass

    def on_train_end(self) -> None:
        """Called when train ends."""
        pass

    def on_validation_end(self) -> None:
        """Called when validation ends."""
        pass

    def on_test_end(self) -> None:
        """Called when test end."""
        pass

    def on_predict_end(self) -> None:
        """Called when predict ends."""
        pass

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        """Called in the training loop before anything happens for that batch."""
        pass

    def on_exception(self, exception: BaseException) -> None:
        """Called when the trainer execution is interrupted by an exception."""
        pass

    def _reset_optimizers_and_schedulers(self) -> None:
        self._optimizers = []
        self._lightning_optimizers = []
        self.lr_scheduler_configs = []

    def __getstate__(self) -> dict:
        # `LightningOptimizer` overrides `self.__class__` so they cannot be pickled
        state = dict(vars(self))  # copy
        state["_lightning_optimizers"] = []
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__ = state
        self.optimizers = self.optimizers  # re-create the `_lightning_optimizers`


class _ForwardRedirection:
    """Implements the `forward-redirection`.

    A method call to a wrapped module gets rerouted through the wrapper's `forward` method instead.

    """

    def __call__(
        self, wrapper_module: Module, original_module: "pl.LightningModule", method_name: str, *args: Any, **kwargs: Any
    ) -> STEP_OUTPUT:
        """Reroutes a method call through the `wrapper_module`'s `forward` method.

        Args:
            wrapper_module: The module that has `original_module` wrapped.
            original_module: The module that was wrapped inside `wrapper_module`.
            method_name: The name of the method that should be called on the `original_module` after inputs get
                redirected through the `wrapper_module`'s `forward` method.
            *args: The positional arguments to the method `method_name`. They will get passed to a patched
                `forward` method instead.
            **kwargs: The keyword arguments to the method `method_name`. They will get passed to a patched
                `forward` method instead.

        """
        assert method_name != "forward"
        original_forward = original_module.forward

        def wrapped_forward(*_args: Any, **_kwargs: Any) -> Any:
            # Unpatch ourselves immediately before calling the method `method_name`
            # because itself may want to call the real `forward`
            original_module.forward = original_forward  # type: ignore[method-assign]
            # Call the actual method e.g. `.training_step(...)`
            method = getattr(original_module, method_name)
            out = method(*_args, **_kwargs)
            self.on_after_inner_forward(wrapper_module, original_module)
            return out

        # Patch the original_module's forward so we can redirect the arguments back to the real method
        original_module.forward = wrapped_forward  # type: ignore[method-assign]

        wrapper_output = wrapper_module(*args, **kwargs)
        self.on_after_outer_forward(wrapper_module, original_module)
        return wrapper_output

    def on_after_inner_forward(self, wrapper_module: Module, original_module: "pl.LightningModule") -> None:
        pass

    def on_after_outer_forward(self, wrapper_module: Module, original_module: "pl.LightningModule") -> None:
        pass
