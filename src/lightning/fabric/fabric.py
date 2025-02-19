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
import inspect
import os
from collections.abc import Generator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    Optional,
    Union,
    cast,
    overload,
)

import torch
import torch.nn as nn
from lightning_utilities.core.apply_func import apply_to_collection
from lightning_utilities.core.overrides import is_overridden
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler, RandomSampler, SequentialSampler

import lightning.fabric
from lightning.fabric.accelerators.accelerator import Accelerator
from lightning.fabric.connector import _PLUGIN_INPUT, _PRECISION_INPUT, _Connector, _is_using_cli
from lightning.fabric.loggers import Logger
from lightning.fabric.plugins import Precision  # avoid circular imports: # isort: split
from lightning.fabric.strategies import (
    DataParallelStrategy,
    DeepSpeedStrategy,
    FSDPStrategy,
    SingleDeviceStrategy,
    Strategy,
    XLAStrategy,
)
from lightning.fabric.strategies.launchers import _MultiProcessingLauncher, _XLALauncher
from lightning.fabric.strategies.strategy import TBroadcast, _Sharded
from lightning.fabric.utilities import move_data_to_device
from lightning.fabric.utilities.apply_func import convert_tensors_to_scalars, convert_to_tensors
from lightning.fabric.utilities.data import (
    _auto_add_worker_init_fn,
    _replace_dunder_methods,
    _update_dataloader,
    has_iterable_dataset,
)
from lightning.fabric.utilities.device_dtype_mixin import _update_properties
from lightning.fabric.utilities.distributed import DistributedSamplerWrapper, _InfiniteBarrier
from lightning.fabric.utilities.init import _has_meta_device_parameters_or_buffers
from lightning.fabric.utilities.rank_zero import rank_zero_deprecation, rank_zero_warn
from lightning.fabric.utilities.registry import _load_external_callbacks
from lightning.fabric.utilities.seed import seed_everything
from lightning.fabric.utilities.types import ReduceOp
from lightning.fabric.utilities.warnings import PossibleUserWarning
from lightning.fabric.wrappers import (
    _FabricDataLoader,
    _FabricModule,
    _FabricOptimizer,
    _to_compiled,
    _unwrap_compiled,
    _unwrap_objects,
)


def _do_nothing(*_: Any) -> None:
    pass


class Fabric:
    r"""Fabric accelerates your PyTorch training or inference code with minimal changes required.

    - Automatic placement of models and data onto the device.
    - Automatic support for mixed and double precision (smaller memory footprint).
    - Seamless switching between hardware (CPU, GPU, TPU) and distributed training strategies
      (data-parallel training, sharded training, etc.).
    - Automated spawning of processes, no launch utilities required.
    - Multi-node support.

    Args:
        accelerator: The hardware to run on. Possible choices are:
            ``"cpu"``, ``"cuda"``, ``"mps"``, ``"gpu"``, ``"tpu"``, ``"auto"``.
        strategy: Strategy for how to run across multiple devices. Possible choices are:
            ``"dp"``, ``"ddp"``, ``"ddp_spawn"``, ``"deepspeed"``, ``"fsdp"``.
        devices: Number of devices to train on (``int``), which GPUs to train on (``list`` or ``str``), or ``"auto"``.
            The value applies per node.
        num_nodes: Number of GPU nodes for distributed training.
        precision: Double precision (``"64"``), full precision (``"32"``), half precision AMP (``"16-mixed"``),
            or bfloat16 precision AMP (``"bf16-mixed"``).
        plugins: One or several custom plugins
        callbacks: A single callback or a list of callbacks. A callback can contain any arbitrary methods that
            can be invoked through :meth:`~lightning.fabric.fabric.Fabric.call` by the user.
        loggers: A single logger or a list of loggers. See :meth:`~lightning.fabric.fabric.Fabric.log` for more
            information.

    """

    def __init__(
        self,
        *,
        accelerator: Union[str, Accelerator] = "auto",
        strategy: Union[str, Strategy] = "auto",
        devices: Union[list[int], str, int] = "auto",
        num_nodes: int = 1,
        precision: Optional[_PRECISION_INPUT] = None,
        plugins: Optional[Union[_PLUGIN_INPUT, list[_PLUGIN_INPUT]]] = None,
        callbacks: Optional[Union[list[Any], Any]] = None,
        loggers: Optional[Union[Logger, list[Logger]]] = None,
    ) -> None:
        self._connector = _Connector(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            num_nodes=num_nodes,
            precision=precision,
            plugins=plugins,
        )
        self._strategy: Strategy = self._connector.strategy
        self._accelerator: Accelerator = self._connector.accelerator
        self._precision: Precision = self._strategy.precision
        self._callbacks = self._configure_callbacks(callbacks)
        loggers = loggers if loggers is not None else []
        self._loggers = loggers if isinstance(loggers, list) else [loggers]
        self._models_setup: int = 0
        self._launched: bool = False

        self._prepare_run_method()
        if _is_using_cli():
            # when the CLI is used to launch the script, we need to set up the environment (init processes) here so
            # that the user can immediately use all functionality in strategies
            self._strategy.setup_environment()
            self._launched = True

    @property
    def accelerator(self) -> Accelerator:
        return self._accelerator

    @property
    def strategy(self) -> Strategy:
        return self._strategy

    @property
    def device(self) -> torch.device:
        """The current device this process runs on.

        Use this to create tensors directly on the device if needed.

        """
        return self._strategy.root_device

    @property
    def global_rank(self) -> int:
        """The global index of the current process across all devices and nodes."""
        return getattr(self._strategy, "global_rank", 0)

    @property
    def local_rank(self) -> int:
        """The index of the current process among the processes running on the local node."""
        return getattr(self._strategy, "local_rank", 0)

    @property
    def node_rank(self) -> int:
        """The index of the current node."""
        return getattr(self._strategy, "node_rank", 0)

    @property
    def world_size(self) -> int:
        """The total number of processes running across all devices and nodes."""
        return getattr(self._strategy, "world_size", 1)

    @property
    def is_global_zero(self) -> bool:
        """Whether this rank is rank zero."""
        return self._strategy.is_global_zero

    @property
    def loggers(self) -> list[Logger]:
        """Returns all loggers passed to Fabric."""
        return self._loggers

    @property
    def logger(self) -> Logger:
        """Returns the first logger in the list passed to Fabric, which is considered the main logger."""
        return self._loggers[0]

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """All the code inside this run method gets accelerated by Fabric.

        You can pass arbitrary arguments to this function when overriding it.

        """

    def setup(
        self,
        module: nn.Module,
        *optimizers: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        move_to_device: bool = True,
        _reapply_compile: bool = True,
    ) -> Any:  # no specific return because the way we want our API to look does not play well with mypy
        r"""Set up a model and its optimizers for accelerated training.

        Args:
            module: A :class:`torch.nn.Module` to set up
            *optimizers: The optimizer(s) to set up (no optimizers is also possible)
            scheduler: The learning rate scheduler to set up (no learning rate scheduler is also possible)
            move_to_device: If set ``True`` (default), moves the model to the correct device. Set this to ``False``
                and alternatively use :meth:`to_device` manually.
            _reapply_compile: If ``True`` (default), and the model was ``torch.compile``d before, the
                corresponding :class:`~torch._dynamo.OptimizedModule` wrapper will be removed and reapplied with the
                same settings after the model was set up by the strategy (e.g., after the model was wrapped by DDP,
                FSDP etc.). Set it to ``False`` if compiling DDP/FSDP is causing issues.

        Returns:
            The tuple containing wrapped module, optimizers, and an optional learning rate scheduler,
            in the same order they were passed in.

        """
        self._validate_setup(module, optimizers)
        module, compile_kwargs = _unwrap_compiled(module) if _reapply_compile else (module, None)
        original_module = module

        module = self._precision.convert_module(module)

        if move_to_device:
            module = self._move_model_to_device(model=module, optimizers=list(optimizers))

        # Let accelerator/plugin wrap and connect the models and optimizers
        if optimizers:
            module, optimizers, scheduler = self._strategy.setup_module_and_optimizers(  # type: ignore[assignment]
                module, list(optimizers), scheduler
            )
        else:
            module = self._strategy.setup_module(module)

        if compile_kwargs is not None:
            module = _to_compiled(module, compile_kwargs)
        module = _FabricModule(module, self._strategy, original_module=original_module)

        # Update the _DeviceDtypeModuleMixin's device parameter
        # NOTE: for sharded strategies or manual device placement, there's no single root device
        _update_properties(
            module, device=self.device if move_to_device else next(module.parameters(), torch.tensor(0)).device
        )

        optimizers = [_FabricOptimizer(optimizer, self._strategy, self._callbacks) for optimizer in optimizers]

        self._models_setup += 1

        if hasattr(original_module, "_fabric"):  # this is probably a LightningModule
            original_module._fabric = self
            original_module._fabric_optimizers = optimizers
            if original_module not in self._callbacks:
                self._callbacks.append(original_module)

        self.call("on_after_setup", fabric=self, module=module)

        if optimizers:
            # join both types in a tuple for API convenience
            return (module, *optimizers, scheduler) if scheduler is not None else (module, *optimizers)
        return module

    def setup_module(
        self, module: nn.Module, move_to_device: bool = True, _reapply_compile: bool = True
    ) -> _FabricModule:
        r"""Set up a model for accelerated training or inference.

        This is the same as calling ``.setup(model)`` with no optimizers. It is useful for inference or for certain
        strategies like `FSDP` that require setting up the module before the optimizer can be created and set up.
        See also :meth:`setup_optimizers`.

        Args:
            module: A :class:`torch.nn.Module` to set up
            move_to_device: If set ``True`` (default), moves the model to the correct device. Set this to ``False``
                and alternatively use :meth:`to_device` manually.
            _reapply_compile: If ``True`` (default), and the model was ``torch.compile``d before, the
                corresponding :class:`~torch._dynamo.OptimizedModule` wrapper will be removed and reapplied with the
                same settings after the model was set up by the strategy (e.g., after the model was wrapped by DDP,
                FSDP etc.). Set it to ``False`` if compiling DDP/FSDP is causing issues.
        Returns:
            The wrapped model.

        """
        self._validate_setup_module(module)
        module, compile_kwargs = _unwrap_compiled(module) if _reapply_compile else (module, None)
        original_module = module

        module = self._precision.convert_module(module)

        if move_to_device:
            module = self._move_model_to_device(model=module, optimizers=[])

        # Let strategy wrap and connect the module alone
        module = self._strategy.setup_module(module)

        if compile_kwargs is not None:
            module = _to_compiled(module, compile_kwargs)
        module = _FabricModule(module, self._strategy, original_module=original_module)

        # Update the _DeviceDtypeModuleMixin's device parameter
        # NOTE: for sharded strategies or manual device placement, there's no single root device
        _update_properties(
            module, device=self.device if move_to_device else next(module.parameters(), torch.tensor(0)).device
        )

        if hasattr(original_module, "_fabric"):  # this is probably a LightningModule
            original_module._fabric = self
            if original_module not in self._callbacks:
                self._callbacks.append(original_module)

        self._models_setup += 1
        return module

    def setup_optimizers(self, *optimizers: Optimizer) -> Union[_FabricOptimizer, tuple[_FabricOptimizer, ...]]:
        r"""Set up one or more optimizers for accelerated training.

        Some strategies do not allow setting up model and optimizer independently. For them, you should call
        ``.setup(model, optimizer, ...)`` instead to jointly set them up.

        Args:
            *optimizers: One or more optmizers to set up.

        Returns:
            The wrapped optimizer(s).

        """
        self._validate_setup_optimizers(optimizers)
        optimizers = [self._strategy.setup_optimizer(optimizer) for optimizer in optimizers]
        optimizers = [
            _FabricOptimizer(optimizer=optimizer, strategy=self._strategy, callbacks=self._callbacks)
            for optimizer in optimizers
        ]
        return optimizers[0] if len(optimizers) == 1 else tuple(optimizers)

    def setup_dataloaders(
        self, *dataloaders: DataLoader, use_distributed_sampler: bool = True, move_to_device: bool = True
    ) -> Union[DataLoader, list[DataLoader]]:
        r"""Set up one or multiple dataloaders for accelerated training. If you need different settings for each
        dataloader, call this method individually for each one.

        Args:
            *dataloaders: A single dataloader or a sequence of dataloaders.
            use_distributed_sampler: If set ``True`` (default), automatically wraps or replaces the sampler on the
                dataloader(s) for distributed training. If you have a custom sampler defined, set this argument
                to ``False``.
            move_to_device: If set ``True`` (default), moves the data returned by the dataloader(s) automatically to
                the correct device. Set this to ``False`` and alternatively use :meth:`to_device` manually on the
                returned data.

        Returns:
            The wrapped dataloaders, in the same order they were passed in.

        """
        self._validate_setup_dataloaders(dataloaders)
        dataloaders = [
            self._setup_dataloader(
                dataloader, use_distributed_sampler=use_distributed_sampler, move_to_device=move_to_device
            )
            for dataloader in dataloaders
        ]
        dataloaders = dataloaders[0] if len(dataloaders) == 1 else dataloaders
        return dataloaders  # type: ignore[return-value]

    def _setup_dataloader(
        self, dataloader: DataLoader, use_distributed_sampler: bool = True, move_to_device: bool = True
    ) -> DataLoader:
        r"""Set up a single dataloader for accelerated training.

        Args:
            dataloader: The dataloader to accelerate.
            use_distributed_sampler: If set ``True`` (default), automatically wraps or replaces the sampler on the
                dataloader for distributed training. If you have a custom sampler defined, set this argument to
                ``False``.
            move_to_device: If set ``True`` (default), moves the data returned by the dataloader automatically to
                the correct device. Set this to ``False`` and alternatively use :meth:`to_device` manually on the
                returned data.

        Returns:
            The wrapped dataloader.

        """
        if use_distributed_sampler and self._requires_distributed_sampler(dataloader):
            sampler = self._get_distributed_sampler(dataloader, **self._strategy.distributed_sampler_kwargs)

            # the dataloader needs to be re-instantiated because we want to update the sampler
            dataloader = _update_dataloader(dataloader, sampler)

        # add worker_init_fn for correct seeding in worker processes
        _auto_add_worker_init_fn(dataloader, self.global_rank)

        dataloader = self._strategy.process_dataloader(dataloader)
        device = self.device if move_to_device and not isinstance(self._strategy, XLAStrategy) else None
        fabric_dataloader = _FabricDataLoader(dataloader=dataloader, device=device)
        fabric_dataloader = cast(DataLoader, fabric_dataloader)
        return fabric_dataloader

    def backward(self, tensor: Tensor, *args: Any, model: Optional[_FabricModule] = None, **kwargs: Any) -> None:
        r"""Replaces ``loss.backward()`` in your training loop. Handles precision and automatically for you.

        Args:
            tensor: The tensor (loss) to back-propagate gradients from.
            *args: Optional positional arguments passed to the underlying backward function.
            model: Optional model instance for plugins that require the model for backward().
            **kwargs: Optional named keyword arguments passed to the underlying backward function.

        Note:
            When using ``strategy="deepspeed"`` and multiple models were set up, it is required to pass in the
            model as argument here.

        """
        module = model._forward_module if model is not None else model
        module, _ = _unwrap_compiled(module)
        if isinstance(self._strategy, DeepSpeedStrategy):
            if model is None:
                if self._models_setup == 0:
                    raise RuntimeError("No models were set up for backward. Did you forget to call `fabric.setup()`?")
                if self._models_setup > 1:
                    raise ValueError(
                        "When using multiple models + deepspeed, please provide the model used to perform"
                        " the optimization: `self.backward(loss, model=model)`"
                    )
                module = self._strategy.model
            else:
                # requires to attach the current `DeepSpeedEngine` for the `_FabricOptimizer.step` call.
                self._strategy._deepspeed_engine = module

        lightning.fabric.wrappers._in_fabric_backward = True
        try:
            self._strategy.backward(tensor, module, *args, **kwargs)
        finally:
            lightning.fabric.wrappers._in_fabric_backward = False

    def clip_gradients(
        self,
        module: Union[torch.nn.Module, _FabricModule],
        optimizer: Union[Optimizer, _FabricOptimizer],
        clip_val: Optional[Union[float, int]] = None,
        max_norm: Optional[Union[float, int]] = None,
        norm_type: Union[float, int] = 2.0,
        error_if_nonfinite: bool = True,
    ) -> Optional[torch.Tensor]:
        """Clip the gradients of the model to a given max value or max norm.

        Args:
            module: The module whose parameters should be clipped.
            optimizer: The optimizer referencing the parameters to be clipped.
            clip_val: If passed, gradients will be clipped to this value.
            max_norm: If passed, clips the gradients in such a way that the p-norm of the resulting parameters is
                no larger than the given value.
            norm_type: The type of norm if `max_norm` was passed. Can be ``'inf'`` for infinity norm.
                Default is the 2-norm.
            error_if_nonfinite: An error is raised if the total norm of the gradients is NaN or infinite.

        Return:
            The total norm of the gradients (before clipping was applied) as a scalar tensor if ``max_norm`` was
            passed, otherwise ``None``.

        """
        if clip_val is not None and max_norm is not None:
            raise ValueError(
                "Only one of `clip_val` or `max_norm` can be set as this specifies the underlying clipping algorithm!"
            )

        if clip_val is not None:
            self.strategy.clip_gradients_value(_unwrap_objects(module), _unwrap_objects(optimizer), clip_val=clip_val)
            return None
        if max_norm is not None:
            return self.strategy.clip_gradients_norm(
                _unwrap_objects(module),
                _unwrap_objects(optimizer),
                max_norm=max_norm,
                norm_type=norm_type,
                error_if_nonfinite=error_if_nonfinite,
            )
        raise ValueError("You have to specify either `clip_val` or `max_norm` to do gradient clipping!")

    def autocast(self) -> AbstractContextManager:
        """A context manager to automatically convert operations for the chosen precision.

        Use this only if the `forward` method of your model does not cover all operations you wish to run with the
        chosen precision setting.

        """
        return self._precision.forward_context()

    @overload
    def to_device(self, obj: nn.Module) -> nn.Module: ...

    @overload
    def to_device(self, obj: Tensor) -> Tensor: ...

    @overload
    def to_device(self, obj: Any) -> Any: ...

    def to_device(self, obj: Union[nn.Module, Tensor, Any]) -> Union[nn.Module, Tensor, Any]:
        r"""Move a :class:`torch.nn.Module` or a collection of tensors to the current device, if it is not already on
        that device.

        Args:
            obj: An object to move to the device. Can be an instance of :class:`torch.nn.Module`, a tensor, or a
                 (nested) collection of tensors (e.g., a dictionary).

        Returns:
            A reference to the object that was moved to the new device.

        """
        if isinstance(obj, nn.Module):
            self._accelerator.setup_device(self.device)
            self._strategy.module_to_device(obj)
            return obj
        return move_data_to_device(obj, device=self.device)

    def print(self, *args: Any, **kwargs: Any) -> None:
        r"""Print something only on the first process. If running on multiple machines, it will print from the first
        process in each machine.

        Arguments passed to this method are forwarded to the Python built-in :func:`print` function.

        """
        if self.local_rank == 0:
            print(*args, **kwargs)

    def barrier(self, name: Optional[str] = None) -> None:
        """Wait for all processes to enter this call.

        Use this to synchronize all parallel processes, but only if necessary, otherwise the overhead of synchronization
        will cause your program to slow down. This method needs to be called on all processes. Failing to do so will
        cause your program to stall forever.

        """
        self._validate_launched()
        self._strategy.barrier(name=name)

    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        r"""Send a tensor from one process to all others.

        This method needs to be called on all processes. Failing to do so will cause your program to stall forever.

        Args:
            obj: The object to broadcast to all other members. Any serializable object is supported, but it is
                most efficient with the object being a :class:`~torch.Tensor`.
            src: The (global) rank of the process that should send the data to all others.

        Return:
            The transferred data, the same value on every rank.

        """
        self._validate_launched()
        return self._strategy.broadcast(obj, src=src)

    def all_gather(
        self, data: Union[Tensor, dict, list, tuple], group: Optional[Any] = None, sync_grads: bool = False
    ) -> Union[Tensor, dict, list, tuple]:
        """Gather tensors or collections of tensors from multiple processes.

        This method needs to be called on all processes and the tensors need to have the same shape across all
        processes, otherwise your program will stall forever.

        Args:
            data: int, float, tensor of shape (batch, ...), or a (possibly nested) collection thereof.
            group: the process group to gather results from. Defaults to all processes (world).
            sync_grads: flag that allows users to synchronize gradients for the ``all_gather`` operation

        Return:
            A tensor of shape (world_size, batch, ...), or if the input was a collection
            the output will also be a collection with tensors of this shape. For the special case where
            world_size is 1, no additional dimension is added to the tensor(s).

        """
        self._validate_launched()
        group = group if group is not None else torch.distributed.group.WORLD
        data = convert_to_tensors(data, device=self.device)
        return apply_to_collection(data, Tensor, self._strategy.all_gather, group=group, sync_grads=sync_grads)

    def all_reduce(
        self,
        data: Union[Tensor, dict, list, tuple],
        group: Optional[Any] = None,
        reduce_op: Optional[Union[ReduceOp, str]] = "mean",
    ) -> Union[Tensor, dict, list, tuple]:
        """Reduce tensors or collections of tensors from multiple processes.

        The reduction on tensors is applied in-place, meaning the result will be placed back into the input tensor.
        This method needs to be called on all processes and the tensors need to have the same shape across all
        processes, otherwise your program will stall forever.

        Args:
            data: int, float, tensor of shape (batch, ...), or a (possibly nested) collection thereof. Tensor will be
                modified in-place.
            group: the process group to reduce results across. Defaults to all processes (world).
            reduce_op: the reduction operation. Defaults to 'mean'. Can also be a string 'sum' or ReduceOp.
                Some strategies may limit the choices here.

        Return:
            A tensor of the same shape as the input with values reduced pointwise across processes. The same is
            applied to tensors in a collection if a collection is given as input.

        """
        self._validate_launched()
        group = group if group is not None else torch.distributed.group.WORLD
        data = convert_to_tensors(data, device=self.device)
        return apply_to_collection(data, Tensor, self._strategy.all_reduce, group=group, reduce_op=reduce_op)

    @contextmanager
    def rank_zero_first(self, local: bool = False) -> Generator:
        r"""The code block under this context manager gets executed first on the main process (rank 0) and only when
        completed, the other processes get to run the code in parallel.

        Args:
            local: Set this to ``True`` if the **local** rank should be the one going first. Useful if you are
                downloading data and the filesystem isn't shared between the nodes.

        Example::

            with fabric.rank_zero_first():
                dataset = MNIST("datasets/", download=True)

        """
        rank = self.local_rank if local else self.global_rank
        with _InfiniteBarrier() as barrier:
            if rank > 0:
                barrier()
            yield
            if rank == 0:
                barrier()

    def no_backward_sync(self, module: _FabricModule, enabled: bool = True) -> AbstractContextManager:
        r"""Skip gradient synchronization during backward to avoid redundant communication overhead.

        Use this context manager when performing gradient accumulation to speed up training with multiple devices.

        Example::

            # Accumulate gradient 8 batches at a time
            with fabric.no_backward_sync(model, enabled=(batch_idx % 8 != 0)):
                output = model(input)
                loss = ...
                fabric.backward(loss)
                ...

        For those strategies that don't support it, a warning is emitted. For single-device strategies, it is a no-op.
        Both the model's ``.forward()`` and the ``fabric.backward()`` call need to run under this context.

        Args:
            module: The module for which to control the gradient synchronization.
            enabled: Whether the context manager is enabled or not. ``True`` means skip the sync, ``False`` means do not
                skip.

        """
        module, _ = _unwrap_compiled(module)
        if not isinstance(module, _FabricModule):
            raise TypeError(
                "You need to set up the model first before you can call `fabric.no_backward_sync()`:"
                " `model = fabric.setup(model, ...)`"
            )
        if isinstance(self._strategy, (SingleDeviceStrategy, XLAStrategy)):
            return nullcontext()
        if self._strategy._backward_sync_control is None:
            rank_zero_warn(
                f"The `{self._strategy.__class__.__name__}` does not support skipping the gradient synchronization."
                f" Remove `.no_backward_sync()` from your code or choose a different strategy.",
                category=PossibleUserWarning,
            )
            return nullcontext()

        forward_module, _ = _unwrap_compiled(module._forward_module)
        return self._strategy._backward_sync_control.no_backward_sync(forward_module, enabled)

    def sharded_model(self) -> AbstractContextManager:
        r"""Instantiate a model under this context manager to prepare it for model-parallel sharding.

        .. deprecated:: This context manager is deprecated in favor of :meth:`init_module`, use it instead.

        """
        rank_zero_deprecation("`Fabric.sharded_model()` is deprecated in favor of `Fabric.init_module()`.")
        self._validate_launched()
        if isinstance(self.strategy, _Sharded):
            return self.strategy.module_sharded_context()
        return nullcontext()

    def init_tensor(self) -> AbstractContextManager:
        """Tensors that you instantiate under this context manager will be created on the device right away and have
        the right data type depending on the precision setting in Fabric."""
        return self._strategy.tensor_init_context()

    def init_module(self, empty_init: Optional[bool] = None) -> AbstractContextManager:
        """Instantiate the model and its parameters under this context manager to reduce peak memory usage.

        The parameters get created on the device and with the right data type right away without wasting memory being
        allocated unnecessarily.

        Args:
            empty_init: Whether to initialize the model with empty weights (uninitialized memory).
                If ``None``, the strategy will decide. Some strategies may not support all options.
                Set this to ``True`` if you are loading a checkpoint into a large model.

        """
        self._validate_launched()
        return self._strategy.module_init_context(empty_init=empty_init)

    def save(
        self,
        path: Union[str, Path],
        state: dict[str, Union[nn.Module, Optimizer, Any]],
        filter: Optional[dict[str, Callable[[str, Any], bool]]] = None,
    ) -> None:
        r"""Save checkpoint contents to a file.

        How and which processes save gets determined by the `strategy`. For example, the `ddp` strategy
        saves checkpoints only on process 0, while the `fsdp` strategy saves files from every rank.
        This method must be called on all processes!

        Args:
            path: A path to where the file(s) should be saved
            state: A dictionary with contents to be saved. If the dict contains modules or optimizers, their
                state-dict will be retrieved and converted automatically.
            filter: An optional dictionary containing filter callables that return a boolean indicating whether the
                given item should be saved (``True``) or filtered out (``False``). Each filter key should match a
                state key, where its filter will be applied to the ``state_dict`` generated.

        """
        if filter is not None:
            if not isinstance(filter, dict):
                raise TypeError(f"Filter should be a dictionary, given {filter!r}")
            if not set(filter).issubset(state):
                raise ValueError(
                    f"The filter keys {filter.keys() - state} are not present in the state keys {set(state)}."
                )
            for k, v in filter.items():
                if not callable(v):
                    raise TypeError(f"Expected `fabric.save(filter=...)` for key {k!r} to be a callable, given {v!r}")
        self._strategy.save_checkpoint(path=path, state=_unwrap_objects(state), filter=filter)
        self.barrier()

    def load(
        self,
        path: Union[str, Path],
        state: Optional[dict[str, Union[nn.Module, Optimizer, Any]]] = None,
        strict: bool = True,
    ) -> dict[str, Any]:
        """Load a checkpoint from a file and restore the state of objects (modules, optimizers, etc.)

        How and which processes load gets determined by the `strategy`.
        This method must be called on all processes!

        Args:
            path: A path to where the file is located
            state: A dictionary of objects whose state will be restored in-place from the checkpoint path.
                If no state is given, then the checkpoint will be returned in full.
            strict: Whether to enforce that the keys in `state` match the keys in the checkpoint.

        Returns:
            The remaining items that were not restored into the given state dictionary. If no state dictionary is
            given, the full checkpoint will be returned.

        """
        unwrapped_state = _unwrap_objects(state)
        remainder = self._strategy.load_checkpoint(path=path, state=unwrapped_state, strict=strict)
        self.barrier()
        if state is not None:
            # We need to unwrap objects (see above) but this creates a new dictionary. In-place updates
            # (for user metadata) wouldn't show up in the original dict, so we need to copy the data back.
            for k in list(unwrapped_state.keys()):
                obj, _ = _unwrap_compiled(state[k])
                if isinstance(obj, (_FabricModule, _FabricOptimizer, _FabricDataLoader)):
                    continue
                state[k] = unwrapped_state[k]
        return remainder

    def load_raw(self, path: Union[str, Path], obj: Union[nn.Module, Optimizer], strict: bool = True) -> None:
        """Load the state of a module or optimizer from a single state-dict file.

        Use this for loading a raw PyTorch model checkpoint created without Fabric.
        This is conceptually equivalent to ``obj.load_state_dict(torch.load(path))``, but is agnostic to the strategy
        being used.

        Args:
            path: A path to where the file is located
            obj: A :class:`~torch.nn.Module` or :class:`~torch.optim.Optimizer` instance.
            strict: Whether to enforce that the keys in the module's state-dict match the keys in the checkpoint.
                Does not apply to optimizers.

        """
        obj = _unwrap_objects(obj)
        self._strategy.load_checkpoint(path=path, state=obj, strict=strict)

    def launch(self, function: Callable[["Fabric"], Any] = _do_nothing, *args: Any, **kwargs: Any) -> Any:
        """Launch and initialize all the processes needed for distributed execution.

        Args:
            function: Optional function to launch when using a spawn/fork-based strategy, for example, when using the
                XLA strategy (``accelerator="tpu"``). The function must accept at least one argument, to which
                the Fabric object itself will be passed.
            *args: Optional positional arguments to be passed to the function.
            **kwargs: Optional keyword arguments to be passed to the function.

        Returns:
            Returns the output of the function that ran in worker process with rank 0.

        The ``launch()`` method should only be used if you intend to specify accelerator, devices, and so on in
        the code (programmatically). If you are launching with the Lightning CLI, ``fabric run ...``, remove
        ``launch()`` from your code.

        The ``launch()`` is a no-op when called multiple times and no function is passed in.

        """
        if _is_using_cli():
            raise RuntimeError(
                "This script was launched through the CLI, and processes have already been created. Calling "
                " `.launch()` again is not allowed."
            )
        if function is not _do_nothing:
            if not callable(function):
                raise TypeError(
                    f"`Fabric.launch(...)` needs to be a callable, but got {function}."
                    " HINT: do `.launch(your_fn)` instead of `.launch(your_fn())`"
                )
            if not inspect.signature(function).parameters:
                raise TypeError(
                    f"`Fabric.launch(function={function})` needs to take at least one argument. The launcher will"
                    " pass in the `Fabric` object so you can use it inside the function."
                )
        elif isinstance(self.strategy.launcher, (_MultiProcessingLauncher, _XLALauncher)):
            raise TypeError(
                f"To spawn processes with the `{type(self.strategy).__name__}` strategy, `.launch()` needs to be called"
                " with a function that contains the code to launch in processes."
            )
        return self._wrap_and_launch(function, self, *args, **kwargs)

    def call(self, hook_name: str, *args: Any, **kwargs: Any) -> None:
        r"""Trigger the callback methods with the given name and arguments.

        Not all objects registered via ``Fabric(callbacks=...)`` must implement a method with the given name. The ones
        that have a matching method name will get called.

        Args:
            hook_name: The name of the callback method.
            *args: Optional positional arguments that get passed down to the callback method.
            **kwargs: Optional keyword arguments that get passed down to the callback method.

        Example::

            class MyCallback:
                def on_train_epoch_end(self, results):
                    ...

            fabric = Fabric(callbacks=[MyCallback()])
            fabric.call("on_train_epoch_end", results={...})

        """
        for callback in self._callbacks:
            method = getattr(callback, hook_name, None)
            if method is None:
                continue
            if not callable(method):
                rank_zero_warn(
                    f"Skipping the callback `{type(callback).__name__}.{hook_name}` because it is not callable."
                )
                continue

            method(*args, **kwargs)

            # TODO(fabric): handle the following signatures
            # method(self, fabric|trainer, x, y=1)
            # method(self, fabric|trainer, *args, x, y=1)
            # method(self, *args, y=1)
            # method(self, *args, **kwargs)

    def log(self, name: str, value: Any, step: Optional[int] = None) -> None:
        """Log a scalar to all loggers that were added to Fabric.

        Args:
            name: The name of the metric to log.
            value: The metric value to collect. If the value is a :class:`torch.Tensor`, it gets detached from the
                graph automatically.
            step: Optional step number. Most Logger implementations auto-increment the step value by one with every
                log call. You can specify your own value here.

        """
        self.log_dict(metrics={name: value}, step=step)

    def log_dict(self, metrics: Mapping[str, Any], step: Optional[int] = None) -> None:
        """Log multiple scalars at once to all loggers that were added to Fabric.

        Args:
            metrics: A dictionary where the key is the name of the metric and the value the scalar to be logged.
                Any :class:`torch.Tensor` in the dictionary get detached from the graph automatically.
            step: Optional step number. Most Logger implementations auto-increment this value by one with every
                log call. You can specify your own value here.

        """
        metrics = convert_tensors_to_scalars(metrics)
        for logger in self._loggers:
            logger.log_metrics(metrics=metrics, step=step)

    @staticmethod
    def seed_everything(seed: Optional[int] = None, workers: Optional[bool] = None, verbose: bool = True) -> int:
        r"""Helper function to seed everything without explicitly importing Lightning.

        See :func:`~lightning.fabric.utilities.seed.seed_everything` for more details.

        """
        if workers is None:
            # Lightning sets `workers=False` by default to avoid breaking reproducibility, but since this is a new
            # release, we can afford to do it.
            workers = True
        return seed_everything(seed=seed, workers=workers, verbose=verbose)

    def _wrap_and_launch(self, to_run: Callable, *args: Any, **kwargs: Any) -> Any:
        self._launched = True
        to_run = partial(self._wrap_with_setup, to_run)
        if (launcher := self._strategy.launcher) is not None:
            return launcher.launch(to_run, *args, **kwargs)
        return to_run(*args, **kwargs)

    def _wrap_with_setup(self, to_run: Callable, *args: Any, **kwargs: Any) -> Any:
        self._strategy.setup_environment()
        with _replace_dunder_methods(DataLoader, "dataset"), _replace_dunder_methods(BatchSampler):
            return to_run(*args, **kwargs)

    def _move_model_to_device(self, model: nn.Module, optimizers: list[Optimizer]) -> nn.Module:
        try:
            initial_name, initial_param = next(model.named_parameters())
        except StopIteration:
            pass
        else:
            initial_device = initial_param.device
            count = 0
            first_name, first_device = None, None
            for name, param in model.named_parameters():
                if param.device != initial_device:
                    count += 1
                    if first_name is None:
                        first_name = name
                        first_device = param.device
            if count > 0:
                rank_zero_warn(
                    f"The model passed to `Fabric.setup()` has {count} parameters on different devices (for example"
                    f" {first_name!r} on {first_device} and {initial_name!r} on {initial_device}). Since"
                    " `move_to_device=True`, all parameters will be moved to the new device. If this is not"
                    " desired, set `Fabric.setup(..., move_to_device=False)`.",
                    category=PossibleUserWarning,
                )

        if isinstance(self._strategy, XLAStrategy):
            # When the user creates the optimizer, they reference the parameters on the CPU.
            # However, when running with TPU the parameters get copied and the reference in the optimizer
            # remains invalid. We need to update the references to point to the parameter tensors on the device.
            params_before_move = dict(model.named_parameters())
            model = self.to_device(model)
            # XLA makes a copy on the parameters, so the device is not the same before and after to_device.
            params_on_device = dict(model.named_parameters())

            mapping = {param: params_on_device[name] for name, param in params_before_move.items()}
            for optimizer in optimizers:
                for param_group in optimizer.param_groups:
                    param_group["params"] = [mapping.get(p, p) for p in param_group["params"]]
        else:
            model = self.to_device(model)
        return model

    def _requires_distributed_sampler(self, dataloader: DataLoader) -> bool:
        return (
            getattr(self.strategy, "distributed_sampler_kwargs", None) is not None
            and not isinstance(dataloader.sampler, DistributedSampler)
            and not has_iterable_dataset(dataloader)
        )

    @staticmethod
    def _get_distributed_sampler(dataloader: DataLoader, **kwargs: Any) -> DistributedSampler:
        kwargs.setdefault("shuffle", isinstance(dataloader.sampler, RandomSampler))
        kwargs.setdefault("seed", int(os.getenv("PL_GLOBAL_SEED", 0)))
        if isinstance(dataloader.sampler, (RandomSampler, SequentialSampler)):
            return DistributedSampler(dataloader.dataset, **kwargs)
        return DistributedSamplerWrapper(dataloader.sampler, **kwargs)

    def _prepare_run_method(self) -> None:
        if is_overridden("run", self, Fabric) and _is_using_cli():
            raise TypeError(
                "Overriding `Fabric.run()` and launching from the CLI is not allowed. Run the script normally,"
                " or change your code to directly call `fabric = Fabric(...); fabric.setup(...)` etc."
            )
        # wrap the run method, so we can inject setup logic or spawn processes for the user
        setattr(self, "run", partial(self._wrap_and_launch, self.run))

    def _validate_launched(self) -> None:
        if not self._launched and not isinstance(self._strategy, (SingleDeviceStrategy, DataParallelStrategy)):
            raise RuntimeError(
                "To use Fabric with more than one device, you must call `.launch()` or use the CLI:"
                " `fabric run --help`."
            )

    def _validate_setup(self, module: nn.Module, optimizers: Sequence[Optimizer]) -> None:
        self._validate_launched()
        if isinstance(module, _FabricModule):
            raise ValueError("A model should be passed only once to the `setup` method.")

        if any(isinstance(opt, _FabricOptimizer) for opt in optimizers):
            raise ValueError("An optimizer should be passed only once to the `setup` method.")

        if isinstance(self._strategy, FSDPStrategy) and any(
            _has_meta_device_parameters_or_buffers(optimizer) for optimizer in optimizers
        ):
            raise RuntimeError(
                "The optimizer has references to the model's meta-device parameters. Materializing them is"
                " is currently not supported unless you to set up the model and optimizer(s) separately."
                " Create and set up the model first through `model = fabric.setup_module(model)`. Then create the"
                " optimizer and set it up: `optimizer = fabric.setup_optimizers(optimizer)`."
            )

    def _validate_setup_module(self, module: nn.Module) -> None:
        self._validate_launched()
        if isinstance(module, _FabricModule):
            raise ValueError("A model should be passed only once to the `setup_module` method.")

    def _validate_setup_optimizers(self, optimizers: Sequence[Optimizer]) -> None:
        self._validate_launched()
        if isinstance(self._strategy, (DeepSpeedStrategy, XLAStrategy)):
            raise RuntimeError(
                f"The `{type(self._strategy).__name__}` requires the model and optimizer(s) to be set up jointly"
                " through `.setup(model, optimizer, ...)`."
            )

        if not optimizers:
            raise ValueError("`setup_optimizers` requires at least one optimizer as input.")

        if any(isinstance(opt, _FabricOptimizer) for opt in optimizers):
            raise ValueError("An optimizer should be passed only once to the `setup_optimizers` method.")

        if any(_has_meta_device_parameters_or_buffers(optimizer) for optimizer in optimizers):
            raise RuntimeError(
                "The optimizer has references to the model's meta-device parameters. Materializing them is"
                " is currently not supported. Create the optimizer after setting up the model, then call"
                " `fabric.setup_optimizers(optimizer)`."
            )

    def _validate_setup_dataloaders(self, dataloaders: Sequence[DataLoader]) -> None:
        self._validate_launched()
        if not dataloaders:
            raise ValueError("`setup_dataloaders` requires at least one dataloader as input.")

        if any(isinstance(dl, _FabricDataLoader) for dl in dataloaders):
            raise ValueError("A dataloader should be passed only once to the `setup_dataloaders` method.")

        if any(not isinstance(dl, DataLoader) for dl in dataloaders):
            raise TypeError("Only PyTorch DataLoader are currently supported in `setup_dataloaders`.")

    @staticmethod
    def _configure_callbacks(callbacks: Optional[Union[list[Any], Any]]) -> list[Any]:
        callbacks = callbacks if callbacks is not None else []
        callbacks = callbacks if isinstance(callbacks, list) else [callbacks]
        callbacks.extend(_load_external_callbacks("lightning.fabric.callbacks_factory"))
        return callbacks
