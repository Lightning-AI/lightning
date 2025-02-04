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
"""The LightningModule - an nn.Module with many additional features."""

import logging
import numbers
import weakref
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Optional,
    Union,
    cast,
    overload,
)

import torch
from lightning_utilities.core.apply_func import apply_to_collection
from lightning_utilities.core.imports import RequirementCache
from torch import ScriptModule, Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torchmetrics import Metric, MetricCollection
from typing_extensions import Self, override

import lightning.fabric as lf
import lightning.pytorch as pl
from lightning.fabric.loggers import Logger as FabricLogger
from lightning.fabric.utilities.apply_func import convert_to_tensors
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.fabric.utilities.device_dtype_mixin import _DeviceDtypeModuleMixin
from lightning.fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH
from lightning.fabric.wrappers import _FabricOptimizer
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.core.hooks import CheckpointHooks, DataHooks, ModelHooks
from lightning.pytorch.core.mixins import HyperparametersMixin
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.core.saving import _load_from_checkpoint
from lightning.pytorch.loggers import Logger
from lightning.pytorch.trainer import call
from lightning.pytorch.trainer.connectors.logger_connector.fx_validator import _FxValidator
from lightning.pytorch.trainer.connectors.logger_connector.result import _get_default_dtype
from lightning.pytorch.utilities import GradClipAlgorithmType
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.imports import _TORCHMETRICS_GREATER_EQUAL_0_9_1
from lightning.pytorch.utilities.model_helpers import _restricted_classmethod
from lightning.pytorch.utilities.rank_zero import WarningCache, rank_zero_warn
from lightning.pytorch.utilities.signature_utils import is_param_in_hook_signature
from lightning.pytorch.utilities.types import (
    _METRIC,
    STEP_OUTPUT,
    LRSchedulerPLType,
    LRSchedulerTypeUnion,
    OptimizerLRScheduler,
)

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

_ONNX_AVAILABLE = RequirementCache("onnx")

warning_cache = WarningCache()
log = logging.getLogger(__name__)

MODULE_OPTIMIZERS = Union[
    Optimizer, LightningOptimizer, _FabricOptimizer, list[Optimizer], list[LightningOptimizer], list[_FabricOptimizer]
]


class LightningModule(
    _DeviceDtypeModuleMixin,
    HyperparametersMixin,
    ModelHooks,
    DataHooks,
    CheckpointHooks,
    Module,
):
    # Below is for property support of JIT
    # since none of these are important when using JIT, we are going to ignore them.
    __jit_unused_properties__: list[str] = (
        [
            "example_input_array",
            "on_gpu",
            "current_epoch",
            "global_step",
            "global_rank",
            "local_rank",
            "logger",
            "loggers",
            "automatic_optimization",
            "trainer",
            "fabric",
            "strict_loading",
            "device_mesh",
        ]
        + _DeviceDtypeModuleMixin.__jit_unused_properties__
        + HyperparametersMixin.__jit_unused_properties__
    )
    _jit_is_scripting = False

    CHECKPOINT_HYPER_PARAMS_KEY = "hyper_parameters"
    CHECKPOINT_HYPER_PARAMS_NAME = "hparams_name"
    CHECKPOINT_HYPER_PARAMS_TYPE = "hparams_type"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # pointer to the trainer object
        self._trainer: Optional[pl.Trainer] = None

        # attributes that can be set by user
        self._example_input_array: Optional[Union[Tensor, tuple, dict]] = None
        self._automatic_optimization: bool = True
        self._strict_loading: Optional[bool] = None

        # attributes used internally
        self._current_fx_name: Optional[str] = None
        self._param_requires_grad_state: dict[str, bool] = {}
        self._metric_attributes: Optional[dict[int, str]] = None
        self._compiler_ctx: Optional[dict[str, Any]] = None

        # attributes only used when using fabric
        self._fabric: Optional[lf.Fabric] = None
        self._fabric_optimizers: list[_FabricOptimizer] = []

        # access to device mesh in `conigure_model()` hook
        self._device_mesh: Optional[DeviceMesh] = None

    @overload
    def optimizers(
        self, use_pl_optimizer: Literal[True] = True
    ) -> Union[LightningOptimizer, list[LightningOptimizer]]: ...

    @overload
    def optimizers(self, use_pl_optimizer: Literal[False]) -> Union[Optimizer, list[Optimizer]]: ...

    @overload
    def optimizers(self, use_pl_optimizer: bool) -> MODULE_OPTIMIZERS: ...

    def optimizers(self, use_pl_optimizer: bool = True) -> MODULE_OPTIMIZERS:
        """Returns the optimizer(s) that are being used during training. Useful for manual optimization.

        Args:
            use_pl_optimizer: If ``True``, will wrap the optimizer(s) in a
                :class:`~lightning.pytorch.core.optimizer.LightningOptimizer` for automatic handling of precision,
                profiling, and counting of step calls for proper logging and checkpointing. It specifically wraps the
                ``step`` method and custom optimizers that don't have this method are not supported.

        Returns:
            A single optimizer, or a list of optimizers in case multiple ones are present.

        """
        if self._fabric:
            opts: MODULE_OPTIMIZERS = self._fabric_optimizers
        elif use_pl_optimizer:
            opts = self.trainer.strategy._lightning_optimizers
        else:
            opts = self.trainer.optimizers

        # single optimizer
        if (
            isinstance(opts, list)
            and len(opts) == 1
            and isinstance(opts[0], (Optimizer, LightningOptimizer, _FabricOptimizer))
        ):
            return opts[0]
        # multiple opts
        return opts

    def lr_schedulers(self) -> Union[None, list[LRSchedulerPLType], LRSchedulerPLType]:
        """Returns the learning rate scheduler(s) that are being used during training. Useful for manual optimization.

        Returns:
            A single scheduler, or a list of schedulers in case multiple ones are present, or ``None`` if no
            schedulers were returned in :meth:`~lightning.pytorch.core.LightningModule.configure_optimizers`.

        """
        if not self.trainer.lr_scheduler_configs:
            return None

        # ignore other keys "interval", "frequency", etc.
        lr_schedulers: list[LRSchedulerPLType] = [config.scheduler for config in self.trainer.lr_scheduler_configs]

        # single scheduler
        if len(lr_schedulers) == 1:
            return lr_schedulers[0]

        # multiple schedulers
        return lr_schedulers

    @property
    def trainer(self) -> "pl.Trainer":
        if self._fabric is not None:
            return _TrainerFabricShim(fabric=self._fabric)  # type: ignore[return-value]
        if not self._jit_is_scripting and self._trainer is None:
            raise RuntimeError(f"{self.__class__.__qualname__} is not attached to a `Trainer`.")
        return self._trainer  # type: ignore[return-value]

    @trainer.setter
    def trainer(self, trainer: Optional["pl.Trainer"]) -> None:
        for v in self.children():
            if isinstance(v, LightningModule):
                v.trainer = trainer  # type: ignore[assignment]
        self._trainer = trainer

    @property
    def fabric(self) -> Optional["lf.Fabric"]:
        return self._fabric

    @fabric.setter
    def fabric(self, fabric: Optional["lf.Fabric"]) -> None:
        for v in self.children():
            if isinstance(v, LightningModule):
                v.fabric = fabric
        if fabric is not None and not isinstance(fabric, weakref.ProxyTypes):
            fabric = weakref.proxy(fabric)
        self._fabric = fabric

    @property
    def example_input_array(self) -> Optional[Union[Tensor, tuple, dict]]:
        """The example input array is a specification of what the module can consume in the :meth:`forward` method. The
        return type is interpreted as follows:

        -   Single tensor: It is assumed the model takes a single argument, i.e.,
            ``model.forward(model.example_input_array)``
        -   Tuple: The input array should be interpreted as a sequence of positional arguments, i.e.,
            ``model.forward(*model.example_input_array)``
        -   Dict: The input array represents named keyword arguments, i.e.,
            ``model.forward(**model.example_input_array)``

        """
        return self._example_input_array

    @example_input_array.setter
    def example_input_array(self, example: Optional[Union[Tensor, tuple, dict]]) -> None:
        self._example_input_array = example

    @property
    def current_epoch(self) -> int:
        """The current epoch in the ``Trainer``, or 0 if not attached."""
        return self.trainer.current_epoch if self._trainer else 0

    @property
    def global_step(self) -> int:
        """Total training batches seen across all epochs.

        If no Trainer is attached, this propery is 0.

        """
        return self.trainer.global_step if self._trainer else 0

    @property
    def global_rank(self) -> int:
        """The index of the current process across all nodes and devices."""
        return self.trainer.global_rank if self._trainer else 0

    @property
    def local_rank(self) -> int:
        """The index of the current process within a single node."""
        return self.trainer.local_rank if self._trainer else 0

    @property
    def on_gpu(self) -> bool:
        """Returns ``True`` if this model is currently located on a GPU.

        Useful to set flags around the LightningModule for different CPU vs GPU behavior.

        """
        return self.device.type == "cuda"

    @property
    def automatic_optimization(self) -> bool:
        """If set to ``False`` you are responsible for calling ``.backward()``, ``.step()``, ``.zero_grad()``."""
        return self._automatic_optimization

    @automatic_optimization.setter
    def automatic_optimization(self, automatic_optimization: bool) -> None:
        self._automatic_optimization = automatic_optimization

    @property
    def strict_loading(self) -> bool:
        """Determines how Lightning loads this model using `.load_state_dict(..., strict=model.strict_loading)`."""
        # We use None as the default internally to determine whether the user has set a value
        return self._strict_loading in (None, True)

    @strict_loading.setter
    def strict_loading(self, strict_loading: bool) -> None:
        self._strict_loading = strict_loading

    @property
    def logger(self) -> Optional[Union[Logger, FabricLogger]]:
        """Reference to the logger object in the Trainer."""
        if self._fabric is not None:
            return self._fabric.logger
        return self._trainer.logger if self._trainer is not None else None

    @property
    def loggers(self) -> Union[list[Logger], list[FabricLogger]]:
        """Reference to the list of loggers in the Trainer."""
        if self._fabric is not None:
            return self._fabric.loggers
        if self._trainer is not None:
            return self._trainer.loggers
        return []

    @property
    def device_mesh(self) -> Optional["DeviceMesh"]:
        """Strategies like ``ModelParallelStrategy`` will create a device mesh that can be accessed in the
        :meth:`~lightning.pytorch.core.hooks.ModelHooks.configure_model` hook to parallelize the LightningModule."""
        return self._device_mesh

    def _call_batch_hook(self, hook_name: str, *args: Any) -> Any:
        trainer = self._trainer
        if trainer:
            datahook_selector = trainer._data_connector._datahook_selector
            assert datahook_selector is not None
            obj = datahook_selector.get_instance(hook_name)
            if isinstance(obj, self.__class__):
                trainer_method = call._call_lightning_module_hook
            else:
                trainer_method = call._call_lightning_datamodule_hook

            return trainer_method(trainer, hook_name, *args)
        hook = getattr(self, hook_name)
        return hook(*args)

    def _on_before_batch_transfer(self, batch: Any, dataloader_idx: int = 0) -> Any:
        return self._call_batch_hook("on_before_batch_transfer", batch, dataloader_idx)

    def _apply_batch_transfer_handler(
        self, batch: Any, device: Optional[torch.device] = None, dataloader_idx: int = 0
    ) -> Any:
        device = device or self.device
        batch = self._call_batch_hook("transfer_batch_to_device", batch, device, dataloader_idx)
        batch = self._call_batch_hook("on_after_batch_transfer", batch, dataloader_idx)
        return batch

    def print(self, *args: Any, **kwargs: Any) -> None:
        r"""Prints only from process 0. Use this in any distributed mode to log only once.

        Args:
            *args: The thing to print. The same as for Python's built-in print function.
            **kwargs: The same as for Python's built-in print function.

        Example::

            def forward(self, x):
                self.print(x, 'in forward')

        """
        if self.trainer.is_global_zero:
            progress_bar = self.trainer.progress_bar_callback
            if progress_bar is not None and progress_bar.is_enabled:
                progress_bar.print(*args, **kwargs)
            else:
                print(*args, **kwargs)

    def log(
        self,
        name: str,
        value: _METRIC,
        prog_bar: bool = False,
        logger: Optional[bool] = None,
        on_step: Optional[bool] = None,
        on_epoch: Optional[bool] = None,
        reduce_fx: Union[str, Callable] = "mean",
        enable_graph: bool = False,
        sync_dist: bool = False,
        sync_dist_group: Optional[Any] = None,
        add_dataloader_idx: bool = True,
        batch_size: Optional[int] = None,
        metric_attribute: Optional[str] = None,
        rank_zero_only: bool = False,
    ) -> None:
        """Log a key, value pair.

        Example::

            self.log('train_loss', loss)

        The default behavior per hook is documented here: :ref:`extensions/logging:Automatic Logging`.

        Args:
            name: key to log. Must be identical across all processes if using DDP or any other distributed strategy.
            value: value to log. Can be a ``float``, ``Tensor``, or a ``Metric``.
            prog_bar: if ``True`` logs to the progress bar.
            logger: if ``True`` logs to the logger.
            on_step: if ``True`` logs at this step. The default value is determined by the hook.
                See :ref:`extensions/logging:Automatic Logging` for details.
            on_epoch: if ``True`` logs epoch accumulated metrics. The default value is determined by the hook.
                See :ref:`extensions/logging:Automatic Logging` for details.
            reduce_fx: reduction function over step values for end of epoch. :meth:`torch.mean` by default.
            enable_graph: if ``True``, will not auto detach the graph.
            sync_dist: if ``True``, reduces the metric across devices. Use with care as this may lead to a significant
                communication overhead.
            sync_dist_group: the DDP group to sync across.
            add_dataloader_idx: if ``True``, appends the index of the current dataloader to
                the name (when using multiple dataloaders). If False, user needs to give unique names for
                each dataloader to not mix the values.
            batch_size: Current batch_size. This will be directly inferred from the loaded batch,
                but for some data structures you might need to explicitly provide it.
            metric_attribute: To restore the metric state, Lightning requires the reference of the
                :class:`torchmetrics.Metric` in your model. This is found automatically if it is a model attribute.
            rank_zero_only: Tells Lightning if you are calling ``self.log`` from every process (default) or only from
                rank 0. If ``True``, you won't be able to use this metric as a monitor in callbacks
                (e.g., early stopping). Warning: Improper use can lead to deadlocks! See
                :ref:`Advanced Logging <visualize/logging_advanced:rank_zero_only>` for more details.

        """
        if self._fabric is not None:
            self._log_dict_through_fabric(dictionary={name: value}, logger=logger)
            return

        # check for invalid values
        apply_to_collection(value, dict, self.__check_not_nested, name)
        apply_to_collection(
            value, object, self.__check_allowed, name, value, wrong_dtype=(numbers.Number, Metric, Tensor)
        )

        trainer = self._trainer
        if trainer is None:
            # not an error to support testing the `*_step` methods without a `Trainer` reference
            rank_zero_warn(
                "You are trying to `self.log()` but the `self.trainer` reference is not registered on the model yet."
                " This is most likely because the model hasn't been passed to the `Trainer`"
            )
            return
        if trainer.barebones:
            rank_zero_warn(
                "You are trying to `self.log()` but `Trainer(barebones=True)` is configured."
                " Logging can impact raw speed so it is disabled under this setting."
            )
            return
        results = trainer._results
        if results is None:
            raise MisconfigurationException(
                "You are trying to `self.log()` but the loop's result collection is not registered"
                " yet. This is most likely because you are trying to log in a `predict` hook,"
                " but it doesn't support logging"
            )
        if self._current_fx_name is None:
            raise MisconfigurationException(
                "You are trying to `self.log()` but it is not managed by the `Trainer` control flow"
            )

        on_step, on_epoch = _FxValidator.check_logging_and_get_default_levels(
            self._current_fx_name, on_step=on_step, on_epoch=on_epoch
        )

        # make sure user doesn't introduce logic for multi-dataloaders
        if "/dataloader_idx_" in name:
            raise MisconfigurationException(
                f"You called `self.log` with the key `{name}`"
                " but it should not contain information about `dataloader_idx`"
            )

        value = apply_to_collection(value, (Tensor, numbers.Number), self.__to_tensor, name)

        if trainer._logger_connector.should_reset_tensors(self._current_fx_name):
            # if we started a new epoch (running its first batch) the hook name has changed
            # reset any tensors for the new hook name
            results.reset(metrics=False, fx=self._current_fx_name)

        if metric_attribute is None and isinstance(value, Metric):
            if self._metric_attributes is None:
                # compute once
                self._metric_attributes = {
                    id(module): name for name, module in self.named_modules() if isinstance(module, Metric)
                }
                if not self._metric_attributes:
                    raise MisconfigurationException(
                        "Could not find the `LightningModule` attribute for the `torchmetrics.Metric` logged."
                        " You can fix this by setting an attribute for the metric in your `LightningModule`."
                    )
            # try to find the passed metric in the LightningModule
            metric_attribute = self._metric_attributes.get(id(value), None)
            if metric_attribute is None:
                raise MisconfigurationException(
                    "Could not find the `LightningModule` attribute for the `torchmetrics.Metric` logged."
                    f" You can fix this by calling `self.log({name}, ..., metric_attribute=name)` where `name` is one"
                    f" of {list(self._metric_attributes.values())}"
                )

        if (
            trainer.training
            and is_param_in_hook_signature(self.training_step, "dataloader_iter", explicit=True)
            and batch_size is None
        ):
            raise MisconfigurationException(
                "With `def training_step(self, dataloader_iter)`, `self.log(..., batch_size=...)` should be provided."
            )

        if logger and trainer.logger is None:
            rank_zero_warn(
                f"You called `self.log({name!r}, ..., logger=True)` but have no logger configured. You can enable one"
                " by doing `Trainer(logger=ALogger(...))`"
            )
        if logger is None:
            # we could set false here if there's no configured logger, however, we still need to compute the "logged"
            # metrics anyway because that's what the evaluation loops use as return value
            logger = True

        results.log(
            self._current_fx_name,
            name,
            value,
            prog_bar=prog_bar,
            logger=logger,
            on_step=on_step,
            on_epoch=on_epoch,
            reduce_fx=reduce_fx,
            enable_graph=enable_graph,
            add_dataloader_idx=add_dataloader_idx,
            batch_size=batch_size,
            sync_dist=sync_dist and trainer._accelerator_connector.is_distributed,
            sync_dist_fn=trainer.strategy.reduce,
            sync_dist_group=sync_dist_group,
            metric_attribute=metric_attribute,
            rank_zero_only=rank_zero_only,
        )

        trainer._logger_connector._current_fx = self._current_fx_name

    def log_dict(
        self,
        dictionary: Union[Mapping[str, _METRIC], MetricCollection],
        prog_bar: bool = False,
        logger: Optional[bool] = None,
        on_step: Optional[bool] = None,
        on_epoch: Optional[bool] = None,
        reduce_fx: Union[str, Callable] = "mean",
        enable_graph: bool = False,
        sync_dist: bool = False,
        sync_dist_group: Optional[Any] = None,
        add_dataloader_idx: bool = True,
        batch_size: Optional[int] = None,
        rank_zero_only: bool = False,
    ) -> None:
        """Log a dictionary of values at once.

        Example::

            values = {'loss': loss, 'acc': acc, ..., 'metric_n': metric_n}
            self.log_dict(values)

        Args:
            dictionary: key value pairs.
                Keys must be identical across all processes if using DDP or any other distributed strategy.
                The values can be a ``float``, ``Tensor``, ``Metric``, or ``MetricCollection``.
            prog_bar: if ``True`` logs to the progress base.
            logger: if ``True`` logs to the logger.
            on_step: if ``True`` logs at this step.
                ``None`` auto-logs for training_step but not validation/test_step.
                The default value is determined by the hook.
                See :ref:`extensions/logging:Automatic Logging` for details.
            on_epoch: if ``True`` logs epoch accumulated metrics.
                ``None`` auto-logs for val/test step but not ``training_step``.
                The default value is determined by the hook.
                See :ref:`extensions/logging:Automatic Logging` for details.
            reduce_fx: reduction function over step values for end of epoch. :meth:`torch.mean` by default.
            enable_graph: if ``True``, will not auto-detach the graph
            sync_dist: if ``True``, reduces the metric across GPUs/TPUs. Use with care as this may lead to a significant
                communication overhead.
            sync_dist_group: the ddp group to sync across.
            add_dataloader_idx: if ``True``, appends the index of the current dataloader to
                the name (when using multiple). If ``False``, user needs to give unique names for
                each dataloader to not mix values.
            batch_size: Current batch size. This will be directly inferred from the loaded batch,
                but some data structures might need to explicitly provide it.
            rank_zero_only: Tells Lightning if you are calling ``self.log`` from every process (default) or only from
                rank 0. If ``True``, you won't be able to use this metric as a monitor in callbacks
                (e.g., early stopping). Warning: Improper use can lead to deadlocks! See
                :ref:`Advanced Logging <visualize/logging_advanced:rank_zero_only>` for more details.

        """
        if self._fabric is not None:
            return self._log_dict_through_fabric(dictionary=dictionary, logger=logger)

        kwargs: dict[str, bool] = {}

        if isinstance(dictionary, MetricCollection):
            kwargs["keep_base"] = False
            if _TORCHMETRICS_GREATER_EQUAL_0_9_1 and dictionary._enable_compute_groups:
                kwargs["copy_state"] = False

        for k, v in dictionary.items(**kwargs):
            self.log(
                name=k,
                value=v,
                prog_bar=prog_bar,
                logger=logger,
                on_step=on_step,
                on_epoch=on_epoch,
                reduce_fx=reduce_fx,
                enable_graph=enable_graph,
                sync_dist=sync_dist,
                sync_dist_group=sync_dist_group,
                add_dataloader_idx=add_dataloader_idx,
                batch_size=batch_size,
                rank_zero_only=rank_zero_only,
            )
        return None

    def _log_dict_through_fabric(
        self, dictionary: Union[Mapping[str, _METRIC], MetricCollection], logger: Optional[bool] = None
    ) -> None:
        if logger is False:
            # Passing `logger=False` with Fabric does not make much sense because there is no other destination to
            # log to, but we support it in case the original code was written for Trainer use
            return

        if any(isinstance(v, dict) for v in dictionary.values()):
            raise ValueError(f"`self.log_dict({dictionary})` was called, but nested dictionaries cannot be logged")
        for name, value in dictionary.items():
            apply_to_collection(value, object, self.__check_allowed, name, value, wrong_dtype=(numbers.Number, Tensor))

        assert self._fabric is not None
        self._fabric.log_dict(metrics=dictionary)  # type: ignore[arg-type]

    @staticmethod
    def __check_not_nested(value: dict, name: str) -> None:
        # self-imposed restriction. for simplicity
        if any(isinstance(v, dict) for v in value.values()):
            raise ValueError(f"`self.log({name}, {value})` was called, but nested dictionaries cannot be logged")

    @staticmethod
    def __check_allowed(v: Any, name: str, value: Any) -> None:
        raise ValueError(f"`self.log({name}, {value})` was called, but `{type(v).__name__}` values cannot be logged")

    def __to_tensor(self, value: Union[Tensor, numbers.Number], name: str) -> Tensor:
        value = (
            value.clone().detach()
            if isinstance(value, Tensor)
            else torch.tensor(value, device=self.device, dtype=_get_default_dtype())
        )
        if not torch.numel(value) == 1:
            raise ValueError(
                f"`self.log({name}, {value})` was called, but the tensor must have a single element."
                f" You can try doing `self.log({name}, {value}.mean())`"
            )
        value = value.squeeze()
        return value

    def all_gather(
        self, data: Union[Tensor, dict, list, tuple], group: Optional[Any] = None, sync_grads: bool = False
    ) -> Union[Tensor, dict, list, tuple]:
        r"""Gather tensors or collections of tensors from multiple processes.

        This method needs to be called on all processes and the tensors need to have the same shape across all
        processes, otherwise your program will stall forever.

        Args:
            data: int, float, tensor of shape (batch, ...), or a (possibly nested) collection thereof.
            group: the process group to gather results from. Defaults to all processes (world)
            sync_grads: flag that allows users to synchronize gradients for the all_gather operation

        Return:
            A tensor of shape (world_size, batch, ...), or if the input was a collection
            the output will also be a collection with tensors of this shape. For the special case where
            world_size is 1, no additional dimension is added to the tensor(s).

        """
        group = group if group is not None else torch.distributed.group.WORLD
        all_gather = self.trainer.strategy.all_gather
        data = convert_to_tensors(data, device=self.device)
        return apply_to_collection(data, Tensor, all_gather, group=group, sync_grads=sync_grads)

    @override
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        r"""Same as :meth:`torch.nn.Module.forward`.

        Args:
            *args: Whatever you decide to pass into the forward method.
            **kwargs: Keyword arguments are also possible.

        Return:
            Your model's output

        """
        return super().forward(*args, **kwargs)

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        r"""Here you compute and return the training loss and some additional metrics for e.g. the progress bar or
        logger.

        Args:
            batch: The output of your data iterable, normally a :class:`~torch.utils.data.DataLoader`.
            batch_idx: The index of this batch.
            dataloader_idx: The index of the dataloader that produced this batch.
                (only if multiple dataloaders used)

        Return:
            - :class:`~torch.Tensor` - The loss tensor
            - ``dict`` - A dictionary which can include any keys, but must include the key ``'loss'`` in the case of
              automatic optimization.
            - ``None`` - In automatic optimization, this will skip to the next batch (but is not supported for
              multi-GPU, TPU, or DeepSpeed). For manual optimization, this has no special meaning, as returning
              the loss is not required.

        In this step you'd normally do the forward pass and calculate the loss for a batch.
        You can also do fancier things like multiple forward passes or something model specific.

        Example::

            def training_step(self, batch, batch_idx):
                x, y, z = batch
                out = self.encoder(x)
                loss = self.loss(out, x)
                return loss

        To use multiple optimizers, you can switch to 'manual optimization' and control their stepping:

        .. code-block:: python

            def __init__(self):
                super().__init__()
                self.automatic_optimization = False


            # Multiple optimizers (e.g.: GANs)
            def training_step(self, batch, batch_idx):
                opt1, opt2 = self.optimizers()

                # do training_step with encoder
                ...
                opt1.step()
                # do training_step with decoder
                ...
                opt2.step()

        Note:
            When ``accumulate_grad_batches`` > 1, the loss returned here will be automatically
            normalized by ``accumulate_grad_batches`` internally.

        """
        rank_zero_warn("`training_step` must be implemented to be used with the Lightning Trainer")

    def validation_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        r"""Operates on a single batch of data from the validation set. In this step you'd might generate examples or
        calculate anything of interest like accuracy.

        Args:
            batch: The output of your data iterable, normally a :class:`~torch.utils.data.DataLoader`.
            batch_idx: The index of this batch.
            dataloader_idx: The index of the dataloader that produced this batch.
                (only if multiple dataloaders used)

        Return:
            - :class:`~torch.Tensor` - The loss tensor
            - ``dict`` - A dictionary. Can include any keys, but must include the key ``'loss'``.
            - ``None`` - Skip to the next batch.

        .. code-block:: python

            # if you have one val dataloader:
            def validation_step(self, batch, batch_idx): ...


            # if you have multiple val dataloaders:
            def validation_step(self, batch, batch_idx, dataloader_idx=0): ...

        Examples::

            # CASE 1: A single validation dataset
            def validation_step(self, batch, batch_idx):
                x, y = batch

                # implement your own
                out = self(x)
                loss = self.loss(out, y)

                # log 6 example images
                # or generated text... or whatever
                sample_imgs = x[:6]
                grid = torchvision.utils.make_grid(sample_imgs)
                self.logger.experiment.add_image('example_images', grid, 0)

                # calculate acc
                labels_hat = torch.argmax(out, dim=1)
                val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

                # log the outputs!
                self.log_dict({'val_loss': loss, 'val_acc': val_acc})

        If you pass in multiple val dataloaders, :meth:`validation_step` will have an additional argument. We recommend
        setting the default value of 0 so that you can quickly switch between single and multiple dataloaders.

        .. code-block:: python

            # CASE 2: multiple validation dataloaders
            def validation_step(self, batch, batch_idx, dataloader_idx=0):
                # dataloader_idx tells you which dataset this is.
                ...

        Note:
            If you don't need to validate you don't need to implement this method.

        Note:
            When the :meth:`validation_step` is called, the model has been put in eval mode
            and PyTorch gradients have been disabled. At the end of validation,
            the model goes back to training mode and gradients are enabled.

        """

    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        r"""Operates on a single batch of data from the test set. In this step you'd normally generate examples or
        calculate anything of interest such as accuracy.

        Args:
            batch: The output of your data iterable, normally a :class:`~torch.utils.data.DataLoader`.
            batch_idx: The index of this batch.
            dataloader_idx: The index of the dataloader that produced this batch.
                (only if multiple dataloaders used)

        Return:
            - :class:`~torch.Tensor` - The loss tensor
            - ``dict`` - A dictionary. Can include any keys, but must include the key ``'loss'``.
            - ``None`` - Skip to the next batch.

        .. code-block:: python

            # if you have one test dataloader:
            def test_step(self, batch, batch_idx): ...


            # if you have multiple test dataloaders:
            def test_step(self, batch, batch_idx, dataloader_idx=0): ...

        Examples::

            # CASE 1: A single test dataset
            def test_step(self, batch, batch_idx):
                x, y = batch

                # implement your own
                out = self(x)
                loss = self.loss(out, y)

                # log 6 example images
                # or generated text... or whatever
                sample_imgs = x[:6]
                grid = torchvision.utils.make_grid(sample_imgs)
                self.logger.experiment.add_image('example_images', grid, 0)

                # calculate acc
                labels_hat = torch.argmax(out, dim=1)
                test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

                # log the outputs!
                self.log_dict({'test_loss': loss, 'test_acc': test_acc})

        If you pass in multiple test dataloaders, :meth:`test_step` will have an additional argument. We recommend
        setting the default value of 0 so that you can quickly switch between single and multiple dataloaders.

        .. code-block:: python

            # CASE 2: multiple test dataloaders
            def test_step(self, batch, batch_idx, dataloader_idx=0):
                # dataloader_idx tells you which dataset this is.
                ...

        Note:
            If you don't need to test you don't need to implement this method.

        Note:
            When the :meth:`test_step` is called, the model has been put in eval mode and
            PyTorch gradients have been disabled. At the end of the test epoch, the model goes back
            to training mode and gradients are enabled.

        """

    def predict_step(self, *args: Any, **kwargs: Any) -> Any:
        """Step function called during :meth:`~lightning.pytorch.trainer.trainer.Trainer.predict`. By default, it calls
        :meth:`~lightning.pytorch.core.LightningModule.forward`. Override to add any processing logic.

        The :meth:`~lightning.pytorch.core.LightningModule.predict_step` is used
        to scale inference on multi-devices.

        To prevent an OOM error, it is possible to use :class:`~lightning.pytorch.callbacks.BasePredictionWriter`
        callback to write the predictions to disk or database after each batch or on epoch end.

        The :class:`~lightning.pytorch.callbacks.BasePredictionWriter` should be used while using a spawn
        based accelerator. This happens for ``Trainer(strategy="ddp_spawn")``
        or training on 8 TPU cores with ``Trainer(accelerator="tpu", devices=8)`` as predictions won't be returned.

        Args:
            batch: The output of your data iterable, normally a :class:`~torch.utils.data.DataLoader`.
            batch_idx: The index of this batch.
            dataloader_idx: The index of the dataloader that produced this batch.
                (only if multiple dataloaders used)

        Return:
            Predicted output (optional).

        Example ::

            class MyModel(LightningModule):

                def predict_step(self, batch, batch_idx, dataloader_idx=0):
                    return self(batch)

            dm = ...
            model = MyModel()
            trainer = Trainer(accelerator="gpu", devices=2)
            predictions = trainer.predict(model, dm)

        """
        # For backwards compatibility
        batch = kwargs.get("batch", args[0])
        return self(batch)

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        """Configure model-specific callbacks. When the model gets attached, e.g., when ``.fit()`` or ``.test()`` gets
        called, the list or a callback returned here will be merged with the list of callbacks passed to the Trainer's
        ``callbacks`` argument. If a callback returned here has the same type as one or several callbacks already
        present in the Trainer's callbacks list, it will take priority and replace them. In addition, Lightning will
        make sure :class:`~lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint` callbacks run last.

        Return:
            A callback or a list of callbacks which will extend the list of callbacks in the Trainer.

        Example::

            def configure_callbacks(self):
                early_stop = EarlyStopping(monitor="val_acc", mode="max")
                checkpoint = ModelCheckpoint(monitor="val_loss")
                return [early_stop, checkpoint]

        """
        return []

    def configure_optimizers(self) -> OptimizerLRScheduler:
        r"""Choose what optimizers and learning-rate schedulers to use in your optimization. Normally you'd need one.
        But in the case of GANs or similar you might have multiple. Optimization with multiple optimizers only works in
        the manual optimization mode.

        Return:
            Any of these 6 options.

            - **Single optimizer**.
            - **List or Tuple** of optimizers.
            - **Two lists** - The first list has multiple optimizers, and the second has multiple LR schedulers
              (or multiple ``lr_scheduler_config``).
            - **Dictionary**, with an ``"optimizer"`` key, and (optionally) a ``"lr_scheduler"``
              key whose value is a single LR scheduler or ``lr_scheduler_config``.
            - **None** - Fit will run without any optimizer.

        The ``lr_scheduler_config`` is a dictionary which contains the scheduler and its associated configuration.
        The default configuration is shown below.

        .. code-block:: python

            lr_scheduler_config = {
                # REQUIRED: The scheduler instance
                "scheduler": lr_scheduler,
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": "epoch",
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": 1,
                # Metric to monitor for schedulers like `ReduceLROnPlateau`
                "monitor": "val_loss",
                # If set to `True`, will enforce that the value specified 'monitor'
                # is available when the scheduler is updated, thus stopping
                # training if not found. If set to `False`, it will only produce a warning
                "strict": True,
                # If using the `LearningRateMonitor` callback to monitor the
                # learning rate progress, this keyword can be used to specify
                # a custom logged name
                "name": None,
            }

        When there are schedulers in which the ``.step()`` method is conditioned on a value, such as the
        :class:`torch.optim.lr_scheduler.ReduceLROnPlateau` scheduler, Lightning requires that the
        ``lr_scheduler_config`` contains the keyword ``"monitor"`` set to the metric name that the scheduler
        should be conditioned on.

        .. testcode::

            # The ReduceLROnPlateau scheduler requires a monitor
            def configure_optimizers(self):
                optimizer = Adam(...)
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": ReduceLROnPlateau(optimizer, ...),
                        "monitor": "metric_to_track",
                        "frequency": "indicates how often the metric is updated",
                        # If "monitor" references validation metrics, then "frequency" should be set to a
                        # multiple of "trainer.check_val_every_n_epoch".
                    },
                }


            # In the case of two optimizers, only one using the ReduceLROnPlateau scheduler
            def configure_optimizers(self):
                optimizer1 = Adam(...)
                optimizer2 = SGD(...)
                scheduler1 = ReduceLROnPlateau(optimizer1, ...)
                scheduler2 = LambdaLR(optimizer2, ...)
                return (
                    {
                        "optimizer": optimizer1,
                        "lr_scheduler": {
                            "scheduler": scheduler1,
                            "monitor": "metric_to_track",
                        },
                    },
                    {"optimizer": optimizer2, "lr_scheduler": scheduler2},
                )

        Metrics can be made available to monitor by simply logging it using
        ``self.log('metric_to_track', metric_val)`` in your :class:`~lightning.pytorch.core.LightningModule`.

        Note:
            Some things to know:

            - Lightning calls ``.backward()`` and ``.step()`` automatically in case of automatic optimization.
            - If a learning rate scheduler is specified in ``configure_optimizers()`` with key
              ``"interval"`` (default "epoch") in the scheduler configuration, Lightning will call
              the scheduler's ``.step()`` method automatically in case of automatic optimization.
            - If you use 16-bit precision (``precision=16``), Lightning will automatically handle the optimizer.
            - If you use :class:`torch.optim.LBFGS`, Lightning handles the closure function automatically for you.
            - If you use multiple optimizers, you will have to switch to 'manual optimization' mode and step them
              yourself.
            - If you need to control how often the optimizer steps, override the :meth:`optimizer_step` hook.

        """
        rank_zero_warn("`configure_optimizers` must be implemented to be used with the Lightning Trainer")

    def manual_backward(self, loss: Tensor, *args: Any, **kwargs: Any) -> None:
        """Call this directly from your :meth:`training_step` when doing optimizations manually. By using this,
        Lightning can ensure that all the proper scaling gets applied when using mixed precision.

        See :ref:`manual optimization<common/optimization:Manual optimization>` for more examples.

        Example::

            def training_step(...):
                opt = self.optimizers()
                loss = ...
                opt.zero_grad()
                # automatically applies scaling, etc...
                self.manual_backward(loss)
                opt.step()

        Args:
            loss: The tensor on which to compute gradients. Must have a graph attached.
            *args: Additional positional arguments to be forwarded to :meth:`~torch.Tensor.backward`
            **kwargs: Additional keyword arguments to be forwarded to :meth:`~torch.Tensor.backward`

        """
        if self._fabric:
            self._fabric.backward(loss, *args, **kwargs)
        else:
            self._verify_is_manual_optimization("manual_backward")
            self.trainer.strategy.backward(loss, None, *args, **kwargs)

    def backward(self, loss: Tensor, *args: Any, **kwargs: Any) -> None:
        """Called to perform backward on the loss returned in :meth:`training_step`. Override this hook with your own
        implementation if you need to.

        Args:
            loss: The loss tensor returned by :meth:`training_step`. If gradient accumulation is used, the loss here
                holds the normalized value (scaled by 1 / accumulation steps).

        Example::

            def backward(self, loss):
                loss.backward()

        """
        if self._fabric:
            self._fabric.backward(loss, *args, **kwargs)
        else:
            loss.backward(*args, **kwargs)

    def toggle_optimizer(self, optimizer: Union[Optimizer, LightningOptimizer]) -> None:
        """Makes sure only the gradients of the current optimizer's parameters are calculated in the training step to
        prevent dangling gradients in multiple-optimizer setup.

        It works with :meth:`untoggle_optimizer` to make sure ``param_requires_grad_state`` is properly reset.

        Args:
            optimizer: The optimizer to toggle.

        """
        # Iterate over all optimizer parameters to preserve their `requires_grad` information
        # in case these are pre-defined during `configure_optimizers`
        param_requires_grad_state = {}
        for opt in self.trainer.optimizers:
            for group in opt.param_groups:
                for param in group["params"]:
                    # If a param already appear in param_requires_grad_state, continue
                    if param in param_requires_grad_state:
                        continue
                    param_requires_grad_state[param] = param.requires_grad
                    param.requires_grad = False

        # Then iterate over the current optimizer's parameters and set its `requires_grad`
        # properties accordingly
        for group in optimizer.param_groups:
            for param in group["params"]:
                param.requires_grad = param_requires_grad_state[param]
        self._param_requires_grad_state = param_requires_grad_state

    def untoggle_optimizer(self, optimizer: Union[Optimizer, LightningOptimizer]) -> None:
        """Resets the state of required gradients that were toggled with :meth:`toggle_optimizer`.

        Args:
            optimizer: The optimizer to untoggle.

        """
        for opt in self.trainer.optimizers:
            if not (opt is optimizer or (isinstance(optimizer, LightningOptimizer) and opt is optimizer.optimizer)):
                for group in opt.param_groups:
                    for param in group["params"]:
                        if param in self._param_requires_grad_state:
                            param.requires_grad = self._param_requires_grad_state[param]
        # save memory
        self._param_requires_grad_state = {}

    def clip_gradients(
        self,
        optimizer: Optimizer,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ) -> None:
        """Handles gradient clipping internally.

        Note:
            - Do not override this method. If you want to customize gradient clipping, consider using
              :meth:`configure_gradient_clipping` method.
            - For manual optimization (``self.automatic_optimization = False``), if you want to use
              gradient clipping, consider calling
              ``self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")``
              manually in the training step.

        Args:
            optimizer: Current optimizer being used.
            gradient_clip_val: The value at which to clip gradients.
            gradient_clip_algorithm: The gradient clipping algorithm to use. Pass ``gradient_clip_algorithm="value"``
                to clip by value, and ``gradient_clip_algorithm="norm"`` to clip by norm.

        """

        if self.fabric is not None:
            self.fabric.clip_gradients(
                self,
                optimizer,
                clip_val=gradient_clip_val if gradient_clip_algorithm == GradClipAlgorithmType.VALUE else None,
                max_norm=None if gradient_clip_algorithm == GradClipAlgorithmType.VALUE else gradient_clip_val,
            )
            return

        if gradient_clip_val is None:
            gradient_clip_val = self.trainer.gradient_clip_val or 0.0
        elif self.trainer.gradient_clip_val is not None and self.trainer.gradient_clip_val != gradient_clip_val:
            raise MisconfigurationException(
                f"You have set `Trainer(gradient_clip_val={self.trainer.gradient_clip_val!r})`"
                f" and have passed `clip_gradients(gradient_clip_val={gradient_clip_val!r})`."
                " Please use only one of them."
            )

        if gradient_clip_algorithm is None:
            gradient_clip_algorithm = self.trainer.gradient_clip_algorithm or "norm"
        else:
            gradient_clip_algorithm = gradient_clip_algorithm.lower()
            if (
                self.trainer.gradient_clip_algorithm is not None
                and self.trainer.gradient_clip_algorithm != gradient_clip_algorithm
            ):
                raise MisconfigurationException(
                    f"You have set `Trainer(gradient_clip_algorithm={self.trainer.gradient_clip_algorithm.value!r})`"
                    f" and have passed `clip_gradients(gradient_clip_algorithm={gradient_clip_algorithm!r})"
                    " Please use only one of them."
                )

        if not isinstance(gradient_clip_val, (int, float)):
            raise TypeError(f"`gradient_clip_val` should be an int or a float. Got {gradient_clip_val}.")

        if not GradClipAlgorithmType.supported_type(gradient_clip_algorithm.lower()):
            raise MisconfigurationException(
                f"`gradient_clip_algorithm` {gradient_clip_algorithm} is invalid."
                f" Allowed algorithms: {GradClipAlgorithmType.supported_types()}."
            )

        gradient_clip_algorithm = GradClipAlgorithmType(gradient_clip_algorithm)
        self.trainer.precision_plugin.unscale_gradients(optimizer)
        self.trainer.precision_plugin.clip_gradients(optimizer, gradient_clip_val, gradient_clip_algorithm)

    def configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ) -> None:
        """Perform gradient clipping for the optimizer parameters. Called before :meth:`optimizer_step`.

        Args:
            optimizer: Current optimizer being used.
            gradient_clip_val: The value at which to clip gradients. By default, value passed in Trainer
                will be available here.
            gradient_clip_algorithm: The gradient clipping algorithm to use. By default, value
                passed in Trainer will be available here.

        Example::

            def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
                # Implement your own custom logic to clip gradients
                # You can call `self.clip_gradients` with your settings:
                self.clip_gradients(
                    optimizer,
                    gradient_clip_val=gradient_clip_val,
                    gradient_clip_algorithm=gradient_clip_algorithm
                )

        """
        self.clip_gradients(
            optimizer, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm
        )

    def lr_scheduler_step(self, scheduler: LRSchedulerTypeUnion, metric: Optional[Any]) -> None:
        r"""Override this method to adjust the default way the :class:`~lightning.pytorch.trainer.trainer.Trainer` calls
        each scheduler. By default, Lightning calls ``step()`` and as shown in the example for each scheduler based on
        its ``interval``.

        Args:
            scheduler: Learning rate scheduler.
            metric: Value of the monitor used for schedulers like ``ReduceLROnPlateau``.

        Examples::

            # DEFAULT
            def lr_scheduler_step(self, scheduler, metric):
                if metric is None:
                    scheduler.step()
                else:
                    scheduler.step(metric)

            # Alternative way to update schedulers if it requires an epoch value
            def lr_scheduler_step(self, scheduler, metric):
                scheduler.step(epoch=self.current_epoch)

        """
        if metric is None:
            scheduler.step()  # type: ignore[call-arg]
        else:
            scheduler.step(metric)

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Union[Optimizer, LightningOptimizer],
        optimizer_closure: Optional[Callable[[], Any]] = None,
    ) -> None:
        r"""Override this method to adjust the default way the :class:`~lightning.pytorch.trainer.trainer.Trainer` calls
        the optimizer.

        By default, Lightning calls ``step()`` and ``zero_grad()`` as shown in the example.
        This method (and ``zero_grad()``) won't be called during the accumulation phase when
        ``Trainer(accumulate_grad_batches != 1)``. Overriding this hook has no benefit with manual optimization.

        Args:
            epoch: Current epoch
            batch_idx: Index of current batch
            optimizer: A PyTorch optimizer
            optimizer_closure: The optimizer closure. This closure must be executed as it includes the
                calls to ``training_step()``, ``optimizer.zero_grad()``, and ``backward()``.

        Examples::

            def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
                # Add your custom logic to run directly before `optimizer.step()`

                optimizer.step(closure=optimizer_closure)

                # Add your custom logic to run directly after `optimizer.step()`

        """
        optimizer.step(closure=optimizer_closure)

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: Optimizer) -> None:
        """Override this method to change the default behaviour of ``optimizer.zero_grad()``.

        Args:
            epoch: Current epoch
            batch_idx: Index of current batch
            optimizer: A PyTorch optimizer

        Examples::

            # DEFAULT
            def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
                optimizer.zero_grad()

            # Set gradients to `None` instead of zero to improve performance (not required on `torch>=2.0.0`).
            def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
                optimizer.zero_grad(set_to_none=True)

        See :meth:`torch.optim.Optimizer.zero_grad` for the explanation of the above example.

        """
        optimizer.zero_grad()

    def freeze(self) -> None:
        r"""Freeze all params for inference.

        Example::

            model = MyLightningModule(...)
            model.freeze()

        """
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    def unfreeze(self) -> None:
        """Unfreeze all parameters for training.

        .. code-block:: python

            model = MyLightningModule(...)
            model.unfreeze()

        """
        for param in self.parameters():
            param.requires_grad = True

        self.train()

    def _verify_is_manual_optimization(self, fn_name: str) -> None:
        if self.automatic_optimization:
            raise MisconfigurationException(
                f"to use {fn_name}, please disable automatic optimization:"
                " set model property `automatic_optimization` as False"
            )

    @torch.no_grad()
    def to_onnx(self, file_path: Union[str, Path, BytesIO], input_sample: Optional[Any] = None, **kwargs: Any) -> None:
        """Saves the model in ONNX format.

        Args:
            file_path: The path of the file the onnx model should be saved to.
            input_sample: An input for tracing. Default: None (Use self.example_input_array)
            **kwargs: Will be passed to torch.onnx.export function.

        Example::

            class SimpleModel(LightningModule):
                def __init__(self):
                    super().__init__()
                    self.l1 = torch.nn.Linear(in_features=64, out_features=4)

                def forward(self, x):
                    return torch.relu(self.l1(x.view(x.size(0), -1)

            model = SimpleModel()
            input_sample = torch.randn(1, 64)
            model.to_onnx("export.onnx", input_sample, export_params=True)

        """
        if not _ONNX_AVAILABLE:
            raise ModuleNotFoundError(f"`{type(self).__name__}.to_onnx()` requires `onnx` to be installed.")

        mode = self.training

        if input_sample is None:
            if self.example_input_array is None:
                raise ValueError(
                    "Could not export to ONNX since neither `input_sample` nor"
                    " `model.example_input_array` attribute is set."
                )
            input_sample = self.example_input_array

        input_sample = self._on_before_batch_transfer(input_sample)
        input_sample = self._apply_batch_transfer_handler(input_sample)

        file_path = str(file_path) if isinstance(file_path, Path) else file_path
        # PyTorch (2.5) declares file_path to be str | PathLike[Any] | None, but
        #               BytesIO does work, too.
        torch.onnx.export(self, input_sample, file_path, **kwargs)  # type: ignore
        self.train(mode)

    @torch.no_grad()
    def to_torchscript(
        self,
        file_path: Optional[Union[str, Path]] = None,
        method: Optional[str] = "script",
        example_inputs: Optional[Any] = None,
        **kwargs: Any,
    ) -> Union[ScriptModule, dict[str, ScriptModule]]:
        """By default compiles the whole model to a :class:`~torch.jit.ScriptModule`. If you want to use tracing,
        please provided the argument ``method='trace'`` and make sure that either the `example_inputs` argument is
        provided, or the model has :attr:`example_input_array` set. If you would like to customize the modules that are
        scripted you should override this method. In case you want to return multiple modules, we recommend using a
        dictionary.

        Args:
            file_path: Path where to save the torchscript. Default: None (no file saved).
            method: Whether to use TorchScript's script or trace method. Default: 'script'
            example_inputs: An input to be used to do tracing when method is set to 'trace'.
              Default: None (uses :attr:`example_input_array`)
            **kwargs: Additional arguments that will be passed to the :func:`torch.jit.script` or
              :func:`torch.jit.trace` function.

        Note:
            - Requires the implementation of the
              :meth:`~lightning.pytorch.core.LightningModule.forward` method.
            - The exported script will be set to evaluation mode.
            - It is recommended that you install the latest supported version of PyTorch
              to use this feature without limitations. See also the :mod:`torch.jit`
              documentation for supported features.

        Example::

            class SimpleModel(LightningModule):
                def __init__(self):
                    super().__init__()
                    self.l1 = torch.nn.Linear(in_features=64, out_features=4)

                def forward(self, x):
                    return torch.relu(self.l1(x.view(x.size(0), -1)))

            model = SimpleModel()
            model.to_torchscript(file_path="model.pt")

            torch.jit.save(model.to_torchscript(
                file_path="model_trace.pt", method='trace', example_inputs=torch.randn(1, 64))
            )

        Return:
            This LightningModule as a torchscript, regardless of whether `file_path` is
            defined or not.

        """
        mode = self.training

        if method == "script":
            with _jit_is_scripting():
                torchscript_module = torch.jit.script(self.eval(), **kwargs)
        elif method == "trace":
            # if no example inputs are provided, try to see if model has example_input_array set
            if example_inputs is None:
                if self.example_input_array is None:
                    raise ValueError(
                        "Choosing method=`trace` requires either `example_inputs`"
                        " or `model.example_input_array` to be defined."
                    )
                example_inputs = self.example_input_array

            # automatically send example inputs to the right device and use trace
            example_inputs = self._on_before_batch_transfer(example_inputs)
            example_inputs = self._apply_batch_transfer_handler(example_inputs)
            with _jit_is_scripting():
                torchscript_module = torch.jit.trace(func=self.eval(), example_inputs=example_inputs, **kwargs)
        else:
            raise ValueError(f"The 'method' parameter only supports 'script' or 'trace', but value given was: {method}")

        self.train(mode)

        if file_path is not None:
            fs = get_filesystem(file_path)
            with fs.open(file_path, "wb") as f:
                torch.jit.save(torchscript_module, f)

        return torchscript_module

    @_restricted_classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[_PATH, IO],
        map_location: _MAP_LOCATION_TYPE = None,
        hparams_file: Optional[_PATH] = None,
        strict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Self:
        r"""Primary way of loading a model from a checkpoint. When Lightning saves a checkpoint it stores the arguments
        passed to ``__init__``  in the checkpoint under ``"hyper_parameters"``.

        Any arguments specified through \*\*kwargs will override args stored in ``"hyper_parameters"``.

        Args:
            checkpoint_path: Path to checkpoint. This can also be a URL, or file-like object
            map_location:
                If your checkpoint saved a GPU model and you now load on CPUs
                or a different number of GPUs, use this to map to the new setup.
                The behaviour is the same as in :func:`torch.load`.
            hparams_file: Optional path to a ``.yaml`` or ``.csv`` file with hierarchical structure
                as in this example::

                    drop_prob: 0.2
                    dataloader:
                        batch_size: 32

                You most likely won't need this since Lightning will always save the hyperparameters
                to the checkpoint.
                However, if your checkpoint weights don't have the hyperparameters saved,
                use this method to pass in a ``.yaml`` file with the hparams you'd like to use.
                These will be converted into a :class:`~dict` and passed into your
                :class:`LightningModule` for use.

                If your model's ``hparams`` argument is :class:`~argparse.Namespace`
                and ``.yaml`` file has hierarchical structure, you need to refactor your model to treat
                ``hparams`` as :class:`~dict`.
            strict: Whether to strictly enforce that the keys in :attr:`checkpoint_path` match the keys
                returned by this module's state dict. Defaults to ``True`` unless ``LightningModule.strict_loading`` is
                set, in which case it defaults to the value of ``LightningModule.strict_loading``.
            \**kwargs: Any extra keyword args needed to init the model. Can also be used to override saved
                hyperparameter values.

        Return:
            :class:`LightningModule` instance with loaded weights and hyperparameters (if available).

        Note:
            ``load_from_checkpoint`` is a **class** method. You should use your :class:`LightningModule`
            **class** to call it instead of the :class:`LightningModule` instance, or a
            ``TypeError`` will be raised.

        Note:
            To ensure all layers can be loaded from the checkpoint, this function will call
            :meth:`~lightning.pytorch.core.hooks.ModelHooks.configure_model` directly after instantiating the
            model if this hook is overridden in your LightningModule. However, note that ``load_from_checkpoint`` does
            not support loading sharded checkpoints, and you may run out of memory if the model is too large. In this
            case, consider loading through the Trainer via ``.fit(ckpt_path=...)``.

        Example::

            # load weights without mapping ...
            model = MyLightningModule.load_from_checkpoint('path/to/checkpoint.ckpt')

            # or load weights mapping all weights from GPU 1 to GPU 0 ...
            map_location = {'cuda:1':'cuda:0'}
            model = MyLightningModule.load_from_checkpoint(
                'path/to/checkpoint.ckpt',
                map_location=map_location
            )

            # or load weights and hyperparameters from separate files.
            model = MyLightningModule.load_from_checkpoint(
                'path/to/checkpoint.ckpt',
                hparams_file='/path/to/hparams_file.yaml'
            )

            # override some of the params with new values
            model = MyLightningModule.load_from_checkpoint(
                PATH,
                num_layers=128,
                pretrained_ckpt_path=NEW_PATH,
            )

            # predict
            pretrained_model.eval()
            pretrained_model.freeze()
            y_hat = pretrained_model(x)

        """
        loaded = _load_from_checkpoint(
            cls,
            checkpoint_path,
            map_location,
            hparams_file,
            strict,
            **kwargs,
        )
        return cast(Self, loaded)

    @override
    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_trainer"] = None
        return state


@contextmanager
def _jit_is_scripting() -> Generator:
    """Workaround for https://github.com/pytorch/pytorch/issues/67146."""
    LightningModule._jit_is_scripting = True
    try:
        yield
    finally:
        LightningModule._jit_is_scripting = False


class _TrainerFabricShim:
    """Intercepts attribute access on LightningModule's trainer reference and redirects it to the Fabric object."""

    def __init__(self, fabric: lf.Fabric) -> None:
        super().__init__()
        self._fabric = fabric

    def __getattr__(self, item: Any) -> Any:
        try:
            return getattr(self._fabric, item)
        except AttributeError:
            raise AttributeError(
                f"Your LightningModule code tried to access `self.trainer.{item}` but this attribute is not available"
                f" when using Fabric with a LightningModule."
            )
