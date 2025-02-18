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
from contextlib import nullcontext
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Union

import torch
import torch.distributed
from lightning_utilities.core.rank_zero import rank_zero_only as utils_rank_zero_only
from torch import Tensor
from torch.nn import Module
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim.optimizer import Optimizer
from typing_extensions import override

import lightning.pytorch as pl
from lightning.fabric.plugins import CheckpointIO, ClusterEnvironment
from lightning.fabric.plugins.collectives.torch_collective import default_pg_timeout
from lightning.fabric.strategies import _StrategyRegistry
from lightning.fabric.utilities.distributed import (
    _distributed_is_initialized,
    _get_default_process_group_backend_for_device,
    _init_dist_connection,
    _sync_ddp_if_available,
)
from lightning.fabric.utilities.distributed import group as _group
from lightning.fabric.utilities.imports import _IS_WINDOWS
from lightning.fabric.utilities.optimizer import _optimizers_to_device
from lightning.fabric.utilities.seed import reset_seed
from lightning.fabric.utilities.types import ReduceOp
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.overrides.distributed import _register_ddp_comm_hook, _sync_module_states, prepare_for_backward
from lightning.pytorch.plugins.precision import Precision
from lightning.pytorch.strategies.launchers import _MultiProcessingLauncher, _SubprocessScriptLauncher
from lightning.pytorch.strategies.parallel import ParallelStrategy
from lightning.pytorch.strategies.strategy import TBroadcast, _ForwardRedirection
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.exceptions import _augment_message
from lightning.pytorch.utilities.rank_zero import rank_zero_deprecation, rank_zero_info, rank_zero_only

if TYPE_CHECKING:
    from torch.distributed.algorithms.model_averaging.averagers import ModelAverager

log = logging.getLogger(__name__)

_DDP_FORK_ALIASES = (
    "ddp_fork",
    "ddp_fork_find_unused_parameters_false",
    "ddp_fork_find_unused_parameters_true",
    "ddp_notebook",
    "ddp_notebook_find_unused_parameters_false",
    "ddp_notebook_find_unused_parameters_true",
)


class DDPStrategy(ParallelStrategy):
    """Strategy for multi-process single-device training on one or multiple nodes."""

    def __init__(
        self,
        accelerator: Optional["pl.accelerators.Accelerator"] = None,
        parallel_devices: Optional[list[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[Precision] = None,
        ddp_comm_state: Optional[object] = None,
        ddp_comm_hook: Optional[Callable] = None,
        ddp_comm_wrapper: Optional[Callable] = None,
        model_averaging_period: Optional[int] = None,
        process_group_backend: Optional[str] = None,
        timeout: Optional[timedelta] = default_pg_timeout,
        start_method: Literal["popen", "spawn", "fork", "forkserver"] = "popen",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
        )
        log.debug(f"{self.__class__.__name__}: initializing DDP strategy")
        self._forward_redirection = _DDPForwardRedirection()
        self._num_nodes = 1
        self._ddp_kwargs = kwargs
        self._ddp_comm_state = ddp_comm_state
        self._ddp_comm_hook = ddp_comm_hook
        self._ddp_comm_wrapper = ddp_comm_wrapper
        self._model_averaging_period = model_averaging_period
        self._model_averager: Optional[ModelAverager] = None
        self._process_group_backend: Optional[str] = process_group_backend
        self._timeout: Optional[timedelta] = timeout
        self._start_method = start_method

    @property
    def is_distributed(self) -> bool:  # pragma: no-cover
        """Legacy property kept for backwards compatibility."""
        rank_zero_deprecation(
            f"`{type(self).__name__}.is_distributed` is deprecated. Use is discouraged.", stacklevel=6
        )
        return True

    @property
    @override
    def root_device(self) -> torch.device:
        assert self.parallel_devices is not None
        return self.parallel_devices[self.local_rank]

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @num_nodes.setter
    def num_nodes(self, num_nodes: int) -> None:
        # note that world ranks is related to num_nodes, when resetting it, need to reset world ranks
        self._num_nodes = num_nodes

    @property
    def num_processes(self) -> int:
        return len(self.parallel_devices) if self.parallel_devices is not None else 0

    @property
    @override
    def distributed_sampler_kwargs(self) -> dict[str, Any]:
        return {"num_replicas": (self.num_nodes * self.num_processes), "rank": self.global_rank}

    @property
    def process_group_backend(self) -> Optional[str]:
        return self._process_group_backend

    @override
    def _configure_launcher(self) -> None:
        assert self.cluster_environment is not None
        if self._start_method == "popen":
            self._launcher = _SubprocessScriptLauncher(self.cluster_environment, self.num_processes, self.num_nodes)
        else:
            self._launcher = _MultiProcessingLauncher(self, start_method=self._start_method)

    @override
    def setup_environment(self) -> None:
        super().setup_environment()
        self.setup_distributed()

    @override
    def setup(self, trainer: "pl.Trainer") -> None:
        assert self.accelerator is not None
        self.accelerator.setup(trainer)

        trainer_fn = trainer.state.fn
        assert self.model is not None
        if trainer_fn == TrainerFn.FITTING and self._layer_sync:
            self.model = self._layer_sync.apply(self.model)

        self.precision_plugin.convert_module(self.model)
        self.model_to_device()

        if trainer_fn == TrainerFn.FITTING:
            # do not wrap with DDP if not fitting as there's no gradients to reduce
            self.configure_ddp()

            # set up optimizers after the wrapped module has been moved to the device
            self.setup_optimizers(trainer)
        else:
            # we need to manually synchronize the module's states since we aren't using the DDP wrapper
            _sync_module_states(self.model)
        self.setup_precision_plugin()
        if trainer_fn == TrainerFn.FITTING:
            _optimizers_to_device(self.optimizers, self.root_device)

            import torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook as post_localSGD

            if isinstance(self._ddp_comm_state, post_localSGD.PostLocalSGDState):
                self._enable_model_averaging()

    @override
    def _setup_model(self, model: Module) -> DistributedDataParallel:
        """Wraps the model into a :class:`~torch.nn.parallel.distributed.DistributedDataParallel` module."""
        device_ids = self.determine_ddp_device_ids()
        log.debug(f"setting up DDP model with device ids: {device_ids}, kwargs: {self._ddp_kwargs}")
        # https://pytorch.org/docs/stable/notes/cuda.html#id5
        ctx = self._create_stream_context(device_ids=device_ids)
        with ctx:
            return DistributedDataParallel(module=model, device_ids=device_ids, **self._ddp_kwargs)

    def setup_distributed(self) -> None:
        log.debug(f"{self.__class__.__name__}: setting up distributed...")
        reset_seed()
        self.set_world_ranks()
        self._process_group_backend = self._get_process_group_backend()
        assert self.cluster_environment is not None
        _init_dist_connection(self.cluster_environment, self._process_group_backend, timeout=self._timeout)

    def _get_process_group_backend(self) -> str:
        return self._process_group_backend or _get_default_process_group_backend_for_device(self.root_device)

    def set_world_ranks(self) -> None:
        if self.cluster_environment is not None:
            self.cluster_environment.set_global_rank(self.node_rank * self.num_processes + self.local_rank)
            self.cluster_environment.set_world_size(self.num_nodes * self.num_processes)
        # `LightningEnvironment.set_global_rank` will do this too, but we cannot rely on that implementation detail
        # additionally, for some implementations, the setter is a no-op, so it's safer to access the getter
        rank_zero_only.rank = utils_rank_zero_only.rank = self.global_rank

    def _register_ddp_hooks(self) -> None:
        log.debug(f"{self.__class__.__name__}: registering ddp hooks")
        # currently, DDP communication hooks only work with NCCL backend and SPSD (single process single device) mode
        # https://github.com/pytorch/pytorch/blob/v1.8.0/torch/nn/parallel/distributed.py#L1080-L1084
        if self.root_device.type == "cuda":
            assert isinstance(self.model, DistributedDataParallel)
            _register_ddp_comm_hook(
                model=self.model,
                ddp_comm_state=self._ddp_comm_state,
                ddp_comm_hook=self._ddp_comm_hook,
                ddp_comm_wrapper=self._ddp_comm_wrapper,
            )

    def _enable_model_averaging(self) -> None:
        log.debug(f"{self.__class__.__name__}: reinitializing optimizers with post localSGD")
        if self._model_averaging_period is None:  # type: ignore[no-untyped-def]
            raise ValueError(
                "Post-localSGD algorithm is used, but model averaging period is not provided to DDP strategy."
            )
        from torch.distributed.optim import DistributedOptimizer, PostLocalSGDOptimizer, ZeroRedundancyOptimizer

        for optimizer in self.optimizers:
            if isinstance(optimizer, LightningOptimizer):
                optimizer = optimizer._optimizer

            is_distributed_optimizer = isinstance(optimizer, DistributedOptimizer) if not _IS_WINDOWS else False
            if isinstance(optimizer, (ZeroRedundancyOptimizer, PostLocalSGDOptimizer)) or is_distributed_optimizer:
                raise ValueError(
                    f"Currently model averaging cannot work with a distributed optimizer of type "
                    f"{optimizer.__class__.__name__}."
                )

        assert self._ddp_comm_state is not None
        self._model_averager = torch.distributed.algorithms.model_averaging.averagers.PeriodicModelAverager(
            period=self._model_averaging_period, warmup_steps=self._ddp_comm_state.start_localSGD_iter
        )

    @override
    def optimizer_step(
        self,
        optimizer: Optimizer,
        closure: Callable[[], Any],
        model: Optional[Union["pl.LightningModule", Module]] = None,
        **kwargs: Any,
    ) -> Any:
        """Performs the actual optimizer step.

        Args:
            optimizer: the optimizer performing the step
            closure: closure calculating the loss value
            model: reference to the model, optionally defining optimizer step related hooks
            **kwargs: Any extra arguments to ``optimizer.step``

        """
        optimizer_output = super().optimizer_step(optimizer, closure, model, **kwargs)

        if self._model_averager is None:
            return optimizer_output

        params = [param for group in optimizer.param_groups for param in group["params"] if param.grad is not None]
        self._model_averager.average_parameters(iter(params))

        return optimizer_output

    def configure_ddp(self) -> None:
        log.debug(f"{self.__class__.__name__}: configuring DistributedDataParallel")
        assert isinstance(self.model, pl.LightningModule)
        self.model = self._setup_model(self.model)
        self._register_ddp_hooks()

    def determine_ddp_device_ids(self) -> Optional[list[int]]:
        if self.root_device.type == "cpu":
            return None
        return [self.root_device.index]

    @override
    def barrier(self, *args: Any, **kwargs: Any) -> None:
        if not _distributed_is_initialized():
            return

        if torch.distributed.get_backend() == "nccl":
            torch.distributed.barrier(device_ids=self.determine_ddp_device_ids())
        else:
            torch.distributed.barrier()

    @override
    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        if not _distributed_is_initialized():
            return obj

        obj = [obj]
        torch.distributed.broadcast_object_list(obj, src, group=_group.WORLD)
        return obj[0]

    @override
    def pre_backward(self, closure_loss: Tensor) -> None:
        """Run before precision plugin executes backward."""
        if not isinstance(self.model, DistributedDataParallel):
            return
        assert self.lightning_module is not None
        if not self.lightning_module.automatic_optimization:
            prepare_for_backward(self.model, closure_loss)

    @override
    def model_to_device(self) -> None:
        log.debug(f"{self.__class__.__name__}: moving model to device [{self.root_device}]...")
        assert self.model is not None
        self.model.to(self.root_device)

    @override
    def reduce(
        self, tensor: Tensor, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = "mean"
    ) -> Tensor:
        """Reduces a tensor from several distributed processes to one aggregated tensor.

        Args:
            tensor: the tensor to sync and reduce
            group: the process group to gather results from. Defaults to all processes (world)
            reduce_op: the reduction operation. Defaults to 'mean'/'avg'.
                Can also be a string 'sum' to calculate the sum during reduction.

        Return:
            reduced value, except when the input was not a tensor the output remains is unchanged

        """
        if isinstance(tensor, Tensor):
            return _sync_ddp_if_available(tensor, group, reduce_op=reduce_op)
        return tensor

    @classmethod
    @override
    def register_strategies(cls, strategy_registry: _StrategyRegistry) -> None:
        entries = (
            ("ddp", "popen"),
            ("ddp_spawn", "spawn"),
            ("ddp_fork", "fork"),
            ("ddp_notebook", "fork"),
        )
        for name, start_method in entries:
            strategy_registry.register(
                name,
                cls,
                description=f"DDP strategy with `start_method` '{start_method}'",
                start_method=start_method,
            )

        entries = (
            ("ddp_find_unused_parameters_false", False, "popen"),
            ("ddp_find_unused_parameters_true", True, "popen"),
            ("ddp_spawn_find_unused_parameters_false", False, "spawn"),
            ("ddp_spawn_find_unused_parameters_true", True, "spawn"),
            ("ddp_fork_find_unused_parameters_false", False, "fork"),
            ("ddp_fork_find_unused_parameters_true", True, "fork"),
            ("ddp_notebook_find_unused_parameters_false", False, "fork"),
            ("ddp_notebook_find_unused_parameters_true", True, "fork"),
        )
        for name, fup, start_method in entries:
            strategy_registry.register(
                name,
                cls,
                description=f"DDP strategy with `find_unused_parameters` as {fup} and `start_method` '{start_method}'",
                find_unused_parameters=fup,
                start_method=start_method,
            )

    @override
    def on_exception(self, exception: BaseException) -> None:
        _augment_message(
            exception,
            pattern=".*Expected to have finished reduction in the prior iteration.*",
            new_message=(
                "It looks like your LightningModule has parameters that were not used in producing the loss returned"
                " by training_step. If this is intentional, you must enable the detection of unused parameters in DDP,"
                " either by setting the string value `strategy='ddp_find_unused_parameters_true'`"
                " or by setting the flag in the strategy with `strategy=DDPStrategy(find_unused_parameters=True)`."
            ),
        )

    @override
    def teardown(self) -> None:
        log.debug(f"{self.__class__.__name__}: tearing down strategy")

        pl_module = self.lightning_module
        if isinstance(self.model, DistributedDataParallel):
            if not self.model.static_graph and self.model._get_ddp_logging_data().get("can_set_static_graph"):
                rank_zero_info(
                    "Your model can run with static graph optimizations. For future training runs, we suggest you"
                    f" pass `Trainer(..., strategy={self.__class__.__name__}(static_graph=True))` to enable them."
                )
            # unwrap model
            self.model = pl_module

        if (
            pl_module is not None
            # `self.lightning_module._trainer` can be None if teardown gets called on an exception before
            # the trainer gets set on the LightningModule
            and pl_module._trainer is not None
            and pl_module._trainer.state.fn == TrainerFn.FITTING
            and self._layer_sync
        ):
            assert self.model is not None
            self.model = self._layer_sync.revert(self.model)

        super().teardown()

    def _create_stream_context(self, device_ids=None):  # type: ignore[no-untyped-def]
        """Create a stream context for the current device, if supported."""

        torch_lib = getattr(torch, self.root_device.type)
        # Check if the device type supports streams and has the necessary attributes.
        if hasattr(torch_lib, "Stream") and hasattr(torch_lib, "stream") and device_ids is not None:
            stream = torch_lib.Stream()
            ctx = torch_lib.stream(stream)
        else:
            ctx = nullcontext()
        return ctx


class _DDPForwardRedirection(_ForwardRedirection):
    @override
    def on_after_inner_forward(self, wrapper_module: Module, original_module: "pl.LightningModule") -> None:
        # In manual_optimization, we need to prevent DDP reducer as
        # it is done manually in `LightningModule.manual_backward`
        if isinstance(wrapper_module, DistributedDataParallel) and not original_module.automatic_optimization:
            wrapper_module.require_backward_grad_sync = False

    @override
    def on_after_outer_forward(self, wrapper_module: Module, original_module: "pl.LightningModule") -> None:
        if isinstance(wrapper_module, DistributedDataParallel) and not original_module.automatic_optimization:
            wrapper_module.require_backward_grad_sync = True
