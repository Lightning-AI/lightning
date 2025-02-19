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
import shutil
import warnings
from collections.abc import Generator
from contextlib import AbstractContextManager, ExitStack, nullcontext
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Optional,
    Union,
)

import torch
from lightning_utilities.core.imports import RequirementCache
from lightning_utilities.core.rank_zero import rank_zero_only as utils_rank_zero_only
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing_extensions import TypeGuard, override

from lightning.fabric.accelerators import Accelerator
from lightning.fabric.plugins import CheckpointIO, ClusterEnvironment, Precision
from lightning.fabric.plugins.collectives.torch_collective import default_pg_timeout
from lightning.fabric.plugins.precision.fsdp import FSDPPrecision
from lightning.fabric.strategies.launchers.subprocess_script import _SubprocessScriptLauncher
from lightning.fabric.strategies.parallel import ParallelStrategy
from lightning.fabric.strategies.registry import _StrategyRegistry
from lightning.fabric.strategies.strategy import (
    TBroadcast,
    _apply_filter,
    _BackwardSyncControl,
    _Sharded,
    _validate_keys_for_strict_loading,
)
from lightning.fabric.utilities.distributed import (
    ReduceOp,
    _distributed_is_initialized,
    _get_default_process_group_backend_for_device,
    _init_dist_connection,
    _sync_ddp_if_available,
)
from lightning.fabric.utilities.distributed import group as _group
from lightning.fabric.utilities.imports import (
    _TORCH_GREATER_EQUAL_2_2,
    _TORCH_GREATER_EQUAL_2_3,
)
from lightning.fabric.utilities.init import _has_meta_device_parameters_or_buffers
from lightning.fabric.utilities.load import _METADATA_FILENAME, _lazy_load, _materialize_tensors, _move_state_into
from lightning.fabric.utilities.rank_zero import rank_zero_deprecation, rank_zero_only, rank_zero_warn
from lightning.fabric.utilities.seed import reset_seed
from lightning.fabric.utilities.types import _PATH, _Stateful

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    _POLICY = Union[set[type[Module]], Callable[[Module, bool, int], bool], ModuleWrapPolicy]
    _SHARDING_STRATEGY = Union[ShardingStrategy, Literal["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"]]


_FSDP_ALIASES = ("fsdp", "fsdp_cpu_offload")

# TODO: Switch to new state-dict APIs
warnings.filterwarnings("ignore", category=FutureWarning, message=".*FSDP.state_dict_type.*")  # from torch >= 2.4


class FSDPStrategy(ParallelStrategy, _Sharded):
    r"""Strategy for Fully Sharded Data Parallel provided by torch.distributed.

    Fully Sharded Training shards the entire model across all available GPUs, allowing you to scale model
    size, whilst using efficient communication to reduce overhead. In practice, this means we can remain
    at parity with PyTorch DDP, whilst scaling our model sizes dramatically. The technique is similar
    to ZeRO-Stage 3.

    For more information check out
    `this blogpost <https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api>`__.

    Defaults have been set and options have been exposed, but may require configuration
    based on your level of memory/speed efficiency. We suggest having a look at
    `this tutorial <https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html>`__ for more information.

    Arguments:
        cpu_offload: See ``cpu_offload`` parameter in :class:`torch.distributed.fsdp.FullyShardedDataParallel`.
        mixed_precision: See ``mixed_precision`` parameter in :class:`torch.distributed.fsdp.FullyShardedDataParallel`.
        auto_wrap_policy: Same as ``auto_wrap_policy`` parameter in
            :class:`torch.distributed.fsdp.FullyShardedDataParallel`. For convenience, this also accepts a set of the
            layer classes to wrap.
        activation_checkpointing: Deprecated. Use ``activation_checkpointing_policy``.
        activation_checkpointing_policy: Same as ``auto_wrap_policy`` parameter in
            :class:`torch.distributed.fsdp.FullyShardedDataParallel` but used when selecting the modules for which you
            want to enable activation checkpointing. Enabling this can free up a significant amount of memory at the
            cost of speed since activations in these layers need to be recomputed during backpropagation. For
            convenience, this also accepts a set of the layer classes to wrap.
        sharding_strategy: Select whether to shard model parameters, gradients, optimizer states, or a combination of
            them. Available values are:

            - ``"FULL_SHARD"``: Shards model parameters, gradients, and optimizer states (default).
            - ``"SHARD_GRAD_OP"``: Shards gradients and optimizer states only. Model parameters get replicated.
            - ``"NO_SHARD"``: No sharding (identical to regular DDP).
            - ``"HYBRID_SHARD"``: Shards model parameters, gradients, and optimizer states within a single machine, but
              replicates across machines. See also the `device_mesh` parameter below.

            Also accepts a :class:`torch.distributed.fsdp.ShardingStrategy` enum value.

        device_mesh: A tuple `(replication size, sharding size)` that defines over how many devices to shard and
            replicate the model. The product of the two numbers must equal the world size. Only valid in combination
            with the `HYBRID_SHARD` sharding strategy.

        state_dict_type: The format in which the state of the model and optimizers gets saved into the checkpoint.

            - ``"full"``: The full weights and optimizer states get assembled on rank 0 and saved to a single file.
            - ``"sharded"``: Each rank saves its shard of weights and optimizer states to a file. The checkpoint is
              a folder with as many files as the world size.

        \**kwargs: See available parameters in :class:`torch.distributed.fsdp.FullyShardedDataParallel`.

    """

    def __init__(
        self,
        accelerator: Optional[Accelerator] = None,
        parallel_devices: Optional[list[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        precision: Optional[Precision] = None,
        process_group_backend: Optional[str] = None,
        timeout: Optional[timedelta] = default_pg_timeout,
        cpu_offload: Union[bool, "CPUOffload", None] = None,
        mixed_precision: Optional["MixedPrecision"] = None,
        auto_wrap_policy: Optional["_POLICY"] = None,
        activation_checkpointing: Optional[Union[type[Module], list[type[Module]]]] = None,
        activation_checkpointing_policy: Optional["_POLICY"] = None,
        sharding_strategy: "_SHARDING_STRATEGY" = "FULL_SHARD",
        state_dict_type: Literal["full", "sharded"] = "sharded",
        device_mesh: Optional[Union[tuple[int], "DeviceMesh"]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            precision=precision,
        )
        self._num_nodes = 1
        self._process_group_backend: Optional[str] = process_group_backend
        self._timeout: Optional[timedelta] = timeout
        self._backward_sync_control = _FSDPBackwardSyncControl()
        self._fsdp_kwargs = _auto_wrap_policy_kwargs(auto_wrap_policy, kwargs)

        # Enables joint setup of model and optimizer, multiple optimizer param groups, and `torch.compile()`
        self._fsdp_kwargs.setdefault("use_orig_params", True)

        if device_mesh is not None:
            if not _TORCH_GREATER_EQUAL_2_2:
                raise ValueError("The `device_mesh` argument is only supported in torch >= 2.2.")
            self._fsdp_kwargs["device_mesh"] = device_mesh

        self._activation_checkpointing_kwargs = _activation_checkpointing_kwargs(
            activation_checkpointing, activation_checkpointing_policy
        )
        self._state_dict_type = state_dict_type
        self.sharding_strategy = _init_sharding_strategy(sharding_strategy, self._fsdp_kwargs)
        self.cpu_offload = _init_cpu_offload(cpu_offload)
        self.mixed_precision = mixed_precision

    @property
    @override
    def checkpoint_io(self) -> CheckpointIO:
        raise NotImplementedError(f"The `{type(self).__name__}` does not use the `CheckpointIO` plugin interface.")

    @checkpoint_io.setter
    @override
    def checkpoint_io(self, io: CheckpointIO) -> None:
        raise NotImplementedError(f"The `{type(self).__name__}` does not support setting a `CheckpointIO` plugin.")

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

    @property
    def mixed_precision_config(self) -> Optional["MixedPrecision"]:
        if self.mixed_precision:
            return self.mixed_precision
        plugin = self.precision
        if isinstance(plugin, FSDPPrecision):
            return plugin.mixed_precision_config
        return None

    @property
    @override
    def precision(self) -> FSDPPrecision:
        plugin = self._precision
        if plugin is not None:
            assert isinstance(plugin, FSDPPrecision)
            return plugin
        return FSDPPrecision("32-true")

    @precision.setter
    @override
    def precision(self, precision: Optional[FSDPPrecision]) -> None:
        if precision is not None and not isinstance(precision, FSDPPrecision):
            raise TypeError(f"The FSDP strategy can only work with the `FSDPPrecision` plugin, found {precision}")
        self._precision = precision

    @override
    def _configure_launcher(self) -> None:
        assert self.cluster_environment is not None
        if not self.cluster_environment.creates_processes_externally:
            self._launcher = _SubprocessScriptLauncher(self.cluster_environment, self.num_processes, self.num_nodes)

    @override
    def setup_environment(self) -> None:
        super().setup_environment()
        self._setup_distributed()

        # if 'device_mesh' in the `_fsdp_kwargs` is provided as a tuple, update it into the `DeviceMesh` object here
        if isinstance(self._fsdp_kwargs.get("device_mesh"), tuple):
            from torch.distributed.device_mesh import init_device_mesh

            self._fsdp_kwargs["device_mesh"] = init_device_mesh("cuda", self._fsdp_kwargs["device_mesh"])

    @override
    def setup_module_and_optimizers(
        self, module: Module, optimizers: list[Optimizer], scheduler: Optional[_LRScheduler] = None
    ) -> tuple[Module, list[Optimizer], Optional[_LRScheduler]]:
        """Wraps the model into a :class:`~torch.distributed.fsdp.fully_sharded_data_parallel.FullyShardedDataParallel`
        module and sets `use_orig_params=True` to keep the reference to the original parameters in the optimizer."""
        use_orig_params = self._fsdp_kwargs.get("use_orig_params")
        if use_orig_params is False:
            raise ValueError(
                f"You set `{type(self).__name__}(use_orig_params=False)` but this is not supported when"
                " setting the model and optimizer up jointly. Either set it to `True` or set the objects"
                " up in this order: Create the model, call `setup_module`, create the optimizer,"
                " call `setup_optimizer`."
            )
        module = self.setup_module(module)
        return module, optimizers, scheduler

    @override
    def setup_module(self, module: Module) -> Module:
        """Wraps the model into a :class:`~torch.distributed.fsdp.fully_sharded_data_parallel.FullyShardedDataParallel`
        module."""
        from torch.distributed.fsdp import FullyShardedDataParallel

        if any(isinstance(mod, FullyShardedDataParallel) for mod in module.modules()):
            # The user has wrapped their submodules manually, don't apply the auto wrap policy.
            if _has_meta_device_parameters_or_buffers(module):
                rank_zero_warn(
                    "The model is already wrapped in `FSDP` but there are still parameters on the meta device."
                )
            if "auto_wrap_policy" in self._fsdp_kwargs:
                rank_zero_warn(
                    "A FSDP `auto_wrap_policy` is set, but the model is already wrapped. The policy will be ignored."
                )
                del self._fsdp_kwargs["auto_wrap_policy"]
        else:
            module = FullyShardedDataParallel(
                module=module,
                cpu_offload=self.cpu_offload,
                mixed_precision=self.mixed_precision_config,
                sharding_strategy=self.sharding_strategy,
                device_id=self.root_device.index,
                **self._fsdp_kwargs,
            )

        _move_torchmetrics_to_device(module, self.root_device)

        # activation checkpointing needs to be set up after wrapping the model
        _setup_activation_checkpointing(module, self._activation_checkpointing_kwargs)

        return module

    @override
    def setup_optimizer(self, optimizer: Optimizer) -> Optimizer:
        """Set up an optimizer for a model wrapped with FSDP.

        This setup method doesn't modify the optimizer or wrap the optimizer. The only thing it currently does is verify
        that the optimizer was created after the model was wrapped with :meth:`setup_module` with a reference to the
        flattened parameters.

        """
        if self._fsdp_kwargs.get("use_orig_params"):
            return super().setup_optimizer(optimizer)
        if not _optimizer_has_flat_params(optimizer):
            # We avoid this limitation by setting `use_orig_params=True`
            raise ValueError(
                "The optimizer does not seem to reference any FSDP parameters. HINT: Make sure to create the optimizer"
                " after setting up the model."
            )
        return optimizer

    @override
    def module_to_device(self, module: Module) -> None:
        pass

    @override
    def module_init_context(self, empty_init: Optional[bool] = None) -> AbstractContextManager:
        precision_init_ctx = self.precision.module_init_context()
        module_sharded_ctx = self.module_sharded_context()
        stack = ExitStack()
        if empty_init:
            # Materialization happens in `setup`. When modules get wrapped by FSDP, the sequence of operations is:
            # 1) materialize module 2) call `reset_parameters()` 3) shard the module.
            # These operations are applied to each submodule 'bottom up' in the module hierarchy.
            stack.enter_context(torch.device("meta"))
        stack.enter_context(precision_init_ctx)
        stack.enter_context(module_sharded_ctx)
        return stack

    @override
    def module_sharded_context(self) -> AbstractContextManager:
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
        from torch.distributed.fsdp.wrap import enable_wrap

        return enable_wrap(
            wrapper_cls=FullyShardedDataParallel,
            cpu_offload=self.cpu_offload,
            mixed_precision=self.mixed_precision_config,
            sharding_strategy=self.sharding_strategy,
            device_id=self.root_device.index,
            **self._fsdp_kwargs,
        )

    @override
    def all_reduce(
        self, tensor: Tensor, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = "mean"
    ) -> Tensor:
        if isinstance(tensor, Tensor):
            return _sync_ddp_if_available(tensor, group, reduce_op=reduce_op)
        return tensor

    @override
    def barrier(self, *args: Any, **kwargs: Any) -> None:
        if not _distributed_is_initialized():
            return
        if torch.distributed.get_backend() == "nccl":
            torch.distributed.barrier(device_ids=[self.root_device.index])
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
    def clip_gradients_norm(
        self,
        module: Module,
        optimizer: Optimizer,
        max_norm: Union[float, int],
        norm_type: Union[float, int] = 2.0,
        error_if_nonfinite: bool = True,
    ) -> Tensor:
        """Clip gradients by norm."""
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel

        if not isinstance(module, FullyShardedDataParallel):
            # the root must be wrapped
            raise TypeError(
                "Gradient clipping with FSDP is only possible if the module passed to"
                f" `{type(self).__name__}.clip_gradients_norm` is wrapped in `FullyShardedDataParallel`."
                f" Got: {module.__class__.__name__}."
            )
        self.precision.unscale_gradients(optimizer)
        return module.clip_grad_norm_(max_norm=max_norm, norm_type=norm_type)

    @override
    def save_checkpoint(
        self,
        path: _PATH,
        state: dict[str, Union[Module, Optimizer, Any]],
        storage_options: Optional[Any] = None,
        filter: Optional[dict[str, Callable[[str, Any], bool]]] = None,
    ) -> None:
        """Save model, optimizer, and other state to a checkpoint on disk.

        If the state-dict-type is ``'full'``, the checkpoint will be written to a single file containing the weights,
        optimizer state and other metadata. If the state-dict-type is ``'sharded'``, the checkpoint gets saved as a
        directory containing one file per process, with model- and optimizer shards stored per file. Additionally, it
        creates a metadata file `meta.pt` with the rest of the user's state (only saved from rank 0).

        """
        if storage_options is not None:
            raise TypeError(
                "`FSDPStrategy.save_checkpoint(..., storage_options=...)` is not supported because"
                " `FSDPStrategy` does not use the `CheckpointIO`."
            )
        if filter is not None and self._state_dict_type == "sharded":
            # https://github.com/pytorch/pytorch/issues/105379
            raise NotImplementedError(
                "FSDP doesn't support loading sharded filtered checkpoints, so saving them is disabled."
            )

        # broadcast the path from rank 0 to ensure all the states are saved in a common path
        path = Path(self.broadcast(path))
        if path.is_dir() and self._state_dict_type == "full" and not _is_sharded_checkpoint(path):
            raise IsADirectoryError(f"The checkpoint path exists and is a directory: {path}")

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        modules = [module for module in state.values() if _has_fsdp_modules(module)]
        if len(modules) == 0:
            raise ValueError(
                "Could not find a FSDP model in the provided checkpoint state. Please provide the model as"
                " part of the state like so: `save_checkpoint(..., state={'model': model, ...})`. Make sure"
                " you set up the model (and optimizers if any) through the strategy before saving the checkpoint."
            )
        if len(modules) > 1:
            raise ValueError(
                "Found multiple FSDP models in the given state. Saving checkpoints with FSDP is"
                " currently limited to a single model per checkpoint. To save multiple models, call the"
                " save method for each model separately with a different path."
            )
        module = modules[0]

        if self._state_dict_type == "sharded":
            if path.is_file():
                path.unlink()
            path.mkdir(parents=True, exist_ok=True)

            state_dict_ctx = _get_sharded_state_dict_context(module)

            # replace the modules and optimizer objects in the state with their local state dict
            # and separate the user's metadata
            converted_state: dict[str, Any] = {}
            metadata: dict[str, Any] = {}
            with state_dict_ctx:
                for key, obj in state.items():
                    converted: Any
                    if isinstance(obj, Module):
                        converted = obj.state_dict()
                        target_dict = converted_state
                    elif isinstance(obj, Optimizer):
                        converted = FSDP.optim_state_dict(module, obj)
                        target_dict = converted_state
                    else:  # everything not a module or optimizer is considered metadata
                        converted = obj.state_dict() if isinstance(obj, _Stateful) else obj
                        target_dict = metadata
                    _apply_filter(key, filter or {}, converted, target_dict)

            _distributed_checkpoint_save(converted_state, path)

            if self.global_rank == 0:
                torch.save(metadata, path / _METADATA_FILENAME)

        elif self._state_dict_type == "full":
            if _is_sharded_checkpoint(path):
                shutil.rmtree(path)

            state_dict_ctx = _get_full_state_dict_context(module, world_size=self.world_size)
            full_state: dict[str, Any] = {}
            with state_dict_ctx:
                for key, obj in state.items():
                    if isinstance(obj, Module):
                        converted = obj.state_dict()
                    elif isinstance(obj, Optimizer):
                        converted = FSDP.optim_state_dict(module, obj)
                    else:  # everything not a module or optimizer is considered metadata
                        converted = obj.state_dict() if isinstance(obj, _Stateful) else obj
                    _apply_filter(key, filter or {}, converted, full_state)

            if self.global_rank == 0:
                torch.save(full_state, path)
        else:
            raise ValueError(f"Unknown state_dict_type: {self._state_dict_type}")

    @override
    def load_checkpoint(
        self,
        path: _PATH,
        state: Optional[Union[Module, Optimizer, dict[str, Union[Module, Optimizer, Any]]]] = None,
        strict: bool = True,
    ) -> dict[str, Any]:
        """Load the contents from a checkpoint and restore the state of the given objects."""
        if not state:
            raise ValueError(
                f"Got FSDPStrategy.load_checkpoint(..., state={state!r}) but a state with at least "
                f" a model instance to reload is required. Pass it in like so:"
                " FSDPStrategy.load_checkpoint(..., state={'model': model, ...})"
            )
        # broadcast the path from rank 0 to ensure all the states are loaded from a common path
        path = Path(self.broadcast(path))

        if isinstance(state, Module):
            from lightning.fabric.strategies.model_parallel import _load_raw_module_state_from_path

            _load_raw_module_state_from_path(path, module=state, world_size=self.world_size, strict=strict)
            return {}

        if isinstance(state, Optimizer):
            raise NotImplementedError(
                "Loading a single optimizer object from a checkpoint is not supported yet with the FSDP strategy."
            )

        from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        modules = {key: module for key, module in state.items() if _has_fsdp_modules(module)}
        if len(modules) == 0:
            raise ValueError(
                "Could not find a FSDP model in the provided checkpoint state. Please provide the model as"
                " part of the state like so: `load_checkpoint(..., state={'model': model, ...})`. Make sure"
                " you set up the model (and optimizers if any) through the strategy before loading the checkpoint."
            )
        optimizers = {key: optim for key, optim in state.items() if isinstance(optim, Optimizer)}
        if len(modules) > 1:
            raise ValueError(
                "Found multiple FSDP models in the given state. Loading checkpoints with FSDP is"
                " currently limited to a single model per checkpoint. To load multiple models, call the"
                " load method for each model separately with a different path."
            )
        module_key, module = list(modules.items())[0]

        if _is_sharded_checkpoint(path):
            state_dict_ctx = _get_sharded_state_dict_context(module)

            with state_dict_ctx:
                module_state = {module_key: module.state_dict()}
                _distributed_checkpoint_load(module_state, path)
                module.load_state_dict(module_state[module_key], strict=strict)

                if optimizers:
                    from torch.distributed.checkpoint import FileSystemReader

                    # TODO: replace with newer APIs
                    # https://github.com/pytorch/pytorch/issues/119800#issuecomment-1942156271
                    reader = FileSystemReader(path=path)
                    # the optimizer states must be loaded separately
                    for optim_key, optim in optimizers.items():
                        optim_state = load_sharded_optimizer_state_dict(
                            model_state_dict=module_state[module_key],
                            optimizer_key=optim_key,
                            storage_reader=reader,
                        )
                        flattened_osd = FSDP.optim_state_dict_to_load(
                            optim_state_dict=optim_state[optim_key],
                            model=module,
                            optim=optim,
                        )
                        optim.load_state_dict(flattened_osd)

            # Load metadata (anything not a module or optimizer)
            metadata = torch.load(path / _METADATA_FILENAME)
            requested_metadata_keys = state.keys() - modules.keys() - optimizers.keys()
            _validate_keys_for_strict_loading(requested_metadata_keys, metadata.keys(), strict=strict)
            for key in requested_metadata_keys:
                if key not in metadata:
                    continue
                state[key] = metadata.pop(key)

            # return the remaining metadata that wasn't requested as part of `state`
            return metadata

        if _is_full_checkpoint(path):
            checkpoint = _lazy_load(path)

            from lightning.fabric.strategies.model_parallel import (
                _load_raw_module_state,
                _rekey_optimizer_state_if_needed,
            )

            _load_raw_module_state(checkpoint.pop(module_key), module=module, world_size=self.world_size, strict=strict)

            if isinstance(state, Module):
                return {}

            # Materialize lazy tensors if there are any left in the checkpoint
            # The `torch.Optimizer.load_state_dict` method can't load lazy tensors because of deepcopy pickle issues
            checkpoint = _materialize_tensors(checkpoint)

            # Load optimizer states
            for optim_key, optim in optimizers.items():
                # rank0_only should be false because we need to load the optimizer state on all ranks
                with _get_full_state_dict_context(module, world_size=self.world_size, rank0_only=False):
                    temp_state_dict = _rekey_optimizer_state_if_needed(checkpoint.pop(optim_key), module)
                    optim_state_dict = FSDP.optim_state_dict_to_load(
                        optim_state_dict=temp_state_dict,
                        model=module,
                        optim=optim,
                    )
                    optim.load_state_dict(optim_state_dict)

            requested_metadata_keys = state.keys() - modules.keys() - optimizers.keys()
            _validate_keys_for_strict_loading(requested_metadata_keys, checkpoint.keys(), strict=strict)

            # Load metadata (anything not a module or optimizer)
            _move_state_into(source=checkpoint, destination=state, keys=requested_metadata_keys)

            # return the remaining metadata that wasn't requested as part of `state`
            return checkpoint

        raise ValueError(
            f"The path {str(path)!r} does not point to a valid checkpoint. Make sure the path points to either a"
            " directory with FSDP checkpoint shards, or a single file with a full checkpoint."
        )

    @classmethod
    @override
    def register_strategies(cls, strategy_registry: _StrategyRegistry) -> None:
        if not torch.distributed.is_available():
            return

        strategy_registry.register(
            "fsdp",
            cls,
            description="Fully Sharded Data Parallel (FSDP) training",
        )
        strategy_registry.register(
            "fsdp_cpu_offload",
            cls,
            description="Fully Sharded Data Parallel (FSDP) training with Full Sharding and CPU Offloading",
            cpu_offload=True,
        )

    def _setup_distributed(self) -> None:
        reset_seed()
        self._set_world_ranks()
        self._process_group_backend = self._get_process_group_backend()
        assert self.cluster_environment is not None
        _init_dist_connection(self.cluster_environment, self._process_group_backend, timeout=self._timeout)

    def _get_process_group_backend(self) -> str:
        return self._process_group_backend or _get_default_process_group_backend_for_device(self.root_device)

    def _set_world_ranks(self) -> None:
        if self.cluster_environment is not None:
            self.cluster_environment.set_global_rank(self.node_rank * self.num_processes + self.local_rank)
            self.cluster_environment.set_world_size(self.num_nodes * self.num_processes)
        # `LightningEnvironment.set_global_rank` will do this too, but we cannot rely on that implementation detail
        # additionally, for some implementations, the setter is a no-op, so it's safer to access the getter
        rank_zero_only.rank = utils_rank_zero_only.rank = self.global_rank


def _activation_checkpointing_kwargs(
    activation_checkpointing: Optional[Union[type[Module], list[type[Module]]]],
    activation_checkpointing_policy: Optional["_POLICY"],
) -> dict:
    if activation_checkpointing is None and activation_checkpointing_policy is None:
        return {}
    if activation_checkpointing is not None and activation_checkpointing_policy is not None:
        raise ValueError(
            "You cannot set both `activation_checkpointing` and `activation_checkpointing_policy`. Use the latter."
        )
    if activation_checkpointing is not None:
        if isinstance(activation_checkpointing, list):
            classes = tuple(activation_checkpointing)
        else:
            classes = (activation_checkpointing,)
        rank_zero_deprecation(
            f"`FSDPStrategy(activation_checkpointing={activation_checkpointing})` is deprecated, use "
            f"`FSDPStrategy(activation_checkpointing_policy={set(classes)})` instead."
        )
        return {"check_fn": lambda submodule: isinstance(submodule, classes)}
    if isinstance(activation_checkpointing_policy, set):
        return _auto_wrap_policy_kwargs(activation_checkpointing_policy, {})
    return {"auto_wrap_policy": activation_checkpointing_policy}


def _auto_wrap_policy_kwargs(policy: Optional["_POLICY"], kwargs: dict) -> dict:
    if policy is None:
        return kwargs
    if isinstance(policy, set):
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy

        policy = ModuleWrapPolicy(policy)

    kwargs["auto_wrap_policy"] = policy
    return kwargs


def _setup_activation_checkpointing(module: Module, activation_checkpointing_kwargs: dict) -> None:
    if not activation_checkpointing_kwargs:
        return

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

    if any(isinstance(mod, CheckpointWrapper) for mod in module.modules()):
        rank_zero_warn(
            "FSDP checkpointing is configured, but the model already contains checkpointed layers."
            " Checkpointing will be ignored."
        )
        return

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    if not _TORCH_GREATER_EQUAL_2_2:
        checkpoint_wrapper = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
    apply_activation_checkpointing(module, checkpoint_wrapper_fn=checkpoint_wrapper, **activation_checkpointing_kwargs)


class _FSDPBackwardSyncControl(_BackwardSyncControl):
    @override
    def no_backward_sync(self, module: Module, enabled: bool) -> AbstractContextManager:
        """Blocks gradient synchronization inside the :class:`~torch.distributed.fsdp.FullyShardedDataParallel`
        wrapper."""
        if not enabled:
            return nullcontext()
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel

        if not isinstance(module, FullyShardedDataParallel):
            # the root must be wrapped
            raise TypeError(
                "Blocking backward sync is only possible if the module passed to"
                f" `{type(self).__name__}.no_backward_sync` is wrapped in `FullyShardedDataParallel`."
                f" Got: {module.__class__.__name__}."
            )
        return module.no_sync()


def _init_cpu_offload(cpu_offload: Optional[Union[bool, "CPUOffload"]]) -> "CPUOffload":
    from torch.distributed.fsdp import CPUOffload

    return cpu_offload if isinstance(cpu_offload, CPUOffload) else CPUOffload(offload_params=bool(cpu_offload))


def _init_sharding_strategy(sharding_strategy: "_SHARDING_STRATEGY", kwargs: dict) -> "ShardingStrategy":
    from torch.distributed.fsdp import ShardingStrategy

    if kwargs.get("process_group") is not None and kwargs.get("device_mesh") is not None:
        raise ValueError(
            "The arguments `FSDPStrategy(process_group=..., device_mesh=...)` are mutually exclusive."
            "Pass only one of them."
        )

    strategy = ShardingStrategy[sharding_strategy.upper()] if isinstance(sharding_strategy, str) else sharding_strategy
    if (
        "HYBRID" in strategy.name
        and kwargs.get("auto_wrap_policy") is None
        and kwargs.get("process_group") is None
        and kwargs.get("device_mesh") is None
    ):
        raise RuntimeError(
            "The hybrid sharding strategy requires you to pass at least one of the parameters: `auto_wrap_policy`,"
            " `process_group` tuple, or `device_mesh`."
        )
    return strategy


def _optimizer_has_flat_params(optimizer: Optimizer) -> bool:
    return any(
        getattr(param, "_fsdp_flattened", False) for group in optimizer.param_groups for param in group["params"]
    )


def _get_sharded_state_dict_context(module: Module) -> Generator[None, None, None]:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.api import ShardedOptimStateDictConfig, ShardedStateDictConfig, StateDictType

    state_dict_config = ShardedStateDictConfig(offload_to_cpu=True)
    optim_state_dict_config = ShardedOptimStateDictConfig(offload_to_cpu=True)
    state_dict_type_context = FSDP.state_dict_type(
        module=module,
        state_dict_type=StateDictType.SHARDED_STATE_DICT,
        state_dict_config=state_dict_config,
        optim_state_dict_config=optim_state_dict_config,
    )
    return state_dict_type_context  # type: ignore[return-value]


def _get_full_state_dict_context(
    module: Module, world_size: int, rank0_only: bool = True
) -> Generator[None, None, None]:
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.api import FullOptimStateDictConfig

    state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=rank0_only)
    optim_state_dict_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=rank0_only)
    state_dict_type_context = FSDP.state_dict_type(
        module=module,
        state_dict_type=StateDictType.FULL_STATE_DICT,
        state_dict_config=state_dict_config,
        optim_state_dict_config=optim_state_dict_config,
    )

    return state_dict_type_context  # type: ignore[return-value]


def _is_sharded_checkpoint(path: Path) -> bool:
    """A heuristic check to determine whether the path points to a directory with checkpoint shards."""
    return path.is_dir() and (path / _METADATA_FILENAME).is_file()


def _is_full_checkpoint(path: Path) -> bool:
    return path.is_file()


def _has_fsdp_modules(module: object) -> TypeGuard[Module]:
    from torch.distributed.fsdp import FullyShardedDataParallel

    return isinstance(module, Module) and any(isinstance(m, FullyShardedDataParallel) for m in module.modules())


def _move_torchmetrics_to_device(module: torch.nn.Module, device: torch.device) -> None:
    # FSDP doesn't move modules without parameters (e.g. Metrics) to the device
    # https://github.com/pytorch/pytorch/issues/113113
    if not RequirementCache("torchmetrics"):
        return

    from torchmetrics import Metric

    for metric in (m for m in module.modules() if isinstance(m, Metric)):
        metric.to(device)  # `.to()` is in-place


def _distributed_checkpoint_save(converted_state: dict[str, Any], path: Path) -> None:
    if _TORCH_GREATER_EQUAL_2_3:
        from torch.distributed.checkpoint import save

        # let torch automatically infer the writer to use. This might also support fsspec paths in the future
        # https://github.com/pytorch/pytorch/issues/118036
        save(converted_state, checkpoint_id=path)
    else:  # deprecated
        from torch.distributed.checkpoint import FileSystemWriter

        if _TORCH_GREATER_EQUAL_2_2:
            from torch.distributed.checkpoint import save
        else:
            from torch.distributed.checkpoint import save_state_dict as save
        # FSDP's FileSystemWriter streams the tensors to disk to minimize memory peaks
        writer = FileSystemWriter(path=path, single_file_per_rank=True)
        save(converted_state, writer)


def _distributed_checkpoint_load(module_state: dict[str, Any], path: Path) -> None:
    if _TORCH_GREATER_EQUAL_2_3:
        from torch.distributed.checkpoint import load

        # let torch automatically infer the reader to use. This might also support fsspec paths in the future
        # https://github.com/pytorch/pytorch/issues/118036
        load(module_state, checkpoint_id=path)
    else:  # deprecated
        from torch.distributed.checkpoint import FileSystemReader

        if _TORCH_GREATER_EQUAL_2_2:
            from torch.distributed.checkpoint import load
        else:
            from torch.distributed.checkpoint import load_state_dict as load
        reader = FileSystemReader(path=path)
        load(module_state, reader)
