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
import io
from contextlib import AbstractContextManager, ExitStack, nullcontext
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from typing_extensions import override

from lightning.fabric.accelerators import Accelerator
from lightning.fabric.accelerators.xla import _XLA_AVAILABLE
from lightning.fabric.plugins import XLAPrecision
from lightning.fabric.plugins.environments import XLAEnvironment
from lightning.fabric.plugins.io.xla import XLACheckpointIO
from lightning.fabric.strategies import ParallelStrategy, _StrategyRegistry
from lightning.fabric.strategies.fsdp import _apply_filter
from lightning.fabric.strategies.launchers.xla import _XLALauncher
from lightning.fabric.strategies.strategy import (
    TBroadcast,
    _BackwardSyncControl,
    _Sharded,
    _validate_keys_for_strict_loading,
)
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.fabric.utilities.init import _EmptyInit
from lightning.fabric.utilities.rank_zero import rank_zero_only, rank_zero_warn
from lightning.fabric.utilities.types import _PATH, Optimizable, ReduceOp

if TYPE_CHECKING:
    from torch_xla.distributed.parallel_loader import MpDeviceLoader

_POLICY_SET = set[type[Module]]
_POLICY = Union[_POLICY_SET, Callable[[Module, bool, int], bool]]


class XLAFSDPStrategy(ParallelStrategy, _Sharded):
    r"""Strategy for training multiple XLA devices using the
    :func:`torch_xla.distributed.xla_fully_sharded_data_parallel.XlaFullyShardedDataParallel` method.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    For more information check out https://github.com/pytorch/xla/blob/v2.5.0/docs/fsdp.md

    Args:
        auto_wrap_policy: Same as ``auto_wrap_policy`` parameter in
            :class:`torch_xla.distributed.fsdp.XlaFullyShardedDataParallel`.
            For convenience, this also accepts a set of the layer classes to wrap.
        activation_checkpointing_policy: Used when selecting the modules for
            which you want to enable activation checkpointing. Enabling this can free up a significant amount of memory
            at the cost of speed since activations in these layers need to be recomputed during backpropagation.
            This accepts a set of the layer classes to wrap.

        state_dict_type: The format in which the state of the model and optimizers gets saved into the checkpoint.

            - ``"full"``: The full weights and optimizer states get assembled on rank 0 and saved to a single file.
            - ``"sharded"``: Each rank saves its shard of weights and optimizer states to a file. The checkpoint is
              a folder with files for each shard in the host. Note that TPU VM multihost does not have a shared
              filesystem.

        sequential_save: With this enabled, individual ranks consecutively save their state dictionary shards, reducing
            peak system RAM usage, although it elongates the saving process.
        \**kwargs: See available parameters in :class:`torch_xla.distributed.fsdp.XlaFullyShardedDataParallel`.

    """

    def __init__(
        self,
        accelerator: Optional[Accelerator] = None,
        parallel_devices: Optional[list[torch.device]] = None,
        checkpoint_io: Optional[XLACheckpointIO] = None,
        precision: Optional[XLAPrecision] = None,
        auto_wrap_policy: Optional[_POLICY] = None,
        activation_checkpointing_policy: Optional[_POLICY_SET] = None,
        state_dict_type: Literal["full", "sharded"] = "sharded",
        sequential_save: bool = False,
        **kwargs: Any,
    ) -> None:
        if not _XLA_AVAILABLE:
            raise ModuleNotFoundError(str(_XLA_AVAILABLE))
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=XLAEnvironment(),
            checkpoint_io=checkpoint_io,
            precision=precision,
        )
        self._backward_sync_control = _XLAFSDPBackwardSyncControl()

        self._auto_wrap_policy = auto_wrap_policy
        self._activation_checkpointing_policy = activation_checkpointing_policy
        self._fsdp_kwargs = kwargs
        self._state_dict_type = state_dict_type
        self._sequential_save = sequential_save
        self._launched = False

    @property
    @override
    def root_device(self) -> torch.device:
        if not self._launched:
            raise RuntimeError("Accessing the XLA device before processes have spawned is not allowed.")
        import torch_xla.core.xla_model as xm

        return xm.xla_device()

    @property
    def num_processes(self) -> int:
        return len(self.parallel_devices) if self.parallel_devices is not None else 0

    @property
    @override
    def checkpoint_io(self) -> XLACheckpointIO:
        plugin = self._checkpoint_io
        if plugin is not None:
            assert isinstance(plugin, XLACheckpointIO)
            return plugin
        return XLACheckpointIO()

    @checkpoint_io.setter
    @override
    def checkpoint_io(self, io: Optional[XLACheckpointIO]) -> None:
        if io is not None and not isinstance(io, XLACheckpointIO):
            raise TypeError(f"The XLA strategy can only work with the `XLACheckpointIO` plugin, found {io}")
        self._checkpoint_io = io

    @property
    @override
    def precision(self) -> XLAPrecision:
        plugin = self._precision
        if plugin is not None:
            assert isinstance(plugin, XLAPrecision)
            return plugin
        return XLAPrecision("32-true")

    @precision.setter
    @override
    def precision(self, precision: Optional[XLAPrecision]) -> None:
        if precision is not None and not isinstance(precision, XLAPrecision):
            raise TypeError(f"The XLA FSDP strategy can only work with the `XLAPrecision` plugin, found {precision}")
        self._precision = precision

    @property
    @override
    def global_rank(self) -> int:
        return super().global_rank if self._launched else 0

    @property
    @override
    def local_rank(self) -> int:
        return super().local_rank if self._launched else 0

    @property
    @override
    def node_rank(self) -> int:
        return super().node_rank if self._launched else 0

    @property
    @override
    def world_size(self) -> int:
        return super().world_size if self._launched else 1

    @override
    def _configure_launcher(self) -> None:
        self._launcher = _XLALauncher(self)

    @override
    def setup_environment(self) -> None:
        assert self.parallel_devices is not None
        if len(self.parallel_devices) == 1:
            # spawning only 1 device with PjRT is not supported:
            # https://github.com/Lightning-AI/lightning/pull/17408#discussion_r1170671732
            raise NotImplementedError(
                f"The {type(self).__name__} does not support running on a single device with the PjRT runtime."
                " Try using all devices or the `SingleDeviceXLAStrategy` strategy"
            )

        self._launched = True
        rank_zero_only.rank = self.global_rank
        super().setup_environment()

    @override
    def setup_module_and_optimizers(
        self, module: Module, optimizers: list[Optimizer], scheduler: Optional[_LRScheduler] = None
    ) -> tuple[Module, list[Optimizer], Optional[_LRScheduler]]:
        """Returns NotImplementedError since for XLAFSDP optimizer setup must happen after module setup."""
        raise NotImplementedError(
            f"The `{type(self).__name__}` does not support the joint setup of module and optimizer(s)."
            " Please do it in this order: Create the model, call `setup_module`, create the optimizer,"
            " call `setup_optimizer`."
        )

    @override
    def setup_module(self, module: Module) -> Module:
        from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as XLAFSDP

        kwargs = self._parse_fsdp_kwargs()
        if any(isinstance(mod, XLAFSDP) for mod in module.modules()) and "auto_wrap_policy" in kwargs:
            rank_zero_warn(
                "A XLAFSDP `auto_wrap_policy` is set, but at least one submodule is already wrapped."
                " The policy will be ignored."
            )
            del kwargs["auto_wrap_policy"]
        # XLA FSDP requires that the root is wrapped, even if submodules are already wrapped
        if not isinstance(module, XLAFSDP):
            module = XLAFSDP(module=module, **kwargs)
        return module

    @override
    def module_to_device(self, module: Module) -> None:
        pass

    def module_init_context(self, empty_init: Optional[bool] = None) -> AbstractContextManager:
        precision_init_ctx = self.precision.module_init_context()
        module_sharded_ctx = self.module_sharded_context()
        stack = ExitStack()
        stack.enter_context(_EmptyInit(enabled=bool(empty_init)))
        stack.enter_context(precision_init_ctx)
        stack.enter_context(module_sharded_ctx)
        return stack

    @override
    def module_sharded_context(self) -> AbstractContextManager:
        return nullcontext()

    @override
    def process_dataloader(self, dataloader: DataLoader) -> "MpDeviceLoader":
        from torch_xla.distributed.parallel_loader import MpDeviceLoader

        if isinstance(dataloader, MpDeviceLoader):
            # dataloader is already wrapped by MpDeviceLoader
            return dataloader

        dataloader = MpDeviceLoader(dataloader, self.root_device)
        # Mimic interface to torch.utils.data.DataLoader
        dataloader.dataset = dataloader._loader.dataset
        dataloader.batch_sampler = getattr(dataloader._loader, "batch_sampler", None)
        return dataloader

    @override
    def setup_optimizer(self, optimizer: Optimizer) -> Optimizer:
        """Set up an optimizer for a model wrapped with XLAFSDP.

        This setup method doesn't modify the optimizer or wrap the optimizer. The only thing it currently does is verify
        that the optimizer was created after the model was wrapped with :meth:`setup_module` with a reference to the
        flattened parameters.

        """
        if any(getattr(p, "_is_sharded", False) for group in optimizer.param_groups for p in group["params"]):
            return optimizer
        raise ValueError(
            "The optimizer does not seem to reference any XLAFSDP parameters. HINT: Make sure to create the optimizer"
            " after setting up the model."
        )

    @override
    def optimizer_step(self, optimizer: Optimizable, **kwargs: Any) -> Any:
        """Overrides default tpu optimizer_step since FSDP should not call `torch_xla.core.xla_model.optimizer_step`.
        Performs the actual optimizer step.

        Args:
            optimizer: the optimizer performing the step
            **kwargs: Any extra arguments to ``optimizer.step``

        """
        loss = optimizer.step(**kwargs)
        import torch_xla.core.xla_model as xm

        xm.mark_step()
        return loss

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
        self.precision.unscale_gradients(optimizer)
        return module.clip_grad_norm_(max_norm=max_norm, norm_type=norm_type)

    @override
    def clip_gradients_value(self, module: Module, optimizer: Optimizer, clip_val: Union[float, int]) -> None:
        """Clip gradients by value."""
        raise NotImplementedError(
            "XLA's FSDP strategy does not support to clip gradients by value."
            " Consider clipping by norm instead or choose another strategy!"
        )

    @override
    def all_gather(self, tensor: Tensor, group: Optional[Any] = None, sync_grads: bool = False) -> Tensor:
        """Function to gather a tensor from several distributed processes.

        Args:
            tensor: tensor to all-gather.
            group: unused.
            sync_grads: flag that allows users to synchronize gradients for the all-gather operation.
        Return:
            A tensor of shape (world_size, ...)

        """
        if not self._launched:
            return tensor
        if not isinstance(tensor, Tensor):
            raise NotImplementedError(
                f"`{type(self).__name__}.all_gather` is only implemented for tensors. Given {tensor}"
            )
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        original_device = tensor.device
        tensor = tensor.to(self.root_device)

        import torch_xla.core.functions as xf
        import torch_xla.core.xla_model as xm

        tensor = xf.all_gather(tensor) if sync_grads else xm.all_gather(tensor)
        tensor = tensor.to(original_device)
        return tensor

    @override
    def all_reduce(
        self, output: Union[Tensor, Any], group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = None
    ) -> Tensor:
        if not isinstance(output, Tensor):
            output = torch.tensor(output, device=self.root_device)

        invalid_reduce_op = isinstance(reduce_op, ReduceOp) and reduce_op != ReduceOp.SUM
        invalid_reduce_op_str = isinstance(reduce_op, str) and reduce_op.lower() not in ("sum", "mean", "avg")
        if invalid_reduce_op or invalid_reduce_op_str:
            raise ValueError(
                "Currently, the XLAFSDPStrategy only supports `sum`, `mean`, `avg` for the reduce operation, got:"
                f" {reduce_op}"
            )
        import torch_xla.core.xla_model as xm

        output = xm.mesh_reduce("reduce", output, sum)

        if isinstance(reduce_op, str) and reduce_op.lower() in ("avg", "mean"):
            output = output / self.world_size

        return output

    @override
    def barrier(self, name: Optional[str] = None, *args: Any, **kwargs: Any) -> None:
        if not self._launched:
            return
        import torch_xla.core.xla_model as xm

        if name is None:
            # `None` is not supported: "TypeError: _xla_rendezvous(): incompatible function arguments"
            name = ""
        xm.rendezvous(name)

    @override
    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        if not self._launched:
            return obj

        import torch_xla.core.xla_model as xm

        is_tensor = isinstance(obj, Tensor)
        if is_tensor:
            if obj.dim() == 0:
                obj = obj.unsqueeze(0)
            original_device = obj.device
            # XLA distributed requires that the data is on the XLA device
            obj = obj.to(self.root_device)
        else:
            # support for arbitrary pickle-ables
            buffer = io.BytesIO()
            torch.save(obj, buffer)
            obj = torch.tensor(  # type: ignore[assignment]
                bytearray(buffer.getbuffer()), device=self.root_device, dtype=torch.float
            )

        obj = [obj]
        xm.collective_broadcast(obj, root_ordinal=src)
        obj = obj[0]

        if not is_tensor:
            # this will preserve the dtype and device of any tensors
            buffer = io.BytesIO(obj.cpu().byte().numpy())
            obj = torch.load(buffer)
        else:
            obj = obj.to(original_device)

        return obj

    @override
    def save_checkpoint(
        self,
        path: _PATH,
        state: dict[str, Union[Module, Optimizer, Any]],
        storage_options: Optional[Any] = None,
        filter: Optional[dict[str, Callable[[str, Any], bool]]] = None,
    ) -> None:
        """Save model, optimizer, and other state in the provided checkpoint directory.

        If the user specifies sharded checkpointing, the directory will contain one file per process, with model- and
        optimizer shards stored per file. If the user specifies full checkpointing, the directory will contain a
        consolidated checkpoint combining all of the sharded checkpoints.

        """
        # broadcast the path from rank 0 to ensure all the states are saved in a common path
        path = Path(self.broadcast(path))
        if path.is_dir() and any(path.iterdir()):
            raise FileExistsError(f"The checkpoint directory already exists and is not empty: {path}")
        from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as XLAFSDP

        modules = [module for module in state.values() if isinstance(module, XLAFSDP)]
        if len(modules) == 0:
            raise ValueError(
                "Could not find a XLAFSDP model in the provided checkpoint state. Please provide the model as"
                " part of the state like so: `save_checkpoint(..., state={'model': model, ...})`. Make sure"
                " you set up the model (and optimizers if any) through the strategy before saving the checkpoint."
            )
        if len(modules) > 1:
            raise ValueError(
                "Found multiple XLAFSDP modules in the given state. Saving checkpoints with FSDP is"
                " currently limited to a single model per checkpoint. To save multiple models, call the"
                " save method for each model separately with a different path."
            )
        import torch_xla.core.xla_model as xm

        # ensure model parameters are updated
        xm.mark_step()

        parallel_devices = self.parallel_devices
        assert parallel_devices is not None
        if self._sequential_save:
            # each host runs this in parallel, but the ranks in the host run it sequentially
            for rank in range(len(parallel_devices)):
                if rank == self.local_rank:
                    self._save_checkpoint_shard(path, state, storage_options, filter)
                self.barrier(f"wait-for-{rank}-save")
        else:
            self._save_checkpoint_shard(path, state, storage_options, filter)

        if self._state_dict_type == "full":
            ckpt_prefix = str(path / "checkpoint")
            ckpt_suffix = "_rank-*-of-*.pth"
            if len(parallel_devices) != self.world_size:  # multihost
                raise OSError(
                    "Multihost setups do not have a shared filesystem, so the checkpoint shards cannot be consolidated"
                    " into a single checkpoint after saving them. Please switch to"
                    " `XLAFSDPStrategy(state_dict_type='sharded')`. TIP: You can consolidate them manually by getting"
                    " them together into a single directory and running `python -m"
                    f" torch_xla.distributed.fsdp.consolidate_sharded_ckpts --ckpt_prefix {ckpt_prefix!r} --ckpt_suffix"
                    f" {ckpt_suffix!r} --save_path 'path/to/consolidated.ckpt'`."
                )

            from torch_xla.distributed.fsdp import consolidate_sharded_model_checkpoints

            self.barrier("before_ckpt_consolidation")
            if self.is_global_zero:
                save_path = path.parent / "consolidated.ckpt"
                # save consolidated checkpoint separate to the shards
                consolidate_sharded_model_checkpoints(ckpt_prefix, ckpt_suffix, str(save_path))
                # remove the shards directory
                self.checkpoint_io.remove_checkpoint(path)
                # mv the consolidated checkpoint where the user would expect it
                get_filesystem(save_path).mv(str(save_path), str(path))
            self.barrier("after_ckpt_consolidation")

    def _save_checkpoint_shard(
        self,
        path: Path,
        state: dict[str, Union[Module, Optimizer, Any]],
        storage_options: Optional[Any],
        filter: Optional[dict[str, Callable[[str, Any], bool]]],
    ) -> None:
        from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as XLAFSDP

        converted_state: dict[str, Any] = {}
        for key, obj in state.items():
            # convert the state
            if isinstance(obj, Module) and isinstance(obj, XLAFSDP):
                converted = obj.state_dict()
                # add shard_metadata to state
                converted_state["shard_metadata"] = obj.get_shard_metadata()
            elif isinstance(obj, Optimizer):
                converted = obj.state_dict()
            else:
                converted = obj
            _apply_filter(key, filter or {}, converted, converted_state)

        self.checkpoint_io.save_checkpoint(
            converted_state,
            path / f"checkpoint_rank-{self.global_rank:08d}-of-{self.world_size:08d}.pth",
            storage_options=storage_options,
        )

    @override
    def load_checkpoint(
        self,
        path: _PATH,
        state: Optional[Union[Module, Optimizer, dict[str, Union[Module, Optimizer, Any]]]] = None,
        strict: bool = True,
    ) -> dict[str, Any]:
        """Given a folder, load the contents from a checkpoint and restore the state of the given objects.

        The strategy currently only supports saving and loading sharded checkpoints which are stored in form of a
        directory of multiple files rather than a single file.

        """
        if not state:
            raise ValueError(
                f"Got `XLAFSDPStrategy.load_checkpoint(..., state={state!r})` but a state with at least "
                " a model instance to reload is required. Pass it in like so:"
                " `FSDPStrategy.load_checkpoint(..., state={'model': model, ...})`"
            )

        # broadcast the path from rank 0 to ensure all the states are loaded from a common path
        path = Path(self.broadcast(path))

        if isinstance(state, (Module, Optimizer)):
            raise NotImplementedError(
                "Loading a single module or optimizer object from a checkpoint"
                " is not supported yet with the XLAFSDP strategy."
            )

        from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as XLAFSDP

        modules = {key: module for key, module in state.items() if isinstance(module, XLAFSDP)}
        optimizers = {key: optim for key, optim in state.items() if isinstance(optim, Optimizer)}
        if self._state_dict_type == "sharded":
            file = path / f"checkpoint_rank-{self.global_rank:08d}-of-{self.world_size:08d}.pth"
            if not file.is_file():
                raise ValueError(
                    f"The path {str(file)!r} does not point to valid sharded checkpoints. Make sure the path points to"
                    " a directory with XLAFSDP checkpoint shards."
                )
            if len(modules) == 0:
                raise ValueError(
                    "Could not find a XLAFSDP model in the provided checkpoint state. Please provide the model as"
                    " part of the state like so: `load_checkpoint(..., state={'model': model, ...})`. Make sure"
                    " you set up the model (and optimizers if any) through the strategy before loading the checkpoint."
                )
            if len(modules) > 1:
                raise ValueError(
                    "Found multiple XLAFSDP modules in the given state. Loading checkpoints with FSDP is"
                    " currently limited to a single model per checkpoint. To load multiple models, call the"
                    " load method for each model separately with a different path."
                )

            _, module = list(modules.items())[0]
            sharded_ckpt = torch.load(file)

            module.load_state_dict(sharded_ckpt["model"], strict=strict)
            for opt_key, opt in optimizers.items():
                opt.load_state_dict(sharded_ckpt[opt_key])

            # Load anything leftover from sharded_ckpt
            loaded_metadata_keys = sharded_ckpt.keys() - modules.keys() - optimizers.keys()
            requested_metadata_keys = state.keys() - modules.keys() - optimizers.keys()
            _validate_keys_for_strict_loading(requested_metadata_keys, loaded_metadata_keys, strict=strict)
            for key in requested_metadata_keys:
                if key in loaded_metadata_keys:
                    state[key] = sharded_ckpt[key]
                    loaded_metadata_keys.remove(key)

            metadata = {}
            if len(loaded_metadata_keys):
                for key in loaded_metadata_keys:
                    metadata[key] = sharded_ckpt[key]

            # remove "shard_metadata" that is loaded in
            if "shard_metadata" in metadata:
                metadata.pop("shard_metadata")

            return metadata

        if self._state_dict_type == "full":
            if not path.is_file():
                raise ValueError(
                    f"The path {str(path)!r} does not point to a valid full checkpoint. Make sure the path points to a"
                    " directory with a full XLAFSDP checkpoint."
                )
            if len(optimizers) > 0 or len(state.keys() - modules.keys() - optimizers.keys()) > 0:
                rank_zero_warn(
                    "Loading a full checkpoint will only load the full model."
                    " The optimizer and any additional metadata are not included."
                )
            if len(modules) > 0:
                raise ValueError(
                    "Found a XLAFSDP model in the provided checkpoint state."
                    " Please provide the model without any XLAFSDP wrapper."
                )
            if "model" not in state or not isinstance(model := state["model"], torch.nn.Module):
                raise NotImplementedError("XLAFSDP only supports a single model instance with 'model' as the key.")
            full_ckpt = torch.load(path)
            model.load_state_dict(full_ckpt.pop("model"), strict=strict)
            return full_ckpt

        raise ValueError(f"Unknown state_dict_type: {self._state_dict_type}")

    @classmethod
    @override
    def register_strategies(cls, strategy_registry: _StrategyRegistry) -> None:
        strategy_registry.register("xla_fsdp", cls, description=cls.__name__)

    def _parse_fsdp_kwargs(self) -> dict:
        # this needs to be delayed because `self.precision` isn't available at init
        kwargs = self._fsdp_kwargs.copy()
        precision = self.precision
        if isinstance(precision, XLAPrecision):
            # the `compute_dtype` will be passed to the `auto_wrapper_callable` automatically, so we don't need to pass
            # it when creating it
            kwargs.setdefault("compute_dtype", precision._desired_dtype)
        kwargs = _auto_wrap_policy_kwargs(self._auto_wrap_policy, kwargs)
        return _activation_checkpointing_kwargs(self._activation_checkpointing_policy, kwargs)


def _auto_wrap_policy_kwargs(policy: Optional["_POLICY"], kwargs: dict) -> dict:
    if policy is None:
        return kwargs
    if isinstance(policy, set):
        from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy

        # this is not transformer specific despite the name
        policy = partial(transformer_auto_wrap_policy, transformer_layer_cls=policy)
    kwargs["auto_wrap_policy"] = policy
    return kwargs


def _activation_checkpointing_auto_wrapper(policy: _POLICY_SET, module: Module, *args: Any, **kwargs: Any) -> Module:
    from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as XLAFSDP
    from torch_xla.distributed.fsdp import checkpoint_module

    module = checkpoint_module(module) if isinstance(module, tuple(policy)) else module
    return XLAFSDP(module, *args, **kwargs)


def _activation_checkpointing_kwargs(policy: Optional[_POLICY_SET], kwargs: dict) -> dict:
    if not policy:
        return kwargs
    if "auto_wrapper_callable" in kwargs:
        raise ValueError(
            "You cannot set both `auto_wrapper_callable` and `activation_checkpointing_policy`. Choose one"
        )
    if not isinstance(policy, set):
        raise TypeError(
            f"`activation_checkpointing_policy` must be a set, found {policy}. You can try defining and"
            " passing `auto_wrapper_callable` instead."
        )
    auto_wrapper_callable = partial(_activation_checkpointing_auto_wrapper, policy)
    kwargs["auto_wrapper_callable"] = auto_wrapper_callable
    return kwargs


class _XLAFSDPBackwardSyncControl(_BackwardSyncControl):
    @override
    def no_backward_sync(self, module: Module, enabled: bool) -> AbstractContextManager:
        """Blocks gradient synchronization inside the :class:`~torch_xla.distributed.fsdp.XlaFullyShardedDataParallel`
        wrapper."""
        if not enabled:
            return nullcontext()
        from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as XLAFSDP

        if not isinstance(module, XLAFSDP):
            raise TypeError(
                "Blocking backward sync is only possible if the module passed to"
                f" `{self.__class__.__name__}.no_backward_sync` is wrapped in `XlaFullyShardedDataParallel`."
                f" Got: {module.__class__.__name__}."
            )
        return module.no_sync()
