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
import argparse
import json
import logging
import os
import platform
from collections.abc import Mapping
from contextlib import AbstractContextManager, ExitStack
from datetime import timedelta
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import torch
from lightning_utilities.core.imports import RequirementCache
from torch.nn import Module
from torch.optim import Optimizer
from typing_extensions import override

from lightning.fabric.accelerators import Accelerator, CUDAAccelerator
from lightning.fabric.plugins.collectives.torch_collective import default_pg_timeout
from lightning.fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning.fabric.plugins.precision import Precision
from lightning.fabric.strategies.ddp import DDPStrategy
from lightning.fabric.strategies.registry import _StrategyRegistry
from lightning.fabric.strategies.strategy import _Sharded
from lightning.fabric.utilities.distributed import log
from lightning.fabric.utilities.load import _move_state_into
from lightning.fabric.utilities.rank_zero import rank_zero_info, rank_zero_warn
from lightning.fabric.utilities.seed import reset_seed
from lightning.fabric.utilities.types import _PATH

if TYPE_CHECKING:
    from deepspeed import DeepSpeedEngine

_DEEPSPEED_AVAILABLE = RequirementCache("deepspeed")
_DEEPSPEED_GREATER_EQUAL_0_14_1 = RequirementCache("deepspeed>=0.14.1")


# TODO(fabric): Links in the docstrings to PL-specific deepspeed user docs need to be replaced.
class DeepSpeedStrategy(DDPStrategy, _Sharded):
    DEEPSPEED_ENV_VAR = "PL_DEEPSPEED_CONFIG_PATH"

    def __init__(
        self,
        accelerator: Optional[Accelerator] = None,
        zero_optimization: bool = True,
        stage: int = 2,
        remote_device: Optional[str] = None,
        offload_optimizer: bool = False,
        offload_parameters: bool = False,
        offload_params_device: str = "cpu",
        nvme_path: str = "/local_nvme",
        params_buffer_count: int = 5,
        params_buffer_size: int = 100_000_000,
        max_in_cpu: int = 1_000_000_000,
        offload_optimizer_device: str = "cpu",
        optimizer_buffer_count: int = 4,
        block_size: int = 1048576,
        queue_depth: int = 8,
        single_submit: bool = False,
        overlap_events: bool = True,
        thread_count: int = 1,
        pin_memory: bool = False,
        sub_group_size: int = 1_000_000_000_000,
        contiguous_gradients: bool = True,
        overlap_comm: bool = True,
        allgather_partitions: bool = True,
        reduce_scatter: bool = True,
        allgather_bucket_size: int = 200_000_000,
        reduce_bucket_size: int = 200_000_000,
        zero_allow_untested_optimizer: bool = True,
        logging_batch_size_per_gpu: Optional[int] = None,
        config: Optional[Union[_PATH, dict[str, Any]]] = None,
        logging_level: int = logging.WARN,
        parallel_devices: Optional[list[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        loss_scale: float = 0,
        initial_scale_power: int = 16,
        loss_scale_window: int = 1000,
        hysteresis: int = 2,
        min_loss_scale: int = 1,
        partition_activations: bool = False,
        cpu_checkpointing: bool = False,
        contiguous_memory_optimization: bool = False,
        synchronize_checkpoint_boundary: bool = False,
        load_full_weights: bool = False,
        precision: Optional[Precision] = None,
        process_group_backend: Optional[str] = None,
        timeout: Optional[timedelta] = default_pg_timeout,
    ) -> None:
        """Provides capabilities to run training using the DeepSpeed library, with training optimizations for large
        billion parameter models. `For more information: https://pytorch-
        lightning.readthedocs.io/en/stable/advanced/model_parallel.html#deepspeed`.

        .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

        Defaults have been set to enable ZeRO-Offload and some have been taken from the link below.
        These defaults have been set generally, but may require tuning for optimum performance based on your model size.
        `For more information: https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training`.

        Arguments:

            zero_optimization: Enable ZeRO optimization. This is compatible with either ``precision="16-mixed"`` or
                ``precision="bf16-mixed"``.

            stage: Different stages of the ZeRO Optimizer. 0 is disabled,
                1 is optimizer state partitioning, 2 is optimizer+gradient state partitioning,
                3 is optimizer+gradient_parameter partitioning using the infinity engine.

            remote_device: Device to instantiate the model on initially (``cpu`` or ``nvme``). Defaults to GPU.

            offload_optimizer: Enable offloading optimizer memory and computation to CPU or NVMe
                based on ``offload_optimizer_device``.

            offload_parameters: When using ZeRO Stage 3, Enable offloading parameter memory and computation
                to CPU or NVMe based on ``offload_params_device``.

            offload_params_device: When offloading parameters choose the device to offload to, ``cpu`` or ``nvme``.

            offload_optimizer_device: When offloading optimizer state choose the device to offload to,
                ``cpu`` or ``nvme``.

            params_buffer_count: Number of buffers in buffer pool for
                parameter offloading when ``offload_params_device`` is ``nvme``.

            params_buffer_size: Size of buffers in buffer pool for parameter offloading
                when ``offload_params_device`` is ``nvme``.

            max_in_cpu: Number of parameter elements to maintain in CPU memory when offloading to NVMe is enabled.

            nvme_path: Filesystem path for NVMe device for optimizer/parameter state offloading.

            optimizer_buffer_count: Number of buffers in buffer pool for optimizer state offloading
                when ``offload_optimizer_device`` is set to ``nvme``.
                This should be at least the number of states maintained per parameter by the optimizer.
                For example, Adam optimizer has 4 states (parameter, gradient, momentum, and variance).

            block_size: When using NVMe Offloading, the I/O block size in bytes.

            queue_depth: When using NVMe Offloading, the I/O queue depth.

            single_submit: When using NVMe Offloading,
                submit requests to storage device as multiple individual requests,
                as opposed to one block of requests.

            overlap_events: When using NVMe Offloading,
                submit requests to storage device in an overlapped fashion
                without waiting for completion of earlier requests.

            thread_count: When using NVMe Offloading,
                Intra-request parallelism for each read/write submitted by a user thread.

            pin_memory: When using ZeRO stage 3, pin optimizer state memory on CPU.
                This could boost throughput at the cost of extra memory overhead.

            sub_group_size: When using ZeRO stage 3, defines the number of parameters
                within a sub group to offload at a time.
                Smaller numbers require more communication, but improve memory efficiency.

            contiguous_gradients: Copies gradients to a continuous buffer as they are produced.
                Avoids memory fragmentation during backwards. Useful when training large models.

            overlap_comm: Overlap the reduction (synchronization) of gradients with the backwards computation.
                This is a speed optimization when training across multiple GPUs/machines.

            allgather_partitions: All gather updated parameters at the end of training step,
                instead of using a series of broadcast collectives.

            reduce_scatter: Use reduce/scatter instead of allreduce to average gradients.

            allgather_bucket_size: Number of elements to allgather at once.
                Used to limit the memory required for larger model sizes, with a tradeoff with speed.

            reduce_bucket_size: Number of elements to reduce at once.
                Used to limit the memory required for larger model sizes, with a tradeoff with speed.

            zero_allow_untested_optimizer: Allow untested optimizers to be used with ZeRO. Currently only Adam is a
                DeepSpeed supported optimizer when using ZeRO.

            logging_batch_size_per_gpu: Config used in DeepSpeed to calculate verbose timing for logging
                on a per sample per second basis (only displayed if logging=logging.INFO).
                To obtain accurate logs when using datasets that do not support batch samplers,
                set this to the actual per gpu batch size.

            config: Pass in a deepspeed formatted config dict,
                or path to a deepspeed config: https://www.deepspeed.ai/docs/config-json.
                All defaults will be ignored if a config is passed in.

            logging_level: Set logging level for deepspeed.

            loss_scale: Loss scaling value for FP16 training.
                0.0 results in dynamic loss scaling, otherwise static.

            initial_scale_power: Power of the initial dynamic loss scale value. Loss scale is computed
                by ``2^initial_scale_power``.

            loss_scale_window: Window in which to raise/lower the dynamic FP16 loss scaling value.

            hysteresis: FP16 Delay shift in Dynamic Loss scaling.

            min_loss_scale: The minimum FP16 dynamic loss scaling value.

            partition_activations: Enables partition activation when used with ZeRO stage 3 and model parallelism.
                Still requires you to wrap your forward functions in deepspeed.checkpointing.checkpoint.
                See `deepspeed tutorial
                <https://www.deepspeed.ai/tutorials/megatron/#deepspeed-activation-checkpoints-optional>`_.

            cpu_checkpointing: Offloads partitioned activations to CPU if ``partition_activations`` is enabled.

            contiguous_memory_optimization: Copies partitioned activations so that they are contiguous in memory.
                Not supported by all models.

            synchronize_checkpoint_boundary: Insert :func:`torch.cuda.synchronize` at each checkpoint boundary.

            load_full_weights: True when loading a single checkpoint file containing the model state dict
                when using ZeRO Stage 3. This differs from the DeepSpeed checkpoint which contains shards
                per worker.

        """
        if not _DEEPSPEED_AVAILABLE:
            raise ImportError(
                "To use the `DeepSpeedStrategy`, you must have DeepSpeed installed."
                " Install it by running `pip install -U deepspeed`."
            )

        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            precision=precision,
            process_group_backend=process_group_backend,
        )
        self._backward_sync_control = None  # DeepSpeed handles gradient accumulation internally
        self._timeout: Optional[timedelta] = timeout

        self.config = self._load_config(config)
        if self.config is None:
            # User has not overridden config, set defaults
            self.config = self._create_default_config(
                zero_optimization,
                zero_allow_untested_optimizer,
                logging_batch_size_per_gpu,
                offload_optimizer=offload_optimizer,
                offload_parameters=offload_parameters,
                nvme_path=nvme_path,
                offload_params_device=offload_params_device,
                params_buffer_count=params_buffer_count,
                params_buffer_size=params_buffer_size,
                max_in_cpu=max_in_cpu,
                pin_memory=pin_memory,
                offload_optimizer_device=offload_optimizer_device,
                optimizer_buffer_count=optimizer_buffer_count,
                block_size=block_size,
                queue_depth=queue_depth,
                single_submit=single_submit,
                overlap_events=overlap_events,
                thread_count=thread_count,
                partition_activations=partition_activations,
                cpu_checkpointing=cpu_checkpointing,
                contiguous_memory_optimization=contiguous_memory_optimization,
                synchronize_checkpoint_boundary=synchronize_checkpoint_boundary,
                stage=stage,
                contiguous_gradients=contiguous_gradients,
                overlap_comm=overlap_comm,
                allgather_partitions=allgather_partitions,
                reduce_scatter=reduce_scatter,
                allgather_bucket_size=allgather_bucket_size,
                reduce_bucket_size=reduce_bucket_size,
                sub_group_size=sub_group_size,
            )

        import deepspeed

        self._config_initialized = False
        deepspeed.utils.logging.logger.setLevel(logging_level)

        self.remote_device = remote_device
        self.load_full_weights = load_full_weights

        # default FP16 parameters.
        self.loss_scale = loss_scale
        self.initial_scale_power = initial_scale_power
        self.loss_scale_window = loss_scale_window
        self.hysteresis = hysteresis
        self.min_loss_scale = min_loss_scale

        self._deepspeed_engine: Optional[DeepSpeedEngine] = None

    @property
    def zero_stage_3(self) -> bool:
        assert isinstance(self.config, dict)
        zero_optimization = self.config.get("zero_optimization")
        return zero_optimization is not None and zero_optimization.get("stage") == 3

    @property
    @override
    def distributed_sampler_kwargs(self) -> dict[str, int]:
        return {"num_replicas": self.world_size, "rank": self.global_rank}

    @property
    def model(self) -> "DeepSpeedEngine":
        return self._deepspeed_engine

    @override
    def setup_module_and_optimizers(
        self, module: Module, optimizers: list[Optimizer]
    ) -> tuple["DeepSpeedEngine", list[Optimizer]]:
        """Set up a model and multiple optimizers together.

        Currently, only a single optimizer is supported.

        Return:
            The model wrapped into a :class:`deepspeed.DeepSpeedEngine` and a list with a single
            deepspeed optimizer.

        """
        if len(optimizers) != 1:
            raise ValueError(
                f"Currently only one optimizer is supported with DeepSpeed. Got {len(optimizers)} optimizers instead."
            )

        self._deepspeed_engine, optimizer = self._initialize_engine(module, optimizers[0])
        self._set_deepspeed_activation_checkpointing()
        return self._deepspeed_engine, [optimizer]

    @override
    def setup_module(self, module: Module) -> "DeepSpeedEngine":
        """Set up a module for inference (no optimizers).

        For training, see :meth:`setup_module_and_optimizers`.

        """
        self._deepspeed_engine, _ = self._initialize_engine(module)
        return self._deepspeed_engine

    @override
    def setup_optimizer(self, optimizer: Optimizer) -> Optimizer:
        """Optimizers can only be set up jointly with the model in this strategy.

        Please use :meth:`setup_module_and_optimizers` to set up both module and optimizer together.

        """
        raise NotImplementedError(self._err_msg_joint_setup_required())

    @override
    def module_init_context(self, empty_init: Optional[bool] = None) -> AbstractContextManager:
        if self.zero_stage_3 and empty_init is False:
            raise NotImplementedError(
                f"`{empty_init=}` is not a valid choice with `DeepSpeedStrategy` when ZeRO stage 3 is enabled."
            )
        module_sharded_ctx = self.module_sharded_context()
        stack = ExitStack()
        if not self.zero_stage_3:
            stack.enter_context(super().module_init_context(empty_init=empty_init))
        stack.enter_context(module_sharded_ctx)
        return stack

    @override
    def module_sharded_context(self) -> AbstractContextManager:
        # Current limitation in Fabric: The config needs to be fully determined at the time of calling the context
        # manager. Later modifications through e.g. `Fabric.setup()` won't have an effect here.

        import deepspeed

        assert self._config_initialized
        assert self.config is not None

        if (
            "zero_optimization" in self.config
            and "mics_shard_size" in self.config["zero_optimization"]
            and self.config["zero_optimization"]["mics_shard_size"] > 0
            and self.zero_stage_3
        ):
            return deepspeed.zero.MiCS_Init(
                enabled=self.zero_stage_3,
                remote_device=self.remote_device,
                config_dict_or_path=self.config,
            )
        else:
            return deepspeed.zero.Init(
                enabled=self.zero_stage_3,
                remote_device=self.remote_device,
                config_dict_or_path=self.config,
            )

    @override
    def save_checkpoint(
        self,
        path: _PATH,
        state: dict[str, Union[Module, Optimizer, Any]],
        storage_options: Optional[Any] = None,
        filter: Optional[dict[str, Callable[[str, Any], bool]]] = None,
    ) -> None:
        """Save model, optimizer, and other state in a checkpoint directory.

        Args:
            path: A path to where the files should be saved
            state: A dictionary with contents to be saved. If the dict contains modules or optimizers, their
                state-dict will be retrieved and converted automatically.
            storage_options: Unused by this strategy, since it doesn't use a ``CheckpointIO`` plugin.
            filter: Unsupported.

        Raises:
            TypeError:
                If the unused ``storage_options`` gets passed.
            ValueError:
                When no :class:`deepspeed.DeepSpeedEngine` objects were found in the state, or when multiple
                :class:`deepspeed.DeepSpeedEngine` objects were found.

        """
        if storage_options is not None:
            raise TypeError(
                "`DeepSpeedStrategy.save_checkpoint(..., storage_options=...)` is not supported because"
                " `DeepSpeedStrategy` does not use the `CheckpointIO`."
            )
        if filter is not None:
            raise TypeError(
                "`DeepSpeedStrategy.save_checkpoint(..., filter=...)` is not supported because"
                " `DeepSpeedStrategy` manages the state serialization internally."
            )

        engines = _get_deepspeed_engines_from_state(state)
        if len(engines) == 0:
            raise ValueError(
                "Could not find a DeepSpeed model in the provided checkpoint state. Please provide the model as"
                " part of the state like so: `save_checkpoint(..., state={'model': model, ...})`. Make sure"
                " you set up the model (and optimizers if any) through the strategy before saving the checkpoint."
            )
        if len(engines) > 1:
            raise ValueError(
                "Found multiple DeepSpeed engine modules in the given state. Saving checkpoints with DeepSpeed is"
                " currently limited to a single model per checkpoint. To save multiple models, call the"
                " save method for each model separately with a different path."
            )
        engine = engines[0]

        # broadcast the path from rank 0 to ensure all the states are saved in a common path
        path = self.broadcast(path)

        # split the checkpoint into two parts:
        # 1) the deepspeed engine encapsulating both the model and optionally the optimizer(s)
        # 2) the rest of the user's state, which in deepspeed is called `client state`
        excluded_objects = (engine, engine.optimizer) if engine.optimizer is not None else (engine,)
        state = {k: v for k, v in state.items() if v not in excluded_objects}
        _validate_state_keys(state)
        # there might be other stateful objects unrelated to the deepspeed engine - convert them to a state_dict
        state = self._convert_stateful_objects_in_state(state, filter={})
        # use deepspeed's internal checkpointing function to handle partitioned weights across processes
        engine.save_checkpoint(path, client_state=state, tag="checkpoint")

    @override
    def load_checkpoint(
        self,
        path: _PATH,
        state: Optional[Union[Module, Optimizer, dict[str, Union[Module, Optimizer, Any]]]] = None,
        strict: bool = True,
    ) -> dict[str, Any]:
        """Load the contents from a checkpoint and restore the state of the given objects.

        Args:
            path: A path to where the file is located
            state: A dictionary of objects whose state will be restored in-place from the checkpoint path.
                This should contain exactly one model, and the model must already be set up by DeepSpeed.
            strict: Whether to enforce that the keys in `state` match the keys in the checkpoint.

        Returns:
            Dictionary with the state inside DeepSpeed's engine

        Raises:
            ValueError:
                If no state is provided, when no :class:`deepspeed.DeepSpeedEngine` objects were found in the
                state, or when multiple :class:`deepspeed.DeepSpeedEngine` objects were found.
            RuntimeError:
                If DeepSpeed was unable to load the checkpoint due to missing files or because the checkpoint is
                not in the expected DeepSpeed format.

        """
        if isinstance(state, (Module, Optimizer)) or self.load_full_weights and self.zero_stage_3:
            # This code path to enables loading a checkpoint from a non-deepspeed checkpoint or from
            # a consolidated checkpoint
            path = self.broadcast(path)
            return super().load_checkpoint(path=path, state=state, strict=strict)

        if not state:
            raise ValueError(
                f"Got DeepSpeedStrategy.load_checkpoint(..., state={state!r}) but a state with at least "
                f" a model instance to reload is required. Pass it in like so:"
                " DeepSpeedStrategy.load_checkpoint(..., state={'model': model, ...})"
            )
        _validate_checkpoint_directory(path)

        engines = _get_deepspeed_engines_from_state(state)
        if len(engines) == 0:
            raise ValueError(
                "Could not find a DeepSpeed model in the provided checkpoint state. Please provide the model as"
                " part of the state like so: `load_checkpoint(..., state={'model': model, ...})`. Make sure"
                " you set up the model (and optimizers if any) through the strategy before loading the checkpoint."
            )
        if len(engines) > 1:
            raise ValueError(
                "Found multiple DeepSpeed engine modules in the given state. Saving and loading checkpoints"
                " with DeepSpeed is currently limited to a single model per checkpoint. To load multiple model"
                " states, call the load method for each model checkpoint separately."
            )
        engine = engines[0]

        if _DEEPSPEED_GREATER_EQUAL_0_14_1:
            from deepspeed.runtime.base_optimizer import DeepSpeedOptimizer
        else:
            from deepspeed.runtime import DeepSpeedOptimizer

        optimzer_state_requested = any(isinstance(item, (Optimizer, DeepSpeedOptimizer)) for item in state.values())

        torch.cuda.empty_cache()
        _, client_state = engine.load_checkpoint(
            path,
            tag="checkpoint",
            load_optimizer_states=optimzer_state_requested,
            load_lr_scheduler_states=False,
            load_module_strict=strict,
        )

        if client_state is None:
            raise RuntimeError(
                "DeepSpeed was unable to load the checkpoint. Ensure you passed in a DeepSpeed compatible checkpoint"
                " or a single checkpoint file by setting `DeepSpeedStrategy(..., load_full_weights=True)`."
            )

        # `Engine.load_checkpoint` adds useless keys 'optimizer' and 'lr_scheduler' to the client state; remove
        # them to avoid name collision with user state
        keys = set(client_state) & set(state) - {"optimizer", "lr_scheduler"}
        _move_state_into(source=client_state, destination=state, keys=keys)
        return client_state

    @override
    def clip_gradients_norm(
        self,
        module: "DeepSpeedEngine",
        optimizer: Optimizer,
        max_norm: Union[float, int],
        norm_type: Union[float, int] = 2.0,
        error_if_nonfinite: bool = True,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "DeepSpeed handles gradient clipping automatically within the optimizer. "
            "Make sure to set the `gradient_clipping` value in your Config."
        )

    @override
    def clip_gradients_value(
        self, module: "DeepSpeedEngine", optimizer: Optimizer, clip_val: Union[float, int]
    ) -> None:
        raise NotImplementedError(
            "DeepSpeed handles gradient clipping automatically within the optimizer. "
            "Make sure to set the `gradient_clipping` value in your Config."
        )

    @classmethod
    @override
    def register_strategies(cls, strategy_registry: _StrategyRegistry) -> None:
        strategy_registry.register("deepspeed", cls, description="Default DeepSpeed Strategy")
        strategy_registry.register("deepspeed_stage_1", cls, description="DeepSpeed with ZeRO Stage 1 enabled", stage=1)
        strategy_registry.register(
            "deepspeed_stage_1_offload",
            cls,
            description="DeepSpeed with ZeRO Stage 1 and optimizer CPU Offload",
            stage=1,
            offload_optimizer=True,
        )
        strategy_registry.register("deepspeed_stage_2", cls, description="DeepSpeed with ZeRO Stage 2 enabled", stage=2)
        strategy_registry.register(
            "deepspeed_stage_2_offload",
            cls,
            description="DeepSpeed ZeRO Stage 2 and CPU Offload",
            stage=2,
            offload_optimizer=True,
        )
        strategy_registry.register("deepspeed_stage_3", cls, description="DeepSpeed ZeRO Stage 3", stage=3)
        strategy_registry.register(
            "deepspeed_stage_3_offload",
            cls,
            description="DeepSpeed ZeRO Stage 3 and CPU Offload",
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
        )
        strategy_registry.register(
            "deepspeed_stage_3_offload_nvme",
            cls,
            description="DeepSpeed ZeRO Stage 3 and NVMe Offload",
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
            remote_device="nvme",
            offload_params_device="nvme",
            offload_optimizer_device="nvme",
        )

    def _initialize_engine(
        self,
        model: Module,
        optimizer: Optional[Optimizer] = None,
    ) -> tuple["DeepSpeedEngine", Optimizer]:
        """Initialize one model and one optimizer with an optional learning rate scheduler.

        This calls ``deepspeed.initialize`` internally.

        """
        import deepspeed

        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        deepspeed_engine, deepspeed_optimizer, _, _ = deepspeed.initialize(
            args=argparse.Namespace(device_rank=self.root_device.index),
            config=self.config,
            model=model,
            model_parameters=model_parameters,
            optimizer=optimizer,
            dist_init_required=False,
        )
        return deepspeed_engine, deepspeed_optimizer

    @override
    def setup_environment(self) -> None:
        if not isinstance(self.accelerator, CUDAAccelerator):
            raise RuntimeError(
                f"The DeepSpeed strategy is only supported on CUDA GPUs but `{self.accelerator.__class__.__name__}`"
                " is used."
            )
        super().setup_environment()

    @override
    def _setup_distributed(self) -> None:
        assert self.parallel_devices is not None
        _validate_device_index_selection(self.parallel_devices)
        reset_seed()
        self._set_world_ranks()
        self._init_deepspeed_distributed()
        if not self._config_initialized:
            self._format_config()
            self._config_initialized = True

    def _init_deepspeed_distributed(self) -> None:
        import deepspeed

        assert self.cluster_environment is not None
        if platform.system() != "Windows":
            # do not set env variables on windows, allow deepspeed to control setup
            self._set_node_environment_variables()
            log.info(
                "initializing deepspeed distributed: "
                f"GLOBAL_RANK: {self.global_rank}, "
                f"MEMBER: {self.global_rank + 1}/{self.world_size}"
            )
        self._process_group_backend = self._get_process_group_backend()
        deepspeed.init_distributed(
            self._process_group_backend, distributed_port=self.cluster_environment.main_port, timeout=self._timeout
        )

    def _set_node_environment_variables(self) -> None:
        assert self.cluster_environment is not None
        os.environ["MASTER_ADDR"] = self.cluster_environment.main_address
        os.environ["MASTER_PORT"] = str(self.cluster_environment.main_port)
        os.environ["RANK"] = str(self.global_rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["LOCAL_RANK"] = str(self.local_rank)

    def _set_deepspeed_activation_checkpointing(self) -> None:
        import deepspeed

        assert isinstance(self.config, dict)
        if self.config.get("activation_checkpointing"):
            checkpoint_config = self.config["activation_checkpointing"]
            deepspeed.checkpointing.configure(
                mpu_=None,
                partition_activations=checkpoint_config.get("partition_activations"),
                contiguous_checkpointing=checkpoint_config.get("contiguous_memory_optimization"),
                checkpoint_in_cpu=checkpoint_config.get("cpu_checkpointing"),
                profile=checkpoint_config.get("profile"),
            )

    def _format_config(self) -> None:
        if self.config is None:
            raise ValueError(
                "To use DeepSpeed you must pass in a DeepSpeed config dict, or a path to a JSON config."
                " See: https://lightning.ai/docs/pytorch/stable/advanced/model_parallel.html#deepspeed"
            )

        self.config.setdefault("train_micro_batch_size_per_gpu", 1)
        _format_precision_config(
            config=self.config,
            precision=self.precision.precision,
            loss_scale=self.loss_scale,
            loss_scale_window=self.loss_scale_window,
            min_loss_scale=self.min_loss_scale,
            initial_scale_power=self.initial_scale_power,
            hysteresis=self.hysteresis,
        )

    def _create_default_config(
        self,
        zero_optimization: bool,
        zero_allow_untested_optimizer: bool,
        logging_batch_size_per_gpu: Optional[int],
        partition_activations: bool,
        cpu_checkpointing: bool,
        contiguous_memory_optimization: bool,
        synchronize_checkpoint_boundary: bool,
        offload_optimizer: bool,
        offload_parameters: bool,
        nvme_path: str,
        offload_params_device: str,
        params_buffer_count: int,
        params_buffer_size: int,
        max_in_cpu: int,
        offload_optimizer_device: str,
        optimizer_buffer_count: int,
        pin_memory: bool,
        block_size: int,
        queue_depth: int,
        single_submit: bool,
        overlap_events: bool,
        thread_count: int,
        **zero_kwargs: Any,
    ) -> dict:
        cfg = {
            "activation_checkpointing": {
                "partition_activations": partition_activations,
                "cpu_checkpointing": cpu_checkpointing,
                "contiguous_memory_optimization": contiguous_memory_optimization,
                "synchronize_checkpoint_boundary": synchronize_checkpoint_boundary,
            },
            "aio": {
                "block_size": block_size,
                "queue_depth": queue_depth,
                "single_submit": single_submit,
                "overlap_events": overlap_events,
                "thread_count": thread_count,
            },
        }
        if zero_optimization:
            zero_config = zero_kwargs

            if offload_optimizer:
                zero_config["offload_optimizer"] = {
                    "device": offload_optimizer_device,
                    "nvme_path": nvme_path,
                    "buffer_count": optimizer_buffer_count,
                    "pin_memory": pin_memory,
                }
            if offload_parameters:
                zero_config["offload_param"] = {
                    "device": offload_params_device,
                    "nvme_path": nvme_path,
                    "buffer_count": params_buffer_count,
                    "buffer_size": params_buffer_size,
                    "max_in_cpu": max_in_cpu,
                    "pin_memory": pin_memory,
                }
            cfg.update({
                "zero_allow_untested_optimizer": zero_allow_untested_optimizer,
                "zero_optimization": zero_config,
            })
        if logging_batch_size_per_gpu:
            cfg["train_micro_batch_size_per_gpu"] = logging_batch_size_per_gpu
        return cfg

    def _restore_zero_state(self, module: Module, ckpt: Mapping[str, Any]) -> None:
        """Overrides the normal load_state_dict behaviour in PyTorch to ensure we gather parameters that may be sharded
        across processes before loading the state dictionary when using ZeRO stage 3. This is then automatically synced
        across processes.

        Args:
            ckpt: The ckpt file.

        """
        import deepspeed

        def load(module: torch.nn.Module, prefix: str = "") -> None:
            missing_keys: list[str] = []
            unexpected_keys: list[str] = []
            error_msgs: list[str] = []
            state_dict = ckpt["state_dict"]

            # copy state_dict so _load_from_state_dict can modify it
            metadata = getattr(state_dict, "_metadata", None)
            state_dict = state_dict.copy()
            if metadata is not None:
                state_dict._metadata = metadata

            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            # because zero3 puts placeholders in model params, this context
            # manager gathers (unpartitions) the params of the current layer, then loads from
            # the state dict and then re-partitions them again
            with deepspeed.zero.GatheredParameters(list(module.parameters(recurse=False)), modifier_rank=0):
                if self.is_global_zero:
                    module._load_from_state_dict(
                        state_dict=state_dict,
                        prefix=prefix,
                        local_metadata=local_metadata,
                        strict=True,
                        missing_keys=missing_keys,
                        unexpected_keys=unexpected_keys,
                        error_msgs=error_msgs,
                    )

            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        load(module, prefix="")

    def _load_config(self, config: Optional[Union[_PATH, dict[str, Any]]]) -> Optional[dict[str, Any]]:
        if config is None and self.DEEPSPEED_ENV_VAR in os.environ:
            rank_zero_info(f"Loading DeepSpeed config from set {self.DEEPSPEED_ENV_VAR} environment variable")
            config = os.environ[self.DEEPSPEED_ENV_VAR]
        if isinstance(config, (str, Path)):
            if not os.path.isfile(config):
                raise FileNotFoundError(
                    f"You passed in a path to a DeepSpeed config but the path does not exist: {config}"
                )
            with open(config) as f:
                config = json.load(f)
        assert isinstance(config, dict) or config is None
        return config


def _get_deepspeed_engines_from_state(state: dict[str, Any]) -> list["DeepSpeedEngine"]:
    from deepspeed import DeepSpeedEngine

    modules = chain(*(module.modules() for module in state.values() if isinstance(module, Module)))
    return [engine for engine in modules if isinstance(engine, DeepSpeedEngine)]


def _validate_state_keys(state: dict[str, Any]) -> None:
    # DeepSpeed merges the client state into its internal engine state when saving, but it does not check for
    # colliding keys from the user. We explicitly check it here:
    deepspeed_internal_keys = {
        "module",
        "buffer_names",
        "optimizer",
        "param_shapes",
        "lr_scheduler",
        "sparse_tensor_module_names",
        "skipped_steps",
        "global_steps",
        "global_samples",
        "dp_world_size",
        "mp_world_size",
        "ds_config",
        "ds_version",
    }
    colliding_keys = deepspeed_internal_keys.intersection(state.keys())
    if colliding_keys:
        rank_zero_warn(
            "Your state has keys that collide with DeepSpeed's internal engine state. This could result in your"
            " values being overwritten by DeepSpeed. Consider changing the name of these keys to something else: "
            + ", ".join(colliding_keys)
        )


def _validate_device_index_selection(parallel_devices: list[torch.device]) -> None:
    selected_device_indices = [device.index for device in parallel_devices]
    expected_device_indices = list(range(len(parallel_devices)))
    if selected_device_indices != expected_device_indices:
        raise RuntimeError(
            f"The selected device indices {selected_device_indices!r} don't match the local rank values of processes."
            " If you need to select GPUs at a specific index, set the `CUDA_VISIBLE_DEVICES` environment variable"
            f" instead. For example: `CUDA_VISIBLE_DEVICES={','.join(str(i) for i in selected_device_indices)}`."
        )


def _is_deepspeed_checkpoint(path: Path) -> bool:
    """Heuristic check whether the path points to a top-level DeepSpeed checkpoint directory."""
    return path.is_dir() and (path / "checkpoint").is_dir()


def _validate_checkpoint_directory(path: _PATH) -> None:
    """Validates that the path points to a DeepSpeed checkpoint directory and suggests fixes for user error."""
    # Example DeepSpeed checkpoint directory:
    #
    # epoch=5-step=10999.ckpt
    # ├── checkpoint
    # │   ├── zero_pp_rank_0_mp_rank_00_model_states.pt
    # │   ├── zero_pp_rank_0_mp_rank_00_optim_states.pt
    # │   ├── zero_pp_rank_1_mp_rank_00_model_states.pt
    # │   └── zero_pp_rank_1_mp_rank_00_optim_states.pt
    # ├── latest
    # └── zero_to_fp32.py

    path = Path(path)
    path_is_ds_checkpoint = _is_deepspeed_checkpoint(path)
    default_message = f"The provided path is not a valid DeepSpeed checkpoint: {path}"

    if not path_is_ds_checkpoint:
        # Case 1: User may have accidentally passed the subfolder "checkpoint"
        parent_is_ds_checkpoint = _is_deepspeed_checkpoint(path.parent)
        if parent_is_ds_checkpoint:
            raise FileNotFoundError(
                f"{default_message}. It looks like you passed the path to a subfolder."
                f" Try to load using this parent directory instead: {path.parent}"
            )
        # Case 2: User may have accidentally passed the path to a file inside the "checkpoint" subfolder
        parent_parent_is_ds_checkpoint = path.is_file() and _is_deepspeed_checkpoint(path.parent.parent)
        if parent_parent_is_ds_checkpoint:
            raise FileNotFoundError(
                f"{default_message}. It looks like you passed the path to a file inside a DeepSpeed checkpoint folder."
                f" Try to load using this parent directory instead: {path.parent.parent}"
            )
        raise FileNotFoundError(default_message)


def _format_precision_config(
    config: dict[str, Any],
    precision: str,
    loss_scale: float,
    loss_scale_window: int,
    min_loss_scale: int,
    initial_scale_power: int,
    hysteresis: int,
) -> None:
    if "fp16" not in config and precision in ("16-mixed", "16-true"):
        # FP16 is a DeepSpeed standalone AMP implementation
        rank_zero_info("Enabling DeepSpeed FP16. Model parameters and inputs will be cast to `float16`.")
        config["fp16"] = {
            "enabled": True,
            "loss_scale": loss_scale,
            "initial_scale_power": initial_scale_power,
            "loss_scale_window": loss_scale_window,
            "hysteresis": hysteresis,
            "min_loss_scale": min_loss_scale,
        }
    elif "bf16" not in config and precision in ("bf16-mixed", "bf16-true"):
        rank_zero_info("Enabling DeepSpeed BF16. Model parameters and inputs will be cast to `bfloat16`.")
        config["bf16"] = {"enabled": True}
