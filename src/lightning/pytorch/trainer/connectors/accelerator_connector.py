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
import os
from collections import Counter
from collections.abc import Iterable
from typing import Literal, Optional, Union

import torch

from lightning.fabric.connector import _PRECISION_INPUT, _PRECISION_INPUT_STR, _convert_precision_to_unified_args
from lightning.fabric.plugins.environments import (
    ClusterEnvironment,
    LightningEnvironment,
    LSFEnvironment,
    MPIEnvironment,
    SLURMEnvironment,
    TorchElasticEnvironment,
)
from lightning.fabric.utilities.device_parser import _determine_root_gpu_device
from lightning.fabric.utilities.imports import _IS_INTERACTIVE
from lightning.pytorch.accelerators import AcceleratorRegistry
from lightning.pytorch.accelerators.accelerator import Accelerator
from lightning.pytorch.accelerators.cuda import CUDAAccelerator
from lightning.pytorch.accelerators.mps import MPSAccelerator
from lightning.pytorch.accelerators.xla import XLAAccelerator
from lightning.pytorch.plugins import (
    _PLUGIN_INPUT,
    BitsandbytesPrecision,
    CheckpointIO,
    DeepSpeedPrecision,
    DoublePrecision,
    FSDPPrecision,
    HalfPrecision,
    MixedPrecision,
    Precision,
    TransformerEnginePrecision,
    XLAPrecision,
)
from lightning.pytorch.plugins.layer_sync import LayerSync, TorchSyncBatchNorm
from lightning.pytorch.strategies import (
    DDPStrategy,
    DeepSpeedStrategy,
    FSDPStrategy,
    ModelParallelStrategy,
    ParallelStrategy,
    SingleDeviceStrategy,
    SingleDeviceXLAStrategy,
    Strategy,
    StrategyRegistry,
    XLAStrategy,
)
from lightning.pytorch.strategies.ddp import _DDP_FORK_ALIASES
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.imports import _habana_available_and_importable
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_warn

log = logging.getLogger(__name__)

_LITERAL_WARN = Literal["warn"]


class _AcceleratorConnector:
    def __init__(
        self,
        devices: Union[list[int], str, int] = "auto",
        num_nodes: int = 1,
        accelerator: Union[str, Accelerator] = "auto",
        strategy: Union[str, Strategy] = "auto",
        plugins: Optional[Union[_PLUGIN_INPUT, Iterable[_PLUGIN_INPUT]]] = None,
        precision: Optional[_PRECISION_INPUT] = None,
        sync_batchnorm: bool = False,
        benchmark: Optional[bool] = None,
        use_distributed_sampler: bool = True,
        deterministic: Optional[Union[bool, _LITERAL_WARN]] = None,
    ) -> None:
        """The AcceleratorConnector parses several Trainer arguments and instantiates the Strategy including other
        components such as the Accelerator and Precision plugins.

            A. accelerator flag could be:
                1. accelerator class
                2. accelerator str
                3. accelerator auto

            B. strategy flag could be:
                1. strategy class
                2. strategy str registered with StrategyRegistry

            C. plugins flag could be:
                1. precision class (should be removed, and precision flag should allow user pass classes)
                2. checkpoint_io class
                3. cluster_environment class

        priorities which to take when:
            A. Class > str
            B. Strategy > Accelerator/precision/plugins

        """
        self.use_distributed_sampler = use_distributed_sampler
        _set_torch_flags(deterministic=deterministic, benchmark=benchmark)

        # 1. Parsing flags
        # Get registered strategies, built-in accelerators and precision plugins
        _register_external_accelerators_and_strategies()
        self._registered_strategies = StrategyRegistry.available_strategies()
        self._accelerator_types = AcceleratorRegistry.available_accelerators()

        # Raise an exception if there are conflicts between flags
        # Set each valid flag to `self._x_flag` after validation
        self._strategy_flag: Union[Strategy, str] = "auto"
        self._accelerator_flag: Union[Accelerator, str] = "auto"
        self._precision_flag: _PRECISION_INPUT_STR = "32-true"
        self._precision_plugin_flag: Optional[Precision] = None
        self._cluster_environment_flag: Optional[Union[ClusterEnvironment, str]] = None
        self._parallel_devices: list[Union[int, torch.device, str]] = []
        self._layer_sync: Optional[LayerSync] = TorchSyncBatchNorm() if sync_batchnorm else None
        self.checkpoint_io: Optional[CheckpointIO] = None

        self._check_config_and_set_final_flags(
            strategy=strategy,
            accelerator=accelerator,
            precision=precision,
            plugins=plugins,
            sync_batchnorm=sync_batchnorm,
        )

        # 2. Instantiate Accelerator
        # handle `auto`, `None` and `gpu`
        if self._accelerator_flag == "auto":
            self._accelerator_flag = self._choose_auto_accelerator()
        elif self._accelerator_flag == "gpu":
            self._accelerator_flag = self._choose_gpu_accelerator_backend()
        elif isinstance(self._accelerator_flag, Accelerator):
            pass  # for 3rd party accelerator, just do nothing

        self._check_device_config_and_set_final_flags(devices=devices, num_nodes=num_nodes)
        self._set_parallel_devices_and_init_accelerator()

        # 3. Instantiate ClusterEnvironment
        self.cluster_environment: ClusterEnvironment = self._choose_and_init_cluster_environment()

        # 4. Instantiate Strategy - Part 1
        if self._strategy_flag == "auto":
            self._strategy_flag = self._choose_strategy()
        # In specific cases, ignore user selection and fall back to a different strategy
        self._check_strategy_and_fallback()
        self._init_strategy()

        # 5. Instantiate Precision Plugin
        self.precision_plugin = self._check_and_init_precision()

        # 6. Instantiate Strategy - Part 2
        self._lazy_init_strategy()

    def _check_config_and_set_final_flags(
        self,
        strategy: Union[str, Strategy],
        accelerator: Union[str, Accelerator],
        precision: Optional[_PRECISION_INPUT],
        plugins: Optional[Union[_PLUGIN_INPUT, Iterable[_PLUGIN_INPUT]]],
        sync_batchnorm: bool,
    ) -> None:
        """This method checks:

        1. strategy: whether the strategy name is valid, and sets the internal flags if it is.
        2. accelerator: if the value of the accelerator argument is a type of accelerator (instance or string),
            set self._accelerator_flag accordingly.
        3. precision: The final value of the precision flag may be determined either by the precision argument or
            by a plugin instance.
        4. plugins: The list of plugins may contain a Precision plugin, CheckpointIO, ClusterEnvironment and others.
            Additionally, other flags such as `precision` or `sync_batchnorm` can populate the list with the
            corresponding plugin instances.

        """
        if plugins is not None:
            plugins = [plugins] if not isinstance(plugins, Iterable) else plugins

        if isinstance(strategy, str):
            strategy = strategy.lower()

        self._strategy_flag = strategy

        if strategy != "auto" and strategy not in self._registered_strategies and not isinstance(strategy, Strategy):
            raise ValueError(
                f"You selected an invalid strategy name: `strategy={strategy!r}`."
                " It must be either a string or an instance of `lightning.pytorch.strategies.Strategy`."
                " Example choices: auto, ddp, ddp_spawn, deepspeed, ..."
                " Find a complete list of options in our documentation at https://lightning.ai"
            )

        if (
            accelerator not in self._accelerator_types
            and accelerator not in ("auto", "gpu")
            and not isinstance(accelerator, Accelerator)
        ):
            raise ValueError(
                f"You selected an invalid accelerator name: `accelerator={accelerator!r}`."
                f" Available names are: auto, {', '.join(self._accelerator_types)}."
            )

        # MPS accelerator is incompatible with DDP family of strategies. It supports single-device operation only.
        is_ddp_str = isinstance(strategy, str) and "ddp" in strategy
        is_deepspeed_str = isinstance(strategy, str) and "deepspeed" in strategy
        is_parallel_strategy = isinstance(strategy, ParallelStrategy) or is_ddp_str or is_deepspeed_str
        is_mps_accelerator = MPSAccelerator.is_available() and (
            accelerator in ("mps", "auto", "gpu", None) or isinstance(accelerator, MPSAccelerator)
        )
        if is_mps_accelerator and is_parallel_strategy:
            raise ValueError(
                f"You set `strategy={strategy}` but strategies from the DDP family are not supported on the"
                f" MPS accelerator. Either explicitly set `accelerator='cpu'` or change the strategy."
            )

        self._accelerator_flag = accelerator

        precision_flag = _convert_precision_to_unified_args(precision)

        if plugins:
            plugins_flags_types: dict[str, int] = Counter()
            for plugin in plugins:
                if isinstance(plugin, Precision):
                    self._precision_plugin_flag = plugin
                    plugins_flags_types[Precision.__name__] += 1
                elif isinstance(plugin, CheckpointIO):
                    self.checkpoint_io = plugin
                    plugins_flags_types[CheckpointIO.__name__] += 1
                elif isinstance(plugin, ClusterEnvironment):
                    self._cluster_environment_flag = plugin
                    plugins_flags_types[ClusterEnvironment.__name__] += 1
                elif isinstance(plugin, LayerSync):
                    if sync_batchnorm and not isinstance(plugin, TorchSyncBatchNorm):
                        raise MisconfigurationException(
                            f"You set `Trainer(sync_batchnorm=True)` and provided a `{plugin.__class__.__name__}`"
                            " plugin, but this is not allowed. Choose one or the other."
                        )
                    self._layer_sync = plugin
                    plugins_flags_types[TorchSyncBatchNorm.__name__] += 1
                else:
                    raise MisconfigurationException(
                        f"Found invalid type for plugin {plugin}. Expected one of: Precision, "
                        "CheckpointIO, ClusterEnviroment, or LayerSync."
                    )

            duplicated_plugin_key = [k for k, v in plugins_flags_types.items() if v > 1]
            if duplicated_plugin_key:
                raise MisconfigurationException(
                    f"Received multiple values for {', '.join(duplicated_plugin_key)} flags in `plugins`."
                    " Expected one value for each type at most."
                )

            if plugins_flags_types.get(Precision.__name__) and precision_flag is not None:
                raise ValueError(
                    f"Received both `precision={precision_flag}` and `plugins={self._precision_plugin_flag}`."
                    f" Choose one."
                )

        self._precision_flag = "32-true" if precision_flag is None else precision_flag

        # handle the case when the user passes in a strategy instance which has an accelerator, precision,
        # checkpoint io or cluster env set up
        # TODO: improve the error messages below
        if self._strategy_flag and isinstance(self._strategy_flag, Strategy):
            if self._strategy_flag._accelerator:
                if self._accelerator_flag != "auto":
                    raise MisconfigurationException(
                        "accelerator set through both strategy class and accelerator flag, choose one"
                    )
                self._accelerator_flag = self._strategy_flag._accelerator
            if self._strategy_flag._precision_plugin:
                # [RFC] handle precision plugin set up conflict?
                if self._precision_plugin_flag:
                    raise MisconfigurationException("precision set through both strategy class and plugins, choose one")
                self._precision_plugin_flag = self._strategy_flag._precision_plugin
            if self._strategy_flag._checkpoint_io:
                if self.checkpoint_io:
                    raise MisconfigurationException(
                        "checkpoint_io set through both strategy class and plugins, choose one"
                    )
                self.checkpoint_io = self._strategy_flag._checkpoint_io
            if getattr(self._strategy_flag, "cluster_environment", None):
                if self._cluster_environment_flag:
                    raise MisconfigurationException(
                        "cluster_environment set through both strategy class and plugins, choose one"
                    )
                self._cluster_environment_flag = getattr(self._strategy_flag, "cluster_environment")

            if hasattr(self._strategy_flag, "parallel_devices") and self._strategy_flag.parallel_devices:
                if self._strategy_flag.parallel_devices[0].type == "cpu":
                    if self._accelerator_flag and self._accelerator_flag not in ("auto", "cpu"):
                        raise MisconfigurationException(
                            f"CPU parallel_devices set through {self._strategy_flag.__class__.__name__} class,"
                            f" but accelerator set to {self._accelerator_flag}, please choose one device type"
                        )
                    self._accelerator_flag = "cpu"
                elif self._strategy_flag.parallel_devices[0].type == "cuda":
                    if self._accelerator_flag and self._accelerator_flag not in ("auto", "cuda", "gpu"):
                        raise MisconfigurationException(
                            f"GPU parallel_devices set through {self._strategy_flag.__class__.__name__} class,"
                            f" but accelerator set to {self._accelerator_flag}, please choose one device type"
                        )
                    self._accelerator_flag = "cuda"
                else:
                    pass  # 3rd party accelerator
                self._parallel_devices = self._strategy_flag.parallel_devices

    def _check_device_config_and_set_final_flags(self, devices: Union[list[int], str, int], num_nodes: int) -> None:
        if not isinstance(num_nodes, int) or num_nodes < 1:
            raise ValueError(f"`num_nodes` must be a positive integer, but got {num_nodes}.")

        self._num_nodes_flag = num_nodes
        self._devices_flag = devices

        if self._devices_flag in ([], 0, "0"):
            accelerator_name = (
                self._accelerator_flag.__class__.__qualname__
                if isinstance(self._accelerator_flag, Accelerator)
                else self._accelerator_flag
            )
            raise MisconfigurationException(
                f"`Trainer(devices={self._devices_flag!r})` value is not a valid input"
                f" using {accelerator_name} accelerator."
            )

    @staticmethod
    def _choose_auto_accelerator() -> str:
        """Choose the accelerator type (str) based on availability."""
        if XLAAccelerator.is_available():
            return "tpu"
        if _habana_available_and_importable():
            from lightning_habana import HPUAccelerator

            if HPUAccelerator.is_available():
                return "hpu"
        if MPSAccelerator.is_available():
            return "mps"
        if CUDAAccelerator.is_available():
            return "cuda"
        return "cpu"

    @staticmethod
    def _choose_gpu_accelerator_backend() -> str:
        if MPSAccelerator.is_available():
            return "mps"
        if CUDAAccelerator.is_available():
            return "cuda"
        raise MisconfigurationException("No supported gpu backend found!")

    def _set_parallel_devices_and_init_accelerator(self) -> None:
        if isinstance(self._accelerator_flag, Accelerator):
            self.accelerator: Accelerator = self._accelerator_flag
        else:
            self.accelerator = AcceleratorRegistry.get(self._accelerator_flag)
        accelerator_cls = self.accelerator.__class__

        if not accelerator_cls.is_available():
            available_accelerator = [
                acc_str
                for acc_str in self._accelerator_types
                if AcceleratorRegistry[acc_str]["accelerator"].is_available()
            ]
            raise MisconfigurationException(
                f"`{accelerator_cls.__qualname__}` can not run on your system"
                " since the accelerator is not available. The following accelerator(s)"
                " is available and can be passed into `accelerator` argument of"
                f" `Trainer`: {available_accelerator}."
            )

        self._set_devices_flag_if_auto_passed()
        self._devices_flag = accelerator_cls.parse_devices(self._devices_flag)
        if not self._parallel_devices:
            self._parallel_devices = accelerator_cls.get_parallel_devices(self._devices_flag)

    def _set_devices_flag_if_auto_passed(self) -> None:
        if self._devices_flag != "auto":
            return
        if (
            _IS_INTERACTIVE
            and isinstance(self.accelerator, CUDAAccelerator)
            and self.accelerator.auto_device_count() > 1
        ):
            self._devices_flag = 1
            rank_zero_info(
                f"Trainer will use only 1 of {self.accelerator.auto_device_count()} GPUs because it is running inside"
                " an interactive / notebook environment. You may try to set `Trainer(devices="
                f"{self.accelerator.auto_device_count()})` but please note that multi-GPU inside interactive /"
                " notebook environments is considered experimental and unstable. Your mileage may vary."
            )
        else:
            self._devices_flag = self.accelerator.auto_device_count()

    def _choose_and_init_cluster_environment(self) -> ClusterEnvironment:
        if isinstance(self._cluster_environment_flag, ClusterEnvironment):
            return self._cluster_environment_flag
        for env_type in (
            # TorchElastic has the highest priority since it can also be used inside SLURM
            TorchElasticEnvironment,
            SLURMEnvironment,
            LSFEnvironment,
            MPIEnvironment,
        ):
            if env_type.detect():
                return env_type()
        return LightningEnvironment()

    def _choose_strategy(self) -> Union[Strategy, str]:
        if _habana_available_and_importable():
            from lightning_habana import HPUAccelerator

            if self._accelerator_flag == "hpu" or isinstance(self._accelerator_flag, HPUAccelerator):
                if self._parallel_devices and len(self._parallel_devices) > 1:
                    from lightning_habana import HPUParallelStrategy

                    return HPUParallelStrategy.strategy_name

                from lightning_habana import SingleHPUStrategy

                return SingleHPUStrategy(device=torch.device("hpu"))
        if self._accelerator_flag == "hpu" and not _habana_available_and_importable():
            raise ImportError(
                "You asked to run with HPU but you are missing a required dependency."
                " Please run `pip install lightning-habana` or seek further instructions"
                " in https://github.com/Lightning-AI/lightning-Habana/."
            )

        if self._accelerator_flag == "tpu" or isinstance(self._accelerator_flag, XLAAccelerator):
            if self._parallel_devices and len(self._parallel_devices) > 1:
                return XLAStrategy.strategy_name
            # TODO: lazy initialized device, then here could be self._strategy_flag = "single_xla"
            return SingleDeviceXLAStrategy(device=self._parallel_devices[0])
        if self._num_nodes_flag > 1:
            return "ddp"
        if len(self._parallel_devices) <= 1:
            if isinstance(self._accelerator_flag, (CUDAAccelerator, MPSAccelerator)) or (
                isinstance(self._accelerator_flag, str) and self._accelerator_flag in ("cuda", "gpu", "mps")
            ):
                device = _determine_root_gpu_device(self._parallel_devices)
            else:
                device = "cpu"
            # TODO: lazy initialized device, then here could be self._strategy_flag = "single_device"
            return SingleDeviceStrategy(device=device)  # type: ignore
        if len(self._parallel_devices) > 1 and _IS_INTERACTIVE:
            return "ddp_fork"
        return "ddp"

    def _check_strategy_and_fallback(self) -> None:
        """Checks edge cases when the strategy selection was a string input, and we need to fall back to a different
        choice depending on other parameters or the environment."""
        # current fallback and check logic only apply to user pass in str config and object config
        # TODO this logic should apply to both str and object config
        strategy_flag = "" if isinstance(self._strategy_flag, Strategy) else self._strategy_flag

        if (
            (strategy_flag in FSDPStrategy.get_registered_strategies() or type(self._strategy_flag) is FSDPStrategy)
            and self._accelerator_flag not in ("cuda", "gpu")
            and isinstance(self._accelerator_flag, str)
        ):
            raise ValueError(
                f"The strategy `{FSDPStrategy.strategy_name}` requires a GPU accelerator, but got:"
                f" {self._accelerator_flag}"
            )
        if isinstance(self._accelerator_flag, Accelerator):
            Warning(
                f"Using a custom accelerator `{self._accelerator_flag.__class__.__name__}`."
                f" Please ensure it is compatible with the selected strategy `{strategy_flag}`."
            )
        if strategy_flag in _DDP_FORK_ALIASES and "fork" not in torch.multiprocessing.get_all_start_methods():
            raise ValueError(
                f"You selected `Trainer(strategy='{strategy_flag}')` but process forking is not supported on this"
                f" platform. We recommed `Trainer(strategy='ddp_spawn')` instead."
            )
        if strategy_flag:
            self._strategy_flag = strategy_flag

    def _init_strategy(self) -> None:
        """Instantiate the Strategy given depending on the setting of ``_strategy_flag``."""
        # The validation of `_strategy_flag` already happened earlier on in the connector
        assert isinstance(self._strategy_flag, (str, Strategy))
        if isinstance(self._strategy_flag, str):
            self.strategy = StrategyRegistry.get(self._strategy_flag)
        else:
            self.strategy = self._strategy_flag

    def _check_and_init_precision(self) -> Precision:
        self._validate_precision_choice()
        if isinstance(self._precision_plugin_flag, Precision):
            return self._precision_plugin_flag

        if _habana_available_and_importable():
            from lightning_habana import HPUAccelerator, HPUPrecisionPlugin

            if isinstance(self.accelerator, HPUAccelerator):
                return HPUPrecisionPlugin(self._precision_flag)

        if isinstance(self.strategy, (SingleDeviceXLAStrategy, XLAStrategy)):
            return XLAPrecision(self._precision_flag)  # type: ignore
        if isinstance(self.strategy, DeepSpeedStrategy):
            return DeepSpeedPrecision(self._precision_flag)  # type: ignore[arg-type]
        if isinstance(self.strategy, FSDPStrategy):
            return FSDPPrecision(
                precision=self._precision_flag,  # type: ignore[arg-type]
                device_type=self._accelerator_flag.get_device_type()
                if isinstance(self._accelerator_flag, Accelerator)
                else None,
            )
        if self._precision_flag in ("16-true", "bf16-true"):
            return HalfPrecision(self._precision_flag)  # type: ignore
        if self._precision_flag == "32-true":
            return Precision()
        if self._precision_flag == "64-true":
            return DoublePrecision()
        if self._precision_flag == "transformer-engine":
            return TransformerEnginePrecision(weights_dtype=torch.bfloat16)
        if self._precision_flag == "transformer-engine-float16":
            return TransformerEnginePrecision(weights_dtype=torch.float16)

        if self._precision_flag == "16-mixed" and self._accelerator_flag == "cpu":
            rank_zero_warn(
                "You passed `Trainer(accelerator='cpu', precision='16-mixed')` but AMP with fp16 is not supported on "
                "CPU. Using `precision='bf16-mixed'` instead."
            )
            self._precision_flag = "bf16-mixed"

        if self._precision_flag in ("16-mixed", "bf16-mixed"):
            rank_zero_info(
                f"Using {'16bit' if self._precision_flag == '16-mixed' else 'bfloat16'} Automatic Mixed Precision (AMP)"
            )
            device = "cpu" if self._accelerator_flag == "cpu" else "cuda"
            if isinstance(self._accelerator_flag, Accelerator):
                device = self._accelerator_flag.get_device_type()
            return MixedPrecision(self._precision_flag, device)  # type: ignore[arg-type]

        raise RuntimeError("No precision set")

    def _validate_precision_choice(self) -> None:
        """Validate the combination of choices for precision, AMP type, and accelerator."""
        if isinstance(self._precision_plugin_flag, BitsandbytesPrecision) and not isinstance(
            self.accelerator, CUDAAccelerator
        ):
            raise RuntimeError("Bitsandbytes is only supported on CUDA GPUs.")
        mp_precision_supported = ("32-true", "bf16-mixed", "bf16-true", "16-true")
        if (
            isinstance(self._strategy_flag, ModelParallelStrategy)
            and self._precision_flag not in mp_precision_supported
        ):
            raise ValueError(
                f"The `ModelParallelStrategy` does not support `Fabric(..., precision={self._precision_flag!r})`."
                f" Choose a different precision among: {', '.join(mp_precision_supported)}."
            )

        if _habana_available_and_importable():
            from lightning_habana import HPUAccelerator

            if isinstance(self.accelerator, HPUAccelerator) and self._precision_flag not in (
                "16-mixed",
                "bf16-mixed",
                "32-true",
            ):
                raise MisconfigurationException(
                    f"`Trainer(accelerator='hpu', precision={self._precision_flag!r})` is not supported."
                )

    def _lazy_init_strategy(self) -> None:
        """Lazily set missing attributes on the previously instantiated strategy."""
        self.strategy.accelerator = self.accelerator
        if self.precision_plugin:
            self.strategy.precision_plugin = self.precision_plugin
        if self.checkpoint_io:
            self.strategy.checkpoint_io = self.checkpoint_io
        if hasattr(self.strategy, "cluster_environment"):
            if self.strategy.cluster_environment is None:
                self.strategy.cluster_environment = self.cluster_environment
            self.cluster_environment = self.strategy.cluster_environment
        if hasattr(self.strategy, "parallel_devices"):
            if self.strategy.parallel_devices:
                self._parallel_devices = self.strategy.parallel_devices
            else:
                self.strategy.parallel_devices = self._parallel_devices
        if hasattr(self.strategy, "num_nodes"):
            self.strategy.num_nodes = self._num_nodes_flag
        if hasattr(self.strategy, "_layer_sync"):
            self.strategy._layer_sync = self._layer_sync
        if hasattr(self.strategy, "set_world_ranks"):
            self.strategy.set_world_ranks()
        self.strategy._configure_launcher()

        if _IS_INTERACTIVE and self.strategy.launcher and not self.strategy.launcher.is_interactive_compatible:
            raise MisconfigurationException(
                f"`Trainer(strategy={self._strategy_flag!r})` is not compatible with an interactive"
                " environment. Run your code as a script, or choose a notebook-compatible strategy:"
                f" `Trainer(strategy='ddp_notebook')`."
                " In case you are spawning processes yourself, make sure to include the Trainer"
                " creation inside the worker function."
            )

        # TODO: should be moved to _check_strategy_and_fallback().
        # Current test check precision first, so keep this check here to meet error order
        if isinstance(self.accelerator, XLAAccelerator) and not isinstance(
            self.strategy, (SingleDeviceXLAStrategy, XLAStrategy)
        ):
            raise ValueError(
                "The `XLAAccelerator` can only be used with a `SingleDeviceXLAStrategy` or `XLAStrategy`,"
                f" found {self.strategy.__class__.__name__}."
            )

        if _habana_available_and_importable():
            from lightning_habana import HPUAccelerator, HPUParallelStrategy, SingleHPUStrategy

            if isinstance(self.accelerator, HPUAccelerator) and not isinstance(
                self.strategy, (SingleHPUStrategy, HPUParallelStrategy)
            ):
                raise ValueError(
                    "The `HPUAccelerator` can only be used with a `SingleHPUStrategy` or `HPUParallelStrategy`,"
                    f" found {self.strategy.__class__.__name__}."
                )

    @property
    def is_distributed(self) -> bool:
        distributed_strategies = [
            DDPStrategy,
            FSDPStrategy,
            DeepSpeedStrategy,
            ModelParallelStrategy,
            XLAStrategy,
        ]
        if _habana_available_and_importable():
            from lightning_habana import HPUParallelStrategy

            distributed_strategies.append(HPUParallelStrategy)
        if isinstance(self.strategy, tuple(distributed_strategies)):
            return True
        if hasattr(self.strategy, "is_distributed"):
            # Used for custom plugins. They should implement this property
            return self.strategy.is_distributed
        return False


def _set_torch_flags(
    *, deterministic: Optional[Union[bool, _LITERAL_WARN]] = None, benchmark: Optional[bool] = None
) -> None:
    if deterministic:
        if benchmark is None:
            # Set benchmark to False to ensure determinism
            benchmark = False
        elif benchmark:
            rank_zero_warn(
                "You passed `deterministic=True` and `benchmark=True`. Note that PyTorch ignores"
                " torch.backends.cudnn.deterministic=True when torch.backends.cudnn.benchmark=True.",
            )
    if benchmark is not None:
        torch.backends.cudnn.benchmark = benchmark

    if deterministic == "warn":
        torch.use_deterministic_algorithms(True, warn_only=True)
    elif isinstance(deterministic, bool):
        # do not call this if deterministic wasn't passed
        torch.use_deterministic_algorithms(deterministic)
    if deterministic:
        # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def _register_external_accelerators_and_strategies() -> None:
    """Registers all known strategies in other packages."""
    if _habana_available_and_importable():
        from lightning_habana import HPUAccelerator, HPUParallelStrategy, SingleHPUStrategy

        # TODO: Prevent registering multiple times
        if "hpu" not in AcceleratorRegistry:
            HPUAccelerator.register_accelerators(AcceleratorRegistry)
        if "hpu_parallel" not in StrategyRegistry:
            HPUParallelStrategy.register_strategies(StrategyRegistry)
        if "hpu_single" not in StrategyRegistry:
            SingleHPUStrategy.register_strategies(StrategyRegistry)
