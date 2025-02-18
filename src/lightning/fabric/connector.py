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
import os
from collections import Counter
from collections.abc import Iterable
from typing import Any, Optional, Union, cast

import torch
from typing_extensions import get_args

from lightning.fabric.accelerators import ACCELERATOR_REGISTRY
from lightning.fabric.accelerators.accelerator import Accelerator
from lightning.fabric.accelerators.cuda import CUDAAccelerator
from lightning.fabric.accelerators.mps import MPSAccelerator
from lightning.fabric.accelerators.xla import XLAAccelerator
from lightning.fabric.plugins import (
    BitsandbytesPrecision,
    CheckpointIO,
    DeepSpeedPrecision,
    HalfPrecision,
    MixedPrecision,
    Precision,
    TransformerEnginePrecision,
    XLAPrecision,
)
from lightning.fabric.plugins.environments import (
    ClusterEnvironment,
    LightningEnvironment,
    LSFEnvironment,
    MPIEnvironment,
    SLURMEnvironment,
    TorchElasticEnvironment,
)
from lightning.fabric.plugins.precision.double import DoublePrecision
from lightning.fabric.plugins.precision.fsdp import FSDPPrecision
from lightning.fabric.plugins.precision.precision import (
    _PRECISION_INPUT,
    _PRECISION_INPUT_INT,
    _PRECISION_INPUT_STR,
    _PRECISION_INPUT_STR_ALIAS,
    _PRECISION_INPUT_STR_ALIAS_CONVERSION,
)
from lightning.fabric.strategies import (
    STRATEGY_REGISTRY,
    DeepSpeedStrategy,
    ParallelStrategy,
    SingleDeviceStrategy,
    SingleDeviceXLAStrategy,
    Strategy,
    XLAFSDPStrategy,
    XLAStrategy,
)
from lightning.fabric.strategies.ddp import _DDP_FORK_ALIASES
from lightning.fabric.strategies.fsdp import _FSDP_ALIASES, FSDPStrategy
from lightning.fabric.strategies.model_parallel import ModelParallelStrategy
from lightning.fabric.utilities import rank_zero_info, rank_zero_warn
from lightning.fabric.utilities.device_parser import _determine_root_gpu_device
from lightning.fabric.utilities.imports import _IS_INTERACTIVE

_PLUGIN_INPUT = Union[Precision, ClusterEnvironment, CheckpointIO]


class _Connector:
    """The Connector parses several Fabric arguments and instantiates the Strategy including its owned components.

        A. accelerator flag could be:
            1. accelerator class
            2. accelerator str
            3. accelerator auto

        B. strategy flag could be:
            1. strategy class
            2. strategy str registered with STRATEGY_REGISTRY
            3. strategy str in _strategy_type enum which listed in each strategy as
               backend (registed these too, and _strategy_type could be deprecated)

        C. plugins flag could be:
            1. precision class (should be removed, and precision flag should allow user pass classes)
            2. checkpoint_io class
            3. cluster_environment class

    priorities which to take when:
        A. Class > str
        B. Strategy > Accelerator/precision/plugins

    """

    def __init__(
        self,
        accelerator: Union[str, Accelerator] = "auto",
        strategy: Union[str, Strategy] = "auto",
        devices: Union[list[int], str, int] = "auto",
        num_nodes: int = 1,
        precision: Optional[_PRECISION_INPUT] = None,
        plugins: Optional[Union[_PLUGIN_INPUT, Iterable[_PLUGIN_INPUT]]] = None,
    ) -> None:
        # These arguments can be set through environment variables set by the CLI
        accelerator = self._argument_from_env("accelerator", accelerator, default="auto")
        strategy = self._argument_from_env("strategy", strategy, default="auto")
        devices = self._argument_from_env("devices", devices, default="auto")
        num_nodes = int(self._argument_from_env("num_nodes", num_nodes, default=1))
        precision = self._argument_from_env("precision", precision, default=None)

        # 1. Parsing flags
        # Get registered strategies, built-in accelerators and precision plugins
        self._registered_strategies = STRATEGY_REGISTRY.available_strategies()
        self._registered_accelerators = ACCELERATOR_REGISTRY.available_accelerators()

        # Raise an exception if there are conflicts between flags
        # Set each valid flag to `self._x_flag` after validation
        # For devices: Assign gpus, etc. to the accelerator flag and devices flag
        self._strategy_flag: Union[Strategy, str] = "auto"
        self._accelerator_flag: Union[Accelerator, str] = "auto"
        self._precision_input: _PRECISION_INPUT_STR = "32-true"
        self._precision_instance: Optional[Precision] = None
        self._cluster_environment_flag: Optional[Union[ClusterEnvironment, str]] = None
        self._parallel_devices: list[Union[int, torch.device, str]] = []
        self.checkpoint_io: Optional[CheckpointIO] = None

        self._check_config_and_set_final_flags(
            strategy=strategy,
            accelerator=accelerator,
            precision=precision,
            plugins=plugins,
        )
        self._check_device_config_and_set_final_flags(devices=devices, num_nodes=num_nodes)

        # 2. Instantiate Accelerator
        # handle `auto`, `None` and `gpu`
        if self._accelerator_flag == "auto":
            self._accelerator_flag = self._choose_auto_accelerator()
        elif self._accelerator_flag == "gpu":
            self._accelerator_flag = self._choose_gpu_accelerator_backend()
        elif isinstance(self._accelerator_flag, Accelerator):
            pass  # for 3rd party accelerator, just do nothing

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
        self.precision = self._check_and_init_precision()

        # 6. Instantiate Strategy - Part 2
        self._lazy_init_strategy()

    def _check_config_and_set_final_flags(
        self,
        strategy: Union[str, Strategy],
        accelerator: Union[str, Accelerator],
        precision: Optional[_PRECISION_INPUT],
        plugins: Optional[Union[_PLUGIN_INPUT, Iterable[_PLUGIN_INPUT]]],
    ) -> None:
        """This method checks:

        1. strategy: whether the strategy name is valid, and sets the internal flags if it is.
        2. accelerator: if the value of the accelerator argument is a type of accelerator (instance or string),
            set self._accelerator_flag accordingly.
        3. precision: The final value of the precision flag may be determined either by the precision argument or
            by a plugin instance.
        4. plugins: The list of plugins may contain a Precision plugin, CheckpointIO, ClusterEnvironment and others.
            Additionally, other flags such as `precision` can populate the list with the
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
                " It must be either a string or an instance of `lightning.fabric.strategies.Strategy`."
                " Example choices: auto, ddp, ddp_spawn, deepspeed, dp, ..."
                " Find a complete list of options in our documentation at https://lightning.ai"
            )

        if (
            accelerator not in self._registered_accelerators
            and accelerator not in ("auto", "gpu")
            and not isinstance(accelerator, Accelerator)
        ):
            raise ValueError(
                f"You selected an invalid accelerator name: `accelerator={accelerator!r}`."
                f" Available names are: auto, {', '.join(self._registered_accelerators)}."
            )

        # MPS accelerator is incompatible with DDP family of strategies. It supports single-device operation only.
        is_ddp_str = isinstance(strategy, str) and "ddp" in strategy
        is_dp_str = isinstance(strategy, str) and "dp" in strategy
        is_deepspeed_str = isinstance(strategy, str) and "deepspeed" in strategy
        is_parallel_strategy = isinstance(strategy, ParallelStrategy) or is_ddp_str or is_dp_str or is_deepspeed_str
        is_mps_accelerator = MPSAccelerator.is_available() and (
            accelerator in ("mps", "auto", "gpu", None) or isinstance(accelerator, MPSAccelerator)
        )
        if is_mps_accelerator and is_parallel_strategy:
            raise ValueError(
                f"You set `strategy={strategy}` but strategies from the DDP family are not supported on the"
                f" MPS accelerator. Either explicitly set `accelerator='cpu'` or change the strategy."
            )

        self._accelerator_flag = accelerator

        precision_input = _convert_precision_to_unified_args(precision)

        if plugins:
            plugins_flags_types: dict[str, int] = Counter()
            for plugin in plugins:
                if isinstance(plugin, Precision):
                    self._precision_instance = plugin
                    plugins_flags_types[Precision.__name__] += 1
                elif isinstance(plugin, CheckpointIO):
                    self.checkpoint_io = plugin
                    plugins_flags_types[CheckpointIO.__name__] += 1
                elif isinstance(plugin, ClusterEnvironment):
                    self._cluster_environment_flag = plugin
                    plugins_flags_types[ClusterEnvironment.__name__] += 1
                else:
                    raise TypeError(
                        f"Found invalid type for plugin {plugin}. Expected one of: Precision, "
                        "CheckpointIO, ClusterEnviroment."
                    )

            duplicated_plugin_key = [k for k, v in plugins_flags_types.items() if v > 1]
            if duplicated_plugin_key:
                raise ValueError(
                    f"Received multiple values for {', '.join(duplicated_plugin_key)} flags in `plugins`."
                    " Expected one value for each type at most."
                )

            if plugins_flags_types.get(Precision.__name__) and precision_input is not None:
                raise ValueError(
                    f"Received both `precision={precision_input}` and `plugins={self._precision_instance}`. Choose one."
                )

        self._precision_input = "32-true" if precision_input is None else precision_input

        # handle the case when the user passes in a strategy instance which has an accelerator, precision,
        # checkpoint io or cluster env set up
        # TODO: improve the error messages below
        if isinstance(self._strategy_flag, Strategy):
            if self._strategy_flag._accelerator:
                if self._accelerator_flag != "auto":
                    raise ValueError("accelerator set through both strategy class and accelerator flag, choose one")
                self._accelerator_flag = self._strategy_flag._accelerator
            if self._strategy_flag._precision:
                # [RFC] handle precision plugin set up conflict?
                if self._precision_instance:
                    raise ValueError("precision set through both strategy class and plugins, choose one")
                self._precision_instance = self._strategy_flag._precision
            if self._strategy_flag._checkpoint_io:
                if self.checkpoint_io:
                    raise ValueError("checkpoint_io set through both strategy class and plugins, choose one")
                self.checkpoint_io = self._strategy_flag._checkpoint_io
            if getattr(self._strategy_flag, "cluster_environment", None):
                if self._cluster_environment_flag:
                    raise ValueError("cluster_environment set through both strategy class and plugins, choose one")
                self._cluster_environment_flag = getattr(self._strategy_flag, "cluster_environment")

            if hasattr(self._strategy_flag, "parallel_devices") and self._strategy_flag.parallel_devices:
                if self._strategy_flag.parallel_devices[0].type == "cpu":
                    if self._accelerator_flag and self._accelerator_flag not in ("auto", "cpu"):
                        raise ValueError(
                            f"CPU parallel_devices set through {self._strategy_flag.__class__.__name__} class,"
                            f" but accelerator set to {self._accelerator_flag}, please choose one device type"
                        )
                    self._accelerator_flag = "cpu"
                if self._strategy_flag.parallel_devices[0].type == "cuda":
                    if self._accelerator_flag and self._accelerator_flag not in ("auto", "cuda", "gpu"):
                        raise ValueError(
                            f"GPU parallel_devices set through {self._strategy_flag.__class__.__name__} class,"
                            f" but accelerator set to {self._accelerator_flag}, please choose one device type"
                        )
                    self._accelerator_flag = "cuda"
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
            raise ValueError(
                f"`Fabric(devices={self._devices_flag!r})` value is not a valid input"
                f" using {accelerator_name} accelerator."
            )

    @staticmethod
    def _choose_auto_accelerator() -> str:
        """Choose the accelerator type (str) based on availability when ``accelerator='auto'``."""
        if XLAAccelerator.is_available():
            return "tpu"
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
        raise RuntimeError("No supported gpu backend found!")

    def _set_parallel_devices_and_init_accelerator(self) -> None:
        if isinstance(self._accelerator_flag, Accelerator):
            self.accelerator: Accelerator = self._accelerator_flag
        else:
            assert self._accelerator_flag is not None
            self.accelerator = ACCELERATOR_REGISTRY.get(self._accelerator_flag)
        accelerator_cls = self.accelerator.__class__

        if not accelerator_cls.is_available():
            available_accelerator = [
                acc_str
                for acc_str in self._registered_accelerators
                if ACCELERATOR_REGISTRY[acc_str]["accelerator"].is_available()
            ]
            raise RuntimeError(
                f"`{accelerator_cls.__qualname__}` can not run on your system"
                " since the accelerator is not available. The following accelerator(s)"
                " is available and can be passed into `accelerator` argument of"
                f" `Fabric`: {available_accelerator}."
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
                f"Fabric will use only 1 of {self.accelerator.auto_device_count()} GPUs because it is running inside"
                " an interactive / notebook environment. You may try to set `Fabric(devices="
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
        if self._accelerator_flag == "tpu" or isinstance(self._accelerator_flag, XLAAccelerator):
            if self._parallel_devices and len(self._parallel_devices) > 1:
                return "xla"
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

        # Change fsdp to xla_fsdp if using TPU
        if strategy_flag == "fsdp" and self._accelerator_flag == "tpu":
            strategy_flag = "xla_fsdp"
        if strategy_flag == "dp" and self._accelerator_flag == "cpu":
            rank_zero_warn(f"{strategy_flag!r} is not supported on CPUs, hence setting `strategy='ddp'`.")
            strategy_flag = "ddp"
        if strategy_flag in _DDP_FORK_ALIASES and "fork" not in torch.multiprocessing.get_all_start_methods():
            raise ValueError(
                f"You selected `Fabric(strategy='{strategy_flag}')` but process forking is not supported on this"
                f" platform. We recommed `Fabric(strategy='ddp_spawn')` instead."
            )
        if (
            strategy_flag in _FSDP_ALIASES or type(self._strategy_flag) is FSDPStrategy
        ) and self._accelerator_flag not in ("cuda", "gpu"):
            raise ValueError(
                "You selected the FSDP strategy but FSDP is only available on GPU. Set `Fabric(accelerator='gpu', ...)`"
                " to continue or select a different strategy."
            )
        if strategy_flag:
            self._strategy_flag = strategy_flag

    def _init_strategy(self) -> None:
        """Instantiate the Strategy given depending on the setting of ``_strategy_flag``."""
        # The validation of `_strategy_flag` already happened earlier on in the connector
        assert isinstance(self._strategy_flag, (str, Strategy))
        if isinstance(self._strategy_flag, str):
            self.strategy = STRATEGY_REGISTRY.get(self._strategy_flag)
        else:
            self.strategy = self._strategy_flag

    def _check_and_init_precision(self) -> Precision:
        if isinstance(self._precision_instance, Precision):
            if isinstance(self._precision_instance, BitsandbytesPrecision) and not isinstance(
                self.accelerator, CUDAAccelerator
            ):
                raise RuntimeError("Bitsandbytes is only supported on CUDA GPUs.")
            return self._precision_instance
        if isinstance(self.strategy, (SingleDeviceXLAStrategy, XLAStrategy, XLAFSDPStrategy)):
            return XLAPrecision(self._precision_input)  # type: ignore
        if isinstance(self.strategy, DeepSpeedStrategy):
            return DeepSpeedPrecision(self._precision_input)  # type: ignore
        if isinstance(self.strategy, FSDPStrategy):
            return FSDPPrecision(
                precision=self._precision_input,  # type: ignore[arg-type]
                device_type=self._accelerator_flag.get_device_type()
                if isinstance(self._accelerator_flag, Accelerator)
                else None,
            )
        mp_precision_supported = ("32-true", "bf16-mixed", "bf16-true", "16-true")
        if isinstance(self.strategy, ModelParallelStrategy) and self._precision_input not in mp_precision_supported:
            raise ValueError(
                f"The `ModelParallelStrategy` does not support `Fabric(..., precision={self._precision_input!r})`."
                f" Choose a different precision among: {', '.join(mp_precision_supported)}."
            )
        if self._precision_input in ("16-true", "bf16-true"):
            return HalfPrecision(self._precision_input)  # type: ignore
        if self._precision_input == "32-true":
            return Precision()
        if self._precision_input == "64-true":
            return DoublePrecision()
        if self._precision_input == "transformer-engine":
            return TransformerEnginePrecision(weights_dtype=torch.bfloat16)
        if self._precision_input == "transformer-engine-float16":
            return TransformerEnginePrecision(weights_dtype=torch.float16)

        if self._precision_input == "16-mixed" and self._accelerator_flag == "cpu":
            rank_zero_warn(
                "You passed `Fabric(accelerator='cpu', precision='16-mixed')` but AMP with fp16 is not supported on "
                "CPU. Using `precision='bf16-mixed'` instead."
            )
            self._precision_input = "bf16-mixed"

        if self._precision_input in ("16-mixed", "bf16-mixed"):
            rank_zero_info(
                "Using 16-bit Automatic Mixed Precision (AMP)"
                if self._precision_input == "16-mixed"
                else "Using bfloat16 Automatic Mixed Precision (AMP)"
            )
            device = "cpu" if self._accelerator_flag == "cpu" else "cuda"
            if isinstance(self._accelerator_flag, Accelerator):
                device = self._accelerator_flag.get_device_type()
            return MixedPrecision(precision=self._precision_input, device=device)  # type: ignore[arg-type]

        raise RuntimeError("No precision set")

    def _lazy_init_strategy(self) -> None:
        """Lazily set missing attributes on the previously instantiated strategy."""
        self.strategy.accelerator = self.accelerator
        if self.precision:
            self.strategy.precision = self.precision
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
            self.strategy._num_nodes = self._num_nodes_flag
        if hasattr(self.strategy, "_set_world_ranks"):
            self.strategy._set_world_ranks()
        self.strategy._configure_launcher()

        if _IS_INTERACTIVE and self.strategy.launcher and not self.strategy.launcher.is_interactive_compatible:
            raise RuntimeError(
                f"`Fabric(strategy={self._strategy_flag!r})` is not compatible with an interactive"
                " environment. Run your code as a script, or choose one of the compatible strategies:"
                f" `Fabric(strategy='dp'|'ddp_notebook')`."
                " In case you are spawning processes yourself, make sure to include the Fabric"
                " creation inside the worker function."
            )

        # TODO: should be moved to _check_strategy_and_fallback().
        # Current test check precision first, so keep this check here to meet error order
        if isinstance(self.accelerator, XLAAccelerator) and not isinstance(
            self.strategy, (SingleDeviceXLAStrategy, XLAStrategy, XLAFSDPStrategy)
        ):
            raise ValueError(
                "The `XLAAccelerator` can only be used with a `SingleDeviceXLAStrategy`, `XLAStrategy`, or"
                f" `XLAFSDPStrategy`. Found {self.strategy.__class__.__name__}."
            )

    @staticmethod
    def _argument_from_env(name: str, current: Any, default: Any) -> Any:
        env_value: Optional[str] = os.environ.get("LT_" + name.upper())

        if env_value is None:
            return current

        if env_value is not None and env_value != str(current) and str(current) != str(default) and _is_using_cli():
            raise ValueError(
                f"Your code has `Fabric({name}={current!r}, ...)` but it conflicts with the value "
                f"`--{name}={env_value}` set through the CLI. "
                " Remove it either from the CLI or from the Lightning Fabric object."
            )
        return env_value


def _convert_precision_to_unified_args(precision: Optional[_PRECISION_INPUT]) -> Optional[_PRECISION_INPUT_STR]:
    if precision is None:
        return None

    supported_precision = (
        get_args(_PRECISION_INPUT_STR) + get_args(_PRECISION_INPUT_INT) + get_args(_PRECISION_INPUT_STR_ALIAS)
    )
    if precision not in supported_precision:
        raise ValueError(f"Precision {repr(precision)} is invalid. Allowed precision values: {supported_precision}")

    precision = str(precision)  # convert int flags to str here to enable the legacy-conversion below

    if precision in get_args(_PRECISION_INPUT_STR_ALIAS):
        if str(precision)[:2] not in ("32", "64"):
            rank_zero_warn(
                f"`precision={precision}` is supported for historical reasons but its usage is discouraged. "
                f"Please set your precision to {_PRECISION_INPUT_STR_ALIAS_CONVERSION[precision]} instead!"
            )
        precision = _PRECISION_INPUT_STR_ALIAS_CONVERSION[precision]
    return cast(_PRECISION_INPUT_STR, precision)


def _is_using_cli() -> bool:
    return bool(int(os.environ.get("LT_CLI_USED", "0")))
