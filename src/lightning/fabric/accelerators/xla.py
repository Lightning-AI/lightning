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
import functools
from typing import Any, Union

import torch
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override

from lightning.fabric.accelerators.accelerator import Accelerator
from lightning.fabric.accelerators.registry import _AcceleratorRegistry
from lightning.fabric.utilities.device_parser import _check_data_type


class XLAAccelerator(Accelerator):
    """Accelerator for XLA devices, normally TPUs.

    .. warning::  Use of this accelerator beyond import and instantiation is experimental.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not _XLA_AVAILABLE:
            raise ModuleNotFoundError(str(_XLA_AVAILABLE))
        if not _using_pjrt():
            raise RuntimeError("The XLA XRT runtime is not supported anymore.")
        super().__init__(*args, **kwargs)

    @override
    def setup_device(self, device: torch.device) -> None:
        pass

    @override
    def teardown(self) -> None:
        pass

    @staticmethod
    @override
    def parse_devices(devices: Union[int, str, list[int]]) -> Union[int, list[int]]:
        """Accelerator device parsing logic."""
        return _parse_tpu_devices(devices)

    @staticmethod
    @override
    def get_parallel_devices(devices: Union[int, list[int]]) -> list[torch.device]:
        """Gets parallel devices for the Accelerator."""
        devices = _parse_tpu_devices(devices)
        if isinstance(devices, int):
            return [torch.device("xla", i) for i in range(devices)]
        # list of devices is not supported, just a specific index, fine to access [0]
        return [torch.device("xla", devices[0])]
        # we cannot create `xla_device` here because processes have not been spawned yet (this is called in the
        # accelerator connector init). However, there doesn't seem to be a problem with instantiating `torch.device`.
        # it will be replaced with `xla_device` (also a torch.device`, but with extra logic) in the strategy

    @staticmethod
    @override
    def get_device_type() -> str:
        return "xla"

    @staticmethod
    @override
    # XLA's multiprocessing will pop the TPU_NUM_DEVICES key, so we need to cache it
    # https://github.com/pytorch/xla/blob/v2.0.0/torch_xla/distributed/xla_multiprocessing.py#L280
    @functools.lru_cache(maxsize=1)
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
        if not _XLA_AVAILABLE:
            return 0
        if _XLA_GREATER_EQUAL_2_1:
            from torch_xla._internal import tpu

            return tpu.num_available_devices()
        from torch_xla.experimental import tpu

        device_count_on_version = {2: 8, 3: 8, 4: 4}
        return device_count_on_version.get(tpu.version(), 8)

    @staticmethod
    @override
    @functools.lru_cache(maxsize=1)
    def is_available() -> bool:
        try:
            return XLAAccelerator.auto_device_count() > 0
        except (ValueError, AssertionError, OSError):
            # XLA may raise these exceptions if it's not properly configured. This needs to be avoided for the cases
            # when `torch_xla` is imported but not used
            return False

    @classmethod
    @override
    def register_accelerators(cls, accelerator_registry: _AcceleratorRegistry) -> None:
        accelerator_registry.register("tpu", cls, description=cls.__name__)


# PJRT support requires this minimum version
_XLA_AVAILABLE = RequirementCache("torch_xla>=1.13", "torch_xla")
_XLA_GREATER_EQUAL_2_1 = RequirementCache("torch_xla>=2.1")
_XLA_GREATER_EQUAL_2_5 = RequirementCache("torch_xla>=2.5")


def _using_pjrt() -> bool:
    # `using_pjrt` is removed in torch_xla 2.5
    if _XLA_GREATER_EQUAL_2_5:
        from torch_xla import runtime as xr

        return xr.device_type() is not None
    # delete me when torch_xla 2.2 is the min supported version, where XRT support has been dropped.
    if _XLA_GREATER_EQUAL_2_1:
        from torch_xla import runtime as xr

        return xr.using_pjrt()

    from torch_xla.experimental import pjrt

    return pjrt.using_pjrt()


def _parse_tpu_devices(devices: Union[int, str, list[int]]) -> Union[int, list[int]]:
    """Parses the TPU devices given in the format as accepted by the
    :class:`~lightning.pytorch.trainer.trainer.Trainer` and :class:`~lightning.fabric.Fabric`.

    Args:
        devices: An int of 1 or string '1' indicates that 1 core with multi-processing should be used
            An int 8 or string '8' indicates that all 8 cores with multi-processing should be used
            A single element list of int or string can be used to indicate the specific TPU core to use.

    Returns:
        A list of tpu cores to be used.

    """
    _check_data_type(devices)
    if isinstance(devices, str):
        devices = _parse_tpu_devices_str(devices)
    _check_tpu_devices_valid(devices)
    return devices


def _check_tpu_devices_valid(devices: object) -> None:
    device_count = XLAAccelerator.auto_device_count()
    if (
        # support number of devices
        isinstance(devices, int)
        and devices in {1, device_count}
        # support picking a specific device
        or isinstance(devices, (list, tuple))
        and len(devices) == 1
        and 0 <= devices[0] <= device_count - 1
    ):
        return
    raise ValueError(
        f"`devices` can only be 'auto', 1, {device_count} or [<0-{device_count - 1}>] for TPUs. Got {devices!r}"
    )


def _parse_tpu_devices_str(devices: str) -> Union[int, list[int]]:
    devices = devices.strip()
    try:
        return int(devices)
    except ValueError:
        try:
            return [int(x.strip()) for x in devices.split(",") if len(x) > 0]
        except ValueError:
            raise ValueError(f"Could not parse the selected TPU devices: {devices!r}")
