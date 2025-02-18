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
from typing import Any, Optional, Union

import torch
from typing_extensions import override

from lightning.fabric.accelerators import _AcceleratorRegistry
from lightning.fabric.accelerators.mps import MPSAccelerator as _MPSAccelerator
from lightning.fabric.utilities.device_parser import _parse_gpu_ids
from lightning.fabric.utilities.types import _DEVICE
from lightning.pytorch.accelerators.accelerator import Accelerator
from lightning.pytorch.accelerators.cpu import _PSUTIL_AVAILABLE
from lightning.pytorch.utilities.exceptions import MisconfigurationException


class MPSAccelerator(Accelerator):
    """Accelerator for Metal Apple Silicon GPU devices.

    .. warning::  Use of this accelerator beyond import and instantiation is experimental.

    """

    @override
    def setup_device(self, device: torch.device) -> None:
        """
        Raises:
            MisconfigurationException:
                If the selected device is not MPS.
        """
        if device.type != "mps":
            raise MisconfigurationException(f"Device should be MPS, got {device} instead.")

    @override
    def get_device_stats(self, device: _DEVICE) -> dict[str, Any]:
        """Get M1 (cpu + gpu) stats from ``psutil`` package."""
        return get_device_stats()

    @override
    def teardown(self) -> None:
        pass

    @staticmethod
    @override
    def parse_devices(devices: Union[int, str, list[int]]) -> Optional[list[int]]:
        """Accelerator device parsing logic."""
        return _parse_gpu_ids(devices, include_mps=True)

    @staticmethod
    @override
    def get_parallel_devices(devices: Union[int, str, list[int]]) -> list[torch.device]:
        """Gets parallel devices for the Accelerator."""
        parsed_devices = MPSAccelerator.parse_devices(devices)
        assert parsed_devices is not None

        return [torch.device("mps", i) for i in range(len(parsed_devices))]

    @staticmethod
    @override
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
        return 1

    @staticmethod
    @override
    def is_available() -> bool:
        """MPS is only available on a machine with the ARM-based Apple Silicon processors."""
        return _MPSAccelerator.is_available()

    @classmethod
    @override
    def register_accelerators(cls, accelerator_registry: _AcceleratorRegistry) -> None:
        accelerator_registry.register(
            "mps",
            cls,
            description=cls.__name__,
        )

    @staticmethod
    @override
    def get_device_type() -> str:
        return "mps"


# device metrics
_VM_PERCENT = "M1_vm_percent"
_PERCENT = "M1_percent"
_SWAP_PERCENT = "M1_swap_percent"


def get_device_stats() -> dict[str, float]:
    if not _PSUTIL_AVAILABLE:
        raise ModuleNotFoundError(
            f"Fetching MPS device stats requires `psutil` to be installed. {str(_PSUTIL_AVAILABLE)}"
        )
    import psutil

    return {
        _VM_PERCENT: psutil.virtual_memory().percent,
        _PERCENT: psutil.cpu_percent(),
        _SWAP_PERCENT: psutil.swap_memory().percent,
    }
