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
from typing import Any, Union

import torch
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override

from lightning.fabric.accelerators import _AcceleratorRegistry
from lightning.fabric.accelerators.cpu import _parse_cpu_cores
from lightning.fabric.utilities.types import _DEVICE
from lightning.pytorch.accelerators.accelerator import Accelerator
from lightning.pytorch.utilities.exceptions import MisconfigurationException


class CPUAccelerator(Accelerator):
    """Accelerator for CPU devices."""

    @override
    def setup_device(self, device: torch.device) -> None:
        """
        Raises:
            MisconfigurationException:
                If the selected device is not CPU.
        """
        if device.type != "cpu":
            raise MisconfigurationException(f"Device should be CPU, got {device} instead.")

    @override
    def get_device_stats(self, device: _DEVICE) -> dict[str, Any]:
        """Get CPU stats from ``psutil`` package."""
        return get_cpu_stats()

    @override
    def teardown(self) -> None:
        pass

    @staticmethod
    @override
    def parse_devices(devices: Union[int, str]) -> int:
        """Accelerator device parsing logic."""
        return _parse_cpu_cores(devices)

    @staticmethod
    @override
    def get_parallel_devices(devices: Union[int, str]) -> list[torch.device]:
        """Gets parallel devices for the Accelerator."""
        devices = _parse_cpu_cores(devices)
        return [torch.device("cpu")] * devices

    @staticmethod
    @override
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
        return 1

    @staticmethod
    @override
    def is_available() -> bool:
        """CPU is always available for execution."""
        return True

    @classmethod
    @override
    def register_accelerators(cls, accelerator_registry: _AcceleratorRegistry) -> None:
        accelerator_registry.register(
            "cpu",
            cls,
            description=cls.__name__,
        )

    @staticmethod
    @override
    def get_device_type() -> str:
        return "cpu"


# CPU device metrics
_CPU_VM_PERCENT = "cpu_vm_percent"
_CPU_PERCENT = "cpu_percent"
_CPU_SWAP_PERCENT = "cpu_swap_percent"

_PSUTIL_AVAILABLE = RequirementCache("psutil")


def get_cpu_stats() -> dict[str, float]:
    if not _PSUTIL_AVAILABLE:
        raise ModuleNotFoundError(
            f"Fetching CPU device stats requires `psutil` to be installed. {str(_PSUTIL_AVAILABLE)}"
        )
    import psutil

    return {
        _CPU_VM_PERCENT: psutil.virtual_memory().percent,
        _CPU_PERCENT: psutil.cpu_percent(),
        _CPU_SWAP_PERCENT: psutil.swap_memory().percent,
    }
