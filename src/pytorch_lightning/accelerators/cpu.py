# Copyright The PyTorch Lightning team.
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
from typing import Any, Dict, List, Union

import torch

from lightning_lite.accelerators.cpu import get_cpu_stats as new_get_cpu_stats
from lightning_lite.utilities import rank_zero_deprecation
from lightning_lite.utilities.device_parser import parse_cpu_cores
from lightning_lite.utilities.types import _DEVICE
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class CPUAccelerator(Accelerator):
    """Accelerator for CPU devices."""

    def init_device(self, device: torch.device) -> None:
        """
        Raises:
            MisconfigurationException:
                If the selected device is not CPU.
        """
        if device.type != "cpu":
            raise MisconfigurationException(f"Device should be CPU, got {device} instead.")

    def get_device_stats(self, device: _DEVICE) -> Dict[str, Any]:
        """Get CPU stats from ``psutil`` package."""
        return get_cpu_stats()

    def teardown(self) -> None:
        pass

    @staticmethod
    def parse_devices(devices: Union[int, str, List[int]]) -> int:
        """Accelerator device parsing logic."""
        devices = parse_cpu_cores(devices)
        return devices

    @staticmethod
    def get_parallel_devices(devices: Union[int, str, List[int]]) -> List[torch.device]:
        """Gets parallel devices for the Accelerator."""
        devices = parse_cpu_cores(devices)
        return [torch.device("cpu")] * devices

    @staticmethod
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
        return 1

    @staticmethod
    def is_available() -> bool:
        """CPU is always available for execution."""
        return True

    @classmethod
    def register_accelerators(cls, accelerator_registry: Dict) -> None:
        accelerator_registry.register(
            "cpu",
            cls,
            description=f"{cls.__class__.__name__}",
        )


def get_cpu_stats() -> Dict[str, float]:
    rank_zero_deprecation(
        "`pytorch_lightning.accelerators.cpu.get_cpu_stats` has been deprecated in v1.8.0 and will be removed in"
        " v1.10.0. Please use `lightning_lite.accelerators.cpu.get_cpu_stats` instead."
    )
    return new_get_cpu_stats()
