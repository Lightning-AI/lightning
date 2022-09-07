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
import platform
from typing import Any, Dict, List, Optional, Union

import torch

from lightning_lite.accelerators.mps import get_device_stats as new_get_device_stats
from lightning_lite.utilities import device_parser, rank_zero_deprecation
from lightning_lite.utilities.types import _DEVICE
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _TORCH_GREATER_EQUAL_1_12

# For using the `MPSAccelerator`, user's machine should have `torch>=1.12`, Metal programming framework and
# the ARM-based Apple Silicon processors.
_MPS_AVAILABLE = (
    _TORCH_GREATER_EQUAL_1_12 and torch.backends.mps.is_available() and platform.processor() in ("arm", "arm64")
)


class MPSAccelerator(Accelerator):
    """Accelerator for Metal Apple Silicon GPU devices."""

    def init_device(self, device: torch.device) -> None:
        """
        Raises:
            MisconfigurationException:
                If the selected device is not MPS.
        """
        if device.type != "mps":
            raise MisconfigurationException(f"Device should be MPS, got {device} instead.")

    def get_device_stats(self, device: _DEVICE) -> Dict[str, Any]:
        """Get M1 (cpu + gpu) stats from ``psutil`` package."""
        return get_device_stats()

    def teardown(self) -> None:
        pass

    @staticmethod
    def parse_devices(devices: Union[int, str, List[int]]) -> Optional[List[int]]:
        """Accelerator device parsing logic."""
        parsed_devices = device_parser.parse_gpu_ids(devices, include_mps=True)
        return parsed_devices

    @staticmethod
    def get_parallel_devices(devices: Union[int, str, List[int]]) -> List[torch.device]:
        """Gets parallel devices for the Accelerator."""
        parsed_devices = MPSAccelerator.parse_devices(devices)
        assert parsed_devices is not None

        return [torch.device("mps", i) for i in range(len(parsed_devices))]

    @staticmethod
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
        return 1

    @staticmethod
    def is_available() -> bool:
        """MPS is only available for certain torch builds starting at torch>=1.12."""
        return _MPS_AVAILABLE

    @classmethod
    def register_accelerators(cls, accelerator_registry: Dict) -> None:
        accelerator_registry.register(
            "mps",
            cls,
            description=cls.__class__.__name__,
        )


def get_device_stats() -> Dict[str, float]:
    rank_zero_deprecation(
        "`pytorch_lightning.accelerators.mps.get_device_stats` has been deprecated in v1.8.0 and will be removed in"
        " v1.10.0. Please use `lightning_lite.accelerators.mps.get_device_stats` instead."
    )
    return new_get_device_stats()
