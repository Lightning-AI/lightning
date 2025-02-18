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
import platform
from functools import lru_cache
from typing import Optional, Union

import torch
from typing_extensions import override

from lightning.fabric.accelerators.accelerator import Accelerator
from lightning.fabric.accelerators.registry import _AcceleratorRegistry


class MPSAccelerator(Accelerator):
    """Accelerator for Metal Apple Silicon GPU devices.

    .. warning::  Use of this accelerator beyond import and instantiation is experimental.

    """

    @override
    def setup_device(self, device: torch.device) -> None:
        """
        Raises:
            ValueError:
                If the selected device is not MPS.
        """
        if device.type != "mps":
            raise ValueError(f"Device should be MPS, got {device} instead.")

    @override
    def teardown(self) -> None:
        pass

    @staticmethod
    @override
    def parse_devices(devices: Union[int, str, list[int]]) -> Optional[list[int]]:
        """Accelerator device parsing logic."""
        from lightning.fabric.utilities.device_parser import _parse_gpu_ids

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
    def get_device_type() -> str:
        return "mps"

    @staticmethod
    @override
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
        return 1

    @staticmethod
    @override
    @lru_cache(1)
    def is_available() -> bool:
        """MPS is only available on a machine with the ARM-based Apple Silicon processors."""
        mps_disabled = os.getenv("DISABLE_MPS", "0") == "1"
        return not mps_disabled and torch.backends.mps.is_available() and platform.processor() in ("arm", "arm64")

    @classmethod
    @override
    def register_accelerators(cls, accelerator_registry: _AcceleratorRegistry) -> None:
        accelerator_registry.register(
            "mps",
            cls,
            description=cls.__name__,
        )


def _get_all_available_mps_gpus() -> list[int]:
    """
    Returns:
        A list of all available MPS GPUs
    """
    return [0] if MPSAccelerator.is_available() else []
