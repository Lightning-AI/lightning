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
from abc import ABC
from typing import Any

import lightning.pytorch as pl
from lightning.fabric.accelerators.accelerator import Accelerator as _Accelerator
from lightning.fabric.utilities.types import _DEVICE


class Accelerator(_Accelerator, ABC):
    """The Accelerator base class for Lightning PyTorch.

    .. warning::  Writing your own accelerator is an :ref:`experimental <versioning:Experimental API>` feature.

    """

    def setup(self, trainer: "pl.Trainer") -> None:
        """Called by the Trainer to set up the accelerator before the model starts running on the device.

        Args:
            trainer: the trainer instance

        """

    def get_device_stats(self, device: _DEVICE) -> dict[str, Any]:
        """Get stats for a given device.

        Args:
            device: device for which to get stats

        Returns:
            Dictionary of device stats

        """
        raise NotImplementedError

    @staticmethod
    def get_device_type() -> str:
        """Get the device for the current process."""
        raise NotImplementedError
