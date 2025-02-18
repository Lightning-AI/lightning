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
from typing import Any

import torch

from lightning.fabric.accelerators import ACCELERATOR_REGISTRY, Accelerator


def test_accelerator_registry_with_new_accelerator():
    accelerator_name = "custom_accelerator"
    accelerator_description = "Custom Accelerator"

    class CustomAccelerator(Accelerator):
        def __init__(self, param1, param2):
            self.param1 = param1
            self.param2 = param2
            super().__init__()

        def setup_device(self, device: torch.device) -> None:
            pass

        def get_device_stats(self, device: torch.device) -> dict[str, Any]:
            pass

        def teardown(self) -> None:
            pass

        @staticmethod
        def parse_devices(devices):
            return devices

        @staticmethod
        def get_parallel_devices(devices):
            return ["foo"] * devices

        @staticmethod
        def get_device_type():
            return "foo"

        @staticmethod
        def auto_device_count():
            return 3

        @staticmethod
        def is_available():
            return True

    ACCELERATOR_REGISTRY.register(
        accelerator_name, CustomAccelerator, description=accelerator_description, param1="abc", param2=123
    )

    assert accelerator_name in ACCELERATOR_REGISTRY

    assert ACCELERATOR_REGISTRY[accelerator_name]["description"] == accelerator_description
    assert ACCELERATOR_REGISTRY[accelerator_name]["init_params"] == {"param1": "abc", "param2": 123}
    assert ACCELERATOR_REGISTRY[accelerator_name]["accelerator_name"] == accelerator_name

    assert isinstance(ACCELERATOR_REGISTRY.get(accelerator_name), CustomAccelerator)

    ACCELERATOR_REGISTRY.remove(accelerator_name)
    assert accelerator_name not in ACCELERATOR_REGISTRY


def test_available_accelerators_in_registry():
    assert ACCELERATOR_REGISTRY.available_accelerators() == ["cpu", "cuda", "mps", "tpu"]
