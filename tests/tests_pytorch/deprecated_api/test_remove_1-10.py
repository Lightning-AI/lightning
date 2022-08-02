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
"""Test deprecated functionality which will be removed in v1.10.0."""
from unittest.mock import Mock

import pytest

from pytorch_lightning.demos.boring_classes import BoringModel
from pytorch_lightning.overrides import LightningDistributedModule, LightningParallelModule
from pytorch_lightning.strategies.bagua import LightningBaguaModule


@pytest.mark.parametrize(
    "wrapper_class",
    [
        LightningParallelModule,
        LightningDistributedModule,
        LightningBaguaModule,
    ],
)
def test_v1_10_deprecated_pl_module_init_parameter(wrapper_class):
    with pytest.deprecated_call(match=rf"The argument `pl_module` in `{wrapper_class.__name__}` is deprecated in v1.8"):
        wrapper_class(BoringModel())

    with pytest.deprecated_call(match=rf"The argument `pl_module` in `{wrapper_class.__name__}` is deprecated in v1.8"):
        wrapper_class(pl_module=BoringModel())
