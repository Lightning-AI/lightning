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
import re
from unittest.mock import Mock

import pytest
import torch

from lightning.fabric.plugins.precision.amp import MixedPrecision
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_4


def test_amp_precision_default_scaler():
    precision = MixedPrecision(precision="16-mixed", device=Mock())
    scaler_cls = torch.amp.GradScaler if _TORCH_GREATER_EQUAL_2_4 else torch.cuda.amp.GradScaler
    assert isinstance(precision.scaler, scaler_cls)


def test_amp_precision_scaler_with_bf16():
    with pytest.raises(ValueError, match="`precision='bf16-mixed'` does not use a scaler"):
        MixedPrecision(precision="bf16-mixed", device=Mock(), scaler=Mock())

    precision = MixedPrecision(precision="bf16-mixed", device=Mock())
    assert precision.scaler is None


def test_amp_precision_forward_context():
    """Test to ensure that the context manager correctly is set to bfloat16 on CPU and CUDA."""
    precision = MixedPrecision(precision="16-mixed", device="cuda")
    assert precision.device == "cuda"
    scaler_cls = torch.amp.GradScaler if _TORCH_GREATER_EQUAL_2_4 else torch.cuda.amp.GradScaler
    assert isinstance(precision.scaler, scaler_cls)
    assert torch.get_default_dtype() == torch.float32
    with precision.forward_context():
        assert torch.get_autocast_gpu_dtype() == torch.float16

    precision = MixedPrecision(precision="bf16-mixed", device="cpu")
    assert precision.device == "cpu"
    assert precision.scaler is None
    with precision.forward_context():
        assert torch.get_autocast_cpu_dtype() == torch.bfloat16

    context_manager = precision.forward_context()
    assert isinstance(context_manager, torch.autocast)
    assert context_manager.fast_dtype == torch.bfloat16


def test_amp_precision_backward():
    precision = MixedPrecision(precision="16-mixed", device="cuda")
    precision.scaler = Mock()
    precision.scaler.scale = Mock(side_effect=(lambda x: x))
    tensor = Mock()
    model = Mock()
    precision.backward(tensor, model, "positional-arg", keyword="arg")
    precision.scaler.scale.assert_called_once_with(tensor)
    tensor.backward.assert_called_once_with("positional-arg", keyword="arg")


def test_amp_precision_optimizer_step_with_scaler():
    precision = MixedPrecision(precision="16-mixed", device="cuda")
    precision.scaler = Mock()
    precision.scaler.get_scale = Mock(return_value=1.0)
    optimizer = Mock()

    precision.optimizer_step(optimizer, keyword="arg")
    precision.scaler.step.assert_called_once_with(optimizer, keyword="arg")
    precision.scaler.update.assert_called_once()


def test_amp_precision_optimizer_step_without_scaler():
    precision = MixedPrecision(precision="bf16-mixed", device="cuda")
    assert precision.scaler is None
    optimizer = Mock()

    precision.optimizer_step(optimizer, keyword="arg")
    optimizer.step.assert_called_once_with(keyword="arg")


def test_amp_precision_parameter_validation():
    MixedPrecision("16-mixed", "cpu")  # should not raise exception
    MixedPrecision("bf16-mixed", "cpu")

    with pytest.raises(
        ValueError,
        match=re.escape("Passed `MixedPrecision(precision='16')`. Precision must be '16-mixed' or 'bf16-mixed'"),
    ):
        MixedPrecision("16", "cpu")

    with pytest.raises(
        ValueError,
        match=re.escape("Passed `MixedPrecision(precision=16)`. Precision must be '16-mixed' or 'bf16-mixed'"),
    ):
        MixedPrecision(16, "cpu")

    with pytest.raises(
        ValueError,
        match=re.escape("Passed `MixedPrecision(precision='bf16')`. Precision must be '16-mixed' or 'bf16-mixed'"),
    ):
        MixedPrecision("bf16", "cpu")
