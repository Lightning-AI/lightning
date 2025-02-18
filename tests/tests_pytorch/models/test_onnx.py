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
import operator
import os
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnxruntime
import pytest
import torch
from lightning_utilities import compare_version

import tests_pytorch.helpers.pipelines as tpipes
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from tests_pytorch.helpers.runif import RunIf
from tests_pytorch.utilities.test_model_summary import UnorderedModel


@RunIf(onnx=True)
def test_model_saves_with_input_sample(tmp_path):
    """Test that ONNX model saves with input sample and size is greater than 3 MB."""
    model = BoringModel()
    input_sample = torch.randn((1, 32))

    file_path = os.path.join(tmp_path, "os.path.onnx")
    model.to_onnx(file_path, input_sample)
    assert os.path.isfile(file_path)
    assert os.path.getsize(file_path) > 4e2

    file_path = Path(tmp_path) / "pathlib.onnx"
    model.to_onnx(file_path, input_sample)
    assert os.path.isfile(file_path)
    assert os.path.getsize(file_path) > 4e2

    file_path = BytesIO()
    model.to_onnx(file_path=file_path, input_sample=input_sample)
    assert len(file_path.getvalue()) > 4e2


@pytest.mark.parametrize(
    "accelerator", [pytest.param("mps", marks=RunIf(mps=True)), pytest.param("gpu", marks=RunIf(min_cuda_gpus=True))]
)
@RunIf(onnx=True)
def test_model_saves_on_gpu(tmp_path, accelerator):
    """Test that model saves on gpu."""
    model = BoringModel()
    trainer = Trainer(accelerator=accelerator, devices=1, fast_dev_run=True)
    trainer.fit(model)

    file_path = os.path.join(tmp_path, "model.onnx")
    input_sample = torch.randn((1, 32))
    model.to_onnx(file_path, input_sample)
    assert os.path.isfile(file_path)
    assert os.path.getsize(file_path) > 4e2


@pytest.mark.parametrize(
    ("modelclass", "input_sample"),
    [
        (BoringModel, torch.randn(1, 32)),
        (UnorderedModel, (torch.rand(2, 3), torch.rand(2, 10))),
    ],
)
@RunIf(onnx=True)
def test_model_saves_with_example_input_array(tmp_path, modelclass, input_sample):
    """Test that ONNX model saves with example_input_array and size is greater than 3 MB."""
    model = modelclass()
    model.example_input_array = input_sample

    file_path = os.path.join(tmp_path, "model.onnx")
    model.to_onnx(file_path)
    assert os.path.exists(file_path) is True
    assert os.path.getsize(file_path) > 4e2


@RunIf(min_cuda_gpus=2, onnx=True)
def test_model_saves_on_multi_gpu(tmp_path):
    """Test that ONNX model saves on a distributed backend."""
    trainer_options = {
        "default_root_dir": tmp_path,
        "max_epochs": 1,
        "limit_train_batches": 10,
        "limit_val_batches": 10,
        "accelerator": "gpu",
        "devices": [0, 1],
        "strategy": "ddp_spawn",
        "enable_progress_bar": False,
    }

    model = BoringModel()
    model.example_input_array = torch.randn(5, 32)

    tpipes.run_model_test(trainer_options, model, min_acc=0.08)

    file_path = os.path.join(tmp_path, "model.onnx")
    model.to_onnx(file_path)
    assert os.path.exists(file_path) is True


# todo: investigate where the logging happening in torch.onnx for PT 2.6+
@RunIf(onnx=True, max_torch="2.6.0")
def test_verbose_param(tmp_path, capsys):
    """Test that output is present when verbose parameter is set."""
    model = BoringModel()
    model.example_input_array = torch.randn(5, 32)
    file_path = os.path.join(tmp_path, "model.onnx")

    with patch("torch.onnx.log", autospec=True) as mocked:
        model.to_onnx(file_path, verbose=True)
    (prefix, _), _ = mocked.call_args
    assert prefix == "Exported graph: "


@RunIf(onnx=True)
def test_error_if_no_input(tmp_path):
    """Test that an error is thrown when there is no input tensor."""
    model = BoringModel()
    model.example_input_array = None
    file_path = os.path.join(tmp_path, "model.onnx")
    with pytest.raises(
        ValueError,
        match=r"Could not export to ONNX since neither `input_sample` nor"
        r" `model.example_input_array` attribute is set.",
    ):
        model.to_onnx(file_path)


@RunIf(onnx=True)
def test_if_inference_output_is_valid(tmp_path):
    """Test that the output inferred from ONNX model is same as from PyTorch."""
    model = BoringModel()
    model.example_input_array = torch.randn(5, 32)

    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model)

    model.eval()
    with torch.no_grad():
        torch_out = model(model.example_input_array)

    file_path = os.path.join(tmp_path, "model.onnx")
    model.to_onnx(file_path, model.example_input_array, export_params=True)

    ort_kwargs = {"providers": "CPUExecutionProvider"} if compare_version("onnxruntime", operator.ge, "1.16.0") else {}
    ort_session = onnxruntime.InferenceSession(file_path, **ort_kwargs)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(model.example_input_array)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    assert np.allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
