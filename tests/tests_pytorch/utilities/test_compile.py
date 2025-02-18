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
import sys
from contextlib import nullcontext
from unittest import mock

import pytest
import torch
from lightning_utilities.core.imports import RequirementCache

from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_2, _TORCH_GREATER_EQUAL_2_4
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.utilities.compile import from_compiled, to_uncompiled
from tests_pytorch.conftest import mock_cuda_count
from tests_pytorch.helpers.runif import RunIf

_PYTHON_GREATER_EQUAL_3_9_0 = (sys.version_info.major, sys.version_info.minor) >= (3, 9)


# https://github.com/pytorch/pytorch/issues/95708
@pytest.mark.skipif(sys.platform == "darwin", reason="fatal error: 'omp.h' file not found")
@RunIf(dynamo=True)
@mock.patch("lightning.pytorch.trainer.call._call_and_handle_interrupt")
def test_trainer_compiled_model(_, tmp_path, monkeypatch, mps_count_0):
    trainer_kwargs = {
        "default_root_dir": tmp_path,
        "fast_dev_run": True,
        "logger": False,
        "enable_checkpointing": False,
        "enable_model_summary": False,
        "enable_progress_bar": False,
    }

    model = BoringModel()
    compiled_model = torch.compile(model)

    # can train with compiled model
    trainer = Trainer(**trainer_kwargs)
    trainer.fit(compiled_model)
    assert isinstance(trainer.strategy.model, torch._dynamo.OptimizedModule)

    # the compiled model can be uncompiled
    to_uncompiled_model = to_uncompiled(compiled_model)

    # the compiled model needs to be passed
    with pytest.raises(ValueError, match="required to be a compiled LightningModule"):
        to_uncompiled(to_uncompiled_model)

    # the uncompiled model can be fitted
    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model)
    assert not isinstance(trainer.strategy.model, torch._dynamo.OptimizedModule)

    # some strategies do not support it
    if RequirementCache("deepspeed"):
        compiled_model = torch.compile(model)
        mock_cuda_count(monkeypatch, 2)

        # TODO: Update deepspeed to avoid deprecation warning for `torch.cuda.amp.custom_fwd` on import
        warn_context = (
            pytest.warns(FutureWarning, match="torch.cuda.amp.*is deprecated")
            if _TORCH_GREATER_EQUAL_2_4
            else nullcontext()
        )

        with warn_context:
            trainer = Trainer(strategy="deepspeed", accelerator="cuda", **trainer_kwargs)

        with pytest.raises(RuntimeError, match="Using a compiled model is incompatible with the current strategy.*"):
            trainer.fit(compiled_model)

    # ddp does
    trainer = Trainer(strategy="ddp", **trainer_kwargs)
    trainer.fit(compiled_model)

    # an exception is raised
    trainer = Trainer(**trainer_kwargs)
    with pytest.raises(TypeError, match="must be a `Light"):
        trainer.fit(object())


@RunIf(dynamo=True)
def test_compile_uncompile():
    model = BoringModel()
    compiled_model = torch.compile(model)

    def has_dynamo(fn):
        return any(el for el in dir(fn) if el.startswith("_torchdynamo"))

    from_compiled_model = from_compiled(compiled_model)
    assert isinstance(from_compiled_model, LightningModule)
    assert from_compiled_model._compiler_ctx is not None
    assert has_dynamo(from_compiled_model.forward)
    assert has_dynamo(from_compiled_model.training_step)
    assert has_dynamo(from_compiled_model.validation_step)
    assert has_dynamo(from_compiled_model.test_step)
    assert has_dynamo(from_compiled_model.predict_step)

    to_uncompiled_model = to_uncompiled(model)
    assert to_uncompiled_model._compiler_ctx is None
    assert to_uncompiled_model.forward == model.forward
    assert to_uncompiled_model.training_step == model.training_step
    assert to_uncompiled_model.validation_step == model.validation_step
    assert to_uncompiled_model.test_step == model.test_step
    assert to_uncompiled_model.predict_step == model.predict_step
    assert not has_dynamo(to_uncompiled_model.forward)
    assert not has_dynamo(to_uncompiled_model.training_step)
    assert not has_dynamo(to_uncompiled_model.validation_step)
    assert not has_dynamo(to_uncompiled_model.test_step)
    assert not has_dynamo(to_uncompiled_model.predict_step)


# https://github.com/pytorch/pytorch/issues/95708
@pytest.mark.skipif(sys.platform == "darwin", reason="fatal error: 'omp.h' file not found")
@pytest.mark.skipif(not _PYTHON_GREATER_EQUAL_3_9_0, reason="AssertionError: failed to reach fixed point")
@pytest.mark.xfail(
    sys.platform == "win32" and _TORCH_GREATER_EQUAL_2_2, strict=False, reason="RuntimeError: Failed to import"
)
@RunIf(dynamo=True)
@mock.patch.dict(os.environ, {})
def test_trainer_compiled_model_that_logs(tmp_path):
    class MyModel(BoringModel):
        def training_step(self, batch, batch_idx):
            loss = self.step(batch)
            self.log("loss", loss)
            return loss

    model = MyModel()
    compiled_model = torch.compile(model)

    trainer = Trainer(
        default_root_dir=tmp_path,
        fast_dev_run=True,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        accelerator="cpu",
    )
    trainer.fit(compiled_model)

    assert set(trainer.callback_metrics) == {"loss"}


# https://github.com/pytorch/pytorch/issues/95708
@pytest.mark.skipif(sys.platform == "darwin", reason="fatal error: 'omp.h' file not found")
@pytest.mark.skipif(not _PYTHON_GREATER_EQUAL_3_9_0, reason="AssertionError: failed to reach fixed point")
@pytest.mark.xfail(
    sys.platform == "win32" and _TORCH_GREATER_EQUAL_2_2, strict=False, reason="RuntimeError: Failed to import"
)
@RunIf(dynamo=True)
@mock.patch.dict(os.environ, {})
def test_trainer_compiled_model_test(tmp_path):
    model = BoringModel()
    compiled_model = torch.compile(model)

    trainer = Trainer(
        default_root_dir=tmp_path,
        fast_dev_run=True,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        accelerator="cpu",
    )
    trainer.test(compiled_model)
