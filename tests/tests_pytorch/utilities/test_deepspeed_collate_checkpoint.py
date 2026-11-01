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
import types
from unittest import mock

import pytest
import torch

from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
    ds_checkpoint_dir,
)
from tests_pytorch.helpers.runif import RunIf


@RunIf(min_cuda_gpus=2, standalone=True, deepspeed=True)
def test_deepspeed_collate_checkpoint(tmp_path):
    """Test to ensure that with DeepSpeed Stage 3 we can collate the sharded checkpoints into a single file."""
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        strategy=DeepSpeedStrategy(stage=3),
        accelerator="gpu",
        devices=2,
        fast_dev_run=True,
        precision="16-mixed",
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model)
    checkpoint_path = os.path.join(tmp_path, "model.pt")
    checkpoint_path = trainer.strategy.broadcast(checkpoint_path)
    trainer.save_checkpoint(checkpoint_path)
    if trainer.is_global_zero:
        # ensure function call works
        output_path = os.path.join(tmp_path, "single_model.pt")
        convert_zero_checkpoint_to_fp32_state_dict(checkpoint_path, output_path)
        _assert_checkpoint_equal(model, output_path)


def _assert_checkpoint_equal(model, output_path):
    assert os.path.exists(output_path)
    single_output = torch.load(output_path, weights_only=False)
    state_dict = model.state_dict()
    for orig_param, saved_model_param in zip(state_dict.values(), single_output["state_dict"].values()):
        if model.dtype == torch.half:
            # moved model to float32 for comparison with single fp32 saved weights
            saved_model_param = saved_model_param.half()
        assert torch.equal(orig_param.cpu(), saved_model_param)


def test_ds_checkpoint_dir_tag_and_missing_dir_error(tmp_path):
    tag_dir = tmp_path / "global_step10"
    tag_dir.mkdir()
    assert ds_checkpoint_dir(tmp_path, tag="global_step10") == str(tag_dir)

    missing_tag = "global_step99"
    expected_dir = os.path.join(tmp_path, missing_tag)
    with pytest.raises(FileNotFoundError, match=f"Directory '{expected_dir}' doesn't exist"):
        ds_checkpoint_dir(tmp_path, tag=missing_tag)


def test_convert_zero_checkpoint_to_fp32_state_dict_passes_tag(tmp_path, monkeypatch):
    import lightning.pytorch.utilities.deepspeed as ds_utils

    # Create two tag directories: 'latest' points to step20, while caller explicitly asks for step10.
    (tmp_path / "latest").write_text("global_step20\n")
    step10_dir = tmp_path / "global_step10"
    step20_dir = tmp_path / "global_step20"
    step10_dir.mkdir()
    step20_dir.mkdir()

    torch.save({"optimizer_state_dict": {"zero_stage": 2}}, step10_dir / "zero_pp_rank_0_mp_rank_00_optim_states.pt")
    torch.save(
        {"epoch": 1, "global_step": 10, "module": {}, "optimizer": {}},
        step10_dir / "mp_rank_00_model_states.pt",
    )
    torch.save({"optimizer_state_dict": {"zero_stage": 2}}, step20_dir / "zero_pp_rank_0_mp_rank_00_optim_states.pt")
    torch.save(
        {"epoch": 2, "global_step": 20, "module": {}, "optimizer": {}},
        step20_dir / "mp_rank_00_model_states.pt",
    )

    fake_zero_to_fp32 = types.ModuleType("deepspeed.utils.zero_to_fp32")
    fake_zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint = lambda ckpt_dir, tag: {
        "_forward_module.layer.weight": torch.tensor([1.0, 2.0])
    }
    fake_zero_to_fp32.get_optim_files = lambda ckpt_dir: [
        os.path.join(ckpt_dir, "zero_pp_rank_0_mp_rank_00_optim_states.pt")
    ]
    fake_zero_to_fp32.get_model_state_file = lambda ckpt_dir, zero_stage: os.path.join(
        ckpt_dir, "mp_rank_00_model_states.pt"
    )

    fake_ds = types.ModuleType("deepspeed")
    fake_ds_utils = types.ModuleType("deepspeed.utils")
    fake_ds_utils.zero_to_fp32 = fake_zero_to_fp32
    fake_ds.utils = fake_ds_utils

    monkeypatch.setattr(ds_utils, "_DEEPSPEED_AVAILABLE", True)
    with mock.patch.dict(
        sys.modules,
        {
            "deepspeed": fake_ds,
            "deepspeed.utils": fake_ds_utils,
            "deepspeed.utils.zero_to_fp32": fake_zero_to_fp32,
        },
    ):
        output_file = tmp_path / "fp32.pt"
        client_state = convert_zero_checkpoint_to_fp32_state_dict(
            tmp_path, output_file, tag="global_step10"
        )

    assert client_state["epoch"] == 1
    assert client_state["global_step"] == 10
    assert torch.equal(client_state["state_dict"]["layer.weight"], torch.tensor([1.0, 2.0]))

