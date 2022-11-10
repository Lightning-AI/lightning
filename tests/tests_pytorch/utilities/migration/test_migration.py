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
from unittest.mock import ANY

import pytest
import torch

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.demos.boring_classes import BoringModel, ManualOptimBoringModel
from pytorch_lightning.utilities.migration import migrate_checkpoint
from pytorch_lightning.utilities.migration.utils import _get_version, _set_legacy_version, _set_version


@pytest.mark.parametrize(
    "old_checkpoint, new_checkpoint",
    [
        (
            {"epoch": 1, "global_step": 23, "checkpoint_callback_best": 0.34},
            {"epoch": 1, "global_step": 23, "callbacks": {ModelCheckpoint: {"best_model_score": 0.34}}, "loops": ANY},
        ),
        (
            {"epoch": 1, "global_step": 23, "checkpoint_callback_best_model_score": 0.99},
            {"epoch": 1, "global_step": 23, "callbacks": {ModelCheckpoint: {"best_model_score": 0.99}}, "loops": ANY},
        ),
        (
            {"epoch": 1, "global_step": 23, "checkpoint_callback_best_model_path": "path"},
            {"epoch": 1, "global_step": 23, "callbacks": {ModelCheckpoint: {"best_model_path": "path"}}, "loops": ANY},
        ),
        (
            {"epoch": 1, "global_step": 23, "early_stop_callback_wait": 2, "early_stop_callback_patience": 4},
            {
                "epoch": 1,
                "global_step": 23,
                "callbacks": {EarlyStopping: {"wait_count": 2, "patience": 4}},
                "loops": ANY,
            },
        ),
    ],
)
def test_migrate_model_checkpoint_early_stopping(tmpdir, old_checkpoint, new_checkpoint):
    _set_version(old_checkpoint, "0.9.0")
    _set_legacy_version(new_checkpoint, "0.9.0")
    _set_version(new_checkpoint, pl.__version__)
    updated_checkpoint, _ = migrate_checkpoint(old_checkpoint)
    assert updated_checkpoint == old_checkpoint == new_checkpoint
    assert _get_version(updated_checkpoint) == pl.__version__


def test_migrate_loop_global_step_to_progress_tracking():
    old_checkpoint = {"global_step": 15, "epoch": 2}
    _set_version(old_checkpoint, "1.5.9")  # pretend a checkpoint prior to 1.6.0
    updated_checkpoint, _ = migrate_checkpoint(old_checkpoint)
    # automatic optimization
    assert (
        updated_checkpoint["loops"]["fit_loop"]["epoch_loop.batch_loop.optimizer_loop.optim_progress"]["optimizer"][
            "step"
        ]["total"]["completed"]
        == 15
    )
    # for manual optimization
    assert (
        updated_checkpoint["loops"]["fit_loop"]["epoch_loop.batch_loop.manual_loop.optim_step_progress"]["total"][
            "completed"
        ]
        == 15
    )


def test_migrate_loop_current_epoch_to_progress_tracking():
    old_checkpoint = {"global_step": 15, "epoch": 2}
    _set_version(old_checkpoint, "1.5.9")  # pretend a checkpoint prior to 1.6.0
    updated_checkpoint, _ = migrate_checkpoint(old_checkpoint)
    assert updated_checkpoint["loops"]["fit_loop"]["epoch_progress"]["current"]["completed"] == 2


@pytest.mark.parametrize("model_class", [BoringModel, ManualOptimBoringModel])
def test_migrate_loop_batches_that_stepped(tmpdir, model_class):
    trainer = Trainer(max_steps=1, limit_val_batches=0, default_root_dir=tmpdir)
    model = model_class()
    trainer.fit(model)
    ckpt_path = trainer.checkpoint_callback.best_model_path

    # pretend we have a checkpoint produced in < v1.6.5; the key "_batches_that_stepped" didn't exist back then
    ckpt = torch.load(ckpt_path)
    del ckpt["loops"]["fit_loop"]["epoch_loop.state_dict"]["_batches_that_stepped"]
    _set_version(ckpt, "1.6.4")
    torch.save(ckpt, ckpt_path)

    class TestModel(model_class):
        def on_train_start(self) -> None:
            assert self.trainer.global_step == 1
            assert self.trainer.fit_loop.epoch_loop._batches_that_stepped == 1

    trainer = Trainer(max_steps=2, limit_val_batches=0, default_root_dir=tmpdir)
    model = TestModel()
    trainer.fit(model, ckpt_path=ckpt_path)
    new_loop = trainer.fit_loop.epoch_loop
    assert new_loop.global_step == new_loop._batches_that_stepped == 2
