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
import logging
from unittest.mock import Mock, patch

import pytest
import torch
from lightning_utilities.test.warning import no_warning_call

from lightning.fabric.utilities.warnings import PossibleUserWarning
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.trainer.trainer import Trainer


def test_no_val_on_train_epoch_loop_restart(tmp_path):
    """Test that training validation loop doesn't get triggered at the beginning of a restart."""
    trainer_kwargs = {
        "max_epochs": 1,
        "limit_train_batches": 1,
        "limit_val_batches": 1,
        "num_sanity_val_steps": 0,
        "logger": False,
        "enable_checkpointing": False,
    }
    trainer = Trainer(**trainer_kwargs)
    model = BoringModel()
    trainer.fit(model)
    ckpt_path = str(tmp_path / "last.ckpt")
    trainer.save_checkpoint(ckpt_path)

    trainer_kwargs["max_epochs"] = 2
    trainer = Trainer(**trainer_kwargs)

    with patch.object(
        trainer.fit_loop.epoch_loop.val_loop,
        "_evaluation_step",
        wraps=trainer.fit_loop.epoch_loop.val_loop._evaluation_step,
    ) as step_mock:
        trainer.fit(model, ckpt_path=ckpt_path)
    assert step_mock.call_count == 1


@pytest.mark.parametrize(
    ("min_epochs", "min_steps", "current_epoch", "global_step", "early_stop", "epoch_loop_done", "raise_info_msg"),
    [
        (None, None, 1, 4, True, True, False),
        (None, None, 1, 10, True, True, False),
        (4, None, 1, 4, False, False, True),
        (4, 2, 1, 4, False, False, True),
        (4, None, 1, 10, False, True, False),
        (4, 3, 1, 3, False, False, True),
        (4, 10, 1, 10, False, True, False),
        (None, 4, 1, 4, True, True, False),
    ],
)
def test_should_stop_early_stopping_conditions_not_met(
    caplog, min_epochs, min_steps, current_epoch, global_step, early_stop, epoch_loop_done, raise_info_msg
):
    """Test that checks that info message is logged when users sets `should_stop` but min conditions are not met."""
    trainer = Trainer(min_epochs=min_epochs, min_steps=min_steps, limit_val_batches=0)
    trainer.fit_loop.max_batches = 10
    trainer.should_stop = True
    trainer.fit_loop.epoch_loop.automatic_optimization.optim_progress.optimizer.step.total.completed = global_step
    trainer.fit_loop.epoch_loop.batch_progress.current.ready = global_step
    trainer.fit_loop.epoch_progress.current.completed = current_epoch - 1

    message = f"min_epochs={min_epochs}` or `min_steps={min_steps}` has not been met. Training will continue"
    with caplog.at_level(logging.INFO, logger="lightning.pytorch.loops"):
        assert trainer.fit_loop.epoch_loop.done is epoch_loop_done

    assert (message in caplog.text) is raise_info_msg
    assert trainer.fit_loop._can_stop_early is early_stop


@pytest.mark.parametrize(("min_epochs", "min_steps", "val_count"), [(3, None, 3), (None, 3, 2)])
def test_should_stop_triggers_validation_once(min_epochs, min_steps, val_count, tmp_path):
    """Regression test for issue #15708.

    Test that the request for `should_stop=True` only triggers validation when Trainer is allowed to stop
    (min_epochs/steps is satisfied).

    """

    class NewBoring(BoringModel):
        def training_step(self, batch, batch_idx):
            self.log("loss", self.step(batch))
            return {"loss": self.step(batch)}

    model = NewBoring()
    # create a stopping condition with a high threshold so it triggers immediately
    # check the condition before validation so the count is unaffected
    stopping = EarlyStopping(monitor="loss", check_on_train_epoch_end=True, stopping_threshold=100)
    trainer = Trainer(
        default_root_dir=tmp_path,
        num_sanity_val_steps=0,
        limit_val_batches=2,
        limit_train_batches=2,
        max_epochs=3,
        min_epochs=min_epochs,
        min_steps=min_steps,
        enable_model_summary=False,
        enable_checkpointing=False,
        callbacks=[stopping],
    )
    trainer.fit_loop.epoch_loop.val_loop.run = Mock()
    trainer.fit(model)
    assert trainer.fit_loop.epoch_loop.val_loop.run.call_count == val_count


def test_training_loop_dataloader_iter_multiple_dataloaders(tmp_path):
    trainer = Trainer(
        default_root_dir=tmp_path,
        limit_train_batches=3,
        limit_val_batches=0,
        max_epochs=1,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
        devices=1,
    )

    class MyModel(BoringModel):
        batch_start_ins = []
        step_outs = []
        batch_end_ins = []

        def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
            self.batch_start_ins.append((batch, batch_idx, dataloader_idx))

        def training_step(self, dataloader_iter):
            self.step_outs.append(next(dataloader_iter))

        def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
            self.batch_end_ins.append((batch, batch_idx, dataloader_idx))

    model = MyModel()
    trainer.fit(model, {"a": [0, 1], "b": [2, 3]})

    assert model.batch_start_ins == [(None, 0, 0)] + model.step_outs[:-1]
    assert model.step_outs == [({"a": 0, "b": 2}, 0, 0), ({"a": 1, "b": 3}, 1, 0)]
    assert model.batch_end_ins == model.step_outs


def test_no_batch_idx_gradient_accumulation():
    """Regression test for an issue where excluding the batch_idx from training_step would disable gradient
    accumulation."""

    class MyModel(BoringModel):
        last_batch_idx = -1

        def training_step(self, batch):  # no batch_idx
            return self.step(batch)

        def optimizer_step(self, epoch, batch_idx, *args, **kwargs):
            assert batch_idx in (1, 3)
            self.last_batch_idx = batch_idx
            return super().optimizer_step(epoch, batch_idx, *args, **kwargs)

    trainer = Trainer(fast_dev_run=4, accumulate_grad_batches=2, limit_val_batches=0)
    model = MyModel()
    trainer.fit(model)
    assert model.last_batch_idx == 3


def test_resume_mid_epoch_warning(tmp_path):
    """Test that resuming from a mid-epoch checkpoint raises a warning unless the dataloader is stateful."""

    class NotStatefulIterable:
        def __init__(self):
            self.index = 0

        def __iter__(self):
            for i in range(self.index, len(self)):
                yield torch.ones(2, 32) * i

        def __len__(self):
            return 3

    class StatefulIterable(NotStatefulIterable):
        def state_dict(self):
            return {"index": self.index}

        def load_state_dict(self, state_dict):
            self.index = state_dict["index"]

    trainer_kwargs = {
        "default_root_dir": tmp_path,
        "accelerator": "cpu",
        "max_epochs": 1,
        "enable_model_summary": False,
        "enable_progress_bar": False,
        "logger": False,
    }

    def train_and_resume(dataloader, resume_step, expected_warning):
        # Initial training
        checkpoint_dir = tmp_path / "checkpoints"
        trainer = Trainer(
            **trainer_kwargs,
            callbacks=ModelCheckpoint(dirpath=checkpoint_dir, every_n_train_steps=1, save_top_k=-1),
        )
        trainer.fit(BoringModel(), dataloader)

        # Resume
        trainer = Trainer(**trainer_kwargs, enable_checkpointing=False)
        resume_from = checkpoint_dir / f"epoch=0-step={resume_step}.ckpt"
        warn_assert = pytest.warns if expected_warning else no_warning_call
        with warn_assert(PossibleUserWarning, match="resuming from a checkpoint that ended before"):
            trainer.fit(BoringModel(), dataloader, ckpt_path=resume_from)

    # Resume mid-epoch, no stateful dataloader -> warning
    train_and_resume(dataloader=NotStatefulIterable(), resume_step=1, expected_warning=True)

    # Resume end-of-epoch, no stateful dataloader -> no warning
    train_and_resume(dataloader=NotStatefulIterable(), resume_step=3, expected_warning=False)

    # Resume mid-epoch, stateful dataloader -> no warning
    train_and_resume(dataloader=StatefulIterable(), resume_step=1, expected_warning=False)
