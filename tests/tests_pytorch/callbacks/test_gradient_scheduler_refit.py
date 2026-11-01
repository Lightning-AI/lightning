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
import copy
import pickle

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks import GradientAccumulationScheduler


class RefitModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.seen = []

    def training_step(self, batch, batch_idx):
        self.seen.append((self.current_epoch, self.trainer.accumulate_grad_batches))
        return (self.weight * batch[0]).square().mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def train_dataloader(self):
        return DataLoader(TensorDataset(torch.ones(8, 1)), batch_size=1)


class StopEveryEpoch(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        trainer.should_stop = True


def make_trainer(tmp_path, callback, **kwargs):
    return Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        devices=1,
        max_epochs=3,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=[callback, StopEveryEpoch()],
        **kwargs,
    )


@pytest.mark.parametrize("schedule", [{0: 1}, {0: 2}, {0: 4}, {0: 2, 1: 4, 2: 1}, {1: 2, 2: 4}])
@pytest.mark.parametrize("copy_mode", ["none", "pickle", "deepcopy"])
def test_scheduled_accumulation_across_fits(tmp_path, schedule, copy_mode):
    callback = GradientAccumulationScheduler(dict(schedule))
    trainer = make_trainer(tmp_path, callback)
    model = RefitModel()
    expected_steps = 0
    for epoch in range(3):
        trainer.fit(model)
        expected_steps += 8 // callback.get_accumulate_grad_batches(epoch)
        assert trainer.current_epoch == epoch + 1
        assert trainer.global_step == expected_steps
        assert model.seen == [(e, callback.get_accumulate_grad_batches(e)) for e in range(epoch + 1) for _ in range(8)]
        if copy_mode == "pickle":
            trainer, model = pickle.loads(pickle.dumps((trainer, model)))
        elif copy_mode == "deepcopy":
            trainer, model = copy.deepcopy((trainer, model))
        callback = next(c for c in trainer.callbacks if isinstance(c, GradientAccumulationScheduler))


@pytest.mark.parametrize("copy_mode", ["none", "pickle", "deepcopy"])
def test_callback_reuse_does_not_retain_trainer_or_accept_conflict(tmp_path, copy_mode):
    callback = GradientAccumulationScheduler({0: 2})
    first = make_trainer(tmp_path / "first", callback)
    first.fit(RefitModel())
    if copy_mode == "pickle":
        callback = pickle.loads(pickle.dumps(callback))
    elif copy_mode == "deepcopy":
        callback = copy.deepcopy(callback)
    assert not any(isinstance(value, Trainer) for value in vars(callback).values())
    conflicting = make_trainer(tmp_path / "conflict", callback, accumulate_grad_batches=2)
    with pytest.raises(ValueError, match="Either remove"):
        conflicting.fit(RefitModel())
    second = make_trainer(tmp_path / "second", callback)
    second.fit(RefitModel())
    assert second.global_step == 4
    first.fit(first.lightning_module)
    assert first.current_epoch == 2
    assert first.global_step == 8


@pytest.mark.parametrize("new_factor", [3, 4])
def test_external_assignment_between_fits_is_rejected(tmp_path, new_factor):
    callback = GradientAccumulationScheduler({0: 2})
    trainer = make_trainer(tmp_path, callback)
    trainer.fit(RefitModel())
    trainer.accumulate_grad_batches = new_factor
    with pytest.raises(ValueError, match="Either remove"):
        trainer.fit(trainer.lightning_module)


def test_replacement_callback_cannot_claim_previous_assignment(tmp_path):
    callback = GradientAccumulationScheduler({0: 2})
    trainer = make_trainer(tmp_path, callback)
    trainer.fit(RefitModel())
    trainer.callbacks = [GradientAccumulationScheduler({0: 2}), StopEveryEpoch()]
    with pytest.raises(ValueError, match="Either remove"):
        trainer.fit(trainer.lightning_module)


@pytest.mark.parametrize("reuse_callback", [False, True])
@pytest.mark.parametrize("schedule", [{0: 2}, {0: 2, 1: 4, 2: 1}])
def test_checkpoint_resume_then_refit(tmp_path, reuse_callback, schedule):
    callback = GradientAccumulationScheduler(dict(schedule))
    trainer = make_trainer(tmp_path / "first", callback)
    trainer.fit(RefitModel())
    path = tmp_path / "state.ckpt"
    trainer.save_checkpoint(path)
    state = torch.load(path, weights_only=False)
    assert callback.state_dict() == {}
    assert not any("gradient_accumulation_scheduler_assignment" in key for key in state)
    if not reuse_callback:
        callback = GradientAccumulationScheduler(dict(schedule))
    resumed = make_trainer(tmp_path / "resumed", callback)
    model = RefitModel()
    resumed.fit(model, ckpt_path=path)
    resumed.fit(model)
    assert resumed.current_epoch == 3
    assert resumed.global_step == sum(8 // callback.get_accumulate_grad_batches(e) for e in range(3))
    assert model.seen == [(e, callback.get_accumulate_grad_batches(e)) for e in (1, 2) for _ in range(8)]


@pytest.mark.parametrize("schedule", [{0: 2}, {0: 2, 1: 4, 2: 1}])
def test_repeated_fit_weights_match_uninterrupted_training(tmp_path, schedule):
    callback = GradientAccumulationScheduler(dict(schedule))
    trainer = make_trainer(tmp_path / "refit", callback)
    model = RefitModel()
    for _ in range(3):
        trainer.fit(model)
    reference_callback = GradientAccumulationScheduler(dict(schedule))
    reference = make_trainer(tmp_path / "reference", reference_callback)
    reference.callbacks = [reference_callback]
    reference_model = RefitModel()
    reference.fit(reference_model)
    assert model.seen == reference_model.seen
    assert trainer.global_step == reference.global_step
    torch.testing.assert_close(model.weight, reference_model.weight)
