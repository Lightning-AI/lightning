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
from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks import LearningRateFinder
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.tuner import Tuner


class Sentinel(Callback):
    def __init__(self):
        self.steps = []

    def on_train_batch_end(self, tr, model, *args):
        self.steps.append(tr.global_step)


class Model(LightningModule):
    def __init__(self, optimizer="sgd", failure=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.lr = 0.01
        self.optimizer = optimizer
        self.failure = failure

    def training_step(self, batch, batch_idx):
        if self.failure == ("batch", batch_idx):
            raise RuntimeError("deliberate LR failure")
        return (self.weight * batch[0] - 1).square().mean()

    def on_after_backward(self):
        if self.failure == ("backward", self.global_step):
            raise RuntimeError("deliberate LR failure")

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.failure == ("end", batch_idx):
            raise RuntimeError("deliberate LR failure")

    def configure_optimizers(self):
        opt = (
            torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
            if self.optimizer == "sgd"
            else torch.optim.Adam(self.parameters(), lr=self.lr)
        )
        return {"optimizer": opt, "lr_scheduler": torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)}

    def train_dataloader(self):
        return DataLoader(TensorDataset(torch.arange(1.0, 9.0) / 8), batch_size=1)


@pytest.mark.parametrize("acc", [1, 2])
@pytest.mark.parametrize("mode", ["linear", "exponential"])
@pytest.mark.parametrize("entry", ["tuner", "callback"])
@pytest.mark.parametrize("optimizer", ["sgd", "adam"])
@pytest.mark.parametrize("failure", [("batch", 0), ("batch", 2), ("backward", 1), ("end", 1)])
def test_failure_restores_and_retry_matches_fresh(tmp_path, entry, optimizer, failure, acc, mode):
    sentinel = Sentinel()
    logger = CSVLogger(tmp_path / "logs")
    finder = LearningRateFinder(num_training_steps=4, update_attr=False, mode=mode)
    cbs = [sentinel] + ([finder] if entry == "callback" else [])
    tr = Trainer(
        default_root_dir=tmp_path / "failed",
        max_steps=6,
        accumulate_grad_batches=acc,
        limit_val_batches=2,
        callbacks=cbs,
        logger=logger,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    callbacks = list(tr.callbacks)
    model = Model(optimizer, failure)
    initial = deepcopy(model.state_dict())
    loggers = list(tr.loggers)
    with pytest.raises(RuntimeError, match="deliberate LR failure"):
        if entry == "tuner":
            Tuner(tr).lr_find(model, num_training=4, mode=mode, update_attr=False, early_stop_threshold=None)
        else:
            tr.fit(model)
    assert tr.callbacks == callbacks
    assert tr.loggers == loggers
    assert tr.max_steps == 6
    assert tr.limit_val_batches == 2
    for key, value in initial.items():
        torch.testing.assert_close(model.state_dict()[key], value)
    assert tr.optimizers[0].param_groups[0]["lr"] == 0.01
    assert not tr.optimizers[0].state
    assert isinstance(tr.lr_scheduler_configs[0].scheduler, torch.optim.lr_scheduler.StepLR)
    assert not list((tmp_path / "failed").glob(".lr_find*"))
    model.failure = None
    tr.callbacks = [c for c in tr.callbacks if c is not finder]
    sentinel.steps = []
    tr.fit(model)
    control = Model(optimizer)
    ct = Trainer(
        default_root_dir=tmp_path / "control",
        max_steps=6,
        accumulate_grad_batches=acc,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    ct.fit(control)
    torch.testing.assert_close(model.weight, control.weight)
    actual = tr.optimizers[0].state[model.weight]
    expected = ct.optimizers[0].state[control.weight]
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[key], value)
        else:
            assert actual[key] == value
    assert tr.lr_scheduler_configs[0].scheduler.state_dict() == ct.lr_scheduler_configs[0].scheduler.state_dict()
    assert tr.global_step == ct.global_step == 6
    assert sentinel.steps == [(i + 1) // acc for i in range(6 * acc)]


@pytest.mark.parametrize("optimizer", ["sgd", "adam"])
@pytest.mark.parametrize("stop", [RuntimeError, KeyboardInterrupt])
def test_existing_optimizer_state_restored(tmp_path, optimizer, stop):

    class WarmModel(Model):
        fail_search = False

        def training_step(self, batch, batch_idx):
            if self.fail_search and batch_idx == 1:
                raise stop("search interruption")
            return super().training_step(batch, batch_idx)

    class ScheduledFinder(LearningRateFinder):
        def on_fit_start(self, *args):
            pass

        def on_train_epoch_start(self, tr, m):
            if tr.current_epoch != 1:
                return
            before_weights = deepcopy(m.state_dict())
            before_opt = deepcopy(tr.optimizers[0].state_dict())
            before_sched = deepcopy(tr.lr_scheduler_configs[0].scheduler.state_dict())
            before_step = tr.global_step
            assert before_opt["state"]
            m.fail_search = True
            try:
                self.lr_find(tr, m)
            except stop:
                for k, v in before_weights.items():
                    torch.testing.assert_close(m.state_dict()[k], v)
                after = tr.optimizers[0].state_dict()
                assert after["param_groups"] == before_opt["param_groups"]
                for p, state in before_opt["state"].items():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            torch.testing.assert_close(after["state"][p][k], v)
                        else:
                            assert after["state"][p][k] == v
                assert tr.lr_scheduler_configs[0].scheduler.state_dict() == before_sched
                assert tr.global_step == before_step
                assert tr.callbacks == [self]
                assert tr.max_steps == -1
                assert tr.limit_val_batches == 1.0
                raise
            else:
                raise AssertionError("expected interruption")

    m = WarmModel(optimizer)
    cb = ScheduledFinder(num_training_steps=4, update_attr=False, early_stop_threshold=None)
    tr = Trainer(
        default_root_dir=tmp_path,
        max_epochs=3,
        callbacks=[cb],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    with pytest.raises((stop, SystemExit), match="search interruption|1"):
        tr.fit(m)
