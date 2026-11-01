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
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback, EMAWeightAveraging, WeightAveraging


class ReuseModel(LightningModule):
    def __init__(self, initial=1.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(initial))

    def training_step(self, batch, batch_idx):
        return self.weight.square()

    def train_dataloader(self):
        return DataLoader(TensorDataset(torch.ones(4, 1)), batch_size=1)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9)


class EpochAveraging(WeightAveraging):
    def should_update(self, step_idx=None, epoch_idx=None):
        return epoch_idx is not None


class DelayedAveraging(WeightAveraging):
    def should_update(self, step_idx=None, epoch_idx=None):
        return step_idx is not None and step_idx >= 2


class InspectFreshStart(WeightAveraging):
    def on_train_start(self, tr, model):
        assert self._latest_update_step == 0
        assert self._latest_update_epoch == -1


@pytest.mark.parametrize(
    "kind", [WeightAveraging, EMAWeightAveraging, EpochAveraging, DelayedAveraging, InspectFreshStart]
)
@pytest.mark.parametrize("acc", [1, 2])
@pytest.mark.parametrize("warmstart", [False, True])
def test_reused_callback_matches_fresh(tmp_path, kind, acc, warmstart):
    def train(model, cb, path):
        tr = Trainer(
            default_root_dir=path,
            max_epochs=2,
            accumulate_grad_batches=acc,
            callbacks=[cb],
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
        )
        tr.fit(model)
        return (
            model.weight.detach().clone(),
            cb._average_model.module.weight.detach().clone(),
            cb._average_model.n_averaged.clone(),
            tr.optimizers[0].state[model.weight]["momentum_buffer"].clone(),
        )

    cb = kind()
    model = ReuseModel()
    train(model, cb, tmp_path / "first")
    initial = model.weight.item() if warmstart else 1.0
    for run in range(2):
        got = train(ReuseModel(initial), cb, tmp_path / f"reused-{run}")
        expected = train(ReuseModel(initial), kind(), tmp_path / f"control-{run}")
        for actual_value, expected_value in zip(got, expected):
            torch.testing.assert_close(actual_value, expected_value)


@pytest.mark.parametrize("kind", [WeightAveraging, EMAWeightAveraging])
@pytest.mark.parametrize("reused", [False, True])
def test_restore_after_setup_reinitialization(tmp_path, kind, reused):
    ck = tmp_path / "partial.ckpt"

    class Save(Callback):
        def on_train_batch_end(self, trainer, pl_module, *args):
            if trainer.global_step == 2:
                trainer.save_checkpoint(ck)

    def tr(cb, steps, path, extra=None):
        return Trainer(
            default_root_dir=path,
            max_steps=steps,
            callbacks=[cb] + (extra or []),
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

    model = ReuseModel()
    cb = kind()
    full = tr(cb, 4, tmp_path / "full")
    full.fit(model)
    expected = model.weight.detach().clone()
    first = ReuseModel()
    c = kind()
    partial = tr(c, 2, tmp_path / "partial", [Save()])
    partial.fit(first)
    final = ReuseModel()
    endcb = c if reused else kind()
    resume = tr(endcb, 4, tmp_path / "resume")
    resume.fit(final, ckpt_path=ck)
    torch.testing.assert_close(final.weight, expected)
    assert endcb._average_model.n_averaged == cb._average_model.n_averaged
    torch.testing.assert_close(
        resume.optimizers[0].state[final.weight]["momentum_buffer"],
        full.optimizers[0].state[model.weight]["momentum_buffer"],
    )


# Explicit checkpoint restore is intentionally exercised after fit setup.
