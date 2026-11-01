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

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import (
    Callback,
    EMAWeightAveraging,
    ModelCheckpoint,
    WeightAveraging,
)


class _ExceptionModel(LightningModule):
    def __init__(self, crash=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.crash = crash

    def training_step(self, batch, batch_idx):
        if self.crash == "train" and self.global_step == 2:
            raise RuntimeError("deliberate failure")
        return self.weight.square()

    def validation_step(self, batch, batch_idx):
        if self.crash == "val":
            raise RuntimeError("deliberate failure")

    def train_dataloader(self):
        return DataLoader(TensorDataset(torch.ones(4, 1)), batch_size=1)

    def val_dataloader(self):
        return DataLoader(TensorDataset(torch.ones(1, 1)), batch_size=1)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def _exception_trainer(path, callbacks, max_steps):
    return Trainer(
        default_root_dir=path,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        val_check_interval=2,
    )


@pytest.mark.parametrize("crash", ["train", "val", None])
@pytest.mark.parametrize("checkpoint_first", [False, True])
def test_exception_checkpoint_slots(tmp_path, crash, checkpoint_first):
    cb = WeightAveraging()
    ckpt = ModelCheckpoint(dirpath=tmp_path, save_on_exception=True, save_top_k=-1)
    cbs = [ckpt, cb] if checkpoint_first else [cb, ckpt]
    model = _ExceptionModel(crash)
    tr = _exception_trainer(tmp_path, cbs, 3 if crash == "train" else 2)
    if crash:
        with pytest.raises(RuntimeError, match="deliberate failure"):
            tr.fit(model)
        path = ckpt._last_checkpoint_saved
    else:
        tr.fit(model)
        path = str(tmp_path / "normal.ckpt")
        tr.save_checkpoint(path)
    saved = torch.load(path, weights_only=False)
    torch.testing.assert_close(saved["state_dict"]["weight"], torch.tensor(0.72))
    if crash:
        torch.testing.assert_close(saved["current_model_state"]["weight"], torch.tensor(0.64))
        torch.testing.assert_close(model.weight, torch.tensor(0.64))
    # Every saved checkpoint should initialize resume slots to their correct roles.
    restored = _ExceptionModel()
    new_cb = WeightAveraging()
    new_cb.setup(tr, restored, "fit")
    new_cb.on_load_checkpoint(tr, restored, deepcopy(saved))
    torch.testing.assert_close(new_cb._average_model.module.weight, torch.tensor(0.72))
    if crash:
        torch.testing.assert_close(restored.weight, torch.tensor(0.64))


@pytest.mark.parametrize("crash", ["train", "val"])
def test_actual_trainer_resume_start_roles(tmp_path, crash):
    cb = WeightAveraging()
    ckpt = ModelCheckpoint(dirpath=tmp_path / "before", save_on_exception=True, save_top_k=-1)
    tr = _exception_trainer(tmp_path, [cb, ckpt], 3 if crash == "train" else 2)
    with pytest.raises(RuntimeError, match="deliberate failure"):
        tr.fit(_ExceptionModel(crash))

    class InspectResume(WeightAveraging):
        def on_train_start(self, trainer, pl_module):
            torch.testing.assert_close(pl_module.weight, torch.tensor(0.64))
            torch.testing.assert_close(self._average_model.module.weight, torch.tensor(0.72))

    model = _ExceptionModel()
    tr2 = _exception_trainer(tmp_path / "after", [InspectResume()], 3)
    tr2.fit(model, ckpt_path=ckpt._last_checkpoint_saved)


class StopValidation(Callback):
    def __init__(self, hook):
        self.hook = hook

    def on_validation_epoch_start(self, t, m):
        if self.hook == "start":
            raise RuntimeError("start hook")

    def on_validation_epoch_end(self, t, m):
        if self.hook == "end":
            raise RuntimeError("end hook")


@pytest.mark.parametrize("hook", ["start", "end"])
@pytest.mark.parametrize("first", [True, False])
def test_callback_validation_failure_restores_slots(tmp_path, hook, first):
    cb = WeightAveraging()
    fail = StopValidation(hook)
    model = _ExceptionModel()
    tr = _exception_trainer(tmp_path, [cb, fail] if first else [fail, cb], 2)
    with pytest.raises(RuntimeError, match="hook"):
        tr.fit(model)
    torch.testing.assert_close(model.weight, torch.tensor(0.64))
    torch.testing.assert_close(cb._average_model.module.weight, torch.tensor(0.72))
    assert not getattr(cb, "_validation_weights_swapped", False)
    before = model.weight.detach().clone()
    cb.on_exception(tr, model, RuntimeError())
    torch.testing.assert_close(model.weight, before)


@pytest.mark.parametrize("buffers", [True, False])
def test_exception_restores_buffers_as_swapped(tmp_path, buffers):
    class Buffered(_ExceptionModel):
        def __init__(self):
            super().__init__("val")
            self.register_buffer("counter", torch.tensor(0.0))

        def training_step(self, b, i):
            self.counter.add_(1)
            return super().training_step(b, i)

    cb = WeightAveraging(use_buffers=buffers)
    model = Buffered()
    tr = _exception_trainer(tmp_path, [cb], 2)
    with pytest.raises(RuntimeError, match="deliberate failure"):
        tr.fit(model)
    torch.testing.assert_close(model.weight, torch.tensor(0.64))
    torch.testing.assert_close(model.counter, torch.tensor(2.0))
    assert not getattr(cb, "_validation_weights_swapped", False)


@pytest.mark.parametrize("sanity", [0, 2])
@pytest.mark.parametrize("kind", ["epoch_start", "step", "epoch_end"])
def test_module_validation_exception(tmp_path, sanity, kind):
    class Failure(_ExceptionModel):
        def on_validation_epoch_start(self):
            if kind == "epoch_start":
                raise RuntimeError("module failure")

        def validation_step(self, b, i):
            if kind == "step":
                raise RuntimeError("module failure")

        def on_validation_epoch_end(self):
            if kind == "epoch_end":
                raise RuntimeError("module failure")

    model = Failure()
    cb = WeightAveraging()
    tr = _exception_trainer(tmp_path, [cb], 2)
    tr.num_sanity_val_steps = sanity
    with pytest.raises(RuntimeError, match="module failure"):
        tr.fit(model)
    expected = 1.0 if sanity else 0.64
    torch.testing.assert_close(model.weight, torch.tensor(expected))
    if not sanity:
        torch.testing.assert_close(cb._average_model.module.weight, torch.tensor(0.72))
    assert not getattr(cb, "_validation_weights_swapped", False)


@pytest.mark.parametrize("callback", [WeightAveraging, EMAWeightAveraging])
def test_validation_failure_repeated_fit(tmp_path, callback):
    cb = callback()
    model = _ExceptionModel("val")
    tr = _exception_trainer(tmp_path, [cb], 2)
    with pytest.raises(RuntimeError, match="deliberate failure"):
        tr.fit(model)
    torch.testing.assert_close(model.weight, torch.tensor(0.64))
    model.crash = None
    tr.fit_loop.epoch_loop.max_steps = 3
    tr.fit(model)
    assert not getattr(cb, "_validation_weights_swapped", False)


@pytest.mark.parametrize("kind", ["swa", "ema"])
@pytest.mark.parametrize("acc", [1, 2])
def test_exception_resume_final_optimizer_parity(tmp_path, kind, acc):
    class Model(_ExceptionModel):
        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9)

        def train_dataloader(self):
            return DataLoader(TensorDataset(torch.ones(8, 1)), batch_size=1)

    def callback():
        return WeightAveraging() if kind == "swa" else EMAWeightAveraging(decay=0.8)

    def make(path, cbs):
        t = _exception_trainer(path, cbs, 4)
        t.accumulate_grad_batches = acc
        t.val_check_interval = 2 * acc
        return t

    full = Model()
    fcb = callback()
    f = make(tmp_path / "full", [fcb])
    f.fit(full)
    cb = callback()
    ck = ModelCheckpoint(dirpath=tmp_path / "crashed", save_on_exception=True, save_top_k=-1)
    partial = Model("val")
    t = make(tmp_path / "crashed", [cb, ck])
    with pytest.raises(RuntimeError, match="deliberate failure"):
        t.fit(partial)
    end = Model()
    ecb = callback()
    resume_checkpoint = ModelCheckpoint(dirpath=tmp_path / "crashed", save_top_k=-1)
    r = make(tmp_path / "resume", [ecb, resume_checkpoint])
    r.fit(end, ckpt_path=ck._last_checkpoint_saved)
    torch.testing.assert_close(end.weight, full.weight)
    torch.testing.assert_close(ecb._average_model.module.weight, fcb._average_model.module.weight)
    assert ecb._average_model.n_averaged == fcb._average_model.n_averaged
    torch.testing.assert_close(
        r.optimizers[0].state[end.weight]["momentum_buffer"], f.optimizers[0].state[full.weight]["momentum_buffer"]
    )
    assert r.global_step == f.global_step == 4
