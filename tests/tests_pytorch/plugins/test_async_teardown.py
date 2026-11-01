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
from pathlib import Path
from threading import Event, Thread

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from lightning.fabric.plugins.io import TorchCheckpointIO
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.plugins.io import AsyncCheckpointIO


class BufferedIO(TorchCheckpointIO):
    def __init__(self, save_error=None, close_error=None):
        super().__init__()
        self.pending = []
        self.events = []
        self.save_error = save_error
        self.close_error = close_error
        self.started = Event()
        self.finished = Event()
        self.release = Event()
        self.release.set()

    def save_checkpoint(self, checkpoint, path, storage_options=None):
        self.started.set()
        assert self.release.wait(10)
        self.events.append("save")
        try:
            if self.save_error is not None:
                raise self.save_error
            self.pending.append((checkpoint, path, storage_options))
        finally:
            self.finished.set()

    def teardown(self):
        self.events.append("teardown")
        assert not self.started.is_set() or self.finished.is_set()
        for checkpoint, path, options in self.pending:
            super().save_checkpoint(checkpoint, path, options)
        self.pending.clear()
        if self.close_error is not None:
            raise self.close_error


def wrap(backend, depth):
    for _ in range(depth):
        backend = AsyncCheckpointIO(backend)
    return backend


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("count", [0, 1, 3])
def test_flush_after_all_workers(tmp_path, depth, count):
    backend = BufferedIO()
    plugin = wrap(backend, depth)
    for cycle in range(2):
        for index in range(count):
            plugin.save_checkpoint({"weight": torch.tensor([cycle, index])}, tmp_path / f"{cycle}-{index}.ckpt")
        plugin.teardown()
        assert backend.events == (["save"] * count + ["teardown"]) * (cycle + 1)
        assert not backend.pending
        for index in range(count):
            checkpoint = torch.load(tmp_path / f"{cycle}-{index}.ckpt", weights_only=True)
            torch.testing.assert_close(checkpoint["weight"], torch.tensor([cycle, index]))
        current = plugin
        for _ in range(depth):
            assert current._executor is None
            current = current.checkpoint_io


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("save_type", [None, ValueError, KeyboardInterrupt])
@pytest.mark.parametrize("close_type", [None, OSError, SystemExit])
def test_error_precedence_and_cleanup(tmp_path, depth, save_type, close_type):
    save_error = save_type("background failure") if save_type else None
    close_error = close_type("cleanup failure") if close_type else None
    backend = BufferedIO(save_error, close_error)
    backend.release.clear()
    plugin = wrap(backend, depth)
    plugin.save_checkpoint({}, tmp_path / "checkpoint.ckpt")
    assert backend.started.wait(10)
    backend.release.set()
    assert backend.finished.wait(10)
    expected = save_error or close_error
    if expected is None:
        plugin.teardown()
    else:
        with pytest.raises(type(expected)) as info:
            plugin.teardown()
        assert info.value is expected
        if save_error is not None and close_error is not None:
            assert info.value.__context__ is close_error
    assert backend.events == ["save", "teardown"]
    assert not backend.pending
    assert plugin._executor is None


def test_cleanup_waits_for_blocked_worker(tmp_path):
    backend = BufferedIO()
    backend.release.clear()
    plugin = AsyncCheckpointIO(backend)
    plugin.save_checkpoint({}, tmp_path / "checkpoint.ckpt")
    assert backend.started.wait(10)
    errors = []
    entered_shutdown = Event()
    original_shutdown = plugin._executor.shutdown

    def shutdown(*args, **kwargs):
        entered_shutdown.set()
        return original_shutdown(*args, **kwargs)

    plugin._executor.shutdown = shutdown

    def close():
        try:
            plugin.teardown()
        except BaseException as error:
            errors.append(error)

    thread = Thread(target=close)
    thread.start()
    try:
        assert entered_shutdown.wait(10)
        assert backend.events == []
    finally:
        backend.release.set()
        thread.join(10)
    assert not thread.is_alive()
    assert not errors
    assert backend.events == ["save", "teardown"]
    assert (tmp_path / "checkpoint.ckpt").is_file()


class TinyModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def training_step(self, batch, batch_idx):
        return (self.weight * batch[0]).square().mean()

    def validation_step(self, batch, batch_idx):
        return self.weight.detach().clone()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def train_dataloader(self):
        return DataLoader(TensorDataset(torch.ones(4, 1)), batch_size=1)

    def val_dataloader(self):
        return self.train_dataloader()


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_trainer_flushes_backend_across_calls(tmp_path, depth):
    backend = BufferedIO()
    plugin = wrap(backend, depth)
    checkpoint = ModelCheckpoint(dirpath=tmp_path, save_top_k=-1)
    trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        plugins=[plugin],
        callbacks=[checkpoint],
        num_sanity_val_steps=0,
    )
    model = TinyModel()
    trainer.validate(model)
    assert backend.events == ["teardown"]
    trainer.fit(model)
    assert backend.events == ["teardown", "save", "save", "teardown"]
    assert not backend.pending
    saved = torch.load(checkpoint.best_model_path, weights_only=False)
    torch.testing.assert_close(saved["state_dict"]["weight"], model.weight)
    assert saved["global_step"] == trainer.global_step == 8
    assert len(list(Path(tmp_path).glob("*.ckpt"))) == 2
    trainer.validate(model)
    assert backend.events[-1] == "teardown"
    assert backend.events.count("teardown") == 3


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("failure", ["train", "save"])
def test_trainer_failure_closes_backend(tmp_path, depth, failure):
    error = RuntimeError(f"deliberate {failure} failure")
    backend = BufferedIO(save_error=error if failure == "save" else None)
    plugin = wrap(backend, depth)

    class FailingModel(TinyModel):
        def training_step(self, batch, batch_idx):
            if failure == "train" and batch_idx == 1:
                raise error
            return super().training_step(batch, batch_idx)

    trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        plugins=[plugin],
        num_sanity_val_steps=0,
    )
    with pytest.raises(RuntimeError, match=f"deliberate {failure} failure"):
        trainer.fit(FailingModel())
    assert backend.events.count("teardown") >= 1
    assert not backend.pending
    assert plugin._executor is None


def test_unconfigured_wrapper_teardown():
    plugin = AsyncCheckpointIO()
    plugin.teardown()
    assert plugin._executor is None
    assert plugin.checkpoint_io is None


@pytest.mark.parametrize("close_type", [OSError, KeyboardInterrupt])
def test_unused_backend_cleanup_error(close_type):
    error = close_type("unused backend cleanup failed")
    backend = BufferedIO(close_error=error)
    plugin = AsyncCheckpointIO(backend)
    with pytest.raises(close_type) as info:
        plugin.teardown()
    assert info.value is error
    assert backend.events == ["teardown"]
