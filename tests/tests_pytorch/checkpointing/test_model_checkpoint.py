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
import math
import os
import pickle
import re
import time
from argparse import Namespace
from datetime import timedelta
from inspect import signature
from pathlib import Path
from typing import Union
from unittest import mock
from unittest.mock import Mock, call, patch

import cloudpickle
import pytest
import torch
import yaml
from jsonargparse import ArgumentParser
from torch import optim

import lightning.pytorch as pl
from lightning.fabric.utilities.cloud_io import _load as pl_load
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.imports import _OMEGACONF_AVAILABLE
from tests_pytorch.helpers.runif import RunIf

if _OMEGACONF_AVAILABLE:
    from omegaconf import Container, OmegaConf


def test_model_checkpoint_state_key():
    early_stopping = ModelCheckpoint(monitor="val_loss")
    expected_id = (
        "ModelCheckpoint{'monitor': 'val_loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1,"
        " 'train_time_interval': None}"
    )
    assert early_stopping.state_key == expected_id


class LogInTwoMethods(BoringModel):
    def training_step(self, batch, batch_idx):
        out = super().training_step(batch, batch_idx)
        self.log("early_stop_on", out["loss"])
        return out

    def on_validation_epoch_end(self):
        self.log("val_acc", torch.tensor(1.23))


def mock_training_epoch_loop(trainer):
    # do not use `unittest.Mock` because we need to store the return value
    calls = {}
    old_get_monitor_value = trainer.fit_loop.epoch_loop._get_monitor_value

    def mock(key):
        value = old_get_monitor_value(key)
        calls[trainer.current_epoch] = {key: value}
        return value

    trainer.fit_loop.epoch_loop._get_monitor_value = mock
    return calls


@pytest.mark.parametrize(
    ("validation_step_none", "val_dataloaders_none", "monitor"),
    [(False, False, "val_log"), (True, False, "train_log_epoch"), (False, True, "val_log")],
)
@pytest.mark.parametrize("reduce_lr_on_plateau", [False, True])
def test_model_checkpoint_score_and_ckpt(
    tmp_path, validation_step_none: bool, val_dataloaders_none: bool, monitor: str, reduce_lr_on_plateau: bool
):
    """Test that when a model checkpoint is saved, it saves with the correct score appended to ckpt_path and checkpoint
    data."""
    max_epochs = 3
    limit_train_batches = 5
    limit_val_batches = 7
    lr, gamma = 1e-1, 2

    class CustomBoringModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.train_log_epochs = torch.randn(max_epochs, limit_train_batches)
            self.val_logs = torch.randn(max_epochs, limit_val_batches)
            self.scores = []

        def training_step(self, batch, batch_idx):
            log_value = self.train_log_epochs[self.current_epoch, batch_idx]
            self.log("train_log", log_value, on_epoch=True)
            return super().training_step(batch, batch_idx)

        def validation_step(self, batch, batch_idx):
            log_value = self.val_logs[self.current_epoch, batch_idx]
            self.log("val_log", log_value)
            return super().validation_step(batch, batch_idx)

        def configure_optimizers(self):
            optimizer = optim.SGD(self.parameters(), lr=lr)

            if reduce_lr_on_plateau:
                lr_scheduler = {
                    "scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer),
                    "monitor": monitor,
                    "strict": True,
                }
            else:
                lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

            return [optimizer], [lr_scheduler]

        def on_train_epoch_end(self):
            if "train" in monitor:
                self.scores.append(self.trainer.logged_metrics[monitor])

        def on_validation_epoch_end(self):
            if not self.trainer.sanity_checking and "val" in monitor:
                self.scores.append(self.trainer.logged_metrics[monitor])

    filename = "{" + f"{monitor}" + ":.4f}-{epoch}"
    checkpoint = ModelCheckpoint(dirpath=tmp_path, filename=filename, monitor=monitor, save_top_k=-1)

    model = CustomBoringModel()

    if validation_step_none:
        model.validation_step = None
    if val_dataloaders_none:
        model.val_dataloaders = None

    trainer = Trainer(
        default_root_dir=tmp_path,
        callbacks=[checkpoint],
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        max_epochs=max_epochs,
        enable_progress_bar=False,
    )
    calls = mock_training_epoch_loop(trainer)
    trainer.fit(model)

    ckpt_files = list(tmp_path.glob("*.ckpt"))
    assert len(ckpt_files) == len(model.scores) == max_epochs

    for epoch in range(max_epochs):
        score = model.scores[epoch]
        expected_score = getattr(model, f"{monitor}s")[epoch].mean().item()
        assert math.isclose(score, expected_score, abs_tol=1e-5)

        expected_filename = f"{monitor}={score:.4f}-epoch={epoch}.ckpt"
        chk = pl_load(os.path.join(checkpoint.dirpath, expected_filename))
        assert chk["epoch"] == epoch
        assert chk["global_step"] == limit_train_batches * (epoch + 1)

        mc_specific_data = chk["callbacks"][
            f"ModelCheckpoint{{'monitor': '{monitor}', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1,"
            " 'train_time_interval': None}"
        ]
        assert mc_specific_data["dirpath"] == checkpoint.dirpath
        assert mc_specific_data["monitor"] == monitor
        assert mc_specific_data["current_score"] == score

        if not reduce_lr_on_plateau:
            actual_step_count = chk["lr_schedulers"][0]["_step_count"]
            actual_lr = chk["lr_schedulers"][0]["_last_lr"][0]
            # checkpoint is saved after updating lr_scheduler states
            assert actual_step_count == epoch + 2  # step_count starts at 1
            assert actual_lr == lr * gamma ** (epoch + 1)
        else:
            assert calls[epoch] == {monitor: score}


@pytest.mark.parametrize(
    ("val_check_interval", "reduce_lr_on_plateau", "epoch_aligned"),
    [(0.25, True, True), (0.25, False, True), (0.42, False, False)],
)
def test_model_checkpoint_score_and_ckpt_val_check_interval(
    tmp_path, val_check_interval, reduce_lr_on_plateau, epoch_aligned
):
    """Test that when a model checkpoint is saved, it saves with the correct score appended to ckpt_path and checkpoint
    data with val_check_interval."""
    seed_everything(0)
    max_epochs = 3
    limit_train_batches = 12
    limit_val_batches = 7
    lr, gamma = 1e-1, 2
    monitor = "val_log"
    per_val_train_batches = int(limit_train_batches * val_check_interval)
    per_epoch_val_checks, leftover_train_batches = divmod(limit_train_batches, per_val_train_batches)

    class CustomBoringModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.val_logs = torch.randn(per_epoch_val_checks * max_epochs, limit_val_batches)
            self.val_loop_count = 0
            self.scores = []

        def validation_step(self, batch, batch_idx):
            log_value = self.val_logs[self.val_loop_count, batch_idx]
            self.log("val_log", log_value)
            return super().validation_step(batch, batch_idx)

        def on_validation_epoch_end(self):
            self.val_loop_count += 1
            self.scores.append(self.trainer.logged_metrics[monitor])

        def configure_optimizers(self):
            optimizer = optim.SGD(self.parameters(), lr=lr)

            if reduce_lr_on_plateau:
                lr_scheduler = {
                    "scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer),
                    "monitor": monitor,
                    "strict": True,
                }
            else:
                lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

            return [optimizer], [lr_scheduler]

    filename = "{" + f"{monitor}" + ":.4f}-{epoch}"
    checkpoint = ModelCheckpoint(dirpath=tmp_path, filename=filename, monitor=monitor, save_top_k=-1)

    model = CustomBoringModel()

    trainer = Trainer(
        default_root_dir=tmp_path,
        callbacks=[checkpoint],
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        max_epochs=max_epochs,
        val_check_interval=val_check_interval,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    calls = mock_training_epoch_loop(trainer)
    trainer.fit(model)

    def _make_assertions(epoch, ix):
        global_ix = ix + per_epoch_val_checks * epoch

        # checkpoint saved at the end of training epoch will have updated lr_scheduler states
        epoch_end_checkpoint = epoch_aligned and ix == (per_epoch_val_checks - 1)

        score = model.scores[global_ix]
        expected_score = getattr(model, f"{monitor}s")[global_ix].mean().item()
        expected_filename = f"{monitor}={score:.4f}-epoch={epoch}.ckpt"
        assert math.isclose(score, expected_score, rel_tol=1e-4)

        chk = pl_load(os.path.join(checkpoint.dirpath, expected_filename))
        assert chk["epoch"] == epoch
        expected_global_step = per_val_train_batches * (global_ix + 1) + (leftover_train_batches * epoch)
        assert chk["global_step"] == expected_global_step

        mc_specific_data = chk["callbacks"][
            f"ModelCheckpoint{{'monitor': '{monitor}', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1,"
            " 'train_time_interval': None}"
        ]
        assert mc_specific_data["dirpath"] == checkpoint.dirpath
        assert mc_specific_data["monitor"] == monitor
        assert mc_specific_data["current_score"] == score

        if not reduce_lr_on_plateau:
            actual_step_count = chk["lr_schedulers"][0]["_step_count"]
            actual_lr = chk["lr_schedulers"][0]["_last_lr"][0]
            assert actual_step_count == epoch + 1 + epoch_end_checkpoint
            assert actual_lr == lr * gamma ** (epoch + epoch_end_checkpoint)

        return score

    ckpt_files = list(tmp_path.glob("*.ckpt"))
    assert len(ckpt_files) == len(model.scores) == per_epoch_val_checks * max_epochs

    for epoch in range(max_epochs):
        for i in range(per_epoch_val_checks):
            score = _make_assertions(epoch, i)

        if reduce_lr_on_plateau:
            assert calls[epoch] == {monitor: score}


@pytest.mark.parametrize("save_top_k", [-1, 0, 1, 2])
def test_model_checkpoint_with_non_string_input(tmp_path, save_top_k: int):
    """Test that dirpath=None in checkpoint callback is valid and that ckpt_path is set correctly."""
    model = LogInTwoMethods()

    checkpoint = ModelCheckpoint(monitor="early_stop_on", dirpath=None, filename="{epoch}", save_top_k=save_top_k)
    max_epochs = 2
    trainer = Trainer(
        default_root_dir=tmp_path, callbacks=[checkpoint], overfit_batches=0.20, max_epochs=max_epochs, logger=False
    )
    trainer.fit(model)
    assert checkpoint.dirpath == str(tmp_path / "checkpoints")

    if save_top_k == -1:
        ckpt_files = os.listdir(checkpoint.dirpath)
        expected_ckpt_files = [f"epoch={i}.ckpt" for i in range(max_epochs)]
        assert len(ckpt_files) == len(expected_ckpt_files) == max_epochs
        assert set(ckpt_files) == set(expected_ckpt_files)


@pytest.mark.parametrize("save_top_k", [-1, 0, 1, 2])
def test_model_checkpoint_to_yaml(tmp_path, save_top_k: int):
    """Test that None in checkpoint callback is valid and that chkp_path is set correctly."""
    model = LogInTwoMethods()

    checkpoint = ModelCheckpoint(dirpath=tmp_path, monitor="early_stop_on", save_top_k=save_top_k)

    trainer = Trainer(default_root_dir=tmp_path, callbacks=[checkpoint], overfit_batches=0.20, max_epochs=2)
    trainer.fit(model)

    path_yaml = tmp_path / "best_k_models.yaml"
    checkpoint.to_yaml(path_yaml)
    with open(path_yaml) as fo:
        d = yaml.full_load(fo)
    best_k = dict(checkpoint.best_k_models.items())
    assert d == best_k


@pytest.mark.parametrize(
    ("logger_version", "expected"), [(None, "version_0"), (1, "version_1"), ("awesome", "awesome")]
)
def test_model_checkpoint_path(tmp_path, logger_version: Union[None, int, str], expected: str):
    """Test that "version_" prefix is only added when logger's version is an integer."""
    model = LogInTwoMethods()
    logger = TensorBoardLogger(tmp_path, version=logger_version)

    trainer = Trainer(default_root_dir=tmp_path, overfit_batches=0.2, max_epochs=2, logger=logger)
    trainer.fit(model)

    ckpt_version = Path(trainer.checkpoint_callback.dirpath).parent.name
    assert ckpt_version == expected


def test_pickling(tmp_path):
    ckpt = ModelCheckpoint(dirpath=tmp_path)

    ckpt_pickled = pickle.dumps(ckpt)
    ckpt_loaded = pickle.loads(ckpt_pickled)
    assert vars(ckpt) == vars(ckpt_loaded)

    ckpt_pickled = cloudpickle.dumps(ckpt)
    ckpt_loaded = cloudpickle.loads(ckpt_pickled)
    assert vars(ckpt) == vars(ckpt_loaded)


class ModelCheckpointTestInvocations(ModelCheckpoint):
    # this class has to be defined outside the test function, otherwise we get pickle error
    # due to the way ddp process is launched

    def __init__(self, expected_count, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expected_count = expected_count
        self.state_dict_count = 0

    def on_train_start(self, trainer, pl_module):
        torch.save = Mock(wraps=torch.save)

    def state_dict(self):
        super().state_dict()
        self.state_dict_count += 1

    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)
        assert self.best_model_path
        assert self.best_model_score
        assert self.state_dict_count == self.expected_count
        if trainer.is_global_zero:
            assert torch.save.call_count == self.expected_count
        else:
            assert torch.save.call_count == 0


@RunIf(skip_windows=True)
def test_model_checkpoint_no_extraneous_invocations(tmp_path):
    """Test to ensure that the model callback saves the checkpoints only once in distributed mode."""
    model = LogInTwoMethods()
    num_epochs = 4
    model_checkpoint = ModelCheckpointTestInvocations(monitor="early_stop_on", expected_count=num_epochs, save_top_k=-1)
    trainer = Trainer(
        strategy="ddp_spawn",
        accelerator="cpu",
        devices=2,
        default_root_dir=tmp_path,
        callbacks=[model_checkpoint],
        max_epochs=num_epochs,
    )
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"


def test_model_checkpoint_format_checkpoint_name(tmp_path, monkeypatch):
    model_checkpoint = ModelCheckpoint(dirpath=tmp_path)

    # empty filename:
    ckpt_name = model_checkpoint._format_checkpoint_name("", {"epoch": 3, "step": 2})
    assert ckpt_name == "epoch=3-step=2"

    ckpt_name = model_checkpoint._format_checkpoint_name(None, {"epoch": 3, "step": 2}, prefix="test")
    assert ckpt_name == "test-epoch=3-step=2"

    # no groups case:
    ckpt_name = model_checkpoint._format_checkpoint_name("ckpt", {}, prefix="test")
    assert ckpt_name == "test-ckpt"

    # no prefix
    ckpt_name = model_checkpoint._format_checkpoint_name("{epoch:03d}-{acc}", {"epoch": 3, "acc": 0.03})
    assert ckpt_name == "epoch=003-acc=0.03"

    # one metric name is substring of another
    ckpt_name = model_checkpoint._format_checkpoint_name("{epoch:03d}-{epoch_test:03d}", {"epoch": 3, "epoch_test": 3})
    assert ckpt_name == "epoch=003-epoch_test=003"

    # prefix
    model_checkpoint.CHECKPOINT_JOIN_CHAR = "@"
    ckpt_name = model_checkpoint._format_checkpoint_name("{epoch},{acc:.5f}", {"epoch": 3, "acc": 0.03}, prefix="test")
    assert ckpt_name == "test@epoch=3,acc=0.03000"
    monkeypatch.undo()

    # non-default char for equals sign
    model_checkpoint.CHECKPOINT_EQUALS_CHAR = ":"
    ckpt_name = model_checkpoint._format_checkpoint_name("{epoch:03d}-{acc}", {"epoch": 3, "acc": 0.03})
    assert ckpt_name == "epoch:003-acc:0.03"
    monkeypatch.undo()

    # no dirpath set
    ckpt_name = ModelCheckpoint(monitor="early_stop_on", dirpath=None).format_checkpoint_name({"epoch": 3, "step": 2})
    assert ckpt_name == "epoch=3-step=2.ckpt"
    ckpt_name = ModelCheckpoint(monitor="early_stop_on", dirpath="").format_checkpoint_name({"epoch": 5, "step": 4})
    assert ckpt_name == "epoch=5-step=4.ckpt"

    # CWD
    ckpt_name = ModelCheckpoint(monitor="early_stop_on", dirpath=".").format_checkpoint_name({"epoch": 3, "step": 4})
    assert ckpt_name == str(Path(".").resolve() / "epoch=3-step=4.ckpt")

    # with version
    ckpt = ModelCheckpoint(monitor="early_stop_on", dirpath=tmp_path, filename="name")
    ckpt_name = ckpt.format_checkpoint_name({}, ver=3)
    assert ckpt_name == str(tmp_path / "name-v3.ckpt")

    # using slashes
    ckpt = ModelCheckpoint(monitor="early_stop_on", dirpath=None, filename="{epoch}_{val/loss:.5f}")
    ckpt_name = ckpt.format_checkpoint_name({"epoch": 4, "val/loss": 0.03})
    assert ckpt_name == "epoch=4_val/loss=0.03000.ckpt"

    # auto_insert_metric_name=False
    ckpt_name = model_checkpoint._format_checkpoint_name(
        "epoch={epoch:03d}-val_acc={val/acc}", {"epoch": 3, "val/acc": 0.03}, auto_insert_metric_name=False
    )
    assert ckpt_name == "epoch=003-val_acc=0.03"

    # dots in the metric name
    ckpt_name = model_checkpoint._format_checkpoint_name(
        "mAP@0.50={val/mAP@0.50:.4f}", {"val/mAP@0.50": 0.2}, auto_insert_metric_name=False
    )
    assert ckpt_name == "mAP@0.50=0.2000"


class ModelCheckpointExtensionTest(ModelCheckpoint):
    FILE_EXTENSION = ".tpkc"


def test_model_checkpoint_file_extension(tmp_path):
    """Test ModelCheckpoint with different file extension."""
    model = LogInTwoMethods()
    model_checkpoint = ModelCheckpointExtensionTest(
        monitor="early_stop_on", dirpath=tmp_path, save_top_k=1, save_last=True
    )
    trainer = Trainer(default_root_dir=tmp_path, callbacks=[model_checkpoint], max_steps=1, logger=False)
    trainer.fit(model)

    expected = ["epoch=0-step=1.tpkc", "last.tpkc"]
    assert set(expected) == set(os.listdir(tmp_path))


@pytest.mark.parametrize("save_last", [True, "link"])
def test_model_checkpoint_save_last(save_last, tmp_path, monkeypatch):
    """Tests that save_last produces only one last checkpoint."""
    seed_everything()
    model = LogInTwoMethods()
    epochs = 3
    monkeypatch.setattr(ModelCheckpoint, "CHECKPOINT_NAME_LAST", "last-{epoch}")
    model_checkpoint = ModelCheckpoint(monitor="early_stop_on", dirpath=tmp_path, save_top_k=-1, save_last=save_last)
    trainer = Trainer(
        default_root_dir=tmp_path,
        callbacks=[model_checkpoint],
        max_epochs=epochs,
        limit_train_batches=10,
        limit_val_batches=10,
        logger=False,
    )
    trainer.fit(model)
    last_filename = model_checkpoint._format_checkpoint_name(
        ModelCheckpoint.CHECKPOINT_NAME_LAST, {"epoch": trainer.current_epoch - 1}
    )
    last_filename = last_filename + ".ckpt"
    assert str(tmp_path / last_filename) == model_checkpoint.last_model_path
    assert set(os.listdir(tmp_path)) == set(
        [f"epoch={i}-step={j}.ckpt" for i, j in zip(range(epochs), [10, 20, 30])] + [last_filename]
    )
    if save_last == "link":
        assert os.path.islink(tmp_path / last_filename)
    else:
        assert os.path.isfile(tmp_path / last_filename)
    assert os.path.realpath(tmp_path / last_filename) == model_checkpoint._last_checkpoint_saved


def test_model_checkpoint_save_last_as_link_not_local(tmp_path):
    callback = ModelCheckpoint(dirpath="memory://not-a-filesystem-path", save_last="link")
    with pytest.raises(ValueError, match="save_last='link'.* is only supported for local file paths"):
        callback.setup(trainer=Trainer(), pl_module=BoringModel(), stage="fit")


def test_model_checkpoint_link_checkpoint(tmp_path):
    """Test that linking a checkpoint works and overwrites an existing link if present."""
    trainer = Mock()

    # link doesn't exist
    file = tmp_path / "file"
    file.touch()
    link = tmp_path / "link"
    ModelCheckpoint._link_checkpoint(trainer, filepath=str(file), linkpath=str(link))
    assert os.path.islink(link)
    assert os.path.realpath(link) == str(file)
    assert not os.path.isabs(os.readlink(link))

    # link exists (is a file)
    new_file1 = tmp_path / "new_file1"
    new_file1.touch()
    ModelCheckpoint._link_checkpoint(trainer, filepath=str(new_file1), linkpath=str(link))
    assert os.path.islink(link)
    assert os.path.realpath(link) == str(new_file1)
    assert not os.path.isabs(os.readlink(link))

    # link exists (is a link)
    new_file2 = tmp_path / "new_file2"
    new_file2.touch()
    ModelCheckpoint._link_checkpoint(trainer, filepath=str(new_file2), linkpath=str(link))
    assert os.path.islink(link)
    assert os.path.realpath(link) == str(new_file2)
    assert not os.path.isabs(os.readlink(link))

    # link exists (is a folder)
    folder = tmp_path / "folder"
    folder.mkdir()
    folder_link = tmp_path / "folder_link"
    folder_link.mkdir()
    ModelCheckpoint._link_checkpoint(trainer, filepath=str(folder), linkpath=str(folder_link))
    assert os.path.islink(folder_link)
    assert os.path.realpath(folder_link) == str(folder)
    assert not os.path.isabs(os.readlink(folder_link))

    # link exists (is a link to a folder)
    new_folder = tmp_path / "new_folder"
    new_folder.mkdir()
    ModelCheckpoint._link_checkpoint(trainer, filepath=str(new_folder), linkpath=str(folder_link))
    assert os.path.islink(folder_link)
    assert os.path.realpath(folder_link) == str(new_folder)
    assert not os.path.isabs(os.readlink(folder_link))

    # simulate permission error on Windows (creation of symbolic links requires privileges)
    file = tmp_path / "win_file"
    file.touch()
    link = tmp_path / "win_link"
    with mock.patch("lightning.pytorch.callbacks.model_checkpoint.os.symlink", Mock(side_effect=OSError)):
        ModelCheckpoint._link_checkpoint(trainer, filepath=str(file), linkpath=str(link))
    assert not os.path.islink(link)
    assert os.path.isfile(link)  # fall back to copying instead of linking


def test_model_checkpoint_link_checkpoint_relative_path(tmp_path, monkeypatch):
    """Test that linking a checkpoint works with relative paths."""
    trainer = Mock()
    monkeypatch.chdir(tmp_path)

    folder = Path("x/z/z")
    folder.mkdir(parents=True)
    file = folder / "file"
    file.touch()
    link = folder / "link"
    ModelCheckpoint._link_checkpoint(trainer, filepath=str(file.absolute()), linkpath=str(link.absolute()))
    assert os.path.islink(link)
    assert Path(os.readlink(link)) == file.relative_to(folder)
    assert not os.path.isabs(os.readlink(link))


def test_invalid_top_k(tmp_path):
    """Make sure that a MisconfigurationException is raised for a negative save_top_k argument."""
    with pytest.raises(MisconfigurationException, match=r".*Must be >= -1"):
        ModelCheckpoint(dirpath=tmp_path, save_top_k=-3)


def test_none_monitor_top_k(tmp_path):
    """Test that a warning appears for positive top_k with monitor=None."""
    with pytest.raises(
        MisconfigurationException, match=r"ModelCheckpoint\(save_top_k=3, monitor=None\) is not a valid*"
    ):
        ModelCheckpoint(dirpath=tmp_path, save_top_k=3)
    # These should not fail
    ModelCheckpoint(dirpath=tmp_path, save_top_k=-1)
    ModelCheckpoint(dirpath=tmp_path, save_top_k=0)
    ModelCheckpoint(dirpath=tmp_path, save_top_k=1)


def test_none_monitor_not_alternating(tmp_path):
    """Regression test for the case where the callback saved alternating `model.ckpt` and `model-v1.ckpt` files."""

    class ListDirModel(BoringModel):
        def on_train_epoch_start(self):
            if self.current_epoch > 0:
                assert os.listdir(tmp_path) == ["model.ckpt"]

    model = ListDirModel()
    model_checkpoint = ModelCheckpoint(dirpath=tmp_path, monitor=None, save_top_k=1, filename="model")
    trainer = Trainer(
        callbacks=model_checkpoint,
        limit_train_batches=1,
        limit_val_batches=0,
        max_epochs=3,
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=False,
    )
    trainer.fit(model)


def test_invalid_every_n_epochs(tmp_path):
    """Make sure that a MisconfigurationException is raised for a negative every_n_epochs argument."""
    with pytest.raises(MisconfigurationException, match=r".*Must be >= 0"):
        ModelCheckpoint(dirpath=tmp_path, every_n_epochs=-3)
    # These should not fail
    ModelCheckpoint(dirpath=tmp_path, every_n_epochs=0)
    ModelCheckpoint(dirpath=tmp_path, every_n_epochs=1)
    ModelCheckpoint(dirpath=tmp_path, every_n_epochs=2)


def test_invalid_every_n_train_steps(tmp_path):
    """Make sure that a MisconfigurationException is raised for a negative every_n_epochs argument."""
    with pytest.raises(MisconfigurationException, match=r".*Must be >= 0"):
        ModelCheckpoint(dirpath=tmp_path, every_n_train_steps=-3)
    # These should not fail
    ModelCheckpoint(dirpath=tmp_path, every_n_train_steps=0)
    ModelCheckpoint(dirpath=tmp_path, every_n_train_steps=1)
    ModelCheckpoint(dirpath=tmp_path, every_n_epochs=2)


def test_invalid_trigger_combination(tmp_path):
    """Test that a MisconfigurationException is raised if more than one of every_n_epochs, every_n_train_steps, and
    train_time_interval are enabled together."""
    with pytest.raises(MisconfigurationException, match=r".*Combination of parameters every_n_train_steps"):
        ModelCheckpoint(dirpath=tmp_path, every_n_train_steps=1, every_n_epochs=2)
    with pytest.raises(MisconfigurationException, match=r".*Combination of parameters every_n_train_steps"):
        ModelCheckpoint(train_time_interval=timedelta(minutes=1), every_n_epochs=2)
    with pytest.raises(MisconfigurationException, match=r".*Combination of parameters every_n_train_steps"):
        ModelCheckpoint(train_time_interval=timedelta(minutes=1), every_n_train_steps=2)

    # These should not fail
    ModelCheckpoint(dirpath=tmp_path, every_n_train_steps=0, every_n_epochs=3)
    ModelCheckpoint(dirpath=tmp_path, every_n_train_steps=4, every_n_epochs=0)
    ModelCheckpoint(dirpath=tmp_path, every_n_train_steps=0, every_n_epochs=0, train_time_interval=timedelta(minutes=1))


def test_none_every_n_train_steps_val_epochs(tmp_path):
    checkpoint_callback = ModelCheckpoint(dirpath=tmp_path)
    assert checkpoint_callback.every_n_epochs == 1
    assert checkpoint_callback._every_n_train_steps == 0


def test_model_checkpoint_save_last_none_monitor(tmp_path, caplog):
    """Test that it is possible to save all checkpoints when monitor=None."""
    seed_everything()
    model = LogInTwoMethods()

    epochs = 2
    checkpoint_callback = ModelCheckpoint(monitor=None, dirpath=tmp_path, save_top_k=-1, save_last=True)
    trainer = Trainer(
        default_root_dir=tmp_path,
        callbacks=[checkpoint_callback],
        limit_train_batches=10,
        limit_val_batches=10,
        max_epochs=epochs,
        logger=False,
    )
    trainer.fit(model)

    # these should not be set if monitor is None
    assert checkpoint_callback.monitor is None
    assert checkpoint_callback.best_model_path == str(tmp_path / "epoch=1-step=20.ckpt")
    assert checkpoint_callback.last_model_path == str(tmp_path / "last.ckpt")
    assert checkpoint_callback.best_model_score is None
    assert checkpoint_callback.best_model_metrics is None
    assert checkpoint_callback.best_k_models == {}
    assert checkpoint_callback.kth_best_model_path == ""

    # check that the correct ckpts were created
    expected = [f"epoch={i}-step={j}.ckpt" for i, j in zip(range(epochs), [10, 20])]
    expected.append("last.ckpt")
    assert set(os.listdir(tmp_path)) == set(expected)
    assert os.path.isfile(tmp_path / "last.ckpt")


@pytest.mark.parametrize("every_n_epochs", list(range(4)))
def test_model_checkpoint_every_n_epochs(tmp_path, every_n_epochs):
    model = LogInTwoMethods()
    epochs = 5
    checkpoint_callback = ModelCheckpoint(
        dirpath=tmp_path, filename="{epoch}", save_top_k=-1, every_n_epochs=every_n_epochs
    )
    trainer = Trainer(
        default_root_dir=tmp_path,
        callbacks=[checkpoint_callback],
        max_epochs=epochs,
        limit_train_batches=1,
        limit_val_batches=1,
        logger=False,
    )
    trainer.fit(model)

    # check that the correct ckpts were created, the modulo condition is checked in `ModelCheckpoint`
    expected = [f"epoch={e}.ckpt" for e in range(epochs) if not (e + 1) % every_n_epochs] if every_n_epochs > 0 else []
    assert set(os.listdir(tmp_path)) == set(expected)


def test_ckpt_every_n_train_steps(tmp_path):
    """Tests that the checkpoints are saved every n training steps."""
    model = LogInTwoMethods()
    every_n_train_steps = 16
    max_epochs = 2
    epoch_length = 64
    checkpoint_callback = ModelCheckpoint(
        filename="{step}",
        every_n_epochs=0,
        every_n_train_steps=every_n_train_steps,
        dirpath=tmp_path,
        save_top_k=-1,
        save_last=False,
    )
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=2,
        enable_progress_bar=False,
        callbacks=[checkpoint_callback],
        logger=False,
    )

    trainer.fit(model)
    expected = [
        f"step={i}.ckpt" for i in range(every_n_train_steps, max_epochs * epoch_length + 1, every_n_train_steps)
    ]
    assert set(os.listdir(tmp_path)) == set(expected)


@mock.patch("lightning.pytorch.callbacks.model_checkpoint.time")
def test_model_checkpoint_train_time_interval(mock_datetime, tmp_path) -> None:
    """Tests that the checkpoints are saved at the specified time interval."""
    seconds_per_batch = 7
    start_time = time.monotonic()
    batches_per_epoch = 64
    num_epochs = 2
    max_batches = batches_per_epoch * num_epochs + 1
    mock_datetime.monotonic.side_effect = [start_time + seconds_per_batch * i for i in range(max_batches)]

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        min_epochs=num_epochs,
        max_epochs=num_epochs,
        enable_progress_bar=False,
        callbacks=[
            ModelCheckpoint(
                filename="{epoch}-{step}",
                dirpath=tmp_path,
                train_time_interval=timedelta(minutes=1),
                save_top_k=-1,
                save_last=False,
            )
        ],
        logger=False,
    )

    trainer.fit(model)
    # Each batch takes 7 sec and we checkpoint every minute. There are 64
    # batches per epoch, so total time to run is 7*64*2 = 896 sec < 14.96 minutes,
    # so we should have 14 checkpoints.
    assert len(os.listdir(tmp_path)) == 14


def test_model_checkpoint_topk_zero(tmp_path):
    """Test that no checkpoints are saved when save_top_k=0."""
    model = LogInTwoMethods()
    checkpoint_callback = ModelCheckpoint(dirpath=tmp_path, save_top_k=0, save_last=True)
    trainer = Trainer(default_root_dir=tmp_path, callbacks=[checkpoint_callback], max_epochs=2, logger=False)
    trainer.fit(model)
    # these should not be set if monitor is None
    assert checkpoint_callback.monitor is None
    assert checkpoint_callback.best_model_path == ""
    assert checkpoint_callback.best_model_score is None
    assert checkpoint_callback.best_model_metrics is None
    assert checkpoint_callback.best_k_models == {}
    assert checkpoint_callback.kth_best_model_path == ""
    # check that only the last ckpt was created
    assert os.listdir(tmp_path) == ["last.ckpt"]
    assert checkpoint_callback.last_model_path == str(tmp_path / "last.ckpt")
    # 'last.ckpt' is not a symlink because there are no top-k checkpoints to link
    assert not os.path.islink(checkpoint_callback.last_model_path)


def test_model_checkpoint_topk_all(tmp_path):
    """Test that save_top_k=-1 tracks the best models when monitor key is provided."""
    seed_everything(1000)
    epochs = 3

    model = BoringModel()
    checkpoint_callback = ModelCheckpoint(
        dirpath=tmp_path, filename="{epoch}", monitor="epoch", mode="max", save_top_k=-1
    )
    trainer = Trainer(
        default_root_dir=tmp_path,
        callbacks=[checkpoint_callback],
        max_epochs=epochs,
        logger=False,
        val_check_interval=1.0,
    )
    trainer.fit(model)

    assert checkpoint_callback.monitor == "epoch"
    assert checkpoint_callback.best_model_path == str(tmp_path / "epoch=2.ckpt")
    assert checkpoint_callback.best_model_score == epochs - 1
    assert len(os.listdir(tmp_path)) == len(checkpoint_callback.best_k_models) == epochs
    assert set(checkpoint_callback.best_k_models.keys()) == {str(tmp_path / f"epoch={i}.ckpt") for i in range(epochs)}
    assert checkpoint_callback.kth_best_model_path == str(tmp_path / "epoch=0.ckpt")


def test_ckpt_metric_names(tmp_path):
    model = LogInTwoMethods()

    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        gradient_clip_val=1.0,
        overfit_batches=0.20,
        enable_progress_bar=False,
        limit_train_batches=0.01,
        limit_val_batches=0.01,
        callbacks=[ModelCheckpoint(monitor="early_stop_on", dirpath=tmp_path, filename="{val_loss:.2f}")],
    )

    trainer.fit(model)

    # make sure the checkpoint we saved has the metric in the name
    ckpts = os.listdir(tmp_path)
    ckpts = [x for x in ckpts if "val_loss" in x]
    assert len(ckpts) == 1
    val = re.sub("[^0-9.]", "", ckpts[0])
    assert len(val) > 3


def test_default_checkpoint_behavior(tmp_path):
    seed_everything(1234)

    model = LogInTwoMethods()
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=3,
        enable_progress_bar=False,
        limit_train_batches=5,
        limit_val_batches=5,
        logger=False,
    )

    with patch.object(trainer, "save_checkpoint", wraps=trainer.save_checkpoint) as save_mock:
        trainer.fit(model)
        results = trainer.test()

    assert len(results) == 1
    save_dir = tmp_path / "checkpoints"
    save_weights_only = trainer.checkpoint_callback.save_weights_only
    save_mock.assert_has_calls([
        call(str(save_dir / "epoch=0-step=5.ckpt"), save_weights_only),
        call(str(save_dir / "epoch=1-step=10.ckpt"), save_weights_only),
        call(str(save_dir / "epoch=2-step=15.ckpt"), save_weights_only),
    ])
    ckpts = os.listdir(save_dir)
    assert len(ckpts) == 1
    assert ckpts[0] == "epoch=2-step=15.ckpt"


def test_model_checkpoint_save_last_checkpoint_contents(tmp_path):
    """Tests that the save_last checkpoint contains the latest information."""
    seed_everything(100)
    model = LogInTwoMethods()
    num_epochs = 3
    model_checkpoint = ModelCheckpoint(
        monitor="early_stop_on", dirpath=tmp_path, filename="{epoch}", save_top_k=num_epochs, save_last=True
    )
    trainer = Trainer(
        default_root_dir=tmp_path,
        callbacks=[model_checkpoint],
        max_epochs=num_epochs,
        limit_train_batches=2,
        limit_val_batches=2,
    )
    trainer.fit(model)

    path_last_epoch = str(tmp_path / f"epoch={num_epochs - 1}.ckpt")
    path_last = str(tmp_path / "last.ckpt")
    assert path_last == model_checkpoint.last_model_path
    assert os.path.isfile(path_last_epoch)
    assert os.path.isfile(path_last)

    ckpt_last_epoch = torch.load(path_last_epoch, weights_only=True)
    ckpt_last = torch.load(path_last, weights_only=True)

    assert ckpt_last_epoch["epoch"] == ckpt_last["epoch"]
    assert ckpt_last_epoch["global_step"] == ckpt_last["global_step"]

    ckpt_id = (
        "ModelCheckpoint{'monitor': 'early_stop_on', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1,"
        " 'train_time_interval': None}"
    )
    assert ckpt_last["callbacks"][ckpt_id] == ckpt_last_epoch["callbacks"][ckpt_id]

    # it is easier to load the model objects than to iterate over the raw dict of tensors
    model_last_epoch = LogInTwoMethods.load_from_checkpoint(path_last_epoch)
    model_last = LogInTwoMethods.load_from_checkpoint(model_checkpoint.last_model_path)
    for w0, w1 in zip(model_last_epoch.parameters(), model_last.parameters()):
        assert w0.eq(w1).all()


@pytest.mark.parametrize("mode", ["min", "max"])
def test_checkpointing_with_nan_as_first(tmp_path, mode):
    monitor = [float("nan")]
    monitor += [5, 7, 8] if mode == "max" else [8, 7, 5]

    class CurrentModel(LogInTwoMethods):
        def on_validation_epoch_end(self):
            val_loss = monitor[self.current_epoch]
            self.log("abc", val_loss)

    model = CurrentModel()

    callback = ModelCheckpoint(monitor="abc", mode=mode, save_top_k=1, dirpath=tmp_path)

    trainer = Trainer(
        callbacks=[callback],
        default_root_dir=tmp_path,
        val_check_interval=1.0,
        max_epochs=len(monitor),
    )
    trainer.save_checkpoint = Mock()

    trainer.fit(model)

    # check that last one is also the best one
    assert trainer.save_checkpoint.call_count == len(monitor)
    assert mode == "min" and callback.best_model_score == 5 or mode == "max" and callback.best_model_score == 8


def test_checkpoint_repeated_strategy(tmp_path):
    """This test validates checkpoint can be called several times without increasing internally its global step if
    nothing run."""
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", dirpath=tmp_path, filename="{epoch:02d}")

    class ExtendedBoringModel(BoringModel):
        def validation_step(self, batch, batch_idx):
            loss = self.step(batch)
            self.log("val_loss", loss)

    model = ExtendedBoringModel()
    trainer_kwargs = {
        "max_epochs": 1,
        "limit_train_batches": 2,
        "limit_val_batches": 2,
        "limit_test_batches": 2,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "log_every_n_steps": 1,
        "default_root_dir": tmp_path,
        "logger": CSVLogger(tmp_path),
    }
    trainer = Trainer(**trainer_kwargs, callbacks=[checkpoint_callback])
    trainer.fit(model)
    assert set(os.listdir(tmp_path)) == {"epoch=00.ckpt", "lightning_logs"}

    for idx in range(4):
        # load from checkpoint
        trainer = Trainer(**trainer_kwargs)
        trainer.fit(model, ckpt_path=checkpoint_callback.best_model_path)
        trainer.test(ckpt_path=checkpoint_callback.best_model_path, verbose=False)

        assert set(os.listdir(tmp_path)) == {"epoch=00.ckpt", "lightning_logs"}

    # no new versions created after the initial fit, because the ones that resume from ckpt do not log anything
    assert set(os.listdir(tmp_path / "lightning_logs")) == {"version_0"}


def test_checkpoint_repeated_strategy_extended(tmp_path):
    """This test validates checkpoint can be called several times without increasing internally its global step if
    nothing run."""

    class ExtendedBoringModel(BoringModel):
        def validation_step(self, batch, batch_idx):
            loss = self.step(batch)
            self.log("val_loss", loss)
            return {"val_loss": loss}

    def assert_trainer_init(trainer):
        assert trainer.global_step == 0
        assert trainer.current_epoch == 0

    def get_last_checkpoint(ckpt_dir):
        last = sorted(ckpt_dir.iterdir())[-1]
        return str(last)

    def assert_checkpoint_content(ckpt_dir):
        chk = pl_load(get_last_checkpoint(ckpt_dir))
        # `-1` because this checkpoint is saved `on_train_epoch_end` which is considered part of the epoch so the
        # `current_epoch` count has not been increased yet
        assert chk["epoch"] == epochs - 1
        assert chk["global_step"] == 4

    def assert_checkpoint_log_dir(idx):
        lightning_logs = tmp_path / "lightning_logs"
        actual = [d.name for d in sorted(lightning_logs.iterdir())]
        assert actual == [f"version_{i}" for i in range(idx + 1)]
        actual = [d.name for d in sorted(ckpt_dir.iterdir())]
        assert len(actual) == epochs, actual

    ckpt_dir = tmp_path / "checkpoints"
    checkpoint_cb = ModelCheckpoint(dirpath=ckpt_dir, save_top_k=-1)
    epochs = 2
    limit_train_batches = 2
    trainer_config = {
        "default_root_dir": tmp_path,
        "max_epochs": epochs,
        "limit_train_batches": limit_train_batches,
        "limit_val_batches": 3,
        "limit_test_batches": 4,
        "callbacks": [checkpoint_cb],
        "logger": TensorBoardLogger(tmp_path),
    }
    trainer = Trainer(**trainer_config)
    assert_trainer_init(trainer)

    model = ExtendedBoringModel()
    trainer.fit(model)
    assert trainer.global_step == epochs * limit_train_batches
    assert trainer.current_epoch == epochs
    assert_checkpoint_log_dir(0)
    assert_checkpoint_content(ckpt_dir)

    trainer.validate(model)
    assert trainer.current_epoch == epochs

    trainer.test(model)
    assert trainer.current_epoch == epochs

    for idx in range(1, 5):
        chk = get_last_checkpoint(ckpt_dir)
        assert_checkpoint_content(ckpt_dir)

        # load from checkpoint
        trainer_config["logger"] = TensorBoardLogger(tmp_path)
        trainer = Trainer(**trainer_config)
        assert_trainer_init(trainer)

        model = ExtendedBoringModel()

        trainer.test(model)
        assert_trainer_init(trainer)

        trainer.fit(model, ckpt_path=chk)
        assert trainer.global_step == epochs * limit_train_batches
        assert trainer.current_epoch == epochs
        assert trainer.fit_loop.epoch_progress.current.processed == epochs

        trainer.validate(model)
        assert trainer.global_step == epochs * limit_train_batches
        assert trainer.current_epoch == epochs
        assert trainer.fit_loop.epoch_progress.current.processed == epochs

        trainer.fit(model)
        assert trainer.global_step == epochs * limit_train_batches
        assert trainer.current_epoch == epochs
        assert trainer.fit_loop.epoch_progress.current.processed == epochs
        assert_checkpoint_log_dir(idx)


def test_configure_model_checkpoint(tmp_path):
    """Test all valid and invalid ways a checkpoint callback can be passed to the Trainer."""
    kwargs = {"default_root_dir": tmp_path}
    callback1 = ModelCheckpoint(monitor="foo")
    callback2 = ModelCheckpoint(monitor="bar")

    # no callbacks
    trainer = Trainer(enable_checkpointing=False, callbacks=[], **kwargs)
    assert not any(isinstance(c, ModelCheckpoint) for c in trainer.callbacks)
    assert trainer.checkpoint_callback is None

    # default configuration
    trainer = Trainer(callbacks=[], **kwargs)
    assert sum(1 for c in trainer.callbacks if isinstance(c, ModelCheckpoint)) == 1
    assert isinstance(trainer.checkpoint_callback, ModelCheckpoint)

    # custom callback passed to callbacks list, enable_checkpointing=True is ignored
    trainer = Trainer(enable_checkpointing=True, callbacks=[callback1], **kwargs)
    assert [c for c in trainer.callbacks if isinstance(c, ModelCheckpoint)] == [callback1]
    assert trainer.checkpoint_callback == callback1

    # multiple checkpoint callbacks
    trainer = Trainer(callbacks=[callback1, callback2], **kwargs)
    assert trainer.checkpoint_callback == callback1
    assert trainer.checkpoint_callbacks == [callback1, callback2]

    with pytest.raises(MisconfigurationException, match="`enable_checkpointing=False` but found `ModelCheckpoint`"):
        Trainer(enable_checkpointing=False, callbacks=[callback1], **kwargs)


def test_val_check_interval_checkpoint_files(tmp_path):
    """Test correct checkpoint naming when validating/checkpointing multiple times per epoch."""
    model = LogInTwoMethods()
    model_checkpoint = ModelCheckpoint(dirpath=tmp_path, save_top_k=-1, monitor="val_acc", mode="max")
    trainer = Trainer(
        default_root_dir=tmp_path,
        val_check_interval=0.2,
        max_epochs=1,
        limit_train_batches=10,
        callbacks=[model_checkpoint],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model)
    files = {p.name for p in tmp_path.iterdir()}
    assert files == {f"epoch=0-step={s}.ckpt" for s in [2, 4, 6, 8, 10]}


def test_current_score(tmp_path):
    """Check that the current_score value is correct and was saved."""

    class TestModel(BoringModel):
        def training_step(self, *args):
            self.log("foo", (self.current_epoch + 1) / 10)
            return super().training_step(*args)

    model_checkpoint = ModelCheckpoint(dirpath=tmp_path, save_top_k=3, monitor="foo", mode="min")
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=3,
        limit_train_batches=1,
        limit_val_batches=1,
        callbacks=[model_checkpoint],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(TestModel())
    assert model_checkpoint.current_score == 0.3
    ckpts = [torch.load(ckpt, weights_only=True) for ckpt in tmp_path.iterdir()]
    ckpts = [
        ckpt["callbacks"][
            "ModelCheckpoint{'monitor': 'foo', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1,"
            " 'train_time_interval': None}"
        ]
        for ckpt in ckpts
    ]
    assert sorted(ckpt["current_score"] for ckpt in ckpts) == [0.1, 0.2, 0.3]


@pytest.mark.parametrize("mode", ["min", "max"])
def test_current_score_when_nan(tmp_path, mode: str):
    """Check that ModelCheckpoint handles NaN values correctly."""

    class TestModel(BoringModel):
        def training_step(self, *args):
            self.log("foo", float("nan"))
            return super().training_step(*args)

    model_checkpoint = ModelCheckpoint(dirpath=tmp_path, save_top_k=1, monitor="foo", mode=mode)
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        callbacks=[model_checkpoint],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(TestModel())
    expected = float("inf" if mode == "min" else "-inf")
    assert model_checkpoint.best_model_score == expected
    assert model_checkpoint.current_score == expected


@pytest.mark.parametrize("use_omegaconf", [False, pytest.param(True, marks=RunIf(omegaconf=True))])
def test_hparams_type(tmp_path, use_omegaconf):
    class TestModel(BoringModel):
        def __init__(self, hparams):
            super().__init__()
            self.save_hyperparameters(hparams)

    model_checkpoint = ModelCheckpoint(dirpath=tmp_path, save_top_k=1)
    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmp_path,
        limit_train_batches=1,
        limit_val_batches=1,
        callbacks=[model_checkpoint],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    hp = {"test_hp_0": 1, "test_hp_1": 2}
    hp = OmegaConf.create(hp) if use_omegaconf else Namespace(**hp)
    model = TestModel(hp)
    trainer.fit(model)
    ckpt = trainer._checkpoint_connector.dump_checkpoint()
    if use_omegaconf:
        assert isinstance(ckpt[model.CHECKPOINT_HYPER_PARAMS_KEY], Container)
    else:
        # make sure it's not AttributeDict
        ckpt_params_type = type(ckpt[model.CHECKPOINT_HYPER_PARAMS_KEY])
        assert ckpt_params_type is dict


def test_ckpt_version_after_rerun_new_trainer(tmp_path):
    """Check that previous checkpoints are renamed to have the correct version suffix when new trainer instances are
    used."""
    epochs = 2
    for i in range(epochs):
        mc = ModelCheckpoint(dirpath=tmp_path, save_top_k=-1, monitor="epoch", filename="{epoch}")
        trainer = Trainer(
            max_epochs=epochs,
            limit_train_batches=1,
            limit_val_batches=1,
            default_root_dir=tmp_path,
            callbacks=[mc],
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(BoringModel())

        # check best_k_models state
        expected = {"epoch=0-v1.ckpt", "epoch=1-v1.ckpt"} if i else {"epoch=0.ckpt", "epoch=1.ckpt"}
        assert {Path(f).name for f in mc.best_k_models} == expected

    # check created ckpts
    actual = {f.name for f in tmp_path.iterdir()}
    assert actual == {"epoch=0.ckpt", "epoch=1.ckpt", "epoch=0-v1.ckpt", "epoch=1-v1.ckpt"}


def test_ckpt_version_after_rerun_same_trainer(tmp_path):
    """Check that previous checkpoints are renamed to have the correct version suffix when the same trainer instance is
    used."""
    mc = ModelCheckpoint(dirpath=tmp_path, save_top_k=-1, monitor="epoch", filename="test")
    mc.STARTING_VERSION = 9
    trainer = Trainer(
        max_epochs=2,
        limit_train_batches=1,
        limit_val_batches=1,
        default_root_dir=tmp_path,
        callbacks=[mc],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(BoringModel())
    trainer.fit_loop.max_epochs = 4
    trainer.fit(BoringModel())

    ckpt_range = range(mc.STARTING_VERSION, trainer.max_epochs + mc.STARTING_VERSION - 1)
    expected = {"test.ckpt", *(f"test-v{i}.ckpt" for i in ckpt_range)}
    # check best_k_models state
    assert {Path(f).name for f in mc.best_k_models} == expected
    # check created ckpts
    assert set(os.listdir(tmp_path)) == expected


def test_ckpt_version_counter_disabled_after_rerun_new_trainer(tmp_path):
    """Check that previous checkpoints get overwritten and no suffixes are generated when new trainer instances are
    used."""
    epochs = 2
    for i in range(epochs):
        mc = ModelCheckpoint(
            dirpath=tmp_path,
            save_top_k=-1,
            save_last=True,
            monitor="epoch",
            filename="{epoch}",
            enable_version_counter=False,
        )
        trainer = Trainer(
            max_epochs=epochs,
            limit_train_batches=1,
            limit_val_batches=1,
            default_root_dir=tmp_path,
            callbacks=[mc],
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(BoringModel())

        # check best_k_models and last state
        assert {Path(f).name for f in mc.best_k_models} == {"epoch=0.ckpt", "epoch=1.ckpt"}
        assert Path(mc.last_model_path).name == "last.ckpt"

    # check created ckpts
    actual = {f.name for f in tmp_path.iterdir()}
    assert actual == {"epoch=0.ckpt", "epoch=1.ckpt", "last.ckpt"}


def test_model_checkpoint_mode_options():
    with pytest.raises(MisconfigurationException, match="`mode` can be .* but got unknown_option"):
        ModelCheckpoint(mode="unknown_option")


def test_check_val_every_n_epochs_top_k_integration(tmp_path):
    model = BoringModel()
    mc = ModelCheckpoint(dirpath=tmp_path, monitor="epoch", save_top_k=-1, filename="{epoch}")
    trainer = Trainer(
        default_root_dir=tmp_path,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        max_epochs=5,
        check_val_every_n_epoch=2,
        callbacks=mc,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(model)
    assert set(os.listdir(tmp_path)) == {"epoch=1.ckpt", "epoch=3.ckpt"}


def test_model_checkpoint_saveload_ckpt(tmp_path):
    def make_assertions(cb_restore, written_ckpt):
        expected_keys = {
            "dirpath": False,
            "best_model_score": False,
            "kth_best_model_path": False,
            "kth_value": False,
            "best_k_models": False,
            "last_model_path": False,
            "best_model_path": True,
        }
        for key, should_match in expected_keys.items():
            if should_match:
                assert getattr(cb_restore, key) == written_ckpt[key]
            else:
                assert getattr(cb_restore, key) != written_ckpt[key]

    class CustomModelCheckpoint(ModelCheckpoint):
        def on_load_checkpoint(self, *args, **kwargs):
            assert self.dirpath is not None
            return super().on_load_checkpoint(*args, **kwargs)

    ckpt = {
        "best_model_path": "epoch=10-step=1436.ckpt",
        "best_model_score": torch.tensor(2.246),
        "best_k_models": {"epoch=10-step=1436.ckpt": torch.tensor(2.246)},
        "kth_best_model_path": "epoch=10-step=1436.ckpt",
        "kth_value": torch.tensor(2.246),
        "last_model_path": "last2245.ckpt",
    }

    # test state_dict
    cb_write = ModelCheckpoint(dirpath=tmp_path, save_top_k=-1, save_last=True)
    for key, val in ckpt.items():
        setattr(cb_write, key, val)
    written_ckpt = cb_write.state_dict()
    for state in ckpt:
        assert ckpt[state] == written_ckpt[state]

    # Case - 1
    # test load_state_dict
    # Notes:
    # 1. "current_score", "dirpath" and "monitor" are currently not restored by load_state_dict.
    #    We therefore set "dirpath" and "monitor" to something different than for ckpt/cb_write so we can assert them.
    # 2. "current_score" is left as initialized, i.e. None, and can therefore also be asserted
    # 3. When a different `dirpath` is passed to `ModelCheckpoint` to resume training, only
    #    `best_model_path` and `last_model_path` are reloaded (reloading for others is stopped).
    cb_restore = ModelCheckpoint(dirpath=(tmp_path / "restore"), monitor=None, save_top_k=-1, save_last=True)
    with pytest.warns(UserWarning, match="The dirpath has changed from*"):
        cb_restore.load_state_dict(written_ckpt)
    make_assertions(cb_restore, written_ckpt)

    # Case - 2
    # Make sure that everything runs when dirpath is not initialized explicitly
    cb_restore = CustomModelCheckpoint()
    cb_restore.setup(Trainer(), BoringModel(), stage="fit")
    with pytest.warns(UserWarning, match="The dirpath has changed from*"):
        cb_restore.load_state_dict(written_ckpt)
    make_assertions(cb_restore, written_ckpt)


def test_resume_training_preserves_old_ckpt_last(tmp_path):
    """Ensures that the last saved checkpoint is not deleted from the previous folder when training is resumed from the
    old checkpoint."""
    model = BoringModel()
    trainer_kwargs = {
        "default_root_dir": tmp_path,
        "max_epochs": 1,
        "limit_train_batches": 3,
        "limit_val_batches": 0,
        "enable_model_summary": False,
        "logger": False,
    }
    mc_kwargs = {
        "filename": "{step}",
        "monitor": "step",
        "mode": "max",
        "save_last": True,
        "save_top_k": 2,
        "every_n_train_steps": 1,
    }
    trainer = Trainer(**trainer_kwargs, callbacks=ModelCheckpoint(**mc_kwargs))
    trainer.fit(model)
    # Make sure that the last checkpoint file exists in the dirpath passed (`tmp_path`)
    assert set(os.listdir(tmp_path / "checkpoints")) == {"last.ckpt", "step=2.ckpt", "step=3.ckpt"}

    # Training it for 2 epochs for extra surety, that nothing gets deleted after multiple epochs
    trainer_kwargs["max_epochs"] += 1
    mc_kwargs["dirpath"] = tmp_path / "new"
    trainer = Trainer(**trainer_kwargs, callbacks=ModelCheckpoint(**mc_kwargs))
    trainer.fit(model, ckpt_path=(tmp_path / "checkpoints" / "step=2.ckpt"))
    # Ensure that the file is not deleted from the old folder
    assert os.path.isfile(tmp_path / "checkpoints" / "last.ckpt")


def test_save_last_saves_correct_last_model_path(tmp_path):
    mc = ModelCheckpoint(dirpath=tmp_path, save_last=True)
    mc.CHECKPOINT_NAME_LAST = "{foo}-last"
    trainer = Trainer(callbacks=mc)
    trainer.strategy.connect(BoringModel())

    mc._save_last_checkpoint(trainer, {"foo": torch.tensor(1)})
    expected = "foo=1-last.ckpt"
    assert os.listdir(tmp_path) == [expected]
    full_path = tmp_path / expected
    ckpt = torch.load(full_path, weights_only=True)
    assert ckpt["callbacks"][mc.state_key]["last_model_path"] == str(full_path)


def test_save_last_versioning(tmp_path):
    model = BoringModel()
    for _ in range(2):
        mc = ModelCheckpoint(dirpath=tmp_path, save_top_k=0, save_last=True)
        trainer = Trainer(
            max_epochs=2,
            callbacks=mc,
            limit_train_batches=1,
            limit_val_batches=0,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )
        trainer.fit(model)
    assert {"last.ckpt", "last-v1.ckpt"} == set(os.listdir(tmp_path))
    # 'last.ckpt' is not a symlink since `save_top_k=0` didn't save any other checkpoints to link to
    assert all(not os.path.islink(tmp_path / path) for path in set(os.listdir(tmp_path)))


def test_none_monitor_saves_correct_best_model_path(tmp_path):
    mc = ModelCheckpoint(dirpath=tmp_path, monitor=None)
    trainer = Trainer(callbacks=mc)
    trainer.strategy.connect(BoringModel())

    mc._save_none_monitor_checkpoint(trainer, {})
    expected = "epoch=0-step=0.ckpt"
    assert os.listdir(tmp_path) == [expected]
    full_path = str(tmp_path / expected)
    ckpt = torch.load(full_path, weights_only=True)
    assert ckpt["callbacks"][mc.state_key]["best_model_path"] == full_path


def test_last_global_step_saved():
    # this should not save anything
    model_checkpoint = ModelCheckpoint(save_top_k=0, save_last=False, monitor="foo")
    trainer = Mock()
    monitor_candidates = {"foo": torch.tensor(123)}
    model_checkpoint._save_topk_checkpoint(trainer, monitor_candidates)
    model_checkpoint._save_last_checkpoint(trainer, monitor_candidates)
    assert model_checkpoint._last_global_step_saved == 0


@pytest.mark.parametrize("every_n_epochs", [0, 5])
def test_save_last_every_n_epochs_interaction(tmp_path, every_n_epochs):
    """Test that `save_last` ignores `every_n_epochs`."""
    mc = ModelCheckpoint(every_n_epochs=every_n_epochs, save_last=True, save_top_k=0, save_on_train_epoch_end=True)
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=2,
        callbacks=mc,
        limit_train_batches=1,
        limit_val_batches=0,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    model = BoringModel()
    with patch.object(trainer, "save_checkpoint") as save_mock:
        trainer.fit(model)
    assert mc.last_model_path  # a "last" ckpt was saved
    assert save_mock.call_count == trainer.max_epochs


def test_train_epoch_end_ckpt_with_no_validation():
    trainer = Trainer(val_check_interval=0.5)
    trainer.fit_loop.epoch_loop.val_loop._max_batches = [0]
    assert trainer.checkpoint_callback._should_save_on_train_epoch_end(trainer)
    trainer.fit_loop.epoch_loop.val_loop._max_batches = [1]
    assert not trainer.checkpoint_callback._should_save_on_train_epoch_end(trainer)
    trainer.val_check_interval = 0.8
    assert not trainer.checkpoint_callback._should_save_on_train_epoch_end(trainer)


@pytest.mark.parametrize("same_resume_folder", [True, False])
def test_resume_and_old_checkpoint_files_remain(same_resume_folder, tmp_path):
    """Test that checkpoints saved in the resume-folder won't be deleted under the save-top-k mechanism."""
    model = BoringModel()
    trainer_kwargs = {
        "default_root_dir": tmp_path,
        "limit_train_batches": 10,
        "limit_val_batches": 0,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "logger": False,
    }
    first = tmp_path / "first"
    second = tmp_path / "second"
    new_dirpath = first if same_resume_folder else second

    # Generate checkpoints in the first folder
    callback = ModelCheckpoint(dirpath=first, monitor="step", mode="max", save_top_k=2, every_n_train_steps=2)
    trainer = Trainer(callbacks=callback, max_steps=5, **trainer_kwargs)
    trainer.fit(model)
    assert set(os.listdir(first)) == {"epoch=0-step=2.ckpt", "epoch=0-step=4.ckpt"}

    # Continue training from checkpoint
    callback = ModelCheckpoint(dirpath=new_dirpath, monitor="step", mode="max", save_top_k=2, every_n_train_steps=2)
    trainer = Trainer(callbacks=callback, max_steps=8, **trainer_kwargs)
    trainer.fit(model, ckpt_path=str(first / "epoch=0-step=4.ckpt"))
    if same_resume_folder:
        assert set(os.listdir(first)) == {
            "epoch=0-step=4.ckpt",  # do not delete checkpoint from which we resume from
            "epoch=0-step=6.ckpt",
            "epoch=0-step=8.ckpt",
        }
    else:
        assert set(os.listdir(first)) == {"epoch=0-step=2.ckpt", "epoch=0-step=4.ckpt"}  # no files deleted
        assert set(os.listdir(second)) == {"epoch=0-step=6.ckpt", "epoch=0-step=8.ckpt"}


@pytest.mark.parametrize(
    ("name", "extension", "folder_contents", "expected"),
    [
        ("last", ".ckpt", {}, {}),
        ("any", ".any", {}, {}),
        ("last", ".ckpt", {"last"}, {}),
        ("any", ".any", {"last"}, {}),
        ("last", ".ckpt", {"last", "last.ckpt"}, {"last.ckpt"}),
        ("other", ".pt", {"last", "last.pt", "other.pt"}, {"other.pt"}),
        ("last", ".ckpt", {"log.txt", "last-v0.ckpt", "last-v1.ckpt"}, {"last-v0.ckpt", "last-v1.ckpt"}),
        ("other", ".pt", {"log.txt", "last-v0.ckpt", "other-v0.pt", "other-v1.pt"}, {"other-v0.pt", "other-v1.pt"}),
    ],
)
def test_find_last_checkpoints(name, extension, folder_contents, expected, tmp_path):
    for file in folder_contents:
        (tmp_path / file).touch()

    trainer = Trainer()
    callback = ModelCheckpoint(dirpath=tmp_path)
    callback.CHECKPOINT_NAME_LAST = name
    callback.FILE_EXTENSION = extension
    files = callback._find_last_checkpoints(trainer)
    assert files == {str(tmp_path / p) for p in expected}


def test_expand_home():
    """Test that the dirpath gets expanded if it contains `~`."""
    home_root = Path.home()

    checkpoint = ModelCheckpoint(dirpath="~/checkpoints")
    assert checkpoint.dirpath == str(home_root / "checkpoints")
    checkpoint = ModelCheckpoint(dirpath=Path("~/checkpoints"))
    assert checkpoint.dirpath == str(home_root / "checkpoints")

    # it is possible to have a folder with the name `~`
    checkpoint = ModelCheckpoint(dirpath="./~/checkpoints")
    assert checkpoint.dirpath == str(Path.cwd() / "~" / "checkpoints")


@pytest.mark.parametrize(
    ("val", "expected"),
    [
        ("yes", True),
        ("True", True),
        ("true", True),
        ("no", False),
        ("false", False),
        ("False", False),
        ("link", "link"),
    ],
)
def test_save_last_cli(val, expected):
    """Test that the CLI can parse the `save_last` argument correctly (composed type)."""
    annot = signature(ModelCheckpoint).parameters["save_last"].annotation
    parser = ArgumentParser()
    parser.add_argument("--a", type=annot)
    args = parser.parse_args(["--a", val])
    assert args.a == expected
