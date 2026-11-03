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
import gc
import logging
import math
import os
import pickle
from argparse import Namespace
from contextlib import nullcontext, suppress
from copy import deepcopy
from pathlib import Path
from unittest import mock
from unittest.mock import ANY, Mock, call, patch

import cloudpickle
import pytest
import torch
import torch.nn as nn
from torch.multiprocessing import ProcessRaisedException
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim import SGD
from torch.utils.data import DataLoader, IterableDataset

import tests_pytorch.helpers.utils as tutils
from lightning.fabric.utilities.cloud_io import _load as pl_load
from lightning.fabric.utilities.imports import _IS_WINDOWS
from lightning.fabric.utilities.seed import seed_everything
from lightning.pytorch import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.accelerators import CPUAccelerator, CUDAAccelerator
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Timer
from lightning.pytorch.callbacks.on_exception_checkpoint import OnExceptionCheckpoint
from lightning.pytorch.callbacks.prediction_writer import BasePredictionWriter
from lightning.pytorch.core.saving import load_hparams_from_tags_csv, load_hparams_from_yaml, save_hparams_to_tags_csv
from lightning.pytorch.demos.boring_classes import (
    BoringDataModule,
    BoringModel,
    RandomDataset,
    RandomIterableDataset,
    RandomIterableDatasetWithLen,
)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.overrides.distributed import UnrepeatedDistributedSampler, _IndexBatchSamplerWrapper
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy
from lightning.pytorch.strategies.launchers import _MultiProcessingLauncher, _SubprocessScriptLauncher
from lightning.pytorch.trainer.states import RunningStage, TrainerFn
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.imports import _OMEGACONF_AVAILABLE, _TORCH_GREATER_EQUAL_2_8
from tests_pytorch.conftest import mock_cuda_count, mock_mps_count
from tests_pytorch.helpers.datamodules import ClassifDataModule
from tests_pytorch.helpers.runif import RunIf
from tests_pytorch.helpers.simple_models import ClassificationModel

if _OMEGACONF_AVAILABLE:
    from omegaconf import OmegaConf


def test_enable_device_summary(caplog):
    """Test that the device summary can be disabled via the `enable_device_summary` flag."""
    with caplog.at_level(logging.INFO):
        trainer = Trainer(enable_device_summary=False)
    assert trainer.enable_device_summary is False
    assert "GPU available" not in caplog.text
    assert "TPU available" not in caplog.text

    caplog.clear()
    with caplog.at_level(logging.INFO):
        trainer = Trainer(enable_device_summary=True)
    assert trainer.enable_device_summary is True
    assert "GPU available" in caplog.text
    assert "TPU available" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.INFO):
        trainer = Trainer()
    assert trainer.enable_device_summary is True
    assert "GPU available" in caplog.text


def test_trainer_error_when_input_not_lightning_module():
    """Test that a useful error gets raised when the Trainer methods receive something other than a LightningModule."""
    trainer = Trainer()

    for method in ("fit", "validate", "test", "predict"):
        with pytest.raises(TypeError, match="must be a `LightningModule`.*got `Linear"):
            run_method = getattr(trainer, method)
            run_method(nn.Linear(2, 2))


@pytest.mark.parametrize("url_ckpt", [True, False])
def test_no_val_module(monkeypatch, tmp_path, tmpdir_server, url_ckpt):
    """Tests use case where trainer saves the model, and user loads it from tags independently."""
    # set $TORCH_HOME, which determines torch hub's cache path, to tmp_path
    monkeypatch.setenv("TORCH_HOME", str(tmp_path))

    class CustomModel(BoringModel):
        def __init__(self, lr=1e-2):
            super().__init__()
            self.save_hyperparameters()

    lr = 1e-3
    model = CustomModel(lr=lr)

    # logger file to get meta
    logger = tutils.get_default_logger(tmp_path)

    trainer = Trainer(default_root_dir=tmp_path, max_steps=1, limit_val_batches=1, logger=logger)
    # fit model
    trainer.fit(model)
    # training complete
    assert trainer.state.finished, f"Training failed with {trainer.state}"

    # save model
    new_weights_path = os.path.join(tmp_path, "save_test.ckpt")
    trainer.save_checkpoint(new_weights_path)

    # assert ckpt has hparams
    ckpt = torch.load(new_weights_path, weights_only=True)
    assert LightningModule.CHECKPOINT_HYPER_PARAMS_KEY in ckpt, "hyper_parameters missing from checkpoints"

    # load new model
    hparams_path = tutils.get_data_path(logger, path_dir=tmp_path)
    hparams_path = os.path.join(hparams_path, "hparams.yaml")
    ckpt_path = (
        f"http://{tmpdir_server[0]}:{tmpdir_server[1]}/{os.path.basename(new_weights_path)}"
        if url_ckpt
        else new_weights_path
    )
    model_2 = CustomModel.load_from_checkpoint(checkpoint_path=ckpt_path, hparams_file=hparams_path)
    assert model_2.hparams.lr == lr


@pytest.mark.parametrize("url_ckpt", [True, False])
def test_strict_model_load(monkeypatch, tmp_path, tmpdir_server, url_ckpt):
    """Tests use case where trainer saves the model, and user loads it from tags independently."""
    # set $TORCH_HOME, which determines torch hub's cache path, to tmp_path
    monkeypatch.setenv("TORCH_HOME", tmp_path)

    model = BoringModel()
    # Extra layer
    model.c_d3 = torch.nn.Linear(10, 12)

    # logger file to get meta
    logger = tutils.get_default_logger(tmp_path)

    # fit model
    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=1, logger=logger)
    trainer.fit(model)

    # training complete
    assert trainer.state.finished, f"Training failed with {trainer.state}"

    # save model
    new_weights_path = os.path.join(tmp_path, "save_test.ckpt")
    trainer.save_checkpoint(new_weights_path)

    # load new model
    hparams_path = tutils.get_data_path(logger, path_dir=tmp_path)
    hparams_path = os.path.join(hparams_path, "hparams.yaml")
    ckpt_path = (
        f"http://{tmpdir_server[0]}:{tmpdir_server[1]}/{os.path.basename(new_weights_path)}"
        if url_ckpt
        else new_weights_path
    )

    try:
        BoringModel.load_from_checkpoint(checkpoint_path=ckpt_path, hparams_file=hparams_path)
    # todo: specify the possible exception
    except Exception:
        failed = True
    else:
        failed = False

    assert failed, "Model should not been loaded since the extra layer added."

    failed = False
    try:
        BoringModel.load_from_checkpoint(checkpoint_path=ckpt_path, hparams_file=hparams_path, strict=False)
    # todo: specify the possible exception
    except Exception:
        failed = True

    assert not failed, "Model should be loaded due to strict=False."


@pytest.mark.parametrize(
    ("accumulate_grad_batches", "limit_train_batches"),
    [
        (3, 1.0),
        (3, 0.8),  # not to be divisible by accumulate_grad_batches on purpose
        (4, 1.0),
        (4, 0.7),  # not to be divisible by accumulate_grad_batches on purpose
    ],
)
def test_gradient_accumulation_scheduling_last_batch(tmp_path, accumulate_grad_batches, limit_train_batches):
    """Verify optimizer.step() applied to last batch while grad accumulation."""

    class TestModel(BoringModel):
        def state_dict(self, *args, **kwargs):
            return deepcopy(super().state_dict(*args, **kwargs))

        def check(self, d1, d2, equal=True):
            keys = d1.keys() | d2.keys()
            values = [torch.equal(d1[k], d2[k]) for k in keys]
            return all(values) if equal else not any(values)

        def backward(self, *args, **kwargs) -> None:
            pre_bwd_state_dict = self.state_dict()
            assert self.check(self.start_state_dict, pre_bwd_state_dict)

            out = super().backward(*args, **kwargs)

            # state dict is equal, just the gradients changed
            assert self.check(pre_bwd_state_dict, self.state_dict())

            return out

        def optimizer_step(self, *args, **kwargs):
            pre_opt_step_state_dict = self.state_dict()
            assert self.check(self.start_state_dict, pre_opt_step_state_dict)

            # this calls `backward` and `on_after_backward` inside the closure
            out = super().optimizer_step(*args, **kwargs)

            # the state dict changed
            assert self.check(pre_opt_step_state_dict, self.state_dict(), equal=False)

            self.opt_step_called = True
            return out

        def on_train_batch_start(self, *_):
            self.start_state_dict = self.state_dict()
            self.opt_step_called = False

        def on_train_batch_end(self, outputs, batch, batch_idx):
            end_state_dict = self.state_dict()
            is_last_batch = (batch_idx + 1) == self.trainer.num_training_batches

            if is_last_batch or self.opt_step_called:
                assert self.check(self.start_state_dict, end_state_dict, equal=False)
            else:
                assert self.check(self.start_state_dict, end_state_dict)

    model = TestModel()
    trainer = Trainer(
        accumulate_grad_batches=accumulate_grad_batches,
        max_epochs=2,
        limit_train_batches=limit_train_batches,
        limit_val_batches=0,
        default_root_dir=tmp_path,
        enable_progress_bar=False,
    )

    trainer.fit(model)


def test_loading_meta_tags(tmp_path):
    """Test for backward compatibility to meta_tags.csv."""
    hparams = {
        "batch_size": 32,
        "learning_rate": 0.001 * 8,
        "optimizer_name": "adam",
    }

    # save tags
    logger = tutils.get_default_logger(tmp_path)
    logger.log_hyperparams(Namespace(some_str="a_str", an_int=1, a_float=2.0))
    logger.log_hyperparams(hparams)
    logger.save()

    # load hparams
    path_expt_dir = tutils.get_data_path(logger, path_dir=tmp_path)
    hparams_path = os.path.join(path_expt_dir, TensorBoardLogger.NAME_HPARAMS_FILE)
    hparams = load_hparams_from_yaml(hparams_path)

    # save as legacy meta_tags.csv
    tags_path = os.path.join(path_expt_dir, "meta_tags.csv")
    save_hparams_to_tags_csv(tags_path, hparams)

    tags = load_hparams_from_tags_csv(tags_path)

    assert hparams == tags


def test_loading_yaml(tmp_path):
    hparams = {
        "batch_size": 32,
        "learning_rate": 0.001 * 8,
        "optimizer_name": "adam",
    }

    # save tags
    logger = tutils.get_default_logger(tmp_path)
    logger.log_hyperparams(Namespace(some_str="a_str", an_int=1, a_float=2.0))
    logger.log_hyperparams(hparams)
    logger.save()

    # load hparams
    path_expt_dir = tutils.get_data_path(logger, path_dir=tmp_path)
    hparams_path = os.path.join(path_expt_dir, "hparams.yaml")
    tags = load_hparams_from_yaml(hparams_path)

    assert tags["batch_size"] == 32
    assert tags["optimizer_name"] == "adam"


@pytest.mark.parametrize(
    ("save_top_k", "save_last", "expected_files"),
    [
        pytest.param(-1, False, [f"epoch={i}.ckpt" for i in range(5)], id="CASE K=-1  (all)"),
        pytest.param(1, False, {"epoch=4.ckpt"}, id="CASE K=1 (2.5, epoch 4)"),
        pytest.param(2, False, [f"epoch={i}.ckpt" for i in (2, 4)], id="CASE K=2 (2.5 epoch 4, 2.8 epoch 2)"),
        pytest.param(4, False, [f"epoch={i}.ckpt" for i in range(1, 5)], id="CASE K=4 (save all 4 base)"),
        pytest.param(3, False, [f"epoch={i}.ckpt" for i in range(2, 5)], id="CASE K=3 (save the 2nd, 3rd, 4th model)"),
        pytest.param(1, True, {"epoch=4.ckpt", "last.ckpt"}, id="CASE K=1 (save the 4th model and the last model)"),
    ],
)
def test_model_checkpoint_options(tmp_path, save_top_k, save_last, expected_files):
    """Test ModelCheckpoint options."""

    def mock_save_function(filepath, *args):
        open(filepath, "a").close()

    # simulated losses
    losses = [10, 9, 2.8, 5, 2.5]

    checkpoint_callback = ModelCheckpoint(
        dirpath=tmp_path,
        filename="{epoch}",
        monitor="checkpoint_on",
        save_top_k=save_top_k,
        save_last=save_last,
        verbose=True,
        save_on_train_epoch_end=False,
    )
    trainer = Trainer()
    trainer.state.fn = TrainerFn.FITTING
    trainer.save_checkpoint = mock_save_function

    # emulate callback's calls during the training
    for i, loss in enumerate(losses, 1):
        # sets `trainer.global_step`
        trainer.fit_loop.epoch_loop.automatic_optimization.optim_progress.optimizer.step.total.completed = i
        trainer.callback_metrics.update({"checkpoint_on": torch.tensor(loss)})
        checkpoint_callback.on_validation_end(trainer, trainer.lightning_module)
        trainer.fit_loop.epoch_progress.current.completed = i  # sets `trainer.current_epoch`

    file_lists = set(os.listdir(tmp_path))

    assert len(file_lists) == len(expected_files), (
        f"Should save {len(expected_files)} models when save_top_k={save_top_k} but found={file_lists}"
    )

    # verify correct naming
    for fname in expected_files:
        assert fname in file_lists


def test_model_checkpoint_only_weights(tmp_path):
    """Tests use case where ModelCheckpoint is configured to save only model weights, and user tries to load checkpoint
    to resume training."""
    model = BoringModel()

    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        callbacks=[ModelCheckpoint(dirpath=tmp_path, save_weights_only=True)],
    )
    # fit model
    trainer.fit(model)
    # training complete
    assert trainer.state.finished, f"Training failed with {trainer.state}"

    checkpoint_path = trainer.checkpoint_callback.best_model_path

    # assert saved checkpoint has no trainer data
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert "optimizer_states" not in checkpoint, "checkpoint should contain only model weights"
    assert "lr_schedulers" not in checkpoint, "checkpoint should contain only model weights"

    # assert loading model works when checkpoint has only weights
    assert BoringModel.load_from_checkpoint(checkpoint_path=checkpoint_path)

    # directly save model
    new_weights_path = os.path.join(tmp_path, "save_test.ckpt")
    trainer.save_checkpoint(new_weights_path, weights_only=True)
    # assert saved checkpoint has no trainer data
    checkpoint = torch.load(new_weights_path, weights_only=True)
    assert "optimizer_states" not in checkpoint, "checkpoint should contain only model weights"
    assert "lr_schedulers" not in checkpoint, "checkpoint should contain only model weights"

    # assert restoring train state fails
    with pytest.raises(KeyError, match="checkpoint contains only the model"):
        trainer._checkpoint_connector.restore(new_weights_path)


def test_model_freeze_unfreeze():
    model = BoringModel()
    freeze_ret = model.freeze()
    assert freeze_ret is model
    assert not model.training
    for param in model.parameters():
        assert not param.requires_grad

    unfreeze_ret = model.unfreeze()
    assert unfreeze_ret is model
    assert model.training
    for param in model.parameters():
        assert param.requires_grad


# TODO: move to `tests/tests_pytorch/models/test_restore.py`
@pytest.mark.parametrize("url_ckpt", [True, False])
def test_fit_ckpt_path_epoch_restored(monkeypatch, tmp_path, tmpdir_server, url_ckpt):
    """Verify resuming from checkpoint runs the right number of epochs."""
    # set $TORCH_HOME, which determines torch hub's cache path, to tmp_path
    monkeypatch.setenv("TORCH_HOME", tmp_path)

    class TestModel(BoringModel):
        # Model that tracks epochs and batches seen
        num_epochs_end_seen = 0
        num_batches_seen = 0
        num_on_load_checkpoint_called = 0

        def on_train_epoch_end(self):
            self.num_epochs_end_seen += 1

        def on_train_batch_start(self, *_):
            self.num_batches_seen += 1

        def on_load_checkpoint(self, _):
            self.num_on_load_checkpoint_called += 1

    model = TestModel()
    max_epochs = 2
    trainer = Trainer(
        max_epochs=max_epochs,
        limit_train_batches=0.65,
        limit_val_batches=1,
        callbacks=ModelCheckpoint(dirpath=tmp_path, save_top_k=-1),
        default_root_dir=tmp_path,
        val_check_interval=1.0,
        enable_progress_bar=False,
        logger=False,
        enable_model_summary=False,
    )
    trainer.fit(model)

    assert model.num_epochs_end_seen == max_epochs
    assert model.num_batches_seen == trainer.num_training_batches * max_epochs == trainer.global_step
    assert model.num_on_load_checkpoint_called == 0

    checkpoints = set(Path(trainer.checkpoint_callback.dirpath).glob("*.ckpt"))
    if url_ckpt:
        # transform local paths into url checkpoints
        ip, port = tmpdir_server
        checkpoints = [f"http://{ip}:{port}/" + ckpt.name for ckpt in checkpoints]

    assert len(checkpoints) == max_epochs
    for ckpt in checkpoints:
        model = TestModel()
        state = pl_load(ckpt)
        # Resume training
        trainer = Trainer(default_root_dir=tmp_path, max_epochs=2, enable_progress_bar=False)
        trainer.fit(model, ckpt_path=ckpt)
        assert state["global_step"] + model.num_batches_seen == trainer.global_step
        assert model.num_on_load_checkpoint_called == 1


def test_trainer_max_steps_and_epochs(tmp_path):
    """Verify model trains according to specified max steps."""
    model = BoringModel()
    num_train_samples = math.floor(len(model.train_dataloader()) * 0.5)

    # define less train steps than epochs
    trainer_kwargs = {
        "limit_train_batches": 0.5,
        "default_root_dir": tmp_path,
        "max_epochs": 3,
        "max_steps": num_train_samples + 10,
        "logger": False,
        "enable_model_summary": False,
        "enable_progress_bar": False,
    }
    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model)

    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.global_step == trainer.max_steps, "Model did not stop at max_steps"

    # define less train epochs than steps
    trainer_kwargs["max_epochs"] = 2
    trainer_kwargs["max_steps"] = 3 * 2 * num_train_samples
    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model)

    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.global_step == num_train_samples * trainer.max_epochs
    assert trainer.current_epoch == trainer.max_epochs, "Model did not stop at max_epochs"

    # if max_steps is positive and max_epochs is negative, use max_steps
    trainer_kwargs["max_epochs"] = -1
    trainer_kwargs["max_steps"] = 3
    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model)

    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.global_step == 3


@pytest.mark.parametrize(
    ("max_epochs", "max_steps", "incorrect_variable"),
    [
        (-100, -1, "max_epochs"),
        (1, -2, "max_steps"),
    ],
)
def test_trainer_max_steps_and_epochs_validation(max_epochs, max_steps, incorrect_variable):
    """Don't allow max_epochs or max_steps to be less than -1 or a float."""
    with pytest.raises(
        MisconfigurationException,
        match=f"`{incorrect_variable}` must be a non-negative integer or -1",
    ):
        Trainer(max_epochs=max_epochs, max_steps=max_steps)


@pytest.mark.parametrize(
    ("max_epochs", "max_steps", "is_done", "correct_trainer_epochs"),
    [
        (None, -1, False, None),
        (-1, -1, False, -1),
        (5, -1, False, 5),
        (-1, 10, False, -1),
        (None, 0, True, None),
        (0, -1, True, 0),
        (-1, 0, True, -1),
    ],
)
def test_trainer_max_steps_and_epochs_fit_loop_done(max_epochs, max_steps, is_done, correct_trainer_epochs):
    trainer = Trainer(max_epochs=max_epochs, max_steps=max_steps)

    assert trainer.max_epochs == correct_trainer_epochs
    assert trainer.max_steps == max_steps

    if isinstance(correct_trainer_epochs, int):
        assert trainer.fit_loop.done is is_done

    # Make sure there is no timer
    timer_callbacks = [c for c in trainer.callbacks if isinstance(c, Timer)]
    assert len(timer_callbacks) == 0


def test_trainer_min_steps_and_epochs(tmp_path):
    """Verify model trains according to specified min steps."""
    num_train_samples = math.floor(len(BoringModel().train_dataloader()) * 0.5)

    class CustomModel(BoringModel):
        def training_step(self, *args, **kwargs):
            # try to force stop right after first step
            if self.global_step > 0:
                self.trainer.should_step = True

            return super().training_step(*args, **kwargs)

    model = CustomModel()

    trainer_kwargs = {
        "limit_train_batches": 0.5,
        "default_root_dir": tmp_path,
        "val_check_interval": 2,
        "min_epochs": 1,
        "max_epochs": 7,
        # define less min steps than 1 epoch
        "min_steps": num_train_samples // 2,
        "logger": False,
        "enable_model_summary": False,
        "enable_progress_bar": False,
    }
    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model)

    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.current_epoch > 0
    assert trainer.global_step >= num_train_samples, "Model did not train for at least min_epochs"

    # define less epochs than min_steps
    trainer_kwargs["min_steps"] = math.floor(num_train_samples * 1.5)
    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model)

    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.current_epoch > 0
    assert trainer.global_step >= math.floor(num_train_samples * 1.5), "Model did not train for at least min_steps"


def test_trainer_min_steps_and_min_epochs_not_reached(tmp_path, caplog):
    """Test that min_epochs/min_steps in Trainer are enforced even if EarlyStopping is triggered."""

    class TestModel(BoringModel):
        training_step_invoked = 0

        def training_step(self, batch, batch_idx):
            output = super().training_step(batch, batch_idx)
            output["loss"] = output["loss"] * 0.0  # force minimal loss to trigger early stopping
            self.log("loss", output["loss"])
            self.training_step_invoked += 1
            if self.current_epoch < 2:
                assert not self.trainer.should_stop
            else:
                assert self.trainer.should_stop
            return output

    model = TestModel()
    early_stop = EarlyStopping(monitor="loss", patience=0, check_on_train_epoch_end=True)
    min_epochs = 5
    trainer = Trainer(
        default_root_dir=tmp_path,
        enable_progress_bar=False,
        min_epochs=min_epochs,
        limit_val_batches=0,
        limit_train_batches=2,
        callbacks=[early_stop],
    )
    with caplog.at_level(logging.INFO, logger="lightning.pytorch.trainer.trainer"):
        trainer.fit(model)

    message = f"min_epochs={min_epochs}` or `min_steps=None` has not been met. Training will continue"
    num_messages = sum(1 for record in caplog.records if message in record.message)
    assert num_messages == 1
    assert model.training_step_invoked == min_epochs * 2


def test_trainer_max_steps_accumulate_batches(tmp_path):
    """Verify model trains according to specified max steps with grad accumulated batches."""
    model = BoringModel()
    num_train_samples = math.floor(len(model.train_dataloader()) * 0.5)

    # define less train steps than epochs
    trainer = Trainer(
        limit_train_batches=0.5,
        default_root_dir=tmp_path,
        max_steps=num_train_samples + 10,
        accumulate_grad_batches=10,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model)

    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.global_step == trainer.max_steps, "Model did not stop at max_steps"


@pytest.mark.parametrize("ckpt_path", [None, "last"])
@pytest.mark.parametrize("fn", [TrainerFn.FITTING, TrainerFn.VALIDATING])
def test_checkpoint_path_input_last_fault_tolerant(tmp_path, ckpt_path, fn):
    mc = ModelCheckpoint()
    mc.best_model_path = "foobar"
    # manually create to simulate fault-tolerant training
    ft_ckpt = OnExceptionCheckpoint(tmp_path)
    Path(ft_ckpt.ckpt_path).touch()

    trainer = Trainer(callbacks=[mc, ft_ckpt])
    trainer.state.fn = fn

    if ckpt_path == "last":
        ctxt = nullcontext()
        final_path = os.path.join(tmp_path, "on_exception.ckpt")
    elif fn == "fit":  # and ckpt_path == best
        ctxt = pytest.warns(UserWarning, match="The last model of the previous `fit")
        final_path = os.path.join(tmp_path, "on_exception.ckpt")
    else:  # ckpt_path == best and fn == validate
        ctxt = pytest.warns(UserWarning, match="There is also an on-exception checkpoint available")
        final_path = "foobar"

    with ctxt:
        ckpt_path = trainer._checkpoint_connector._parse_ckpt_path(
            fn, ckpt_path, model_provided=fn == "fit", model_connected=True
        )
    assert ckpt_path == final_path


@pytest.mark.parametrize("ckpt_path", [None, "last"])
@pytest.mark.parametrize("save_last", [True, False])
@pytest.mark.parametrize("fn", ["fit", "validate"])
def test_checkpoint_path_input_last(tmp_path, ckpt_path, save_last, fn):
    model = BoringModel()
    mc = ModelCheckpoint(save_last=save_last)
    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=False,
        default_root_dir=tmp_path,
        callbacks=[mc],
    )
    assert trainer.ckpt_path is None
    trainer_fn = getattr(trainer, fn)

    if fn == "fit":
        ctxt = nullcontext() if ckpt_path is None else pytest.warns(UserWarning, match="No checkpoint will be loaded")
        with ctxt:
            trainer_fn(model, ckpt_path=ckpt_path)
        assert trainer.ckpt_path is None
    else:
        trainer.fit(model)
        if ckpt_path is None:
            ctxt = pytest.warns(
                UserWarning,
                match=r"(?!.*however it is default only when fitting)^"
                r".*The best model of the previous `fit` call will be used",
            )
            final_path = mc.best_model_path
        else:
            if save_last:
                ctxt = nullcontext()
                final_path = mc.last_model_path
            else:
                ctxt = pytest.warns(UserWarning, match="No checkpoint will be loaded")
                final_path = None

        with ctxt:
            trainer_fn(ckpt_path=ckpt_path)
        assert trainer.ckpt_path == final_path


def test_checkpoint_find_last(tmp_path):
    """Test that the last checkpoint is found correctly."""
    model = BoringModel()
    mc = ModelCheckpoint(save_last=True)
    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=0,
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=False,
        default_root_dir=tmp_path,
        callbacks=[mc],
    )
    assert trainer.ckpt_path is None
    trainer.fit(model)

    model = BoringModel()
    mc = ModelCheckpoint()
    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=0,
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=False,
        default_root_dir=tmp_path,
        callbacks=[mc],
    )
    assert trainer.ckpt_path is None
    trainer.fit(model, ckpt_path="last")
    assert trainer.ckpt_path == str(tmp_path / "checkpoints" / "last.ckpt")


@pytest.mark.parametrize("ckpt_path", [None, "best", "specific"])
@pytest.mark.parametrize("save_top_k", [-1, 0, 1, 2])
@pytest.mark.parametrize("fn", ["validate", "test", "predict"])
def test_checkpoint_path_input(tmp_path, ckpt_path, save_top_k, fn):
    class TestModel(BoringModel):
        def validation_step(self, batch, batch_idx):
            self.log("foo", -batch_idx)
            return super().validation_step(batch, batch_idx)

        def test_step(self, *args):
            return self.validation_step(*args)

        def predict_step(self, batch, *_):
            return self(batch)

    model = TestModel()
    trainer = Trainer(
        max_epochs=2,
        limit_val_batches=1,
        limit_test_batches=1,
        limit_predict_batches=1,
        enable_progress_bar=False,
        default_root_dir=tmp_path,
        callbacks=[ModelCheckpoint(monitor="foo", save_top_k=save_top_k)],
    )
    trainer.fit(model)

    trainer_fn = getattr(trainer, fn)
    assert trainer.ckpt_path is None

    if ckpt_path == "best":
        # ckpt_path is 'best', meaning we load the best weights
        if save_top_k == 0:
            with pytest.raises(ValueError, match=".*is not configured to save the best.*"):
                trainer_fn(ckpt_path=ckpt_path)
            with pytest.raises(ValueError, match=".*is not configured to save the best.*"):
                trainer_fn(model, ckpt_path=ckpt_path)
        else:
            trainer_fn(ckpt_path=ckpt_path)
            assert trainer.ckpt_path == trainer.checkpoint_callback.best_model_path

            trainer_fn(model, ckpt_path=ckpt_path)
            assert trainer.ckpt_path == trainer.checkpoint_callback.best_model_path
    elif ckpt_path is None:
        # ckpt_path is None, meaning we don't load any checkpoints and use the provided model
        trainer_fn(model, ckpt_path=ckpt_path)
        assert trainer.ckpt_path is None

        if save_top_k > 0:
            # ckpt_path is None with no model provided means load the best weights
            with pytest.warns(UserWarning, match="The best model of the previous `fit` call will be used"):
                trainer_fn(ckpt_path=ckpt_path)
            assert trainer.ckpt_path == trainer.checkpoint_callback.best_model_path
    else:
        # specific checkpoint, pick one from saved ones
        if save_top_k == 0:
            with pytest.raises(FileNotFoundError):
                trainer_fn(ckpt_path="random.ckpt")
        else:
            ckpt_path = str(
                list((Path(tmp_path) / f"lightning_logs/version_{trainer.logger.version}/checkpoints").iterdir())[
                    0
                ].absolute()
            )
            trainer_fn(ckpt_path=ckpt_path)
            assert trainer.ckpt_path == ckpt_path

            trainer_fn(model, ckpt_path=ckpt_path)
            assert trainer.ckpt_path == ckpt_path


@pytest.mark.parametrize("enable_checkpointing", [False, True])
@pytest.mark.parametrize("fn", ["validate", "test", "predict"])
def test_tested_checkpoint_path_best(tmp_path, enable_checkpointing, fn):
    class TestModel(BoringModel):
        def validation_step(self, batch, batch_idx):
            self.log("foo", -batch_idx)
            return super().validation_step(batch, batch_idx)

        def test_step(self, *args):
            return self.validation_step(*args)

        def predict_step(self, batch, *_):
            return self(batch)

    model = TestModel()
    trainer = Trainer(
        max_epochs=2,
        limit_val_batches=1,
        limit_test_batches=1,
        limit_predict_batches=1,
        enable_progress_bar=False,
        default_root_dir=tmp_path,
        enable_checkpointing=enable_checkpointing,
    )
    trainer.fit(model)

    trainer_fn = getattr(trainer, fn)
    assert trainer.ckpt_path is None

    if enable_checkpointing:
        trainer_fn(ckpt_path="best")
        assert trainer.ckpt_path == trainer.checkpoint_callback.best_model_path

        trainer_fn(model, ckpt_path="best")
        assert trainer.ckpt_path == trainer.checkpoint_callback.best_model_path
    else:
        with pytest.raises(ValueError, match="`ModelCheckpoint` is not configured."):
            trainer_fn(ckpt_path="best")
        with pytest.raises(ValueError, match="`ModelCheckpoint` is not configured."):
            trainer_fn(model, ckpt_path="best")


def test_best_ckpt_evaluate_raises_warning_with_multiple_ckpt_callbacks():
    """Test that a warning is raised if best ckpt callback is used for evaluation configured with multiple
    checkpoints."""

    ckpt_callback1 = ModelCheckpoint(monitor="foo")
    ckpt_callback1.best_model_path = "foo_best_model.ckpt"
    ckpt_callback2 = ModelCheckpoint(monitor="bar")
    ckpt_callback2.best_model_path = "bar_best_model.ckpt"
    trainer = Trainer(callbacks=[ckpt_callback1, ckpt_callback2])
    trainer.state.fn = TrainerFn.TESTING

    with pytest.warns(UserWarning, match="best checkpoint path from first checkpoint callback"):
        trainer._checkpoint_connector._parse_ckpt_path(
            trainer.state.fn, ckpt_path="best", model_provided=False, model_connected=True
        )


def test_disabled_training(tmp_path):
    """Verify that `limit_train_batches=0` disables the training loop unless `fast_dev_run=True`."""

    class CurrentModel(BoringModel):
        training_step_invoked = False

        def training_step(self, *args, **kwargs):
            self.training_step_invoked = True
            return super().training_step(*args, **kwargs)

    model = CurrentModel()

    trainer_options = {
        "default_root_dir": tmp_path,
        "enable_progress_bar": False,
        "max_epochs": 2,
        "limit_train_batches": 0.0,
        "limit_val_batches": 0.2,
        "fast_dev_run": False,
    }

    before_state_dict = deepcopy(model.state_dict())

    trainer = Trainer(**trainer_options)
    trainer.fit(model)

    after_state_dict = model.state_dict()

    for key in before_state_dict:
        assert torch.all(torch.eq(before_state_dict[key], after_state_dict[key]))

    # check that limit_train_batches=0 turns off training
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.current_epoch == 0
    assert not model.training_step_invoked, "`training_step` should not run when `limit_train_batches=0`"

    # check that limit_train_batches has no influence when fast_dev_run is turned on
    model = CurrentModel()
    trainer_options.update(fast_dev_run=True)
    before_state_dict = deepcopy(model.state_dict())

    trainer = Trainer(**trainer_options)
    trainer.fit(model)

    after_state_dict = model.state_dict()

    for key in before_state_dict:
        assert not torch.all(torch.eq(before_state_dict[key], after_state_dict[key]))

    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.current_epoch == 1
    assert model.training_step_invoked, "did not run `training_step` with `fast_dev_run=True`"


def test_disabled_validation(tmp_path):
    """Verify that `limit_val_batches=0` disables the validation loop unless `fast_dev_run=True`."""

    class CurrentModel(BoringModel):
        validation_step_invoked = False

        def validation_step(self, *args, **kwargs):
            self.validation_step_invoked = True
            return super().validation_step(*args, **kwargs)

    model = CurrentModel()

    trainer_options = {
        "default_root_dir": tmp_path,
        "enable_progress_bar": False,
        "max_epochs": 2,
        "limit_train_batches": 0.4,
        "limit_val_batches": 0.0,
        "fast_dev_run": False,
    }

    trainer = Trainer(**trainer_options)
    trainer.fit(model)

    # check that limit_val_batches=0 turns off validation
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.current_epoch == 2
    assert not model.validation_step_invoked, "`validation_step` should not run when `limit_val_batches=0`"

    # check that limit_val_batches has no influence when fast_dev_run is turned on
    model = CurrentModel()
    trainer_options.update(fast_dev_run=True)
    trainer = Trainer(**trainer_options)
    trainer.fit(model)

    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.current_epoch == 1
    assert model.validation_step_invoked, "did not run `validation_step` with `fast_dev_run=True`"


def test_on_exception_hook(tmp_path):
    """Test the on_exception callback hook and the trainer interrupted flag."""
    model = BoringModel()

    class InterruptCallback(Callback):
        def __init__(self):
            super().__init__()

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            raise KeyboardInterrupt

        def on_test_start(self, trainer, pl_module):
            raise MisconfigurationException

    class HandleInterruptCallback(Callback):
        def __init__(self):
            super().__init__()
            self.exception = None

        def on_exception(self, trainer, pl_module, exception):
            self.exception = exception

    interrupt_callback = InterruptCallback()
    handle_interrupt_callback = HandleInterruptCallback()

    trainer = Trainer(
        callbacks=[interrupt_callback, handle_interrupt_callback],
        max_epochs=1,
        limit_val_batches=0.1,
        limit_train_batches=0.2,
        enable_progress_bar=False,
        logger=False,
        default_root_dir=tmp_path,
    )
    assert not trainer.interrupted
    assert handle_interrupt_callback.exception is None
    with pytest.raises(SystemExit):
        trainer.fit(model)
    assert trainer.interrupted
    assert isinstance(handle_interrupt_callback.exception, KeyboardInterrupt)
    with pytest.raises(MisconfigurationException):
        trainer.test(model)
    assert trainer.interrupted
    assert isinstance(handle_interrupt_callback.exception, MisconfigurationException)


def test_keyboard_interrupt(tmp_path):
    class InterruptCallback(Callback):
        def __init__(self):
            super().__init__()

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            raise KeyboardInterrupt

    model = BoringModel()
    trainer = Trainer(
        callbacks=[InterruptCallback()],
        barebones=True,
        default_root_dir=tmp_path,
    )

    trainer.strategy._launcher = Mock(spec=_SubprocessScriptLauncher)
    trainer.strategy._launcher.launch = lambda function, *args, trainer, **kwargs: function(*args, **kwargs)

    with pytest.raises(SystemExit) as exc_info:
        trainer.fit(model)
    assert exc_info.value.args[0] == 1
    trainer.strategy._launcher.kill.assert_called_once_with(15 if _IS_WINDOWS else 9)


@pytest.mark.parametrize("precision", ["32-true", pytest.param("16-mixed", marks=RunIf(min_cuda_gpus=1))])
@RunIf(sklearn=True)
def test_gradient_clipping_by_norm(tmp_path, precision):
    """Test gradient clipping by norm."""
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_steps=1,
        max_epochs=1,
        accelerator="auto",
        devices=1,
        precision=precision,
        gradient_clip_algorithm="norm",
        gradient_clip_val=0.05,
    )

    class TestModel(ClassificationModel):
        def configure_gradient_clipping(self, *args, **kwargs):
            super().configure_gradient_clipping(*args, **kwargs)
            # test that gradient is clipped correctly
            parameters = self.parameters()
            grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2)
            torch.testing.assert_close(grad_norm, torch.tensor(0.05, device=self.device))
            self.assertion_called = True

    model = TestModel()
    trainer.fit(model, ClassifDataModule())
    assert model.assertion_called


@pytest.mark.parametrize("precision", ["32-true", pytest.param("16-mixed", marks=RunIf(min_cuda_gpus=1))])
def test_gradient_clipping_by_value(tmp_path, precision):
    """Test gradient clipping by value."""
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_steps=1,
        max_epochs=1,
        accelerator="auto",
        devices=1,
        precision=precision,
        gradient_clip_algorithm="value",
        gradient_clip_val=1e-10,
    )

    class TestModel(BoringModel):
        def configure_gradient_clipping(self, *args, **kwargs):
            super().configure_gradient_clipping(*args, **kwargs)
            # test that gradient is clipped correctly
            parameters = self.parameters()
            grad_max_list = [torch.max(p.grad.detach().abs()) for p in parameters]
            grad_max = torch.max(torch.stack(grad_max_list))
            torch.testing.assert_close(grad_max.abs(), torch.tensor(1e-10, device=self.device))
            self.assertion_called = True

    model = TestModel()
    trainer.fit(model)
    assert model.assertion_called


def test_invalid_gradient_clip_value(tmp_path):
    with pytest.raises(TypeError, match="`gradient_clip_val` should be an int or a float"):
        Trainer(default_root_dir=tmp_path, gradient_clip_val=(1, 2))


def test_invalid_gradient_clip_algo(tmp_path):
    with pytest.raises(MisconfigurationException, match="`gradient_clip_algorithm` norm2 is invalid"):
        Trainer(default_root_dir=tmp_path, gradient_clip_algorithm="norm2")


@pytest.mark.parametrize("limit_val_batches", [0.0, 1, 1.0, 0.5, 5])
def test_num_sanity_val_steps(tmp_path, limit_val_batches):
    """Test that the number of sanity check batches is clipped to `limit_val_batches`."""
    num_sanity_val_steps = 4
    trainer = Trainer(
        default_root_dir=tmp_path,
        num_sanity_val_steps=num_sanity_val_steps,
        limit_val_batches=limit_val_batches,
        max_steps=1,
    )
    assert trainer.num_sanity_val_steps == num_sanity_val_steps

    class CustomModelMixedVal(BoringModel):
        def validation_step(self, batch, batch_idx, dataloader_idx):
            return super().validation_step(batch, batch_idx)

        def val_dataloader(self):
            return [DataLoader(RandomDataset(32, 64), batch_size=8), DataLoader(RandomDataset(32, 64))]

    model = CustomModelMixedVal()

    with patch.object(
        trainer.fit_loop.epoch_loop.val_loop,
        "_evaluation_step",
        wraps=trainer.fit_loop.epoch_loop.val_loop._evaluation_step,
    ) as mocked:
        trainer.fit(model)
    assert mocked.call_count == sum(trainer.num_sanity_val_batches)


@pytest.mark.parametrize("limit_val_batches", [0.0, 1, 1.0, 0.3])
def test_num_sanity_val_steps_neg_one(tmp_path, limit_val_batches):
    """Test that `num_sanity_val_steps=-1` runs through all validation data once, and as many batches as limited by
    `limit_val_batches` Trainer argument."""

    class CustomModel(BoringModel):
        def validation_step(self, batch, batch_idx, dataloader_idx):
            return super().validation_step(batch, batch_idx)

        def val_dataloader(self):
            return [DataLoader(RandomDataset(32, 64)), DataLoader(RandomDataset(32, 64))]

    model = CustomModel()
    trainer = Trainer(
        default_root_dir=tmp_path, num_sanity_val_steps=-1, limit_val_batches=limit_val_batches, max_steps=1
    )
    assert trainer.num_sanity_val_steps == float("inf")

    with patch.object(
        trainer.fit_loop.epoch_loop.val_loop,
        "_evaluation_step",
        wraps=trainer.fit_loop.epoch_loop.val_loop._evaluation_step,
    ) as mocked:
        val_dataloaders = model.val_dataloader()
        trainer.fit(model, val_dataloaders=val_dataloaders)

        assert mocked.call_count == sum(trainer.num_val_batches)


def test_trainer_subclassing():
    model = BoringModel()

    # First way of pulling out args from signature is to list them
    class TrainerSubclass(Trainer):
        def __init__(self, custom_arg, *args, custom_kwarg="test", **kwargs):
            super().__init__(*args, **kwargs)
            self.custom_arg = custom_arg
            self.custom_kwarg = custom_kwarg

    trainer = TrainerSubclass(123, custom_kwarg="custom", fast_dev_run=True)
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.custom_arg == 123
    assert trainer.custom_kwarg == "custom"
    assert trainer.fast_dev_run

    # Second way is to pop from the dict
    # It's a special case because Trainer does not have any positional args
    class TrainerSubclass(Trainer):
        def __init__(self, **kwargs):
            self.custom_arg = kwargs.pop("custom_arg", 0)
            self.custom_kwarg = kwargs.pop("custom_kwarg", "test")
            super().__init__(**kwargs)

    trainer = TrainerSubclass(custom_kwarg="custom", fast_dev_run=True)
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.custom_kwarg == "custom"
    assert trainer.fast_dev_run

    # when we pass in an unknown arg, the base class should complain
    with pytest.raises(TypeError, match=r"__init__\(\) got an unexpected keyword argument 'abcdefg'"):
        TrainerSubclass(abcdefg="unknown_arg")


@RunIf(omegaconf=True)
@pytest.mark.parametrize(
    "trainer_params",
    [{"max_epochs": 1, "accelerator": "gpu", "devices": 1}, {"max_epochs": 1, "accelerator": "gpu", "devices": [0]}],
)
def test_trainer_omegaconf(cuda_count_1, trainer_params):
    config = OmegaConf.create(trainer_params)
    Trainer(**config)


def test_trainer_pickle(tmp_path):
    trainer = Trainer(max_epochs=1, default_root_dir=tmp_path)
    pickle.dumps(trainer)
    cloudpickle.dumps(trainer)


@pytest.mark.parametrize("stage", ["fit", "validate", "test"])
def test_trainer_setup_call(tmp_path, stage):
    """Test setup call gets the correct stage."""

    class CurrentModel(BoringModel):
        def setup(self, stage):
            self.stage = stage

    class CurrentCallback(Callback):
        def setup(self, trainer, model, stage):
            assert model is not None
            self.stage = stage

    model = CurrentModel()
    callback = CurrentCallback()
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1, enable_checkpointing=False, callbacks=[callback])

    if stage == "fit":
        trainer.fit(model)
    elif stage == "validate":
        trainer.validate(model)
    else:
        trainer.test(model)

    assert callback.stage == stage
    assert model.stage == stage


@pytest.mark.parametrize(("train_batches", "max_steps", "log_interval"), [(10, 10, 1), (3, 10, 1), (3, 10, 5)])
@patch("lightning.pytorch.loggers.tensorboard.TensorBoardLogger.log_metrics")
def test_log_every_n_steps(log_metrics_mock, tmp_path, train_batches, max_steps, log_interval):
    class TestModel(BoringModel):
        def training_step(self, *args, **kwargs):
            self.log("foo", -1)
            return super().training_step(*args, **kwargs)

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        log_every_n_steps=log_interval,
        limit_train_batches=train_batches,
        limit_val_batches=0,
        max_steps=max_steps,
        logger=TensorBoardLogger(tmp_path),
    )
    trainer.fit(model)
    expected_calls = [call(metrics=ANY, step=s) for s in range(log_interval - 1, max_steps, log_interval)]
    log_metrics_mock.assert_has_calls(expected_calls)


class TestLightningDataModule(LightningDataModule):
    def __init__(self, dataloaders):
        super().__init__()
        self._dataloaders = dataloaders

    def test_dataloader(self):
        return self._dataloaders

    def predict_dataloader(self):
        return self._dataloaders


class CustomPredictionWriter(BasePredictionWriter):
    write_on_batch_end_called = False
    write_on_epoch_end_called = False

    def __init__(self, output_dir: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_dir = output_dir

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, *_):
        assert prediction.shape == torch.Size([1, 2])
        assert len(batch_indices) == 1
        self.write_on_batch_end_called = True

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        expected = 1 if trainer._accelerator_connector.is_distributed else 2
        assert len(predictions) == 2
        assert len(predictions[0]) == expected
        assert len(batch_indices) == 2
        assert len(batch_indices[0]) == expected
        self.write_on_epoch_end_called = True

    def on_predict_epoch_end(self, trainer, pl_module):
        if trainer._accelerator_connector.is_distributed:
            for idx in range(2):
                assert isinstance(trainer.predict_dataloaders[idx].batch_sampler.sampler, UnrepeatedDistributedSampler)
                assert isinstance(trainer.predict_dataloaders[idx].batch_sampler, _IndexBatchSamplerWrapper)
        super().on_predict_epoch_end(trainer, pl_module)


def predict(
    tmp_path,
    strategy="auto",
    accelerator="auto",
    devices="auto",
    model=None,
    plugins=None,
    datamodule=True,
    enable_progress_bar=True,
    use_callbacks=True,
):
    dataloaders = [torch.utils.data.DataLoader(RandomDataset(32, 2)), torch.utils.data.DataLoader(RandomDataset(32, 2))]

    model = model or BoringModel()
    dm = TestLightningDataModule(dataloaders)

    cb = CustomPredictionWriter(tmp_path, write_interval="batch")
    cb_1 = CustomPredictionWriter(tmp_path, write_interval="epoch")

    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        log_every_n_steps=1,
        enable_model_summary=False,
        strategy=strategy,
        accelerator=accelerator,
        devices=devices,
        plugins=plugins,
        enable_progress_bar=enable_progress_bar,
        callbacks=[cb, cb_1] if use_callbacks else [],
    )
    if strategy == "ddp_spawn":
        with pytest.raises(ProcessRaisedException, match="`return_predictions` should be set to `False`"):
            trainer.predict(model, datamodule=dm, return_predictions=True)

    results = trainer.predict(model, datamodule=dm) if datamodule else trainer.predict(model, dataloaders=dataloaders)

    if not isinstance(trainer.strategy.launcher, _MultiProcessingLauncher):
        if use_callbacks:
            assert cb.write_on_batch_end_called
            assert not cb.write_on_epoch_end_called

            assert not cb_1.write_on_batch_end_called
            assert cb_1.write_on_epoch_end_called

        num_samples = 1 if strategy == "ddp" else 2
        assert len(results) == 2
        assert len(results[0]) == num_samples
        assert results[0][0].shape == torch.Size([1, 2])


def test_trainer_predict_no_return(tmp_path):
    """Test trainer.predict warns when nothing is returned."""

    class CustomBoringModel(BoringModel):
        def predict_step(self, batch, batch_idx, dataloader_idx=0):
            if (batch_idx + 1) % 2 == 0:
                return None

            return super().predict_step(batch, batch_idx, dataloader_idx)

    with pytest.warns(UserWarning, match="predict returned None"):
        predict(tmp_path, model=CustomBoringModel(), use_callbacks=False)


def test_trainer_predict_grad(tmp_path):
    class CustomBoringModel(BoringModel):
        def predict_step(self, batch, batch_idx, dataloader_idx=0):
            assert batch.expand_as(batch).grad_fn is None
            return super().predict_step(batch, batch_idx, dataloader_idx)

    predict(tmp_path, model=CustomBoringModel(), use_callbacks=False)

    x = torch.zeros(1, requires_grad=True)
    assert x.expand_as(x).grad_fn is not None


@pytest.mark.parametrize("enable_progress_bar", [False, True])
@pytest.mark.parametrize("datamodule", [False, True])
def test_trainer_predict_cpu(tmp_path, datamodule, enable_progress_bar):
    predict(tmp_path, datamodule=datamodule, enable_progress_bar=enable_progress_bar)


@RunIf(min_cuda_gpus=2, standalone=True)
@pytest.mark.parametrize(
    "kwargs",
    [
        {"strategy": "ddp", "devices": 2},
    ],
)
def test_trainer_predict_standalone(tmp_path, kwargs):
    predict(tmp_path, accelerator="gpu", **kwargs)


@pytest.mark.parametrize(
    "accelerator",
    [
        pytest.param("gpu", marks=RunIf(min_cuda_gpus=1)),
        pytest.param("mps", marks=RunIf(mps=True)),
    ],
)
def test_trainer_predict_1_gpu(tmp_path, accelerator):
    predict(tmp_path, accelerator=accelerator, devices=1)


@RunIf(skip_windows=True)
@pytest.mark.parametrize("accelerator", ["cpu", pytest.param("gpu", marks=RunIf(min_cuda_gpus=2))])
def test_trainer_predict_ddp_spawn(tmp_path, accelerator):
    predict(tmp_path, strategy="ddp_spawn", accelerator=accelerator, devices=2)


@pytest.mark.parametrize("dataset_cls", [RandomDataset, RandomIterableDatasetWithLen, RandomIterableDataset])
def test_index_batch_sampler_wrapper_with_iterable_dataset(dataset_cls, tmp_path):
    ds = dataset_cls(32, 8)
    loader = DataLoader(ds)
    is_iterable_dataset = isinstance(ds, IterableDataset)

    class CustomPredictionWriter(BasePredictionWriter):
        def __init__(self, output_dir: str, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.output_dir = output_dir

        def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, *_):
            assert not batch_indices if is_iterable_dataset else batch_indices

    cb = CustomPredictionWriter(tmp_path)
    trainer = Trainer(default_root_dir=tmp_path, callbacks=cb)
    predictions = trainer.predict(BoringModel(), dataloaders=loader)
    assert len(predictions) == 8


def test_spawn_predict_return_predictions(tmp_path):
    """Test that `return_predictions=True` raise a MisconfigurationException with spawn strategies."""
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmp_path, accelerator="cpu", strategy="ddp_spawn", devices=2, fast_dev_run=True)
    assert isinstance(trainer.strategy, DDPStrategy)
    with pytest.raises(ProcessRaisedException, match="`return_predictions` should be set to `False`"):
        trainer.predict(model, dataloaders=model.train_dataloader(), return_predictions=True)


@pytest.mark.parametrize("return_predictions", [None, False, True])
@pytest.mark.parametrize("precision", ["32-true", pytest.param("64-true", marks=RunIf(mps=False))])
def test_predict_return_predictions_cpu(return_predictions, precision, tmp_path):
    """Test that `return_predictions=True`."""
    seed_everything(42)
    model = BoringModel()

    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=True, precision=precision)
    preds = trainer.predict(model, dataloaders=model.train_dataloader(), return_predictions=return_predictions)
    if return_predictions or return_predictions is None:
        assert len(preds) == 1
        assert preds[0].shape == torch.Size([1, 2])
        assert preds[0].dtype == (torch.float64 if precision == "64-true" else torch.float32)


@pytest.mark.parametrize(("max_steps", "max_epochs", "global_step"), [(10, 5, 10), (20, None, 20)])
def test_repeated_fit_calls_with_max_epochs_and_steps(tmp_path, max_steps, max_epochs, global_step):
    """Ensure that the training loop is bound by `max_steps` and `max_epochs` for repeated calls of `trainer.fit`, and
    disabled if the limit is reached."""

    dataset_len = 200
    batch_size = 10

    train_data = DataLoader(RandomDataset(32, dataset_len), batch_size=batch_size)

    model = BoringModel()

    trainer = Trainer(default_root_dir=tmp_path, max_steps=max_steps, max_epochs=max_epochs)
    trainer.fit(model, train_data)
    assert trainer.global_step == global_step
    trainer.fit(model, train_data)
    assert trainer.global_step == global_step


def test_trainer_access_in_configure_optimizers(tmp_path):
    """Verify that the configure optimizer function can reference the trainer."""

    class TestModel(BoringModel):
        def configure_optimizers(self):
            assert self.trainer is not None, "Expect to have access to the trainer within `configure_optimizers`"

    train_data = torch.utils.data.DataLoader(RandomDataset(32, 64))

    model = TestModel()
    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=True)
    trainer.fit(model, train_data)


@pytest.mark.parametrize(
    "accelerator",
    [
        pytest.param("cuda", marks=RunIf(min_cuda_gpus=1)),
        pytest.param("mps", marks=RunIf(mps=True)),
    ],
)
def test_setup_hook_device_and_layers(tmp_path, accelerator):
    """Test `LightningModule.device` access and creation of layers in `LightningModule.setup` hook."""
    expected_device = torch.device(accelerator, 0)

    class TestModel(BoringModel):
        def setup(self, stage: str) -> None:
            # The `self.device` attribute already points to what device the model will land on
            assert self.device == expected_device
            # However, the model parameters have not yet been moved to that device
            assert self.layer.weight.device == torch.device("cpu")
            # Can create new layers in this hook (on CPU)
            self.new_layer = torch.nn.Linear(2, 2)
            assert self.new_layer.weight.device == torch.device("cpu")

        def training_step(self, batch, batch_idx):
            output = self.layer(batch)
            # will crash if not moved to correct device
            output = self.new_layer(output)
            loss = self.loss(output)
            return {"loss": loss}

    # fake data
    train_data = torch.utils.data.DataLoader(RandomDataset(32, 64))

    # model
    model = TestModel()
    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=True, accelerator=accelerator, devices=1)
    trainer.fit(model, train_data)


def test_train_loop_system(tmp_path):
    """
    Test the following methods are called in the order in automatic optimization.
    1. optimizer.step (skip when gradient accumulation)
    2. model.training_step
    3. optimizer.zero_grad (run when the first batch of gradient accumulation)
    4. model.backward

    Note that the order is NOT `training_step`->`zero_grad`->`backward`->`step`.
    This is because `optimizer.step(closure)` calls `closure()` which then calls
    the three remaining methods `training_step`, `zero_grad` and `backward` inside.
    """
    called_methods = []

    trainer_options = {
        "default_root_dir": tmp_path,
        "max_epochs": 1,
        "limit_train_batches": 5,
        "limit_val_batches": 1,
        "limit_test_batches": 1,
        "enable_progress_bar": False,
    }

    class TestOptimizer(SGD):
        def step(self, *args, **kwargs):
            called_methods.append("step")
            return super().step(*args, **kwargs)

        def zero_grad(self, *args, **kwargs):
            called_methods.append("zero_grad")
            return super().zero_grad(*args, **kwargs)

    class TestModel(BoringModel):
        def configure_optimizers(self):
            return TestOptimizer(self.parameters(), lr=0.1)

        def training_step(self, *args, **kwargs):
            called_methods.append("training_step")
            return super().training_step(*args, **kwargs)

        def backward(self, *args, **kwargs):
            called_methods.append("backward")
            return super().backward(*args, **kwargs)

    model = TestModel()
    trainer = Trainer(**trainer_options)

    # No methods are called yet.
    assert called_methods == []

    trainer.fit(model)
    assert called_methods == ["step", "training_step", "zero_grad", "backward"] * trainer.limit_train_batches

    called_methods.clear()
    trainer = Trainer(**trainer_options, accumulate_grad_batches=3)

    # No methods are called yet.
    assert called_methods == []

    trainer.fit(model)
    assert called_methods == [
        # 0
        "training_step",
        "zero_grad",
        "backward",
        # 1
        "training_step",
        "backward",
        # 2
        "step",
        "training_step",
        "backward",
        # 3
        "training_step",
        "zero_grad",
        "backward",
        # 4
        "step",
        "training_step",
        "backward",
    ]


def test_check_val_every_n_epoch_exception(tmp_path):
    with pytest.raises(MisconfigurationException, match="should be an integer."):
        Trainer(default_root_dir=tmp_path, max_epochs=1, check_val_every_n_epoch=1.2)


def test_exception_when_testing_or_validating_with_fast_dev_run():
    trainer = Trainer(fast_dev_run=True)
    trainer.state.fn = TrainerFn.TESTING
    with pytest.raises(ValueError, match=r"with `fast_dev_run=True`. .* pass an exact checkpoint path"):
        trainer._checkpoint_connector._parse_ckpt_path(
            trainer.state.fn, ckpt_path="best", model_provided=False, model_connected=True
        )


class TrainerStagesModel(BoringModel):
    def on_train_start(self) -> None:
        assert self.trainer.model.training
        assert self.training

    def on_validation_start(self) -> None:
        assert not self.trainer.model.training
        assert not self.training

    def on_test_start(self) -> None:
        assert not self.trainer.model.training
        assert not self.training

    def on_predict_start(self) -> None:
        assert not self.trainer.model.training
        assert not self.training


@pytest.mark.parametrize(
    ("strategy", "devices"), [("auto", 1), pytest.param("ddp_spawn", 1, marks=RunIf(skip_windows=True))]
)
def test_model_in_correct_mode_during_stages(tmp_path, strategy, devices):
    model = TrainerStagesModel()
    trainer = Trainer(
        default_root_dir=tmp_path, strategy=strategy, accelerator="cpu", devices=devices, fast_dev_run=True
    )
    trainer.fit(model)
    trainer.validate(model)
    trainer.test(model)
    trainer.predict(model, model.val_dataloader())


class TestDummyModelForCheckpoint(BoringModel):
    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("x", loss)


@RunIf(skip_windows=True)
def test_fit_test_synchronization(tmp_path):
    """Test that the trainer synchronizes processes before returning control back to the caller."""
    model = TestDummyModelForCheckpoint()
    checkpoint = ModelCheckpoint(dirpath=tmp_path, monitor="x", mode="min", save_top_k=1)
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=2,
        strategy="ddp_spawn",
        accelerator="cpu",
        devices=2,
        callbacks=[checkpoint],
    )
    trainer.fit(model)
    assert os.path.exists(checkpoint.best_model_path), f"Could not find checkpoint at rank {trainer.global_rank}"
    trainer.test()


class CustomCallbackOnLoadCheckpoint(Callback):
    def state_dict(self) -> dict:
        return {"a": None}


def test_on_load_checkpoint_missing_callbacks(tmp_path):
    """Test a warning appears when callbacks in the checkpoint don't match callbacks provided when resuming."""
    model = BoringModel()
    chk = ModelCheckpoint(dirpath=tmp_path, save_last=True)

    trainer = Trainer(default_root_dir=tmp_path, max_epochs=3, callbacks=[chk, CustomCallbackOnLoadCheckpoint()])
    trainer.fit(model)

    trainer = Trainer(default_root_dir=tmp_path, max_epochs=5)
    with pytest.warns(UserWarning, match="CustomCallbackOnLoadCheckpoint"):
        trainer.fit(model, ckpt_path=chk.last_model_path)


def test_module_current_fx_attributes_reset(tmp_path):
    """Ensure that lightning module's attributes related to current fx are reset at the end of execution."""
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=1, enable_checkpointing=False, logger=False)

    trainer.fit(model)
    assert model._current_fx_name is None

    trainer.test(model)
    assert model._current_fx_name is None


@pytest.mark.parametrize("fn", ["validate", "test", "predict"])
def test_exception_when_lightning_module_is_not_set_on_trainer(fn):
    trainer = Trainer()
    trainer_fn = getattr(trainer, fn)
    with pytest.raises(TypeError, match=rf"{fn}\(\)` requires a `LightningModule"):
        trainer_fn()


@RunIf(min_cuda_gpus=1)
# FixMe: the memory raises to 1024 from expected 512
@pytest.mark.xfail(
    AssertionError,
    strict=True,
    condition=_TORCH_GREATER_EQUAL_2_8,
    reason="temporarily disabled for torch >= 2.8",
)
def test_multiple_trainer_constant_memory_allocated(tmp_path):
    """This tests ensures calling the trainer several times reset the memory back to 0."""

    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            loss = super().training_step(batch, batch_idx)
            self.log("train_loss", loss["loss"])
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.layer.parameters(), lr=0.1)

    class Check(Callback):
        def on_train_epoch_start(self, trainer, *_):
            assert isinstance(trainer.strategy.model, DistributedDataParallel)

    def current_memory():
        # before measuring the memory force release any leftover allocations, including CUDA tensors
        gc.collect()
        return torch.cuda.memory_allocated(0)

    model = TestModel()
    trainer_kwargs = {
        "default_root_dir": tmp_path,
        "fast_dev_run": True,
        "accelerator": "gpu",
        "devices": 1,
        "strategy": "ddp",
        "enable_progress_bar": False,
        "callbacks": Check(),
    }
    trainer = Trainer(**trainer_kwargs)
    initial = current_memory()
    trainer.fit(model)

    assert trainer.strategy.model is model
    assert list(trainer.optimizers[0].state.values())[0]["exp_avg_sq"].device == torch.device("cpu")
    assert trainer.callback_metrics["train_loss"].device == torch.device("cpu")

    assert current_memory() <= initial

    deepcopy(trainer)

    assert current_memory() <= initial

    trainer_2 = Trainer(**trainer_kwargs)
    trainer_2.fit(model)

    assert current_memory() <= initial


class TrainerStagesErrorsModel(BoringModel):
    def on_train_start(self) -> None:
        raise Exception("Error during train")

    def on_validation_start(self) -> None:
        raise Exception("Error during validation")

    def on_test_start(self) -> None:
        raise Exception("Error during test")

    def on_predict_start(self) -> None:
        raise Exception("Error during predict")


class ExceptionCounter(Callback):
    exceptions = 0

    def on_exception(self, *_):
        self.exceptions += 1


@pytest.mark.parametrize("strategy", ["auto", pytest.param("ddp_spawn", marks=RunIf(skip_windows=True, mps=False))])
def test_error_handling_all_stages(tmp_path, strategy):
    model = TrainerStagesErrorsModel()
    counter = ExceptionCounter()

    trainer = Trainer(
        default_root_dir=tmp_path,
        strategy=strategy,
        devices=1,
        callbacks=counter,
        fast_dev_run=True,
    )

    with pytest.raises(Exception, match=r"Error during train"):
        trainer.fit(model)
    assert counter.exceptions == 1

    with pytest.raises(Exception, match=r"Error during validation"):
        trainer.validate(model)
    assert counter.exceptions == 2

    with pytest.raises(Exception, match=r"Error during test"):
        trainer.test(model)
    assert counter.exceptions == 3

    with pytest.raises(Exception, match=r"Error during predict"):
        trainer.predict(model, model.val_dataloader(), return_predictions=False)
    assert counter.exceptions == 4


def test_trainer_metrics_reset_before_each_task(tmp_path):
    """Test that callback, logged and progress bar metrics are reset before each task starts."""

    class TestMetricRestartCallback(Callback):
        def _make_assertions(self, trainer):
            assert trainer.callback_metrics == {}
            assert trainer.progress_bar_metrics == {}
            assert trainer.logged_metrics == {}

        def on_train_start(self, trainer, *args, **kwargs):
            self._make_assertions(trainer)

        def on_validation_start(self, trainer, *args, **kwargs):
            if trainer.state.fn == TrainerFn.VALIDATING:
                self._make_assertions(trainer)

        def on_test_start(self, trainer, *args, **kwargs):
            self._make_assertions(trainer)

        def on_predict_start(self, trainer, *args, **kwargs):
            self._make_assertions(trainer)

    class CustomBoringModel(BoringModel):
        def __init__(self):
            super().__init__()

        def training_step(self, *args, **kwargs):
            self.log("train/metric", 7.0)
            return super().training_step(*args, **kwargs)

        def validation_step(self, *args, **kwargs):
            self.log("val/metric", 14.0)
            return super().validation_step(*args, **kwargs)

        def test_step(self, *args, **kwargs):
            self.log("test/metric", 21.0)
            return super().test_step(*args, **kwargs)

    model = CustomBoringModel()
    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=4, callbacks=[TestMetricRestartCallback()])
    trainer.fit(model)
    trainer.validate(model)
    trainer.test(model)
    trainer.predict(model)


def test_detect_anomaly_nan(tmp_path):
    class NanModel(BoringModel):
        def training_step(self, batch, batch_idx):
            output = super().training_step(batch, batch_idx)
            output["loss"] = output["loss"] * torch.tensor(float("nan"))
            return output

    model = NanModel()
    trainer = Trainer(default_root_dir=tmp_path, detect_anomaly=True)
    with (
        pytest.raises(RuntimeError, match=r"returned nan values in its 0th output."),
        pytest.warns(UserWarning, match=r".*Error detected in.* Traceback of forward call that caused the error.*"),
    ):
        trainer.fit(model)


@pytest.mark.parametrize(
    ("trainer_kwargs", "strategy_cls", "accelerator_cls", "devices"),
    [
        pytest.param({"strategy": "auto"}, SingleDeviceStrategy, CPUAccelerator, 1, marks=RunIf(mps=False)),
        pytest.param({"strategy": "ddp"}, DDPStrategy, CPUAccelerator, 1, marks=RunIf(mps=False)),
        pytest.param({"strategy": "ddp", "num_nodes": 2}, DDPStrategy, CPUAccelerator, 1, marks=RunIf(mps=False)),
        (
            {"strategy": "auto", "accelerator": "cuda", "devices": 1},
            SingleDeviceStrategy,
            CUDAAccelerator,
            1,
        ),
        ({"strategy": "ddp", "accelerator": "cuda", "devices": 1}, DDPStrategy, CUDAAccelerator, 1),
        (
            {"strategy": "ddp_spawn", "accelerator": "cuda", "devices": 1},
            DDPStrategy,
            CUDAAccelerator,
            1,
        ),
        ({"strategy": "auto", "accelerator": "cuda", "devices": 2}, DDPStrategy, CUDAAccelerator, 2),
        ({"strategy": "ddp", "accelerator": "cuda", "devices": 2}, DDPStrategy, CUDAAccelerator, 2),
        ({"strategy": "ddp", "accelerator": "cpu", "devices": 2}, DDPStrategy, CPUAccelerator, 2),
        (
            {"strategy": "ddp_spawn", "accelerator": "cpu", "devices": 2},
            DDPStrategy,
            CPUAccelerator,
            2,
        ),
        (
            {"strategy": "ddp_spawn", "accelerator": "cpu", "devices": 1},
            DDPStrategy,
            CPUAccelerator,
            1,
        ),
        (
            {"strategy": DDPStrategy(), "accelerator": "cpu", "devices": 2},
            DDPStrategy,
            CPUAccelerator,
            2,
        ),
        (
            {"strategy": DDPStrategy(), "accelerator": "cuda", "devices": 2},
            DDPStrategy,
            CUDAAccelerator,
            2,
        ),
        pytest.param({"strategy": DDPStrategy()}, DDPStrategy, CPUAccelerator, 1, marks=RunIf(mps=False)),
        (
            {"strategy": "ddp_spawn", "accelerator": "cuda", "devices": 2, "num_nodes": 2},
            DDPStrategy,
            CUDAAccelerator,
            2,
        ),
    ],
)
def test_trainer_config_strategy(monkeypatch, trainer_kwargs, strategy_cls, accelerator_cls, devices):
    if trainer_kwargs.get("accelerator") == "cuda":
        mock_cuda_count(monkeypatch, trainer_kwargs["devices"])
    if trainer_kwargs.get("accelerator") == "auto":
        # current parametrizations assume non-CUDA env
        mock_cuda_count(monkeypatch, 0)

    trainer = Trainer(**trainer_kwargs)

    assert isinstance(trainer.strategy, strategy_cls)
    assert isinstance(trainer.accelerator, accelerator_cls)
    assert trainer.num_devices == devices
    assert trainer.num_nodes == trainer_kwargs.get("num_nodes", 1)

    trainer_kwargs.pop("accelerator", None)
    trainer_kwargs.pop("devices", None)

    assert isinstance(trainer.strategy, strategy_cls)
    assert isinstance(trainer.accelerator, accelerator_cls)
    assert trainer.num_devices == devices
    assert trainer.num_nodes == trainer_kwargs.get("num_nodes", 1)


@pytest.mark.parametrize(
    "running_stage", [RunningStage.TRAINING, RunningStage.VALIDATING, RunningStage.TESTING, RunningStage.PREDICTING]
)
def test_dataloaders_are_not_loaded_if_disabled_through_limit_batches(running_stage):
    dl_prefix = running_stage.dataloader_prefix
    argument = f"limit_{dl_prefix}_batches"
    trainer_kwargs = {argument: 0}
    trainer = Trainer(**trainer_kwargs)
    model = BoringModel()
    trainer.strategy.connect(model)
    trainer._data_connector.attach_data(model)

    trainer.state.stage = running_stage
    if running_stage == "train":
        fn = trainer.fit_loop.setup_data
    elif running_stage == "validate":
        fn = trainer.validate_loop.setup_data
    elif running_stage == "test":
        fn = trainer.test_loop.setup_data
    else:
        fn = trainer.predict_loop.setup_data

    # with no limit, the attribute is None
    fn()
    dataloader_attribute = f"{dl_prefix}_dataloader{'' if running_stage == 'train' else 's'}"
    assert getattr(trainer, dataloader_attribute) is None

    # validate it would've worked if a limit was set
    setattr(trainer, argument, 1)
    fn()
    assert isinstance(getattr(trainer, dataloader_attribute), DataLoader)


@pytest.mark.parametrize(
    ("trainer_kwargs", "expected_device_ids"),
    [
        ({}, [0]),
        ({"devices": 1}, [0]),
        ({"devices": "1"}, [0]),
        pytest.param({"devices": 2}, [0, 1], marks=RunIf(mps=False)),
        ({"accelerator": "gpu", "devices": 1}, [0]),
        ({"accelerator": "cuda", "devices": 1}, [0]),
        ({"accelerator": "cuda", "devices": 2}, [0, 1]),
        ({"accelerator": "cuda", "devices": "2"}, [0, 1]),
        ({"accelerator": "cuda", "devices": [2]}, [2]),
        ({"accelerator": "cuda", "devices": "2,"}, [2]),
        ({"accelerator": "cuda", "devices": [0, 2]}, [0, 2]),
        ({"accelerator": "cuda", "devices": "0, 2"}, [0, 2]),
        ({"accelerator": "mps", "devices": 1}, [0]),
    ],
)
def test_trainer_config_device_ids(monkeypatch, trainer_kwargs, expected_device_ids):
    if trainer_kwargs.get("accelerator") in ("cuda", "gpu"):
        mock_cuda_count(monkeypatch, 4)
    elif trainer_kwargs.get("accelerator") in ("mps", "gpu"):
        mock_mps_count(monkeypatch, 1)

    trainer = Trainer(**trainer_kwargs)
    assert trainer.device_ids == expected_device_ids
    assert trainer.num_devices == len(expected_device_ids)


def test_trainer_save_checkpoint_no_model_attached():
    trainer = Trainer()
    assert trainer.model is None
    with pytest.raises(AttributeError, match="Saving a checkpoint is only possible if a model is attached"):
        trainer.save_checkpoint("checkpoint.ckpt")


def test_trainer_calls_logger_finalize_on_exception(tmp_path):
    class CustomModel(BoringModel):
        def on_fit_start(self):
            super().on_fit_start()
            raise Exception("logger-finalize")

    model = CustomModel()
    logger = TensorBoardLogger(save_dir=tmp_path)
    logger.finalize = Mock()
    trainer = Trainer(logger=logger)

    with pytest.raises(Exception, match="logger-finalize"):
        trainer.fit(model)

    logger.finalize.assert_called_once_with("failed")


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, RuntimeError])
def test_trainer_calls_strategy_on_exception(exception_type, tmp_path):
    """Test that when an exception occurs, the Trainer lets the strategy process it."""
    exception = exception_type("Test exception")

    class ExceptionModel(BoringModel):
        def on_fit_start(self):
            raise exception

    trainer = Trainer(default_root_dir=tmp_path)
    with (
        mock.patch("lightning.pytorch.strategies.strategy.Strategy.on_exception") as on_exception_mock,
        suppress(Exception, SystemExit),
    ):
        trainer.fit(ExceptionModel())
    on_exception_mock.assert_called_once_with(exception)


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, RuntimeError])
def test_trainer_calls_datamodule_on_exception(exception_type, tmp_path):
    """Test that when an exception occurs, the Trainer lets the data module process it."""
    exception = exception_type("Test exception")

    class ExceptionModel(BoringModel):
        def on_fit_start(self):
            raise exception

    datamodule = BoringDataModule()
    datamodule.on_exception = Mock()
    trainer = Trainer(default_root_dir=tmp_path)

    with suppress(Exception, SystemExit):
        trainer.fit(ExceptionModel(), datamodule=datamodule)
    datamodule.on_exception.assert_called_once_with(exception)


def test_init_module_context(monkeypatch):
    """Test that the strategy returns the context manager for initializing the module."""
    trainer = Trainer(accelerator="cpu", devices=1)
    strategy = SingleDeviceStrategy(device=torch.device("cuda"))
    strategy.tensor_init_context = Mock(wraps=strategy.tensor_init_context)
    trainer._accelerator_connector.strategy = strategy
    with trainer.init_module():
        pass
    strategy.tensor_init_context.assert_called_once_with(empty_init=None)
    strategy.tensor_init_context.reset_mock()


@pytest.mark.parametrize(
    ("target_device", "accelerator", "devices"),
    [
        ("cpu", "cpu", "auto"),
        pytest.param("cuda:0", "gpu", [0], marks=RunIf(min_cuda_gpus=1)),
        pytest.param("cuda:1", "gpu", [1], marks=RunIf(min_cuda_gpus=2)),
    ],
)
def test_init_module_device_type(target_device, accelerator, devices):
    """Test that the strategy returns the context manager for initializing the module."""
    trainer = Trainer(accelerator=accelerator, devices=devices)
    with trainer.init_module():
        model = BoringModel()
        assert model.device == torch.device(target_device)


def test_expand_home_trainer():
    """Test that the dirpath gets expanded if it contains `~`."""
    home_root = Path.home()

    trainer = Trainer(default_root_dir="~/trainer")
    assert trainer.default_root_dir == str(home_root / "trainer")
    trainer = Trainer(default_root_dir=Path("~/trainer"))
    assert trainer.default_root_dir == str(home_root / "trainer")


@pytest.mark.parametrize("suggest_integrations", [True, False])
def test_trainer_integration_suggestions(tmp_path, caplog, suggest_integrations):
    caplog.set_level("INFO", logger="lightning.pytorch.utilities.rank_zero")

    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        enable_progress_bar=False,
        suggest_integrations=suggest_integrations,
    )

    trainer.fit(BoringModel())

    messages = [r.getMessage() for r in caplog.records if r.name == "lightning.pytorch.utilities.rank_zero"]

    has_suggestion = any("litmodels" in m or "litlogger" in m for m in messages)

    assert has_suggestion == suggest_integrations
