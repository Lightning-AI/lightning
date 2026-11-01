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
import glob
import logging
import math
import os
import pickle
from argparse import Namespace
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from lightning_utilities.test.warning import no_warning_call
from torch.utils.data import DataLoader

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.batch_size_finder import BatchSizeFinder
from lightning.pytorch.demos.boring_classes import BoringDataModule, BoringModel, RandomDataset
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from tests_pytorch.helpers.runif import RunIf


class BatchSizeDataModule(BoringDataModule):
    def __init__(self, batch_size):
        super().__init__()
        if batch_size is not None:
            self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.random_train, batch_size=getattr(self, "batch_size", 1))


class BatchSizeModel(BoringModel):
    def __init__(self, batch_size):
        super().__init__()
        if batch_size is not None:
            self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(RandomDataset(32, 64), batch_size=getattr(self, "batch_size", 1))

    def val_dataloader(self):
        return DataLoader(RandomDataset(32, 64), batch_size=getattr(self, "batch_size", 1))

    def test_dataloader(self):
        return DataLoader(RandomDataset(32, 64), batch_size=getattr(self, "batch_size", 1))

    def predict_dataloader(self):
        return DataLoader(RandomDataset(32, 64), batch_size=getattr(self, "batch_size", 1))


@pytest.mark.parametrize(
    ("entrypoint", "method"),
    [("tuner", "fit"), ("tuner", "validate"), ("tuner", "test"), ("tuner", "predict"), ("callback", "fit")],
)
@pytest.mark.parametrize("weights_only", [None, True, False])
def test_scale_batch_size_weights_only(tmp_path, entrypoint, method, weights_only):
    class Model(BatchSizeModel):
        def on_save_checkpoint(self, checkpoint):
            checkpoint["custom_state"] = Namespace(value=42)

        def on_load_checkpoint(self, checkpoint):
            self.restored_value = checkpoint["custom_state"].value

    model = Model(batch_size=2)
    original_state = deepcopy(model.state_dict())
    trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    kwargs = {"steps_per_trial": 1, "max_trials": 1}
    if weights_only is not None:
        kwargs["weights_only"] = weights_only

    if entrypoint == "callback":

        class RecordingBatchSizeFinder(BatchSizeFinder):
            def scale_batch_size(self, trainer, pl_module):
                super().scale_batch_size(trainer, pl_module)
                self.restored_state = deepcopy(pl_module.state_dict())

        callback = RecordingBatchSizeFinder(**kwargs)
        trainer.callbacks.append(callback)

    if weights_only is not False:
        with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
            if entrypoint == "tuner":
                Tuner(trainer).scale_batch_size(model, method=method, **kwargs)
            else:
                trainer.fit(model)
        return

    if entrypoint == "tuner":
        batch_size = Tuner(trainer).scale_batch_size(model, method=method, **kwargs)
        restored_state = model.state_dict()
    else:
        trainer.fit(model)
        batch_size = callback.optimal_batch_size
        restored_state = callback.restored_state

    assert batch_size == model.batch_size
    assert model.restored_value == 42
    for name, value in restored_state.items():
        torch.testing.assert_close(value, original_state[name], rtol=0, atol=0)
    assert not list(tmp_path.glob(".scale_batch_size_*.ckpt"))


@pytest.mark.parametrize(("model_bs", "dm_bs"), [(2, -1), (2, 2), (2, None), (None, 2), (16, 16)])
def test_scale_batch_size_method_with_model_or_datamodule(tmp_path, model_bs, dm_bs):
    """Test the tuner method `Tuner.scale_batch_size` with a datamodule."""
    trainer = Trainer(default_root_dir=tmp_path, limit_train_batches=1, limit_val_batches=0, max_epochs=1)

    model = BatchSizeModel(model_bs)
    datamodule = BatchSizeDataModule(dm_bs) if dm_bs != -1 else None

    tuner = Tuner(trainer)
    new_batch_size = tuner.scale_batch_size(model, mode="binsearch", init_val=4, max_trials=2, datamodule=datamodule)
    assert new_batch_size == 7  # applied margin of 5% on 8 -> int(8 * 0.95) = 7

    if model_bs is not None:
        assert model.batch_size == new_batch_size
        if dm_bs == -1:
            # datamodule batch size takes precedence
            assert trainer.train_dataloader.batch_size == new_batch_size
    if dm_bs not in (-1, None):
        assert datamodule.batch_size == new_batch_size
        assert trainer.train_dataloader.batch_size == new_batch_size


@pytest.mark.parametrize("trainer_fn", ["fit", "validate", "test", "predict"])
def test_trainer_reset_correctly(tmp_path, trainer_fn):
    """Check that model and all trainer parameters are reset correctly after scaling batch size."""
    model = BatchSizeModel(batch_size=2)
    before_state_dict = deepcopy(model.state_dict())

    # logger file to get meta
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1)
    tuner = Tuner(trainer)

    changed_attributes = [
        "loggers",
        "callbacks",
        "global_step",
        "max_steps",
        "limit_train_batches",
        "limit_val_batches",
        "limit_test_batches",
        "limit_predict_batches",
    ]

    expected = {ca: getattr(trainer, ca) for ca in changed_attributes}
    expected_loop_state_dict = trainer.fit_loop.state_dict()

    with no_warning_call(UserWarning, match="Please add the following callbacks"):
        tuner.scale_batch_size(model, max_trials=64, method=trainer_fn)

    actual = {ca: getattr(trainer, ca) for ca in changed_attributes}
    actual_loop_state_dict = trainer.fit_loop.state_dict()
    assert expected_loop_state_dict == actual_loop_state_dict
    assert actual == expected

    after_state_dict = model.state_dict()
    for key in before_state_dict:
        assert torch.all(torch.eq(before_state_dict[key], after_state_dict[key])), (
            "Model was not reset correctly after scaling batch size"
        )

    assert not any(f for f in os.listdir(tmp_path) if f.startswith(".scale_batch_size_temp_model"))


@RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize("scale_arg", ["power", "binsearch", True])
def test_auto_scale_batch_size_trainer_arg(tmp_path, scale_arg):
    """Test possible values for 'batch size auto scaling' Trainer argument."""
    before_batch_size = 2
    model = BatchSizeModel(batch_size=before_batch_size)
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1, accelerator="gpu", devices=1)
    tuner = Tuner(trainer)
    tuner.scale_batch_size(model)
    after_batch_size = model.batch_size
    assert before_batch_size != after_batch_size, "Batch size was not altered after running auto scaling of batch size"

    assert not any(f for f in os.listdir(tmp_path) if f.startswith(".scale_batch_size_temp_model"))


@pytest.mark.parametrize("use_hparams", [True, False])
def test_auto_scale_batch_size_set_model_attribute(tmp_path, use_hparams):
    """Test that new batch size gets written to the correct hyperparameter attribute for model."""
    hparams = {"batch_size": 2}
    before_batch_size = hparams["batch_size"]

    class HparamsBatchSizeModel(BoringModel):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.save_hyperparameters()

        def train_dataloader(self):
            return DataLoader(RandomDataset(32, 64), batch_size=self.hparams.batch_size)

        def val_dataloader(self):
            return DataLoader(RandomDataset(32, 64), batch_size=self.hparams.batch_size)

    model_class = HparamsBatchSizeModel if use_hparams else BatchSizeModel
    model = model_class(**hparams)

    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1)
    tuner = Tuner(trainer)
    tuner.scale_batch_size(model, steps_per_trial=2, max_trials=4)
    after_batch_size = model.hparams.batch_size if use_hparams else model.batch_size
    assert before_batch_size != after_batch_size
    assert after_batch_size <= len(trainer.train_dataloader.dataset)


@pytest.mark.parametrize("use_hparams", [True, False])
def test_auto_scale_batch_size_set_datamodule_attribute(tmp_path, use_hparams):
    """Test that new batch size gets written to the correct hyperparameter attribute for datamodule."""
    hparams = {"batch_size": 2}
    before_batch_size = hparams["batch_size"]

    class HparamsBatchSizeDataModule(BoringDataModule):
        def __init__(self, batch_size):
            super().__init__()
            self.save_hyperparameters()

        def train_dataloader(self):
            return DataLoader(self.random_train, batch_size=self.hparams.batch_size)

        def val_dataloader(self):
            return DataLoader(RandomDataset(32, 64), batch_size=self.hparams.batch_size)

    datamodule_class = HparamsBatchSizeDataModule if use_hparams else BatchSizeDataModule
    datamodule = datamodule_class(batch_size=before_batch_size)
    model = BatchSizeModel(**hparams)

    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1)
    tuner = Tuner(trainer)
    tuner.scale_batch_size(model, datamodule=datamodule, steps_per_trial=2, max_trials=4)
    after_batch_size = datamodule.hparams.batch_size if use_hparams else datamodule.batch_size
    assert trainer.datamodule == datamodule
    assert before_batch_size < after_batch_size
    assert after_batch_size <= len(trainer.train_dataloader.dataset)


def test_auto_scale_batch_size_duplicate_attribute_warning(tmp_path):
    """Test for a warning when model.batch_size and model.hparams.batch_size both present."""

    class TestModel(BoringModel):
        def __init__(self, batch_size=1):
            super().__init__()
            # now we have model.batch_size and model.hparams.batch_size
            self.batch_size = 1
            self.save_hyperparameters()

    model = TestModel()
    trainer = Trainer(default_root_dir=tmp_path, max_steps=1, max_epochs=1000)
    tuner = Tuner(trainer)

    expected_message = "Field `model.batch_size` and `model.hparams.batch_size` are mutually exclusive!"
    with pytest.warns(UserWarning, match=expected_message):
        tuner.scale_batch_size(model)


@pytest.mark.parametrize("scale_method", ["power", "binsearch"])
def test_call_to_trainer_method(tmp_path, scale_method):
    """Test that calling the trainer method itself works."""
    before_batch_size = 2
    model = BatchSizeModel(batch_size=before_batch_size)

    # logger file to get meta
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1)
    tuner = Tuner(trainer)

    after_batch_size = tuner.scale_batch_size(model, mode=scale_method, max_trials=5)
    model.batch_size = after_batch_size
    trainer.fit(model)

    assert before_batch_size != after_batch_size, "Batch size was not altered after running auto scaling of batch size"


def test_error_on_dataloader_passed_to_fit(tmp_path):
    """Verify that when the auto-scale batch size feature raises an error if a train dataloader is passed to fit."""

    # only train passed to fit
    model = BatchSizeModel(batch_size=2)
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        limit_val_batches=0.1,
        limit_train_batches=0.2,
    )
    tuner = Tuner(trainer)

    with pytest.raises(
        MisconfigurationException,
        match="Batch size finder cannot be used with dataloaders passed directly",
    ):
        tuner.scale_batch_size(model, train_dataloaders=model.train_dataloader(), mode="power")


@RunIf(min_cuda_gpus=1)
def test_auto_scale_batch_size_with_amp(tmp_path):
    before_batch_size = 2
    model = BatchSizeModel(batch_size=before_batch_size)
    trainer = Trainer(default_root_dir=tmp_path, max_steps=1, accelerator="gpu", devices=1, precision="16-mixed")
    tuner = Tuner(trainer)
    tuner.scale_batch_size(model)
    after_batch_size = model.batch_size
    assert trainer.scaler is not None
    assert after_batch_size != before_batch_size


def test_scale_batch_size_no_trials(tmp_path):
    """Check the result is correct even when no trials are run."""
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1, limit_val_batches=1, limit_train_batches=1)
    tuner = Tuner(trainer)
    model = BatchSizeModel(batch_size=2)
    result = tuner.scale_batch_size(model, max_trials=0, mode="power")
    assert result == 2


def test_scale_batch_size_fails_with_unavailable_mode(tmp_path):
    """Check the tuning raises error when called with mode that does not exist."""

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.batch_size = 2

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        limit_val_batches=1,
        limit_train_batches=1,
    )
    tuner = Tuner(trainer)

    with pytest.raises(ValueError, match="should be either of"):
        tuner.scale_batch_size(model, mode="ThisModeDoesNotExist")


@pytest.mark.parametrize("scale_method", ["power", "binsearch"])
@pytest.mark.parametrize("init_batch_size", [8, 17, 64])
def test_dataloader_reset_with_scale_batch_size(tmp_path, caplog, scale_method, init_batch_size):
    """Test that train and val dataloaders are reset at every update in scale batch size."""
    model = BatchSizeModel(batch_size=16)
    max_trials = 2
    scale_batch_size_kwargs = {
        "max_trials": max_trials,
        "steps_per_trial": 2,
        "init_val": init_batch_size,
        "mode": scale_method,
    }

    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1)
    tuner = Tuner(trainer)

    with caplog.at_level(logging.INFO):
        new_batch_size = tuner.scale_batch_size(model, **scale_batch_size_kwargs)

    dataset_len = len(trainer.train_dataloader.dataset)
    assert dataset_len == 64
    # With our fix, when max_trials is reached, we don't try the doubled batch size, so we get max_trials - 1 messages
    expected_tries = max_trials - 1 if init_batch_size < dataset_len and max_trials > 0 else 0
    assert caplog.text.count("trying batch size") == expected_tries

    # Determine the largest batch size that was actually tested.
    # For "power" this is the final found size; for "binsearch" we applied a 5% margin
    # when storing the final value, so the largest tested value is the one before applying margin.
    largest_tested_batch_size = new_batch_size if scale_method == "power" else int(math.ceil(new_batch_size * 100 / 95))
    assert caplog.text.count("greater or equal than the length") == int(largest_tested_batch_size >= dataset_len)

    assert trainer.train_dataloader.batch_size == new_batch_size
    assert trainer.val_dataloaders.batch_size == new_batch_size


@pytest.mark.parametrize("trainer_fn", ["validate", "test", "predict"])
def test_tuner_with_evaluation_methods(tmp_path, trainer_fn):
    """Test batch size tuner with Trainer's evaluation methods."""
    before_batch_size = 2
    max_trials = 4
    # With our fix, we return the last successful batch size, not the doubled untested one
    expected_scaled_batch_size = before_batch_size**max_trials  # 2^4 = 16, not 2^5 = 32

    model = BatchSizeModel(batch_size=before_batch_size)
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=100)
    tuner = Tuner(trainer)
    tuner.scale_batch_size(model, max_trials=max_trials, batch_arg_name="batch_size", method=trainer_fn)

    after_batch_size = model.batch_size
    loop = getattr(trainer, f"{trainer_fn}_loop")

    assert trainer.global_step == 0
    assert trainer.current_epoch == 0
    assert loop.batch_progress.current.completed == 0
    assert expected_scaled_batch_size == after_batch_size
    assert not any(f for f in os.listdir(tmp_path) if f.startswith(".scale_batch_size_temp_model"))


@pytest.mark.parametrize("trainer_fn", ["fit", "validate", "test", "predict"])
def test_batch_size_finder_callback(tmp_path, trainer_fn):
    """Test batch size finder callback with different trainer methods."""
    before_batch_size = 2
    max_trials = 4
    max_epochs = 2
    # With our fix, we return the last successful batch size, not the doubled untested one
    expected_scaled_batch_size = before_batch_size**max_trials  # 2^4 = 16, not 2^5 = 32

    model = BatchSizeModel(batch_size=before_batch_size)
    batch_size_finder = BatchSizeFinder(max_trials=max_trials, batch_arg_name="batch_size")
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=max_epochs, callbacks=[batch_size_finder])
    fn = getattr(trainer, trainer_fn)

    fn(model)
    after_batch_size = model.batch_size
    loop = getattr(trainer, f"{trainer_fn}_loop")

    if trainer_fn == "fit":
        expected_steps = trainer.train_dataloader.dataset.len // after_batch_size
        assert trainer.global_step == expected_steps * max_epochs
        assert trainer.current_epoch == max_epochs
        assert loop.epoch_loop.batch_progress.total.completed == expected_steps * max_epochs
    else:
        if trainer_fn == "validate":
            dl = trainer.val_dataloaders
        elif trainer_fn == "test":
            dl = trainer.test_dataloaders
        elif trainer_fn == "predict":
            dl = trainer.predict_dataloaders

        expected_steps = dl.dataset.len // after_batch_size
        assert trainer.global_step == 0
        assert trainer.current_epoch == 0
        assert loop.batch_progress.current.completed == expected_steps

    assert expected_scaled_batch_size == after_batch_size
    assert not any(f for f in os.listdir(tmp_path) if f.startswith(".scale_batch_size_temp_model"))


def test_invalid_method_in_tuner():
    """Test that an invalid value for `method` raises an error in `Tuner`"""
    trainer = Trainer()
    tuner = Tuner(trainer)
    model = BoringModel()

    with pytest.raises(ValueError, match="method .* is invalid."):
        tuner.scale_batch_size(model, method="prediction")


def test_error_if_train_or_val_dataloaders_passed_with_eval_method():
    """Test that an error is raised if `train_dataloaders` or `val_dataloaders` is passed with eval method inside
    `Tuner`"""
    trainer = Trainer()
    tuner = Tuner(trainer)
    model = BoringModel()
    dl = model.train_dataloader()

    with pytest.raises(MisconfigurationException, match="please consider setting `dataloaders` instead"):
        tuner.scale_batch_size(model, train_dataloaders=dl, method="validate")

    with pytest.raises(MisconfigurationException, match="please consider setting `dataloaders` instead"):
        tuner.scale_batch_size(model, val_dataloaders=dl, method="validate")


def test_error_if_dataloaders_passed_with_fit_method():
    """Test that an error is raised if `dataloaders` is passed with fit method inside `Tuner`"""
    trainer = Trainer()
    tuner = Tuner(trainer)
    model = BoringModel()
    dl = model.val_dataloader()

    with pytest.raises(
        MisconfigurationException, match="please consider setting `train_dataloaders` and `val_dataloaders` instead"
    ):
        tuner.scale_batch_size(model, dataloaders=dl, method="fit")


def test_batch_size_finder_with_distributed_strategies():
    """Test that an error is raised when batch size finder is used with multi-device strategy."""
    trainer = Trainer(devices=2, strategy="ddp", accelerator="cpu")
    model = BoringModel()
    bs_finder = BatchSizeFinder()

    with pytest.raises(
        MisconfigurationException, match="Batch size finder is not supported with distributed strategies."
    ):
        bs_finder.setup(trainer, model)


def test_batch_size_finder_with_multiple_eval_dataloaders(tmp_path):
    """Test that an error is raised with batch size finder is called with multiple eval dataloaders."""

    class CustomModel(BoringModel):
        def val_dataloader(self):
            return [super().val_dataloader(), super().val_dataloader()]

    trainer = Trainer(logger=False, enable_checkpointing=False)
    tuner = Tuner(trainer)
    model = CustomModel()

    with pytest.raises(
        MisconfigurationException, match="Batch size finder cannot be used with multiple .* dataloaders"
    ):
        tuner.scale_batch_size(model, method="validate")


@pytest.mark.parametrize(("scale_method", "expected_batch_size"), [("power", 62), ("binsearch", 95)])
@patch("lightning.pytorch.tuner.batch_size_scaling.is_oom_error", return_value=True)
def test_dataloader_batch_size_updated_on_failure(_, tmp_path, scale_method, expected_batch_size):
    class CustomBatchSizeModel(BatchSizeModel):
        def training_step(self, *_, **__):
            if self.batch_size > 100:
                raise RuntimeError

        def train_dataloader(self):
            return DataLoader(RandomDataset(32, 1000), batch_size=self.batch_size)

    model = CustomBatchSizeModel(batch_size=16)
    model.validation_step = None
    scale_batch_size_kwargs = {"max_trials": 10, "steps_per_trial": 1, "init_val": 500, "mode": scale_method}

    trainer = Trainer(default_root_dir=tmp_path, max_epochs=2)
    tuner = Tuner(trainer)
    new_batch_size = tuner.scale_batch_size(model, **scale_batch_size_kwargs)
    assert new_batch_size == model.batch_size
    assert new_batch_size == expected_batch_size
    assert trainer.train_dataloader.batch_size == expected_batch_size


def test_batch_size_finder_callback_val_batches(tmp_path):
    """Test that `BatchSizeFinder` does not limit the number of val batches during training."""
    steps_per_trial = 2
    model = BatchSizeModel(batch_size=16)
    trainer = Trainer(
        default_root_dir=tmp_path,
        num_sanity_val_steps=0,
        max_epochs=1,
        enable_model_summary=False,
        callbacks=[BatchSizeFinder(steps_per_trial=steps_per_trial, max_trials=1)],
    )
    trainer.fit(model)

    assert trainer.num_val_batches[0] == len(trainer.val_dataloaders)
    assert trainer.num_val_batches[0] != steps_per_trial


@pytest.mark.parametrize("margin", [0.0, 0.1, 0.2])
def test_scale_batch_size_margin_and_max_val(tmp_path, margin):
    """Test margin feature for batch size scaling by comparing results with and without margin."""
    # First, find the batch size without margin
    model1 = BatchSizeModel(batch_size=2)
    trainer1 = Trainer(default_root_dir=tmp_path, max_epochs=1, logger=False, enable_checkpointing=False)
    tuner1 = Tuner(trainer1)

    result_without_margin = tuner1.scale_batch_size(
        model1, mode="binsearch", max_trials=2, steps_per_trial=1, margin=0.0
    )

    model2 = BatchSizeModel(batch_size=2)
    trainer2 = Trainer(default_root_dir=tmp_path, max_epochs=1, logger=False, enable_checkpointing=False)
    tuner2 = Tuner(trainer2)

    result_with_margin = tuner2.scale_batch_size(
        model2, mode="binsearch", max_trials=2, steps_per_trial=1, margin=margin
    )

    assert result_without_margin is not None
    assert result_with_margin is not None

    if margin == 0.0:
        assert result_with_margin == result_without_margin
    else:
        expected_with_margin = max(1, int(result_without_margin * (1 - margin)))
        assert result_with_margin == expected_with_margin
        assert result_with_margin <= result_without_margin


@pytest.mark.parametrize("mode", ["power", "binsearch"])
def test_scale_batch_size_max_val_limit(tmp_path, mode):
    """Test that max_val limits the batch size for both power and binsearch modes."""
    model = BatchSizeModel(batch_size=2)
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1)
    tuner = Tuner(trainer)

    max_val = 8  # Set a low max value
    result = tuner.scale_batch_size(model, mode=mode, max_trials=5, steps_per_trial=1, max_val=max_val)

    assert result is not None
    assert result <= max_val


def test_scale_batch_size_checkpoint_cleanup_on_error(tmp_path):
    """Test that temporary checkpoint files are cleaned up even when an error occurs during batch size scaling."""

    class FailingModel(BoringModel):
        def __init__(self, fail_on_step=2):
            super().__init__()
            self.fail_on_step = fail_on_step
            self.current_step = 0
            self.batch_size = 2

        def training_step(self, batch, batch_idx):
            self.current_step += 1
            if self.current_step >= self.fail_on_step:
                raise RuntimeError("Intentional failure for testing cleanup")
            return super().training_step(batch, batch_idx)

        def train_dataloader(self):
            return DataLoader(RandomDataset(32, 64), batch_size=self.batch_size)

    model = FailingModel()
    batch_size_finder = BatchSizeFinder(max_trials=3, steps_per_trial=2)
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        callbacks=[batch_size_finder],
    )

    # Check no scale_batch_size checkpoint files exist initially
    scale_checkpoints = glob.glob(os.path.join(tmp_path, ".scale_batch_size_*.ckpt"))
    assert len(scale_checkpoints) == 0, "No scale_batch_size checkpoint files should exist initially"

    # Run batch size scaler and expect it to fail
    with pytest.raises(RuntimeError, match="Intentional failure for testing cleanup"):
        trainer.fit(model)

    # Check that no scale_batch_size checkpoint files are left behind
    scale_checkpoints = glob.glob(os.path.join(tmp_path, ".scale_batch_size_*.ckpt"))
    assert len(scale_checkpoints) == 0, (
        f"scale_batch_size checkpoint files should be cleaned up, but found: {scale_checkpoints}"
    )


class AlwaysSucceedingBoringModel(BoringModel):
    """A BoringModel that never fails with OOM errors for batch size scaling tests."""

    def __init__(self, batch_size=2):
        super().__init__()
        self.batch_size = batch_size


class FailsAtBatchSizeBoringModel(BoringModel):
    """A BoringModel that fails when batch size reaches a certain threshold."""

    def __init__(self, batch_size=2, fail_at=16):
        super().__init__()
        self.batch_size = batch_size
        self.fail_at = fail_at

    def training_step(self, batch, batch_idx):
        # Simulate OOM error when batch size is too large
        if self.batch_size >= self.fail_at:
            raise RuntimeError("CUDA error: out of memory")
        return super().training_step(batch, batch_idx)


@pytest.mark.parametrize(
    ("max_trials", "mode", "init_val", "expected"),
    [
        (3, "power", 2, 8),
        (3, "binsearch", 2, 7),  # applied margin of 5% on 8 -> int(8 * 0.95) = 7
        (1, "power", 4, 4),
        (0, "power", 2, 2),
    ],
)
def test_scale_batch_size_max_trials_modes(tmp_path, max_trials, mode, init_val, expected):
    model = AlwaysSucceedingBoringModel(batch_size=init_val)
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1)
    tuner = Tuner(trainer)
    result = tuner.scale_batch_size(
        model,
        mode=mode,
        steps_per_trial=1,
        max_trials=max_trials,
        init_val=init_val,
    )
    assert result == expected
