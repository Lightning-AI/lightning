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
import os
import pickle
from argparse import Namespace
from copy import deepcopy
from typing import Any, Optional
from unittest.mock import patch

import numpy as np
import pytest
import torch

from lightning.fabric.utilities.logger import _convert_params, _sanitize_params
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringDataModule, BoringModel
from lightning.pytorch.loggers import Logger, TensorBoardLogger
from lightning.pytorch.loggers.logger import DummyExperiment, DummyLogger, merge_dicts
from lightning.pytorch.loggers.utilities import _scan_checkpoints
from lightning.pytorch.utilities.rank_zero import rank_zero_only


class CustomLogger(Logger):
    def __init__(self, experiment: str = "test", name: str = "name", version: str = "1"):
        super().__init__()
        self._experiment = experiment
        self._name = name
        self._version = version
        self.hparams_logged = None
        self.metrics_logged = {}
        self.finalized = False
        self.after_save_checkpoint_called = False

    @property
    def experiment(self):
        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params):
        self.hparams_logged = params

    @rank_zero_only
    def log_metrics(self, metrics, step):
        self.metrics_logged = metrics

    @rank_zero_only
    def finalize(self, status):
        self.finalized_status = status

    @property
    def save_dir(self) -> Optional[str]:
        """Return the root directory where experiment logs get saved, or `None` if the logger does not save data
        locally."""
        return None

    @property
    def name(self):
        return self._name

    @property
    def version(self):
        return self._version

    def after_save_checkpoint(self, checkpoint_callback):
        self.after_save_checkpoint_called = True


def test_custom_logger(tmp_path):
    class CustomModel(BoringModel):
        def training_step(self, batch, batch_idx):
            loss = self.step(batch)
            self.log("train_loss", loss)
            return {"loss": loss}

    logger = CustomLogger()
    model = CustomModel()
    trainer = Trainer(max_steps=2, log_every_n_steps=1, logger=logger, default_root_dir=tmp_path)
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert logger.metrics_logged != {}
    assert logger.after_save_checkpoint_called
    assert logger.finalized_status == "success"


def test_multiple_loggers(tmp_path):
    class CustomModel(BoringModel):
        def training_step(self, batch, batch_idx):
            loss = self.step(batch)
            self.log("train_loss", loss)
            return {"loss": loss}

    model = CustomModel()
    logger1 = CustomLogger()
    logger2 = CustomLogger()

    trainer = Trainer(max_steps=2, log_every_n_steps=1, logger=[logger1, logger2], default_root_dir=tmp_path)
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"

    assert logger1.hparams_logged is None
    assert logger1.metrics_logged != {}
    assert logger1.finalized_status == "success"

    assert logger2.hparams_logged is None
    assert logger2.metrics_logged != {}
    assert logger2.finalized_status == "success"


def test_multiple_loggers_pickle(tmp_path):
    """Verify that pickling trainer with multiple loggers works."""
    logger1 = CustomLogger()
    logger2 = CustomLogger()

    trainer = Trainer(logger=[logger1, logger2])
    pkl_bytes = pickle.dumps(trainer)
    trainer2 = pickle.loads(pkl_bytes)
    for logger in trainer2.loggers:
        logger.log_metrics({"acc": 1.0}, 0)

    for logger in trainer2.loggers:
        assert logger.metrics_logged == {"acc": 1.0}


def test_adding_step_key(tmp_path):
    class CustomTensorBoardLogger(TensorBoardLogger):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.logged_step = 0

        def log_metrics(self, metrics, step):
            if "val_acc" in metrics:
                assert step == self.logged_step

            super().log_metrics(metrics, step)

    class CustomModel(BoringModel):
        def on_train_epoch_end(self):
            self.logger.logged_step += 1
            self.log_dict({"step": self.logger.logged_step, "train_acc": self.logger.logged_step / 10})

        def on_validation_epoch_end(self):
            self.logger.logged_step += 1
            self.log_dict({"step": self.logger.logged_step, "val_acc": self.logger.logged_step / 10})

    model = CustomModel()
    trainer = Trainer(
        max_epochs=3,
        logger=CustomTensorBoardLogger(save_dir=tmp_path),
        default_root_dir=tmp_path,
        limit_train_batches=0.1,
        limit_val_batches=0.1,
        num_sanity_val_steps=0,
    )
    trainer.fit(model)


def test_dummylogger_noop_method_calls():
    """Test that the DummyLogger methods can be called with arbitrary arguments."""
    logger = DummyLogger()
    logger.log_hyperparams("1", 2, three="three")
    logger.log_metrics("1", 2, three="three")


def test_dummlogger_arbitrary_method_calls():
    """Test that the DummyLogger can be called with non existing methods."""
    logger = DummyLogger()
    # Example method from WandbLogger
    assert hasattr(logger, "log_text")
    assert callable(logger.log_text)


def test_dummyexperiment_support_item_assignment():
    """Test that the DummyExperiment supports item assignment."""
    experiment = DummyExperiment()
    experiment["variable"] = "value"
    assert experiment["variable"] != "value"  # this is only a stateless mock experiment


def test_np_sanitization():
    class CustomParamsLogger(CustomLogger):
        def __init__(self):
            super().__init__()
            self.logged_params = None

        @rank_zero_only
        def log_hyperparams(self, params):
            params = _convert_params(params)
            params = _sanitize_params(params)
            self.logged_params = params

    logger = CustomParamsLogger()
    np_params = {
        "np.bool_": np.bool_(1),
        "np.byte": np.byte(2),
        "np.intc": np.intc(3),
        "np.int_": np.int_(4),
        "np.longlong": np.longlong(5),
        "np.single": np.single(6.0),
        "np.double": np.double(8.9),
        "np.csingle": np.csingle(7 + 2j),
        "np.cdouble": np.cdouble(9 + 4j),
    }
    sanitized_params = {
        "np.bool_": True,
        "np.byte": 2,
        "np.intc": 3,
        "np.int_": 4,
        "np.longlong": 5,
        "np.single": 6.0,
        "np.double": 8.9,
        "np.csingle": "(7+2j)",
        "np.cdouble": "(9+4j)",
    }
    logger.log_hyperparams(Namespace(**np_params))
    assert logger.logged_params == sanitized_params


@pytest.mark.parametrize("logger", [True, False])
@patch("lightning.pytorch.loggers.tensorboard.TensorBoardLogger.log_hyperparams")
def test_log_hyperparams_being_called(log_hyperparams_mock, tmp_path, logger):
    class TestModel(BoringModel):
        def __init__(self, param_one, param_two):
            super().__init__()
            self.save_hyperparameters(logger=logger)

    model = TestModel("pytorch", "lightning")
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        limit_train_batches=0.1,
        limit_val_batches=0.1,
        num_sanity_val_steps=0,
        logger=TensorBoardLogger(tmp_path),
    )
    trainer.fit(model)

    if logger:
        log_hyperparams_mock.assert_called()
    else:
        log_hyperparams_mock.assert_not_called()


@patch("lightning.pytorch.loggers.tensorboard.TensorBoardLogger.log_hyperparams")
def test_log_hyperparams_key_collision(_, tmp_path):
    class TestModel(BoringModel):
        def __init__(self, hparams: dict[str, Any]) -> None:
            super().__init__()
            self.save_hyperparameters(hparams)

    class TestDataModule(BoringDataModule):
        def __init__(self, hparams: dict[str, Any]) -> None:
            super().__init__()
            self.save_hyperparameters(hparams)

    class _Test: ...

    same_params = {1: 1, "2": 2, "three": 3.0, "test": _Test(), "4": torch.tensor(4)}
    model = TestModel(same_params)

    trainer = Trainer(
        default_root_dir=tmp_path,
        logger=TensorBoardLogger(tmp_path),
        max_epochs=1,
        limit_train_batches=0.1,
        limit_val_batches=0.1,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    # there should be no exceptions raised for the same key/value pair in the hparams of both
    # the lightning module and data module
    trainer.fit(model)

    obj_params = deepcopy(same_params)
    obj_params["test"] = _Test()
    model = TestModel(same_params)
    trainer.fit(model)

    diff_params = deepcopy(same_params)
    diff_params.update({1: 0, "test": _Test()})
    model = TestModel(same_params)
    dm = TestDataModule(diff_params)
    trainer = Trainer(
        default_root_dir=tmp_path,
        logger=TensorBoardLogger(tmp_path),
        max_epochs=1,
        limit_train_batches=0.1,
        limit_val_batches=0.1,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    with pytest.raises(RuntimeError, match="Error while merging hparams"):
        trainer.fit(model, dm)

    tensor_params = deepcopy(same_params)
    tensor_params.update({"4": torch.tensor(3)})
    model = TestModel(same_params)
    dm = TestDataModule(tensor_params)
    trainer = Trainer(
        default_root_dir=tmp_path,
        logger=TensorBoardLogger(tmp_path),
        max_epochs=1,
        limit_train_batches=0.1,
        limit_val_batches=0.1,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    with pytest.raises(RuntimeError, match="Error while merging hparams"):
        trainer.fit(model, dm)


@pytest.mark.parametrize("save_top_k", [0, 1, 2, 5])
@patch("lightning.pytorch.callbacks.ModelCheckpoint")
def test_scan_checkpoints(checkpoint_callback_mock, tmp_path, save_top_k: int):
    """Checks if the expected number of checkpoints is returned."""
    # Test first condition of _scan_checkpoints: if c[1] not in logged_model_time.keys()
    # Test if the returned list of checkpoints has length save_top_k
    best_k_models = {}
    for i in range(save_top_k):
        ckpt_path = tmp_path / f"{i}.ckpt"
        with open(ckpt_path, "w") as f:
            f.write("")
        best_k_models[ckpt_path] = i
    checkpoint_callback_mock.best_k_models = best_k_models

    logged_model_time = {}
    checkpoints = _scan_checkpoints(checkpoint_callback_mock, logged_model_time)
    assert len(checkpoints) == save_top_k

    # Test second condition of _scan_checkpoints: or logged_model_time[c[1]] < c[0]]
    # Test if the returned list of checkpoints has still size 0
    for c in checkpoints:
        logged_model_time[c[1]] = c[0] + 1000
    checkpoints = _scan_checkpoints(checkpoint_callback_mock, logged_model_time)
    assert len(checkpoints) == 0


@patch("lightning.pytorch.callbacks.ModelCheckpoint")
def test_scan_checkpoints_single_stat_and_missing_paths(checkpoint_callback_mock, tmp_path):
    """Verify _scan_checkpoints performs a single stat syscall per path and ignores empty/deleted paths."""
    ckpt_path = str(tmp_path / "model.ckpt")
    with open(ckpt_path, "w") as f:
        f.write("data")

    checkpoint_callback_mock.last_model_path = ""
    checkpoint_callback_mock.current_score = None
    checkpoint_callback_mock.best_model_path = str(tmp_path / "deleted.ckpt")
    checkpoint_callback_mock.best_model_score = 0.5
    checkpoint_callback_mock.best_k_models = {ckpt_path: 0.9, str(tmp_path): 0.1}

    with patch("lightning.pytorch.loggers.utilities.os.stat", wraps=os.stat) as stat_mock:
        checkpoints = _scan_checkpoints(checkpoint_callback_mock, {})

    assert len(checkpoints) == 1
    assert checkpoints[0][1] == ckpt_path
    assert checkpoints[0][2] == 0.9
    assert checkpoints[0][3] == "best_k"
    # Empty path ("") is skipped; each remaining path is stat'd once instead of twice
    assert stat_mock.call_count == 3


def test_merge_dicts_deterministic_order_and_edge_cases():
    """Verify merge_dicts preserves insertion order and handles empty/None inputs."""
    assert merge_dicts([]) == {}
    assert merge_dicts([{"a": None, "b": 2.0}, {"a": None, "b": 4.0}]) == {"b": 3.0}

    d1 = {"z": 1.0, "a": 2.0, "nested": {"y": 10.0, "x": 20.0}}
    d2 = {"a": 4.0, "m": 5.0, "nested": {"x": 40.0, "w": 30.0}}
    merged = merge_dicts([d1, d2])
    assert list(merged.keys()) == ["z", "a", "nested", "m"]
    assert list(merged["nested"].keys()) == ["y", "x", "w"]
    assert merged == {"z": 1.0, "a": 3.0, "m": 5.0, "nested": {"y": 10.0, "x": 30.0, "w": 30.0}}

