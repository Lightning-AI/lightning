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
from typing import Any
from unittest import mock
from unittest.mock import MagicMock, Mock

import pytest

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.loggers.mlflow import (
    _MLFLOW_AVAILABLE,
    MLFlowLogger,
    _get_resolve_tags,
)
from lightning.pytorch.utilities.types import STEP_OUTPUT


def mock_mlflow_run_creation(logger, experiment_name=None, experiment_id=None, run_id=None):
    """Helper function to simulate mlflow client creating a new (or existing) experiment."""
    run = MagicMock()
    run.info.run_id = run_id
    logger._mlflow_client.get_experiment_by_name = MagicMock(return_value=experiment_name)
    logger._mlflow_client.create_experiment = MagicMock(return_value=experiment_id)
    logger._mlflow_client.create_run = MagicMock(return_value=run)
    return logger


@mock.patch("lightning.pytorch.loggers.mlflow._get_resolve_tags", Mock())
def test_mlflow_logger_exists(mlflow_mock, tmp_path):
    """Test launching three independent loggers with either same or different experiment name."""
    client = mlflow_mock.tracking.MlflowClient

    run1 = MagicMock()
    run1.info.run_id = "run-id-1"
    run1.info.experiment_id = "exp-id-1"

    run2 = MagicMock()
    run2.info.run_id = "run-id-2"

    run3 = MagicMock()
    run3.info.run_id = "run-id-3"

    # simulate non-existing experiment creation
    client.return_value.get_experiment_by_name = MagicMock(return_value=None)
    client.return_value.create_experiment = MagicMock(return_value="exp-id-1")  # experiment_id
    client.return_value.create_run = MagicMock(return_value=run1)

    logger = MLFlowLogger("test", save_dir=str(tmp_path))
    assert logger._experiment_id is None
    assert logger._run_id is None
    _ = logger.experiment
    assert logger.experiment_id == "exp-id-1"
    assert logger.run_id == "run-id-1"
    assert logger.experiment.create_experiment.asset_called_once()
    client.reset_mock(return_value=True)

    # simulate existing experiment returns experiment id
    exp1 = MagicMock()
    exp1.experiment_id = "exp-id-1"
    client.return_value.get_experiment_by_name = MagicMock(return_value=exp1)
    client.return_value.create_run = MagicMock(return_value=run2)

    # same name leads to same experiment id, but different runs get recorded
    logger2 = MLFlowLogger("test", save_dir=str(tmp_path))
    assert logger2.experiment_id == logger.experiment_id
    assert logger2.run_id == "run-id-2"
    assert logger2.experiment.create_experiment.call_count == 0
    assert logger2.experiment.create_run.asset_called_once()
    client.reset_mock(return_value=True)

    # simulate a 3rd experiment with new name
    client.return_value.get_experiment_by_name = MagicMock(return_value=None)
    client.return_value.create_experiment = MagicMock(return_value="exp-id-3")
    client.return_value.create_run = MagicMock(return_value=run3)

    # logger with new experiment name causes new experiment id and new run id to be created
    logger3 = MLFlowLogger("new", save_dir=str(tmp_path))
    assert logger3.experiment_id == "exp-id-3" != logger.experiment_id
    assert logger3.run_id == "run-id-3"


def test_mlflow_run_name_setting(tmp_path):
    """Test that the run_name argument makes the MLFLOW_RUN_NAME tag."""
    if not _MLFLOW_AVAILABLE:
        pytest.skip("test for explicit file creation requires mlflow dependency to be installed.")

    from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME

    resolve_tags = _get_resolve_tags()
    tags = resolve_tags({MLFLOW_RUN_NAME: "run-name-1"})

    # run_name is appended to tags
    logger = MLFlowLogger("test", run_name="run-name-1", save_dir=str(tmp_path))
    logger._mlflow_client = client = Mock()

    logger = mock_mlflow_run_creation(logger, experiment_id="exp-id")
    _ = logger.experiment
    client.create_run.assert_called_with(experiment_id="exp-id", tags=tags)

    # run_name overrides tags[MLFLOW_RUN_NAME]
    logger = MLFlowLogger("test", run_name="run-name-1", tags={MLFLOW_RUN_NAME: "run-name-2"}, save_dir=str(tmp_path))
    logger = mock_mlflow_run_creation(logger, experiment_id="exp-id")
    _ = logger.experiment
    client.create_run.assert_called_with(experiment_id="exp-id", tags=tags)

    # default run_name (= None) does not append new tag
    logger = MLFlowLogger("test", save_dir=str(tmp_path))
    logger = mock_mlflow_run_creation(logger, experiment_id="exp-id")
    _ = logger.experiment
    default_tags = resolve_tags(None)
    client.create_run.assert_called_with(experiment_id="exp-id", tags=default_tags)


@mock.patch("lightning.pytorch.loggers.mlflow._get_resolve_tags", Mock())
def test_mlflow_run_id_setting(mlflow_mock, tmp_path):
    """Test that the run_id argument uses the provided run_id."""
    client = mlflow_mock.tracking.MlflowClient

    run = MagicMock()
    run.info.run_id = "run-id"
    run.info.experiment_id = "experiment-id"

    # simulate existing run
    client.return_value.get_run = MagicMock(return_value=run)

    # run_id exists uses the existing run
    logger = MLFlowLogger("test", run_id=run.info.run_id, save_dir=str(tmp_path))
    _ = logger.experiment
    client.return_value.get_run.assert_called_with(run.info.run_id)
    assert logger.experiment_id == run.info.experiment_id
    assert logger.run_id == run.info.run_id
    client.reset_mock(return_value=True)


@mock.patch("lightning.pytorch.loggers.mlflow._get_resolve_tags", Mock())
def test_mlflow_log_dir(mlflow_mock, tmp_path):
    """Test that the trainer saves checkpoints in the logger's save dir."""
    client = mlflow_mock.tracking.MlflowClient

    # simulate experiment creation with mlflow client mock
    run = MagicMock()
    run.info.run_id = "run-id"
    client.return_value.get_experiment_by_name = MagicMock(return_value=None)
    client.return_value.create_experiment = MagicMock(return_value="exp-id")
    client.return_value.create_run = MagicMock(return_value=run)

    # test construction of default log dir path
    logger = MLFlowLogger("test", save_dir=str(tmp_path))
    assert logger.save_dir == str(tmp_path)
    assert logger.version == "run-id"
    assert logger.name == "exp-id"

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmp_path, logger=logger, max_epochs=1, limit_train_batches=1, limit_val_batches=3
    )
    assert trainer.log_dir == logger.save_dir
    trainer.fit(model)
    assert trainer.checkpoint_callback.dirpath == str(tmp_path / "exp-id" / "run-id" / "checkpoints")
    assert set(os.listdir(trainer.checkpoint_callback.dirpath)) == {"epoch=0-step=1.ckpt"}
    assert trainer.log_dir == logger.save_dir


def test_mlflow_logger_dirs_creation(tmp_path):
    """Test that the logger creates the folders and files in the right place."""
    if not _MLFLOW_AVAILABLE:
        pytest.skip("test for explicit file creation requires mlflow dependency to be installed.")

    assert not os.listdir(tmp_path)
    logger = MLFlowLogger("test", save_dir=str(tmp_path))
    assert logger.save_dir == str(tmp_path)
    assert set(os.listdir(tmp_path)) == {".trash"}
    run_id = logger.run_id
    exp_id = logger.experiment_id

    # multiple experiment calls should not lead to new experiment folders
    for i in range(2):
        _ = logger.experiment
        assert set(os.listdir(tmp_path)) == {".trash", exp_id}
        assert set(os.listdir(tmp_path / exp_id)) == {run_id, "meta.yaml"}

    class CustomModel(BoringModel):
        def on_train_epoch_end(self, *args, **kwargs):
            self.log("epoch", self.current_epoch)

    model = CustomModel()
    limit_batches = 5
    trainer = Trainer(
        default_root_dir=tmp_path,
        logger=logger,
        max_epochs=1,
        limit_train_batches=limit_batches,
        limit_val_batches=limit_batches,
    )
    trainer.fit(model)
    assert set(os.listdir(tmp_path / exp_id)) == {run_id, "meta.yaml"}
    assert "epoch" in os.listdir(tmp_path / exp_id / run_id / "metrics")
    assert set(os.listdir(tmp_path / exp_id / run_id / "params")) == model.hparams.keys()
    assert trainer.checkpoint_callback.dirpath == str(tmp_path / exp_id / run_id / "checkpoints")
    assert os.listdir(trainer.checkpoint_callback.dirpath) == [f"epoch=0-step={limit_batches}.ckpt"]


@mock.patch("lightning.pytorch.loggers.mlflow._get_resolve_tags", Mock())
@mock.patch("lightning.pytorch.loggers.mlflow._MLFLOW_AVAILABLE", return_value=True)
def test_mlflow_experiment_id_retrieved_once(_, mlflow_mock, tmp_path):
    """Test that the logger experiment_id retrieved only once."""
    logger = MLFlowLogger("test", save_dir=str(tmp_path))
    _ = logger.experiment
    _ = logger.experiment
    _ = logger.experiment
    assert logger.experiment.get_experiment_by_name.call_count == 1


@mock.patch("lightning.pytorch.loggers.mlflow._get_resolve_tags", Mock())
def test_mlflow_logger_with_unexpected_characters(mlflow_mock, tmp_path):
    """Test that the logger raises warning with special characters not accepted by MLFlow."""
    logger = MLFlowLogger("test", save_dir=str(tmp_path))
    metrics = {"[some_metric]": 10}

    with pytest.warns(RuntimeWarning, match="special characters in metric name"):
        logger.log_metrics(metrics)


@mock.patch("lightning.pytorch.loggers.mlflow._get_resolve_tags", Mock())
def test_mlflow_logger_experiment_calls(mlflow_mock, tmp_path):
    """Test that the logger calls methods on the mlflow experiment correctly."""
    time = mlflow_mock.entities.time
    metric = mlflow_mock.entities.Metric
    param = mlflow_mock.entities.Param

    time.return_value = 1

    logger = MLFlowLogger("test", save_dir=str(tmp_path), artifact_location="my_artifact_location")
    logger._mlflow_client.get_experiment_by_name.return_value = None

    params = {"test": "test_param"}
    logger.log_hyperparams(params)

    logger.experiment.log_batch.assert_called_once_with(
        run_id=logger.run_id, params=[param(key="test", value="test_param")]
    )
    param.assert_called_with(key="test", value="test_param")

    metrics = {"some_metric": 10}
    logger.log_metrics(metrics)

    logger.experiment.log_batch.assert_called_with(
        run_id=logger.run_id, metrics=[metric(key="some_metric", value=10, timestamp=1000, step=0)]
    )
    metric.assert_called_with(key="some_metric", value=10, timestamp=1000, step=0)

    logger._mlflow_client.create_experiment.assert_called_once_with(
        name="test", artifact_location="my_artifact_location"
    )


@pytest.mark.parametrize("synchronous", [False, True])
@mock.patch("lightning.pytorch.loggers.mlflow._get_resolve_tags", Mock())
def test_mlflow_logger_experiment_calls_with_synchronous(mlflow_mock, tmp_path, synchronous):
    """Test that the logger calls methods on the mlflow experiment with the specified synchronous flag."""

    time = mlflow_mock.entities.time
    metric = mlflow_mock.entities.Metric
    param = mlflow_mock.entities.Param
    time.return_value = 1

    mlflow_client = mlflow_mock.tracking.MlflowClient.return_value
    mlflow_client.get_experiment_by_name.return_value = None
    logger = MLFlowLogger(
        "test", save_dir=str(tmp_path), artifact_location="my_artifact_location", synchronous=synchronous
    )

    params = {"test": "test_param"}
    logger.log_hyperparams(params)

    mlflow_client.log_batch.assert_called_once_with(
        run_id=logger.run_id, params=[param(key="test", value="test_param")], synchronous=synchronous
    )
    param.assert_called_with(key="test", value="test_param")

    metrics = {"some_metric": 10}
    logger.log_metrics(metrics)

    mlflow_client.log_batch.assert_called_with(
        run_id=logger.run_id,
        metrics=[metric(key="some_metric", value=10, timestamp=1000, step=0)],
        synchronous=synchronous,
    )
    metric.assert_called_with(key="some_metric", value=10, timestamp=1000, step=0)

    mlflow_client.create_experiment.assert_called_once_with(name="test", artifact_location="my_artifact_location")


@mock.patch("lightning.pytorch.loggers.mlflow._get_resolve_tags", Mock())
@mock.patch.dict("lightning.pytorch.loggers.mlflow.__dict__", {"_MLFLOW_SYNCHRONOUS_AVAILABLE": False})
def test_mlflow_logger_no_synchronous_support(mlflow_mock, tmp_path):
    """Test that the logger does not support synchronous flag."""
    time = mlflow_mock.entities.time
    time.return_value = 1

    mlflow_client = mlflow_mock.tracking.MlflowClient.return_value
    mlflow_client.get_experiment_by_name.return_value = None
    with pytest.raises(ModuleNotFoundError):
        MLFlowLogger("test", save_dir=str(tmp_path), artifact_location="my_artifact_location", synchronous=True)


@mock.patch("lightning.pytorch.loggers.mlflow._get_resolve_tags", Mock())
def test_mlflow_logger_with_long_param_value(mlflow_mock, tmp_path):
    """Test that long parameter values are truncated to 250 characters."""

    def _check_value_length(value, *args, **kwargs):
        assert len(value) <= 250

    mlflow_mock.entities.Param.side_effect = _check_value_length

    logger = MLFlowLogger("test", save_dir=str(tmp_path))

    params = {"test": "test_param" * 50}
    logger.log_hyperparams(params)

    # assert_called_once_with() won't properly check the parameter value.
    logger.experiment.log_batch.assert_called_once()


@mock.patch("lightning.pytorch.loggers.mlflow._get_resolve_tags", Mock())
def test_mlflow_logger_with_many_params(mlflow_mock, tmp_path):
    """Test that when logging more than 100 parameters, it will be split into batches of at most 100 parameters."""
    logger = MLFlowLogger("test", save_dir=str(tmp_path))

    params = {f"test_{idx}": f"test_param_{idx}" for idx in range(150)}
    logger.log_hyperparams(params)

    assert logger.experiment.log_batch.call_count == 2


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("success", "FINISHED"),
        ("failed", "FAILED"),
        ("finished", "FINISHED"),
    ],
)
@mock.patch("lightning.pytorch.loggers.mlflow._get_resolve_tags", Mock())
def test_mlflow_logger_finalize(mlflow_mock, status, expected):
    logger = MLFlowLogger("test")

    # Pretend we are in a worker process and finalizing
    _ = logger.experiment
    assert logger._initialized

    logger.finalize(status)
    logger.experiment.set_terminated.assert_called_once_with(logger.run_id, expected)


@mock.patch("lightning.pytorch.loggers.mlflow._get_resolve_tags", Mock())
def test_mlflow_logger_finalize_when_exception(mlflow_mock):
    logger = MLFlowLogger("test")

    # Pretend we are on the main process and failing
    assert logger._mlflow_client
    assert not logger._initialized
    logger.finalize("failed")
    logger.experiment.set_terminated.assert_not_called()

    # Pretend we are in a worker process and failing
    _ = logger.experiment
    assert logger._initialized
    logger.finalize("failed")
    logger.experiment.set_terminated.assert_called_once_with(logger.run_id, "FAILED")


@pytest.mark.parametrize("log_model", ["all", True, False])
@mock.patch("lightning.pytorch.loggers.mlflow._get_resolve_tags", Mock())
def test_mlflow_log_model(mlflow_mock, log_model, tmp_path):
    """Test that the logger creates the folders and files in the right place."""
    client = mlflow_mock.tracking.MlflowClient

    # Get model, logger, trainer and train
    model = BoringModel()
    logger = MLFlowLogger("test", save_dir=str(tmp_path), log_model=log_model)
    logger = mock_mlflow_run_creation(logger, experiment_id="test-id")

    trainer = Trainer(
        default_root_dir=tmp_path,
        logger=logger,
        max_epochs=2,
        limit_train_batches=3,
        limit_val_batches=3,
    )
    trainer.fit(model)

    if log_model == "all":
        # Checkpoint log
        assert client.return_value.log_artifact.call_count == 2
        # Metadata and aliases log
        assert client.return_value.log_artifacts.call_count == 2

    elif log_model is True:
        # Checkpoint log
        client.return_value.log_artifact.assert_called_once()
        # Metadata and aliases log
        client.return_value.log_artifacts.assert_called_once()

    elif log_model is False:
        # Checkpoint log
        assert not client.return_value.log_artifact.called
        # Metadata and aliases log
        assert not client.return_value.log_artifacts.called


@mock.patch("lightning.pytorch.loggers.mlflow._get_resolve_tags", Mock())
def test_set_tracking_uri(mlflow_mock):
    """Test that the tracking uri is set for logging artifacts to MLFlow server."""
    logger = MLFlowLogger(tracking_uri="the_tracking_uri")
    mlflow_mock.set_tracking_uri.assert_not_called()
    _ = logger.experiment
    mlflow_mock.set_tracking_uri.assert_called_with("the_tracking_uri")


@mock.patch("lightning.pytorch.loggers.mlflow._get_resolve_tags", Mock())
def test_mlflow_multiple_checkpoints_top_k(mlflow_mock, tmp_path):
    """Test that multiple ModelCheckpoint callbacks with top_k parameters work correctly with MLFlowLogger.

    This test verifies that when using multiple ModelCheckpoint callbacks with save_top_k, both callbacks function
    correctly and save the expected number of checkpoints when using MLFlowLogger with log_model=True.

    """

    class CustomBoringModel(BoringModel):
        def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
            loss = self.step(batch)
            self.log("train_loss", loss)
            return {"loss": loss}

        def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
            loss = self.step(batch)
            self.log("val_loss", loss)
            return {"loss": loss}

    client = mlflow_mock.tracking.MlflowClient

    model = CustomBoringModel()
    logger = MLFlowLogger("test", save_dir=str(tmp_path), log_model=True)
    logger = mock_mlflow_run_creation(logger, experiment_id="test-id")

    # Create two ModelCheckpoint callbacks monitoring different metrics
    train_ckpt = ModelCheckpoint(
        dirpath=str(tmp_path / "train_checkpoints"),
        monitor="train_loss",
        filename="best_train_model-{epoch:02d}-{train_loss:.2f}",
        save_top_k=2,
        mode="min",
    )
    val_ckpt = ModelCheckpoint(
        dirpath=str(tmp_path / "val_checkpoints"),
        monitor="val_loss",
        filename="best_val_model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=2,
        mode="min",
    )

    # Create trainer with both callbacks
    trainer = Trainer(
        default_root_dir=tmp_path,
        logger=logger,
        callbacks=[train_ckpt, val_ckpt],
        max_epochs=5,
        limit_train_batches=3,
        limit_val_batches=3,
    )
    trainer.fit(model)

    # Verify both callbacks saved their checkpoints
    assert len(train_ckpt.best_k_models) > 0, "Train checkpoint callback did not save any models"
    assert len(val_ckpt.best_k_models) > 0, "Validation checkpoint callback did not save any models"

    # Get all artifact paths that were logged
    logged_artifacts = [call_args[0][1] for call_args in client.return_value.log_artifact.call_args_list]

    # Verify MLFlow logged artifacts from both callbacks
    # Get all artifact paths that were logged
    logged_artifacts = [call_args[0][1] for call_args in client.return_value.log_artifact.call_args_list]

    # Verify MLFlow logged artifacts from both callbacks
    train_artifacts = [path for path in logged_artifacts if "train_checkpoints" in path]
    val_artifacts = [path for path in logged_artifacts if "val_checkpoints" in path]

    assert len(train_artifacts) > 0, "MLFlow did not log any train checkpoint artifacts"
    assert len(val_artifacts) > 0, "MLFlow did not log any validation checkpoint artifacts"

    # Verify the number of logged artifacts matches the save_top_k for each callback
    assert len(train_artifacts) == train_ckpt.save_top_k, "Number of logged train artifacts doesn't match save_top_k"
    assert len(val_artifacts) == val_ckpt.save_top_k, "Number of logged val artifacts doesn't match save_top_k"
