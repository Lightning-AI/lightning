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
from unittest import mock
from unittest.mock import MagicMock

import fsspec
import pytest
import torch

from lightning.pytorch import Trainer
from lightning.pytorch.core.saving import load_hparams_from_yaml
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers.csv_logs import ExperimentWriter
from tests_pytorch.helpers.datamodules import ClassifDataModule
from tests_pytorch.helpers.runif import RunIf
from tests_pytorch.helpers.simple_models import ClassificationModel


def test_automatic_versioning(tmp_path):
    """Verify that automatic versioning works."""
    (tmp_path / "exp" / "version_0").mkdir(parents=True)
    (tmp_path / "exp" / "version_1").mkdir()
    (tmp_path / "exp" / "version_nonumber").mkdir()
    (tmp_path / "exp" / "other").mkdir()

    logger = CSVLogger(save_dir=tmp_path, name="exp")

    assert logger.version == 2


def test_manual_versioning(tmp_path):
    """Verify that manual versioning works."""
    root_dir = tmp_path / "exp"
    (root_dir / "version_0").mkdir(parents=True)
    (root_dir / "version_1").mkdir()
    (root_dir / "version_2").mkdir()

    logger = CSVLogger(save_dir=tmp_path, name="exp", version=1)

    assert logger.version == 1


def test_manual_versioning_file_exists(tmp_path):
    """Test that a warning is emitted and existing files get overwritten."""

    # Simulate an existing 'version_0' vrom a previous run
    (tmp_path / "exp" / "version_0").mkdir(parents=True)
    previous_metrics_file = tmp_path / "exp" / "version_0" / "metrics.csv"
    previous_metrics_file.touch()

    logger = CSVLogger(save_dir=tmp_path, name="exp", version=0)
    assert previous_metrics_file.exists()
    with pytest.warns(UserWarning, match="Experiment logs directory .* exists and is not empty"):
        _ = logger.experiment
    assert not previous_metrics_file.exists()


def test_named_version(tmp_path):
    """Verify that manual versioning works for string versions, e.g. '2020-02-05-162402'."""
    exp_name = "exp"
    (tmp_path / exp_name).mkdir()
    expected_version = "2020-02-05-162402"

    logger = CSVLogger(save_dir=tmp_path, name=exp_name, version=expected_version)
    logger.log_hyperparams({"a": 1, "b": 2})
    assert logger.version == expected_version
    assert os.listdir(tmp_path / exp_name) == [expected_version]
    assert os.listdir(tmp_path / exp_name / expected_version)


@pytest.mark.parametrize("name", ["", None])
def test_no_name(tmp_path, name):
    """Verify that None or empty name works."""
    logger = CSVLogger(save_dir=tmp_path, name=name)
    logger.log_hyperparams()
    assert os.path.normpath(logger.root_dir) == str(tmp_path)  # use os.path.normpath to handle trailing /
    assert os.listdir(tmp_path / "version_0")


@pytest.mark.parametrize("step_idx", [10, None])
def test_log_metrics(tmp_path, step_idx):
    logger = CSVLogger(tmp_path)
    metrics = {"float": 0.3, "int": 1, "FloatTensor": torch.tensor(0.1), "IntTensor": torch.tensor(1)}
    logger.log_metrics(metrics, step_idx)
    logger.save()

    path_csv = os.path.join(logger.log_dir, ExperimentWriter.NAME_METRICS_FILE)
    with open(path_csv) as fp:
        lines = fp.readlines()
    assert len(lines) == 2
    assert all(n in lines[0] for n in metrics)


def test_log_hyperparams(tmp_path):
    logger = CSVLogger(tmp_path)
    hparams = {
        "float": 0.3,
        "int": 1,
        "string": "abc",
        "bool": True,
        "dict": {"a": {"b": "c"}},
        "list": [1, 2, 3],
        "layer": torch.nn.BatchNorm1d,
    }
    logger.log_hyperparams(hparams)

    path_yaml = os.path.join(logger.log_dir, ExperimentWriter.NAME_HPARAMS_FILE)
    params = load_hparams_from_yaml(path_yaml)
    assert all(n in params for n in hparams)


@RunIf(sklearn=True)
def test_fit_csv_logger(tmp_path):
    dm = ClassifDataModule()
    model = ClassificationModel()
    logger = CSVLogger(save_dir=tmp_path)
    trainer = Trainer(default_root_dir=tmp_path, max_steps=10, logger=logger, log_every_n_steps=1)
    trainer.fit(model, datamodule=dm)
    metrics_file = os.path.join(logger.log_dir, ExperimentWriter.NAME_METRICS_FILE)
    assert os.path.isfile(metrics_file)


def test_csv_logger_remotefs():
    logger = CSVLogger(save_dir="memory://test_fit_csv_logger_remotefs")
    fs, _ = fsspec.core.url_to_fs("memory://test_fit_csv_logger_remotefs")
    exp = logger.experiment
    exp.log_metrics({"loss": 0.1})
    exp.save()
    metrics_file = os.path.join(logger.log_dir, ExperimentWriter.NAME_METRICS_FILE)
    assert fs.isfile(metrics_file)


def test_flush_n_steps(tmp_path):
    logger = CSVLogger(tmp_path, flush_logs_every_n_steps=2)
    metrics = {"float": 0.3, "int": 1, "FloatTensor": torch.tensor(0.1), "IntTensor": torch.tensor(1)}
    logger.save = MagicMock()
    logger.log_metrics(metrics, step=0)

    logger.save.assert_not_called()
    logger.log_metrics(metrics, step=1)
    logger.save.assert_called_once()


def test_metrics_reset_after_save(tmp_path):
    logger = CSVLogger(tmp_path, flush_logs_every_n_steps=2)
    metrics = {"test": 1}
    logger.log_metrics(metrics, step=0)
    assert logger.experiment.metrics
    logger.log_metrics(metrics, step=1)  # flush triggered
    assert not logger.experiment.metrics


@mock.patch(
    # Mock the existance check, so we can simulate appending to the metrics file
    "lightning.fabric.loggers.csv_logs._ExperimentWriter._check_log_dir_exists"
)
def test_append_metrics_file(_, tmp_path):
    """Test that the logger appends to the file instead of rewriting it on every save."""
    logger = CSVLogger(tmp_path, name="test", version=0, flush_logs_every_n_steps=1)

    # initial metrics
    logger.log_metrics({"a": 1, "b": 2})
    logger.log_metrics({"a": 3, "b": 4})

    # create a new logger to show we append to the existing file
    logger = CSVLogger(tmp_path, name="test", version=0, flush_logs_every_n_steps=1)
    logger.log_metrics({"a": 100, "b": 200})

    with open(logger.experiment.metrics_file_path) as file:
        lines = file.readlines()
    assert len(lines) == 4  # 1 header + 3 lines of metrics


def test_append_columns(tmp_path):
    """Test that the CSV file gets rewritten with new headers if the columns change."""
    logger = CSVLogger(tmp_path, flush_logs_every_n_steps=1)

    # initial metrics
    logger.log_metrics({"a": 1, "b": 2})

    # new key appears
    logger.log_metrics({"a": 1, "b": 2, "c": 3})
    with open(logger.experiment.metrics_file_path) as file:
        header = file.readline().strip()
        assert set(header.split(",")) == {"step", "a", "b", "c"}

    # key disappears
    logger.log_metrics({"a": 1, "c": 3})
    with open(logger.experiment.metrics_file_path) as file:
        header = file.readline().strip()
        assert set(header.split(",")) == {"step", "a", "b", "c"}
