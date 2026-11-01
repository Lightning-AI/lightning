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
from argparse import Namespace
from unittest import mock
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from lightning.fabric.loggers import TensorBoardLogger
from lightning.fabric.loggers.tensorboard import _TENSORBOARD_AVAILABLE
from lightning.fabric.wrappers import _FabricModule
from tests_fabric.test_fabric import BoringModel


def test_tensorboard_automatic_versioning(tmp_path):
    """Verify that automatic versioning works."""
    root_dir = tmp_path / "tb_versioning"
    root_dir.mkdir()
    (root_dir / "version_0").mkdir()
    (root_dir / "version_1").mkdir()
    (root_dir / "version_nonumber").mkdir()
    (root_dir / "other").mkdir()

    logger = TensorBoardLogger(root_dir=tmp_path, name="tb_versioning")
    assert logger.version == 2


def test_tensorboard_manual_versioning(tmp_path):
    """Verify that manual versioning works."""
    root_dir = tmp_path / "tb_versioning"
    root_dir.mkdir()
    (root_dir / "version_0").mkdir()
    (root_dir / "version_1").mkdir()
    (root_dir / "version_2").mkdir()

    logger = TensorBoardLogger(root_dir=tmp_path, name="tb_versioning", version=1)
    assert logger.version == 1


def test_tensorboard_named_version(tmp_path):
    """Verify that manual versioning works for string versions, e.g. '2020-02-05-162402'."""
    name = "tb_versioning"
    (tmp_path / name).mkdir()
    expected_version = "2020-02-05-162402"

    logger = TensorBoardLogger(root_dir=tmp_path, name=name, version=expected_version)
    logger.log_hyperparams({"a": 1, "b": 2, 123: 3, 3.5: 4, 5j: 5})  # Force data to be written

    assert logger.version == expected_version
    assert os.listdir(tmp_path / name) == [expected_version]
    assert os.listdir(tmp_path / name / expected_version)


@pytest.mark.parametrize("name", ["", None])
def test_tensorboard_no_name(tmp_path, name):
    """Verify that None or empty name works."""
    logger = TensorBoardLogger(root_dir=tmp_path, name=name)
    logger.log_hyperparams({"a": 1, "b": 2, 123: 3, 3.5: 4, 5j: 5})  # Force data to be written
    assert os.path.normpath(logger.root_dir) == str(tmp_path)  # use os.path.normpath to handle trailing /
    assert os.listdir(tmp_path / "version_0")


def test_tensorboard_log_sub_dir(tmp_path):
    # no sub_dir specified
    root_dir = tmp_path / "logs"
    logger = TensorBoardLogger(root_dir, name="name", version="version")
    assert logger.log_dir == os.path.join(root_dir, "name", "version")

    # sub_dir specified
    logger = TensorBoardLogger(root_dir, name="name", version="version", sub_dir="sub_dir")
    assert logger.log_dir == os.path.join(root_dir, "name", "version", "sub_dir")


def test_tensorboard_expand_home():
    """Test that the home dir (`~`) gets expanded properly."""
    root_dir = "~/tmp"
    explicit_root_dir = os.path.expanduser(root_dir)
    logger = TensorBoardLogger(root_dir, name="name", version="version", sub_dir="sub_dir")
    assert logger.root_dir == root_dir
    assert logger.log_dir == os.path.join(explicit_root_dir, "name", "version", "sub_dir")


@mock.patch.dict(os.environ, {"TEST_ENV_DIR": "some_directory"})
def test_tensorboard_expand_env_vars():
    """Test that the env vars in path names (`$`) get handled properly."""
    test_env_dir = os.environ["TEST_ENV_DIR"]
    root_dir = "$TEST_ENV_DIR/tmp"
    explicit_root_dir = f"{test_env_dir}/tmp"
    logger = TensorBoardLogger(root_dir, name="name", version="version", sub_dir="sub_dir")
    assert logger.log_dir == os.path.join(explicit_root_dir, "name", "version", "sub_dir")


@pytest.mark.parametrize("step_idx", [10, None])
def test_tensorboard_log_metrics(tmp_path, step_idx):
    logger = TensorBoardLogger(tmp_path)
    metrics = {"float": 0.3, "int": 1, "FloatTensor": torch.tensor(0.1), "IntTensor": torch.tensor(1)}
    logger.log_metrics(metrics, step_idx)


def test_tensorboard_log_hyperparams(tmp_path):
    logger = TensorBoardLogger(tmp_path)
    hparams = {
        "float": 0.3,
        "int": 1,
        "string": "abc",
        "bool": True,
        "dict": {"a": {"b": "c"}},
        "list": [1, 2, 3],
        "namespace": Namespace(foo=Namespace(bar="buzz")),
        "layer": torch.nn.BatchNorm1d,
        "tensor": torch.empty(2, 2, 2),
        "array": np.empty([2, 2, 2]),
    }
    logger.log_hyperparams(hparams)


def test_tensorboard_log_hparams_and_metrics(tmp_path):
    logger = TensorBoardLogger(tmp_path, default_hp_metric=False)
    hparams = {
        "float": 0.3,
        "int": 1,
        "string": "abc",
        "bool": True,
        "dict": {"a": {"b": "c"}},
        "list": [1, 2, 3],
        "namespace": Namespace(foo=Namespace(bar="buzz")),
        "layer": torch.nn.BatchNorm1d,
        "tensor": torch.empty(2, 2, 2),
        "array": np.empty([2, 2, 2]),
    }
    metrics = {"abc": torch.tensor([0.54])}
    logger.log_hyperparams(hparams, metrics)


@pytest.mark.parametrize("example_input_array", [None, torch.rand(2, 32)])
def test_tensorboard_log_graph_plain_module(tmp_path, example_input_array):
    model = BoringModel()
    logger = TensorBoardLogger(tmp_path)
    logger._experiment = Mock()

    logger.log_graph(model, example_input_array)
    if example_input_array is not None:
        logger.experiment.add_graph.assert_called_with(model, example_input_array)
    else:
        logger.experiment.add_graph.assert_not_called()

    logger._experiment.reset_mock()

    wrapped = _FabricModule(model, strategy=Mock())
    logger.log_graph(wrapped, example_input_array)
    if example_input_array is not None:
        logger.experiment.add_graph.assert_called_with(model, example_input_array)


@pytest.mark.parametrize("example_input_array", [None, torch.rand(2, 32)])
def test_tensorboard_log_graph_with_batch_transfer_hooks(tmp_path, example_input_array):
    model = pytest.importorskip("lightning.pytorch.demos.boring_classes").BoringModel()
    logger = TensorBoardLogger(tmp_path)
    logger._experiment = Mock()

    with (
        mock.patch.object(model, "_on_before_batch_transfer", return_value=example_input_array) as before_mock,
        mock.patch.object(model, "_apply_batch_transfer_handler", return_value=example_input_array) as transfer_mock,
    ):
        logger.log_graph(model, example_input_array)
        logger._experiment.reset_mock()

        wrapped = _FabricModule(model, strategy=Mock())
        logger.log_graph(wrapped, example_input_array)

        if example_input_array is not None:
            assert before_mock.call_count == 2
            assert transfer_mock.call_count == 2
            logger.experiment.add_graph.assert_called_with(model, example_input_array)
        else:
            before_mock.assert_not_called()
            transfer_mock.assert_not_called()
            logger.experiment.add_graph.assert_not_called()


@pytest.mark.skipif(not _TENSORBOARD_AVAILABLE, reason="tensorboard is required")
def test_tensorboard_log_graph_warning_no_example_input_array(tmp_path):
    """Test that log graph throws warning if model.example_input_array is None."""
    model = BoringModel()
    model.example_input_array = None
    logger = TensorBoardLogger(tmp_path, log_graph=True)
    with pytest.warns(
        UserWarning,
        match="Could not log computational graph to TensorBoard: The `model.example_input_array` .* was not given",
    ):
        logger.log_graph(model)

    model.example_input_array = {"x": 1, "y": 2}
    with pytest.warns(
        UserWarning, match="Could not log computational graph to TensorBoard: .* can't be traced by TensorBoard"
    ):
        logger.log_graph(model)


def test_tensorboard_finalize(monkeypatch, tmp_path):
    """Test that the SummaryWriter closes in finalize."""
    if _TENSORBOARD_AVAILABLE:
        import torch.utils.tensorboard as tb
    else:
        import tensorboardX as tb

    monkeypatch.setattr(tb, "SummaryWriter", Mock())
    logger = TensorBoardLogger(root_dir=tmp_path)
    assert logger._experiment is None
    logger.finalize("any")

    # no log calls, no experiment created -> nothing to flush
    logger.experiment.assert_not_called()

    logger = TensorBoardLogger(root_dir=tmp_path)
    logger.log_metrics({"flush_me": 11.1})  # trigger creation of an experiment
    logger.finalize("any")

    # finalize flushes to experiment directory
    logger.experiment.flush.assert_called()
    logger.experiment.close.assert_called()


def test_tensorboard_with_symlink(tmp_path, monkeypatch):
    """Tests a specific failure case when tensorboard logger is used with empty name, symbolic link ``save_dir``, and
    relative paths."""
    monkeypatch.chdir(tmp_path)  # need to use relative paths
    source = os.path.join(".", "lightning_logs")
    dest = os.path.join(".", "sym_lightning_logs")

    os.makedirs(source, exist_ok=True)
    os.symlink(source, dest)

    logger = TensorBoardLogger(root_dir=dest, name="")
    _ = logger.version


def test_tensorboard_numpy_24_scalar_compatibility(tmp_path):
    """Test TensorBoard logger compatibility with numpy 2.4.0+ scalar handling.

    Addresses issue #21503: TensorBoard logging breaks with certain scalar values with numpy >= 2.4.0 due to changes in
    how .item() behaves on 0-dimensional arrays.

    """
    logger = TensorBoardLogger(root_dir=tmp_path, name="numpy_compat_test")

    # Test various numpy scalar types that could cause issues
    test_metrics = {
        "numpy_float64": np.array(3.14159),  # 0-dimensional float64 array
        "numpy_float32": np.array(2.71828, dtype=np.float32),  # 0-dimensional float32 array
        "numpy_int64": np.array(42),  # 0-dimensional int64 array
        "numpy_int32": np.array(123, dtype=np.int32),  # 0-dimensional int32 array
        "numpy_bool": np.array(True),  # 0-dimensional bool array
        "pytorch_tensor": torch.tensor(1.23),  # PyTorch tensor (should still work)
        "native_float": 4.56,  # Native Python float (control)
        "native_int": 789,  # Native Python int (control)
    }

    # All of these should log without raising exceptions
    logger.log_metrics(test_metrics, step=0)

    # Test with a mock that simulates numpy 2.4.0 TypeError behavior
    problematic_array = np.array(9.87654)

    # Temporarily replace the .item() method to raise TypeError like numpy 2.4.0
    def mock_item_raises_typeerror():
        raise TypeError("Cannot convert 0-d array to scalar")

    import unittest.mock

    with unittest.mock.patch.object(problematic_array, "item", side_effect=mock_item_raises_typeerror):
        # This should use the fallback mechanism and not raise an exception
        fallback_metrics = {"simulated_numpy24_error": problematic_array}
        logger.log_metrics(fallback_metrics, step=1)


def test_tensorboard_numpy_dtype_coverage(tmp_path):
    """Test TensorBoard logger with comprehensive numpy dtypes for robustness.

    Ensures that the numpy 2.4.0 compatibility fix works across all common numpy data types that users might log as
    metrics.

    """
    logger = TensorBoardLogger(root_dir=tmp_path, name="dtype_coverage_test")

    # Test comprehensive numpy data types
    numpy_types_metrics = {
        "float16": np.array(1.0, dtype=np.float16),
        "float32": np.array(2.0, dtype=np.float32),
        "float64": np.array(3.0, dtype=np.float64),
        "int8": np.array(4, dtype=np.int8),
        "int16": np.array(5, dtype=np.int16),
        "int32": np.array(6, dtype=np.int32),
        "int64": np.array(7, dtype=np.int64),
        "uint8": np.array(8, dtype=np.uint8),
        "uint16": np.array(9, dtype=np.uint16),
        "uint32": np.array(10, dtype=np.uint32),
        "uint64": np.array(11, dtype=np.uint64),
        "bool_true": np.array(True, dtype=bool),
        "bool_false": np.array(False, dtype=bool),
    }

    # All dtypes should log successfully without exceptions
    logger.log_metrics(numpy_types_metrics, step=0)
