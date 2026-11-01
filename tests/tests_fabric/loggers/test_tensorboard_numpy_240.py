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
"""Tests for numpy >= 2.4.0 compatibility in TensorBoard logger."""

from unittest import mock

import numpy as np
import pytest
import torch

from lightning.fabric.loggers import TensorBoardLogger


class MockArray:
    """Mock array-like object to test fallback behavior."""

    def __init__(self, value, shape=(), should_fail_item=False):
        self.value = value
        self.shape = shape
        self.ndim = len(shape)
        self.size = np.prod(shape) if shape else 1
        self.should_fail_item = should_fail_item

    def item(self):
        if self.should_fail_item:
            raise TypeError("Mock TypeError to simulate numpy >= 2.4.0 behavior")
        return self.value

    def __float__(self):
        return float(self.value)

    @property
    def flat(self):
        """Simple flat property for 1-dimensional case."""
        return [self.value] if self.ndim <= 1 else None


def test_tensorboard_log_metrics_numpy_arrays(tmp_path):
    """Test logging various numpy array types."""
    logger = TensorBoardLogger(tmp_path)

    # Test 0-dimensional arrays (scalars)
    zero_dim = np.array(0.5)
    logger.log_metrics({"numpy_0dim": zero_dim}, step=1)

    # Test 1-dimensional single-element arrays
    one_dim_single = np.array([0.7])
    logger.log_metrics({"numpy_1dim_single": one_dim_single}, step=1)

    # Test different numpy dtypes
    logger.log_metrics(
        {
            "numpy_float32": np.float32(0.8),
            "numpy_float64": np.float64(0.9),
            "numpy_int32": np.int32(42),
            "numpy_int64": np.int64(123),
        },
        step=2,
    )


def test_tensorboard_log_metrics_numpy_240_fallback(tmp_path):
    """Test fallback behavior for numpy >= 2.4.0 TypeError."""
    logger = TensorBoardLogger(tmp_path)

    # Test mock array that raises TypeError on .item() (simulates numpy >= 2.4.0)
    mock_0dim = MockArray(0.6, shape=(), should_fail_item=True)
    logger.log_metrics({"mock_0dim_fail": mock_0dim}, step=1)

    # Test mock 1-dimensional array that raises TypeError on .item()
    mock_1dim = MockArray(0.8, shape=(1,), should_fail_item=True)
    # Mock the flat property to return an indexable object
    with mock.patch.object(mock_1dim, "flat", [0.8]):
        logger.log_metrics({"mock_1dim_fail": mock_1dim}, step=1)


def test_tensorboard_log_metrics_invalid_arrays(tmp_path):
    """Test that multi-dimensional arrays raise appropriate errors."""
    logger = TensorBoardLogger(tmp_path)

    # Test 2D array - should fail
    two_dim = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="Cannot log multi-dimensional array"):
        logger.log_metrics({"numpy_2dim": two_dim}, step=1)

    # Test 1D array with multiple elements - should fail
    one_dim_multi = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="Cannot log multi-dimensional array"):
        logger.log_metrics({"numpy_1dim_multi": one_dim_multi}, step=1)


def test_tensorboard_log_metrics_torch_tensors_unchanged(tmp_path):
    """Test that torch tensor behavior remains unchanged."""
    logger = TensorBoardLogger(tmp_path)

    # Test various torch tensor types - these should work as before
    logger.log_metrics(
        {
            "torch_scalar": torch.tensor(0.5),
            "torch_0dim": torch.tensor(0.7),
            "torch_float32": torch.tensor(0.8, dtype=torch.float32),
            "torch_float64": torch.tensor(0.9, dtype=torch.float64),
        },
        step=1,
    )


def test_tensorboard_log_metrics_mixed_types(tmp_path):
    """Test logging mixed types in a single call."""
    logger = TensorBoardLogger(tmp_path)

    logger.log_metrics(
        {
            "python_float": 0.1,
            "python_int": 42,
            "numpy_0dim": np.array(0.2),
            "numpy_1dim_single": np.array([0.3]),
            "torch_tensor": torch.tensor(0.4),
        },
        step=1,
    )


def test_tensorboard_log_metrics_backwards_compatibility(tmp_path):
    """Test that existing functionality still works (backwards compatibility)."""
    logger = TensorBoardLogger(tmp_path)

    # All the existing test cases from the original test file should still work
    metrics = {"float": 0.3, "int": 1, "FloatTensor": torch.tensor(0.1), "IntTensor": torch.tensor(1)}
    logger.log_metrics(metrics, 10)

    # Test with None step
    logger.log_metrics(metrics, None)
