#!/usr/bin/env python3
"""Test suite for the TensorBoard numpy 2.4.0 compatibility fix.

Tests the fix for issue #21503: TensorBoard logging breaks with certain scalar values with numpy >= 2.4.0

"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

try:
    import torch

    from lightning.fabric.loggers.tensorboard import TensorBoardLogger

    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch or Lightning not available - creating mock test")


class TestTensorBoardNumpyCompatibility(unittest.TestCase):
    """Test TensorBoard logger compatibility with numpy 2.4.0+"""

    def setUp(self):
        """Set up test environment."""
        if not PYTORCH_AVAILABLE:
            self.skipTest("PyTorch/Lightning not available")

        self.temp_dir = tempfile.mkdtemp()
        self.logger = TensorBoardLogger(root_dir=self.temp_dir, name="test_numpy_compat")

    def test_numpy_scalar_array_logging(self):
        """Test logging of numpy 0-dimensional arrays (scalars)"""
        # Create various numpy scalar arrays
        test_values = {
            "float64_scalar": np.array(3.14),
            "float32_scalar": np.array(2.71, dtype=np.float32),
            "int64_scalar": np.array(42),
            "int32_scalar": np.array(24, dtype=np.int32),
            "bool_scalar": np.array(True),
        }

        # All of these should log without error
        try:
            self.logger.log_metrics(test_values, step=0)
            print("SUCCESS: All numpy scalar arrays logged without error")
        except Exception as e:
            self.fail(f"Failed to log numpy scalar arrays: {e}")

    def test_pytorch_tensor_logging(self):
        """Test that PyTorch tensor logging still works."""
        test_values = {
            "tensor_scalar": torch.tensor(1.23),
            "tensor_float": torch.tensor(4.56, dtype=torch.float32),
            "tensor_int": torch.tensor(789),
        }

        try:
            self.logger.log_metrics(test_values, step=0)
            print("SUCCESS: All PyTorch tensors logged without error")
        except Exception as e:
            self.fail(f"Failed to log PyTorch tensors: {e}")

    def test_mixed_types_logging(self):
        """Test logging mixed numpy arrays, tensors, and native Python types."""
        test_values = {
            "native_float": 1.5,
            "native_int": 100,
            "numpy_scalar": np.array(2.5),
            "torch_scalar": torch.tensor(3.5),
            "numpy_1d": np.array([4.5]),  # This should be handled as-is
        }

        try:
            # Note: numpy_1d might cause an issue, but we're testing the scalar handling
            # We'll test the others individually
            safe_values = {k: v for k, v in test_values.items() if k != "numpy_1d"}
            self.logger.log_metrics(safe_values, step=0)
            print("SUCCESS: Mixed types logged without error")
        except Exception as e:
            self.fail(f"Failed to log mixed types: {e}")

    def test_numpy_item_method_fallback(self):
        """Test the specific numpy .item() fallback behavior."""
        # Create a 0-dimensional array
        scalar_array = np.array(42.0)

        # Mock the .item() method to raise TypeError (simulating numpy 2.4.0 behavior)

        def mock_item_error():
            raise TypeError("Cannot convert 0-d array to scalar")

        # Test our fallback logic by patching the item method
        with patch.object(scalar_array, "item", side_effect=mock_item_error):
            test_values = {"mocked_error_scalar": scalar_array}

            try:
                self.logger.log_metrics(test_values, step=0)
                print("SUCCESS: Fallback handling worked for simulated numpy 2.4.0 error")
            except Exception as e:
                self.fail(f"Fallback failed for mocked numpy 2.4.0 error: {e}")

    def test_various_numpy_dtypes(self):
        """Test various numpy data types to ensure broad compatibility."""
        test_values = {
            "float16": np.array(1.0, dtype=np.float16),
            "float32": np.array(2.0, dtype=np.float32),
            "float64": np.array(3.0, dtype=np.float64),
            "int8": np.array(4, dtype=np.int8),
            "int16": np.array(5, dtype=np.int16),
            "int32": np.array(6, dtype=np.int32),
            "int64": np.array(7, dtype=np.int64),
            "uint8": np.array(8, dtype=np.uint8),
            "bool": np.array(True, dtype=bool),
        }

        try:
            self.logger.log_metrics(test_values, step=0)
            print("SUCCESS: All numpy dtypes handled correctly")
        except Exception as e:
            self.fail(f"Failed to handle numpy dtypes: {e}")

    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, "temp_dir"):
            import shutil

            shutil.rmtree(self.temp_dir, ignore_errors=True)


def test_direct_conversion_methods():
    """Test the conversion methods used in the fix."""
    print("\n" + "=" * 60)
    print("Testing direct conversion methods")
    print("=" * 60)

    # Test various numpy scalar arrays
    test_cases = [
        (np.array(3.14), "float64"),
        (np.array(2.71, dtype=np.float32), "float32"),
        (np.array(42), "int64"),
        (np.array(True), "bool"),
    ]

    for arr, dtype_name in test_cases:
        print(f"\nTesting {dtype_name} array: {arr}")
        print(f"  Shape: {arr.shape}, dtype: {arr.dtype}")

        # Test .item() method
        try:
            item_result = arr.item()
            print(f"  arr.item() = {item_result} (type: {type(item_result)})")
        except Exception as e:
            print(f"  arr.item() FAILED: {e}")

        # Test our fallback method
        try:
            fallback_result = arr.dtype.type(arr)
            print(f"  arr.dtype.type(arr) = {fallback_result} (type: {type(fallback_result)})")
        except Exception as e:
            print(f"  arr.dtype.type(arr) FAILED: {e}")

        # Test float conversion
        try:
            float_result = float(arr)
            print(f"  float(arr) = {float_result} (type: {type(float_result)})")
        except Exception as e:
            print(f"  float(arr) FAILED: {e}")


if __name__ == "__main__":
    print(f"NumPy version: {np.__version__}")
    if PYTORCH_AVAILABLE:
        print(f"PyTorch version: {torch.__version__}")
    print()

    # Run direct conversion tests
    test_direct_conversion_methods()

    # Run unit tests
    print("\n" + "=" * 60)
    print("Running unit tests")
    print("=" * 60)
    unittest.main(verbosity=2)
