#!/usr/bin/env python
"""Test reproduction for numpy 2.4.0+ TypeError issue with TensorBoard logging."""

import tempfile

import numpy as np
import torch

from lightning.fabric.loggers import TensorBoardLogger


def test_numpy_240_issue():
    """Reproduce the numpy 2.4.0+ TypeError issue with 0-dimensional arrays."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger = TensorBoardLogger(tmp_dir)

        # This should work fine - regular scalar
        logger.log_metrics({"scalar_float": 0.5}, step=1)
        logger.log_metrics({"scalar_int": 42}, step=1)

        # This should work fine - torch tensor
        logger.log_metrics({"tensor_scalar": torch.tensor(0.7)}, step=1)

        # This is what breaks in numpy >= 2.4.0
        # 0-dimensional numpy array
        zero_dim_array = np.array(0.8)
        print(f"Zero-dim array type: {type(zero_dim_array)}")
        print(f"Zero-dim array shape: {zero_dim_array.shape}")
        print(f"Zero-dim array ndim: {zero_dim_array.ndim}")

        try:
            # Test the .item() method that numpy 2.4.0 changed
            scalar_val = zero_dim_array.item()
            print(f"zero_dim_array.item() succeeded: {scalar_val}")
        except TypeError as e:
            print(f"zero_dim_array.item() failed with TypeError: {e}")

        # This might break with numpy >= 2.4.0 if the code tries to convert
        # numpy arrays the same way it does torch tensors
        try:
            logger.log_metrics({"numpy_0dim": zero_dim_array}, step=1)
            print("Logging 0-dimensional numpy array succeeded")
        except Exception as e:
            print(f"Logging 0-dimensional numpy array failed: {e}")

        # Also test 1-dimensional arrays with single elements
        one_dim_single = np.array([0.9])
        print(f"One-dim single element array type: {type(one_dim_single)}")
        print(f"One-dim single element array shape: {one_dim_single.shape}")

        try:
            logger.log_metrics({"numpy_1dim_single": one_dim_single}, step=1)
            print("Logging 1-dimensional single-element numpy array succeeded")
        except Exception as e:
            print(f"Logging 1-dimensional single-element numpy array failed: {e}")


if __name__ == "__main__":
    print(f"NumPy version: {np.__version__}")
    test_numpy_240_issue()
