#!/usr/bin/env python3
"""Reproduces issue #21503: TensorBoard logging breaks with numpy >= 2.4.0.

The issue occurs when logging numpy scalars that are 0-dimensional arrays. In numpy 2.4.0, .item() on 0-dimensional
arrays raises TypeError.

"""

import tempfile

import numpy as np


def test_numpy_item_behavior():
    """Test how numpy 2.4.0 behaves with .item() on 0-dimensional arrays."""
    print(f"NumPy version: {np.__version__}")

    # Create a 0-dimensional numpy array (scalar)
    scalar_array = np.array(3.14)
    print(f"Scalar array: {scalar_array}")
    print(f"Scalar array shape: {scalar_array.shape}")
    print(f"Scalar array ndim: {scalar_array.ndim}")

    # Try to call .item() - this fails in numpy 2.4.0
    try:
        result = scalar_array.item()
        print(f"scalar_array.item() = {result} (SUCCESS)")
    except TypeError as e:
        print(f"scalar_array.item() FAILED: {e}")
        print("This is the numpy 2.4.0 behavior causing the Lightning issue")

        # Show the correct way to extract the scalar in numpy 2.4.0
        correct_result = float(scalar_array)
        print(f"Correct approach: float(scalar_array) = {correct_result}")


def test_tensorboard_logging_issue():
    """Test the specific TensorBoard logging scenario."""
    print("\n" + "=" * 50)
    print("Testing TensorBoard logging scenario")
    print("=" * 50)

    try:
        from lightning.fabric.loggers.tensorboard import TensorBoardLogger
    except ImportError:
        print("Lightning not installed - skipping TensorBoard test")
        return

    # Create a temporary directory for logs
    with tempfile.TemporaryDirectory() as temp_dir:
        logger = TensorBoardLogger(root_dir=temp_dir, name="test")

        # This is the problematic case: numpy 0-d array as metric value
        metrics = {
            "loss": np.array(0.5),  # 0-dimensional numpy array
            "accuracy": np.array(0.95),
        }

        print(f"Logging metrics: {metrics}")
        print(f"metric types: {[type(v) for v in metrics.values()]}")

        try:
            logger.log_metrics(metrics, step=0)
            print("TensorBoard logging SUCCESS")
        except Exception as e:
            print(f"TensorBoard logging FAILED: {e}")
            print(f"Exception type: {type(e)}")


if __name__ == "__main__":
    test_numpy_item_behavior()
    test_tensorboard_logging_issue()
