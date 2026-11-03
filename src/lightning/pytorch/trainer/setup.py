# Copyright Lightning AI.
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
"""Houses the methods used to set up the Trainer."""

from datetime import timedelta
from typing import Optional, Union

import lightning.pytorch as pl
from lightning.fabric.utilities.warnings import PossibleUserWarning
from lightning.pytorch.accelerators import CUDAAccelerator, MPSAccelerator, XLAAccelerator
from lightning.pytorch.loggers.logger import DummyLogger
from lightning.pytorch.profilers import (
    AdvancedProfiler,
    PassThroughProfiler,
    Profiler,
    PyTorchProfiler,
    SimpleProfiler,
    XLAProfiler,
)
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_warn


def _init_debugging_flags(
    trainer: "pl.Trainer",
    limit_train_batches: Optional[Union[int, float]],
    limit_val_batches: Optional[Union[int, float]],
    limit_test_batches: Optional[Union[int, float]],
    limit_predict_batches: Optional[Union[int, float]],
    fast_dev_run: Union[int, bool],
    overfit_batches: Union[int, float],
    val_check_interval: Optional[Union[int, float, str, timedelta, dict]],
    num_sanity_val_steps: int,
) -> None:
    # init debugging flags
    if isinstance(fast_dev_run, int) and (fast_dev_run < 0):
        raise MisconfigurationException(
            f"fast_dev_run={fast_dev_run!r} is not a valid configuration. It should be >= 0."
        )
    trainer.fast_dev_run = fast_dev_run

    # set fast_dev_run=True when it is 1, used while logging
    if fast_dev_run == 1:
        trainer.fast_dev_run = True

    trainer.overfit_batches = _determine_batch_limits(overfit_batches, "overfit_batches")
    overfit_batches_enabled = overfit_batches > 0

    if fast_dev_run:
        num_batches = int(fast_dev_run)
        if not overfit_batches_enabled:
            trainer.limit_train_batches = num_batches
            trainer.limit_val_batches = num_batches

        trainer.limit_test_batches = num_batches
        trainer.limit_predict_batches = num_batches
        trainer.fit_loop.epoch_loop.max_steps = num_batches
        trainer.num_sanity_val_steps = 0
        trainer.fit_loop.max_epochs = 1
        trainer.val_check_interval = 1.0
        trainer._val_check_time_interval = None  # time not applicable in fast_dev_run
        trainer.check_val_every_n_epoch = 1
        trainer.loggers = [DummyLogger()] if trainer.loggers else []
        rank_zero_info(
            f"Running in `fast_dev_run` mode: will run the requested loop using {num_batches} batch(es). "
            "Logging and checkpointing is suppressed."
        )
    else:
        if not overfit_batches_enabled:
            trainer.limit_train_batches = _determine_batch_limits(limit_train_batches, "limit_train_batches")
            trainer.limit_val_batches = _determine_batch_limits(limit_val_batches, "limit_val_batches")
        trainer.limit_test_batches = _determine_batch_limits(limit_test_batches, "limit_test_batches")
        trainer.limit_predict_batches = _determine_batch_limits(limit_predict_batches, "limit_predict_batches")
        trainer.num_sanity_val_steps = float("inf") if num_sanity_val_steps == -1 else num_sanity_val_steps
        # Support time-based validation intervals:
        # If `val_check_interval` is str/dict/timedelta, parse and store seconds on the trainer
        # for the loops to consume.
        trainer._val_check_time_interval = None  # default
        if isinstance(val_check_interval, (str, dict, timedelta)):
            trainer._val_check_time_interval = _parse_time_interval_seconds(val_check_interval)
        else:
            trainer.val_check_interval = _determine_batch_limits(val_check_interval, "val_check_interval")

    if overfit_batches_enabled:
        trainer.limit_train_batches = overfit_batches
        trainer.limit_val_batches = overfit_batches


def _determine_batch_limits(batches: Optional[Union[int, float]], name: str) -> Union[int, float]:
    if batches is None:
        # batches is optional to know if the user passed a value so that we can show the above info messages only to the
        # users that set a value explicitly
        return 1.0

    # differentiating based on the type can be error-prone for users. show a message describing the chosen behaviour
    if isinstance(batches, int) and batches == 1:
        if name == "limit_train_batches":
            message = "1 batch per epoch will be used."
        elif name == "val_check_interval":
            message = "validation will run after every batch."
        else:
            message = "1 batch will be used."
        rank_zero_info(f"`Trainer({name}=1)` was configured so {message}")
    elif isinstance(batches, float) and batches == 1.0:
        if name == "limit_train_batches":
            message = "100% of the batches per epoch will be used."
        elif name == "val_check_interval":
            message = "validation will run at the end of the training epoch."
        else:
            message = "100% of the batches will be used."
        rank_zero_info(f"`Trainer({name}=1.0)` was configured so {message}.")

    if 0 <= batches <= 1:
        return batches
    if batches > 1 and batches % 1.0 == 0:
        return int(batches)
    raise MisconfigurationException(
        f"You have passed invalid value {batches} for {name}, it has to be in [0.0, 1.0] or an int."
    )


def _init_profiler(trainer: "pl.Trainer", profiler: Optional[Union[Profiler, str]]) -> None:
    if isinstance(profiler, str):
        PROFILERS = {
            "simple": SimpleProfiler,
            "advanced": AdvancedProfiler,
            "pytorch": PyTorchProfiler,
            "xla": XLAProfiler,
        }
        profiler = profiler.lower()
        if profiler not in PROFILERS:
            raise MisconfigurationException(
                "When passing string value for the `profiler` parameter of `Trainer`,"
                f" it can only be one of {list(PROFILERS.keys())}"
            )
        profiler_class = PROFILERS[profiler]
        profiler = profiler_class()
    trainer.profiler = profiler or PassThroughProfiler()


def _log_device_info(trainer: "pl.Trainer") -> None:
    if not getattr(trainer, "enable_device_summary", True):
        return

    if CUDAAccelerator.is_available():
        gpu_available = True
        gpu_type = " (cuda)"
    elif MPSAccelerator.is_available():
        gpu_available = True
        gpu_type = " (mps)"
    else:
        gpu_available = False
        gpu_type = ""

    gpu_used = isinstance(trainer.accelerator, (CUDAAccelerator, MPSAccelerator))
    rank_zero_info(f"GPU available: {gpu_available}{gpu_type}, used: {gpu_used}")

    num_tpu_cores = trainer.num_devices if isinstance(trainer.accelerator, XLAAccelerator) else 0
    rank_zero_info(f"TPU available: {XLAAccelerator.is_available()}, using: {num_tpu_cores} TPU cores")

    if (
        CUDAAccelerator.is_available()
        and not isinstance(trainer.accelerator, CUDAAccelerator)
        or MPSAccelerator.is_available()
        and not isinstance(trainer.accelerator, MPSAccelerator)
    ):
        rank_zero_warn(
            "GPU available but not used. You can set it by doing `Trainer(accelerator='gpu')`.",
            category=PossibleUserWarning,
        )

    if XLAAccelerator.is_available() and not isinstance(trainer.accelerator, XLAAccelerator):
        rank_zero_warn("TPU available but not used. You can set it by doing `Trainer(accelerator='tpu')`.")


def _parse_time_interval_seconds(value: Union[str, timedelta, dict]) -> float:
    """Convert a time interval into seconds.

    This helper parses different representations of a time interval and
    normalizes them into a float number of seconds.

    Supported input formats:
      * `timedelta`: The total seconds are returned directly.
      * `dict`: A dictionary of keyword arguments accepted by
        `datetime.timedelta`, e.g. `{"days": 1, "hours": 2}`.
      * `str`: A string in the format `"DD:HH:MM:SS"`, where each
        component must be an integer.

    Args:
        value (Union[str, timedelta, dict]): The time interval to parse.

    Returns:
        float: The duration represented by `value` in seconds.

    Raises:
        MisconfigurationException: If the input type is unsupported, the
        string format is invalid, or any string component is not an integer.

    Examples:
        >>> _parse_time_interval_seconds("01:02:03:04")
        93784.0

        >>> _parse_time_interval_seconds({"hours": 2, "minutes": 30})
        9000.0

        >>> from datetime import timedelta
        >>> _parse_time_interval_seconds(timedelta(days=1, seconds=30))
        86430.0

    """
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, dict):
        td = timedelta(**value)
        return td.total_seconds()
    if isinstance(value, str):
        parts = value.split(":")
        if len(parts) != 4:
            raise MisconfigurationException(
                f"Invalid time format for `val_check_interval`: {value!r}. Expected 'DD:HH:MM:SS'."
            )
        d, h, m, s = parts
        try:
            days = int(d)
            hours = int(h)
            minutes = int(m)
            seconds = int(s)
        except ValueError:
            raise MisconfigurationException(
                f"Non-integer component in `val_check_interval` string: {value!r}. Use 'DD:HH:MM:SS'."
            )
        td = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
        return td.total_seconds()
    # Should not happen given the caller guards
    raise MisconfigurationException(f"Unsupported type for `val_check_interval`: {type(value)!r}")
