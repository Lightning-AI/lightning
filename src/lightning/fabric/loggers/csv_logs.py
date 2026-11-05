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

import csv
import logging
import os
from argparse import Namespace
from typing import Any, Optional, Union

from torch import Tensor
from typing_extensions import override

from lightning.fabric.loggers.logger import Logger, rank_zero_experiment
from lightning.fabric.utilities.cloud_io import _is_dir, get_filesystem
from lightning.fabric.utilities.logger import _add_prefix
from lightning.fabric.utilities.rank_zero import rank_zero_only, rank_zero_warn
from lightning.fabric.utilities.types import _PATH

log = logging.getLogger(__name__)


class CSVLogger(Logger):
    r"""Log to the local file system in CSV format.

    Logs are saved to ``os.path.join(root_dir, name, version)``.

    Args:
        root_dir: The root directory in which all your experiments with different names and versions will be stored.
        name: Experiment name. Defaults to ``'lightning_logs'``. If name is ``None``, logs
            (versions) will be stored to the save dir directly.
        version: Experiment version. If version is not specified the logger inspects the save
            directory for existing versions, then automatically assigns the next available version.
            If the version is specified, and the directory already contains a metrics file for that version, it will be
            overwritten.
        prefix: A string to put at the beginning of metric keys.
        flush_logs_every_n_steps: How often to flush logs to disk (defaults to every 100 steps).

    Example::

        from lightning.fabric.loggers import CSVLogger

        logger = CSVLogger("path/to/logs/root", name="my_model")
        logger.log_metrics({"loss": 0.235, "acc": 0.75})
        logger.finalize("success")

    """

    LOGGER_JOIN_CHAR = "-"

    def __init__(
        self,
        root_dir: _PATH,
        name: Optional[str] = "lightning_logs",
        version: Optional[Union[int, str]] = None,
        prefix: str = "",
        flush_logs_every_n_steps: int = 100,
    ):
        super().__init__()
        root_dir = os.fspath(root_dir)
        self._root_dir = root_dir
        self._name = name or ""
        self._version = version
        self._prefix = prefix
        self._fs = get_filesystem(root_dir)
        self._experiment: Optional[_ExperimentWriter] = None
        self._flush_logs_every_n_steps = flush_logs_every_n_steps

    @property
    @override
    def name(self) -> str:
        """Gets the name of the experiment.

        Returns:
            The name of the experiment.

        """
        return self._name

    @property
    @override
    def version(self) -> Union[int, str]:
        """Gets the version of the experiment.

        Returns:
            The version of the experiment if it is specified, else the next version.

        """
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    @property
    @override
    def root_dir(self) -> str:
        """Gets the save directory where the versioned CSV experiments are saved."""
        return self._root_dir

    @property
    @override
    def log_dir(self) -> str:
        """The log directory for this run.

        By default, it is named ``'version_${self.version}'`` but it can be overridden by passing a string value for the
        constructor's version parameter instead of ``None`` or an int.

        """
        # create a pseudo standard path
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        return os.path.join(self._root_dir, self.name, version)

    @property
    @rank_zero_experiment
    def experiment(self) -> "_ExperimentWriter":
        """Actual ExperimentWriter object. To use ExperimentWriter features anywhere in your code, do the following.

        Example::

            self.logger.experiment.some_experiment_writer_function()

        """
        if self._experiment is not None:
            return self._experiment

        os.makedirs(self._root_dir, exist_ok=True)
        self._experiment = _ExperimentWriter(log_dir=self.log_dir)
        return self._experiment

    @override
    @rank_zero_only
    def log_hyperparams(self, params: Union[dict[str, Any], Namespace]) -> None:
        raise NotImplementedError("The `CSVLogger` does not yet support logging hyperparameters.")

    @override
    @rank_zero_only
    def log_metrics(  # type: ignore[override]
        self, metrics: dict[str, Union[Tensor, float]], step: Optional[int] = None
    ) -> None:
        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)
        if step is None:
            step = self.experiment._step_idx
        self.experiment.log_metrics(metrics, step)
        if (step + 1) % self._flush_logs_every_n_steps == 0:
            self.save()

    @override
    @rank_zero_only
    def save(self) -> None:
        super().save()
        self.experiment.save()

    @override
    @rank_zero_only
    def finalize(self, status: str) -> None:
        if self._experiment is None:
            # When using multiprocessing, finalize() should be a no-op on the main process, as no experiment has been
            # initialized there
            return
        self.save()

    def _get_next_version(self) -> int:
        versions_root = os.path.join(self._root_dir, self.name)

        if not _is_dir(self._fs, versions_root, strict=True):
            return 0

        existing_versions = []
        for d in self._fs.listdir(versions_root):
            full_path = d["name"]
            name = os.path.basename(full_path)
            if _is_dir(self._fs, full_path) and name.startswith("version_"):
                dir_ver = name.split("_")[1]
                if dir_ver.isdigit():
                    existing_versions.append(int(dir_ver))

        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1


class _ExperimentWriter:
    r"""Experiment writer for CSVLogger.

    Args:
        log_dir: Directory for the experiment logs

    """

    NAME_METRICS_FILE = "metrics.csv"

    def __init__(self, log_dir: str) -> None:
        self.metrics: list[dict[str, float]] = []
        self.metrics_keys: list[str] = []
        self._step_idx: int = 0

        self._fs = get_filesystem(log_dir)
        self.log_dir = log_dir
        self.metrics_file_path = os.path.join(self.log_dir, self.NAME_METRICS_FILE)

        self._check_log_dir_exists()
        self._fs.makedirs(self.log_dir, exist_ok=True)

    def log_metrics(self, metrics_dict: dict[str, float], step: Optional[int] = None) -> None:
        """Record metrics."""

        def _handle_value(value: Union[Tensor, Any]) -> Any:
            if isinstance(value, Tensor):
                return value.item()
            return value

        if step is None:
            step = self._step_idx

        metrics = {k: _handle_value(v) for k, v in metrics_dict.items()}
        metrics["step"] = step
        self.metrics.append(metrics)
        self._step_idx += 1

    def save(self) -> None:
        """Save recorded metrics into files."""
        if not self.metrics:
            return

        new_keys = self._record_new_keys()
        file_exists = self._fs.isfile(self.metrics_file_path)

        if new_keys and file_exists:
            # we need to re-write the file if the keys (header) change
            self._rewrite_with_new_header(self.metrics_keys)

        with self._fs.open(self.metrics_file_path, mode=("a" if file_exists else "w"), newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.metrics_keys)
            if not file_exists:
                # only write the header if we're writing a fresh file
                writer.writeheader()
            writer.writerows(self.metrics)

        self.metrics = []  # reset

    def _record_new_keys(self) -> set[str]:
        """Records new keys that have not been logged before."""
        current_keys = set().union(*self.metrics)
        new_keys = current_keys - set(self.metrics_keys)
        if new_keys:
            self.metrics_keys.extend(new_keys)
            self.metrics_keys.sort()
        return new_keys

    def _rewrite_with_new_header(self, fieldnames: list[str]) -> None:
        with self._fs.open(self.metrics_file_path, "r", newline="") as file:
            metrics = list(csv.DictReader(file))

        with self._fs.open(self.metrics_file_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics)

    def _check_log_dir_exists(self) -> None:
        if self._fs.exists(self.log_dir) and self._fs.listdir(self.log_dir):
            rank_zero_warn(
                f"Experiment logs directory {self.log_dir} exists and is not empty."
                " Previous log files in this directory will be deleted when the new ones are saved!"
            )
            if self._fs.isfile(self.metrics_file_path):
                self._fs.rm_file(self.metrics_file_path)
