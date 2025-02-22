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
"""
CSV logger
----------

CSV logger for basic experiment logging that does not require opening ports

"""

import os
from argparse import Namespace
from typing import Any, Optional, Union

from typing_extensions import override

from lightning.fabric.loggers.csv_logs import CSVLogger as FabricCSVLogger
from lightning.fabric.loggers.csv_logs import _ExperimentWriter as _FabricExperimentWriter
from lightning.fabric.loggers.logger import rank_zero_experiment
from lightning.fabric.utilities.logger import _convert_params
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.core.saving import save_hparams_to_yaml
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities.rank_zero import rank_zero_only


class ExperimentWriter(_FabricExperimentWriter):
    r"""Experiment writer for CSVLogger.

    Currently, supports to log hyperparameters and metrics in YAML and CSV
    format, respectively.

    This logger supports logging to remote filesystems via ``fsspec``. Make sure you have it installed.

    Args:
        log_dir: Directory for the experiment logs

    """

    NAME_HPARAMS_FILE = "hparams.yaml"

    def __init__(self, log_dir: str) -> None:
        super().__init__(log_dir=log_dir)
        self.hparams: dict[str, Any] = {}

    def log_hparams(self, params: dict[str, Any]) -> None:
        """Record hparams and save into files."""
        self.hparams.update(params)
        hparams_file = os.path.join(self.log_dir, self.NAME_HPARAMS_FILE)
        save_hparams_to_yaml(hparams_file, self.hparams)


class CSVLogger(Logger, FabricCSVLogger):
    r"""Log to local file system in yaml and CSV format.

    Logs are saved to ``os.path.join(save_dir, name, version)``.

    Example:
        >>> from lightning.pytorch import Trainer
        >>> from lightning.pytorch.loggers import CSVLogger
        >>> logger = CSVLogger("logs", name="my_exp_name")
        >>> trainer = Trainer(logger=logger)

    Args:
        save_dir: Save directory
        name: Experiment name, optional. Defaults to ``'lightning_logs'``. If name is ``None``, logs
            (versions) will be stored to the save dir directly.
        version: Experiment version. If version is not specified the logger inspects the save
            directory for existing versions, then automatically assigns the next available version.
        prefix: A string to put at the beginning of metric keys.
        flush_logs_every_n_steps: How often to flush logs to disk (defaults to every 100 steps).

    """

    LOGGER_JOIN_CHAR = "-"

    def __init__(
        self,
        save_dir: _PATH,
        name: Optional[str] = "lightning_logs",
        version: Optional[Union[int, str]] = None,
        prefix: str = "",
        flush_logs_every_n_steps: int = 100,
    ):
        super().__init__(
            root_dir=save_dir,
            name=name,
            version=version,
            prefix=prefix,
            flush_logs_every_n_steps=flush_logs_every_n_steps,
        )
        self._save_dir = os.fspath(save_dir)

    @property
    @override
    def root_dir(self) -> str:
        """Parent directory for all checkpoint subdirectories.

        If the experiment name parameter is an empty string, no experiment subdirectory is used and the checkpoint will
        be saved in "save_dir/version"

        """
        return os.path.join(self.save_dir, self.name)

    @property
    @override
    def log_dir(self) -> str:
        """The log directory for this run.

        By default, it is named ``'version_${self.version}'`` but it can be overridden by passing a string value for the
        constructor's version parameter instead of ``None`` or an int.

        """
        # create a pseudo standard path
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        return os.path.join(self.root_dir, version)

    @property
    @override
    def save_dir(self) -> str:
        """The current directory where logs are saved.

        Returns:
            The path to current directory where logs are saved.

        """
        return self._save_dir

    @override
    @rank_zero_only
    def log_hyperparams(self, params: Optional[Union[dict[str, Any], Namespace]] = None) -> None:
        params = _convert_params(params)
        self.experiment.log_hparams(params)

    @property
    @override
    @rank_zero_experiment
    def experiment(self) -> _FabricExperimentWriter:
        r"""Actual _ExperimentWriter object. To use _ExperimentWriter features in your
        :class:`~lightning.pytorch.core.LightningModule` do the following.

        Example::

            self.logger.experiment.some_experiment_writer_function()

        """
        if self._experiment is not None:
            return self._experiment

        self._fs.makedirs(self.root_dir, exist_ok=True)
        self._experiment = ExperimentWriter(log_dir=self.log_dir)
        return self._experiment
