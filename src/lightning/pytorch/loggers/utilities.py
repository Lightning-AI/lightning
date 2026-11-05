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
"""Utilities for loggers."""

import os
import stat
from typing import Any, Union

from torch import Tensor

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Checkpoint


def _version(loggers: list[Any], separator: str = "_") -> Union[int, str]:
    if len(loggers) == 1:
        return loggers[0].version
    # Concatenate versions together, removing duplicates and preserving order
    return separator.join(dict.fromkeys(str(logger.version) for logger in loggers))


def _scan_checkpoints(checkpoint_callback: Checkpoint, logged_model_time: dict) -> list[tuple[float, str, float, str]]:
    """Return the checkpoints to be logged.

    Args:
        checkpoint_callback: Checkpoint callback reference.
        logged_model_time: dictionary containing the logged model times.

    """
    # get checkpoints to be saved with associated score
    checkpoints = {}
    if hasattr(checkpoint_callback, "last_model_path") and hasattr(checkpoint_callback, "current_score"):
        checkpoints[checkpoint_callback.last_model_path] = (checkpoint_callback.current_score, "latest")

    if hasattr(checkpoint_callback, "best_model_path") and hasattr(checkpoint_callback, "best_model_score"):
        checkpoints[checkpoint_callback.best_model_path] = (checkpoint_callback.best_model_score, "best")

    if hasattr(checkpoint_callback, "best_k_models"):
        for key, value in checkpoint_callback.best_k_models.items():
            checkpoints[key] = (value, "best_k")

    unlogged_checkpoints: list[tuple[float, str, float, str]] = []
    for p, (s, tag) in checkpoints.items():
        if not p:
            continue
        try:
            st = os.stat(os.fspath(p))
        except (OSError, TypeError):
            continue
        if not stat.S_ISREG(st.st_mode):
            continue
        mtime = st.st_mtime
        if p not in logged_model_time or logged_model_time[p] < mtime:
            unlogged_checkpoints.append((mtime, p, s, tag))

    unlogged_checkpoints.sort()
    return unlogged_checkpoints


def _log_hyperparams(trainer: "pl.Trainer") -> None:
    if not trainer.loggers:
        return

    pl_module = trainer.lightning_module
    datamodule_log_hyperparams = trainer.datamodule._log_hyperparams if trainer.datamodule is not None else False

    hparams_initial = None
    if pl_module._log_hyperparams and datamodule_log_hyperparams:
        datamodule_hparams = trainer.datamodule.hparams_initial
        lightning_hparams = pl_module.hparams_initial
        inconsistent_keys = []
        for key in lightning_hparams.keys() & datamodule_hparams.keys():
            if key == "_class_path":
                # Skip LightningCLI's internal hparam
                continue
            lm_val, dm_val = lightning_hparams[key], datamodule_hparams[key]
            if (
                type(lm_val) != type(dm_val)  # noqa: E721
                or (isinstance(lm_val, Tensor) and id(lm_val) != id(dm_val))
                or lm_val != dm_val
            ):
                inconsistent_keys.append(key)
        if inconsistent_keys:
            raise RuntimeError(
                f"Error while merging hparams: the keys {inconsistent_keys} are present "
                "in both the LightningModule's and LightningDataModule's hparams "
                "but have different values."
            )
        hparams_initial = {**lightning_hparams, **datamodule_hparams}
    elif pl_module._log_hyperparams:
        hparams_initial = pl_module.hparams_initial
    elif datamodule_log_hyperparams:
        hparams_initial = trainer.datamodule.hparams_initial

    # Don't log LightningCLI's internal hparam
    if hparams_initial is not None:
        hparams_initial = {k: v for k, v in hparams_initial.items() if k != "_class_path"}

    for logger in trainer.loggers:
        if hparams_initial is not None:
            logger.log_hyperparams(hparams_initial)
        logger.log_graph(pl_module)
        logger.save()
