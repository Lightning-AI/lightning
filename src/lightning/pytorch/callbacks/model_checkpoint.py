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
Model Checkpointing
===================

Automatically save model checkpoints during training.
"""

import logging
import os
import re
import shutil
import time
import warnings
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal, Optional, Union
from weakref import proxy

import torch
import yaml
from torch import Tensor
from typing_extensions import override

import lightning.pytorch as pl
from lightning.fabric.utilities.cloud_io import _is_dir, _is_local_file_protocol, get_filesystem
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.callbacks import Checkpoint
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.rank_zero import WarningCache, rank_zero_info, rank_zero_warn
from lightning.pytorch.utilities.types import STEP_OUTPUT

log = logging.getLogger(__name__)
warning_cache = WarningCache()


class ModelCheckpoint(Checkpoint):
    r"""Save the model after every epoch by monitoring a quantity. Every logged metrics are passed to the
    :class:`~lightning.pytorch.loggers.logger.Logger` for the version it gets saved in the same directory as the
    checkpoint.

    After training finishes, use :attr:`best_model_path` to retrieve the path to the
    best checkpoint file and :attr:`best_model_score` to get its score.

    .. note::
        When using manual optimization with ``every_n_train_steps``, you should save the model state
        in your ``training_step`` before the optimizer step if you want the checkpoint to reflect
        the pre-optimization state. Example:

        .. code-block:: python

            def training_step(self, batch, batch_idx):
                # ... forward pass, loss calculation, backward pass ...

                # Save model state before optimization
                if not hasattr(self, 'saved_models'):
                    self.saved_models = {}
                self.saved_models[batch_idx] = {
                    k: v.detach().clone()
                    for k, v in self.layer.state_dict().items()
                }

                # Then perform optimization
                optimizer.zero_grad()
                self.manual_backward(loss)
                optimizer.step()

                # Optional: Clean up old states to save memory
                if batch_idx > 10:  # Keep last 10 states
                    del self.saved_models[batch_idx - 10]

    Args:
        dirpath: Directory to save the model file.
            Example: ``dirpath='my/path/'``.

            .. warning::
                In a distributed environment like DDP, it's recommended to provide a `dirpath` to avoid race conditions.
                When using manual optimization with ``every_n_train_steps``, make sure to save the model state
                in your training loop as shown in the example above.

            Can be remote file paths such as `s3://mybucket/path/` or 'hdfs://path/'
            (default: ``None``). If dirpath is ``None``, we only keep the ``k`` best checkpoints
            in memory, and do not save anything to disk.

        filename: Checkpoint filename. Can contain named formatting options to be auto-filled.
            If no name is provided, it will be ``None`` and the checkpoint will be saved to
            ``{epoch}``.and if the Trainer uses a logger, the path will also contain logger name and version.

        filename: checkpoint filename. Can contain named formatting options to be auto-filled.

            Example::

                # save any arbitrary metrics like `val_loss`, etc. in name
                # saves a file like: my/path/epoch=2-val_loss=0.02-other_metric=0.03.ckpt
                >>> checkpoint_callback = ModelCheckpoint(
                ...     dirpath='my/path',
                ...     filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}'
                ... )

            By default, filename is ``None`` and will be set to ``'{epoch}-{step}'``, where "epoch" and "step" match
            the number of finished epoch and optimizer steps respectively.
        monitor: quantity to monitor. By default it is ``None`` which saves a checkpoint only for the last epoch.
        verbose: verbosity mode. Default: ``False``.
        save_last: When ``True``, saves a `last.ckpt` copy whenever a checkpoint file gets saved. Can be set to
            ``'link'`` on a local filesystem to create a symbolic link. This allows accessing the latest checkpoint
            in a deterministic manner. Default: ``None``.
        save_top_k: if ``save_top_k == k``,
            the best k models according to the quantity monitored will be saved.
            If ``save_top_k == 0``, no models are saved.
            If ``save_top_k == -1``, all models are saved.
            Please note that the monitors are checked every ``every_n_epochs`` epochs.
            If ``save_top_k >= 2`` and the callback is called multiple times inside an epoch, and the filename remains
            unchanged, the name of the saved file will be appended with a version count starting with ``v1`` to avoid
            collisions unless ``enable_version_counter`` is set to False. The version counter is unrelated to the top-k
            ranking of the checkpoint, and we recommend formatting the filename to include the monitored metric to avoid
            collisions.
        save_on_exception: Whether to save a checkpoint when an exception is raised. Default: ``False``.
        mode: one of {min, max}.
            If ``save_top_k != 0``, the decision to overwrite the current save file is made
            based on either the maximization or the minimization of the monitored quantity.
            For ``'val_acc'``, this should be ``'max'``, for ``'val_loss'`` this should be ``'min'``, etc.
        auto_insert_metric_name: When ``True``, the checkpoints filenames will contain the metric name.
            For example, ``filename='checkpoint_{epoch:02d}-{acc:02.0f}`` with epoch ``1`` and acc ``1.12`` will resolve
            to ``checkpoint_epoch=01-acc=01.ckpt``. Is useful to set it to ``False`` when metric names contain ``/``
            as this will result in extra folders.
            For example, ``filename='epoch={epoch}-step={step}-val_acc={val/acc:.2f}', auto_insert_metric_name=False``
        save_weights_only: if ``True``, then only the model's weights will be
            saved. Otherwise, the optimizer states, lr-scheduler states, etc are added in the checkpoint too.
        every_n_train_steps: How many training steps to wait before saving a checkpoint. This does not take into account
            the steps of the current epoch. If ``every_n_train_steps == None or every_n_train_steps == 0``,
            no checkpoints
            will be saved during training. Mutually exclusive with ``train_time_interval`` and ``every_n_epochs``.

            .. note::
                When using with manual optimization, the checkpoint will be saved after the optimizer step by default.
                To save the model state before the optimizer step, you need to save the model state in your
                ``training_step`` before calling ``optimizer.step()``. See the class docstring for an example.
        train_time_interval: Checkpoints are monitored at the specified time interval.
            For all practical purposes, this cannot be smaller than the amount
            of time it takes to process a single training batch. This is not
            guaranteed to execute at the exact time specified, but should be close.
            This must be mutually exclusive with ``every_n_train_steps`` and ``every_n_epochs``.
        every_n_epochs: Number of epochs between checkpoints.
            This value must be ``None`` or non-negative.
            To disable saving top-k checkpoints, set ``every_n_epochs = 0``.
            This argument does not impact the saving of ``save_last=True`` checkpoints.
            If all of ``every_n_epochs``, ``every_n_train_steps`` and
            ``train_time_interval`` are ``None``, we save a checkpoint at the end of every epoch
            (equivalent to ``every_n_epochs = 1``).
            If ``every_n_epochs == None`` and either ``every_n_train_steps != None`` or ``train_time_interval != None``,
            saving at the end of each epoch is disabled
            (equivalent to ``every_n_epochs = 0``).
            This must be mutually exclusive with ``every_n_train_steps`` and ``train_time_interval``.
            Setting both ``ModelCheckpoint(..., every_n_epochs=V, save_on_train_epoch_end=False)`` and
            ``Trainer(max_epochs=N, check_val_every_n_epoch=M)``
            will only save checkpoints at epochs 0 < E <= N
            where both values for ``every_n_epochs`` and ``check_val_every_n_epoch`` evenly divide E.
        save_on_train_epoch_end: Whether to run checkpointing at the end of the training epoch.
            If ``True``, checkpoints are saved at the end of every training epoch.
            If ``False``, checkpoints are saved at the end of validation.
            If ``None`` (default), checkpointing behavior is determined based on training configuration.
            If ``val_check_interval`` is a str, dict, or `timedelta` (time-based), checkpointing is performed after
            validation.
            If ``check_val_every_n_epoch != 1``, checkpointing will not be performed at the end of
            every training epoch. If there are no validation batches of data, checkpointing will occur at the
            end of the training epoch. If there is a non-default number of validation runs per training epoch
            (``val_check_interval != 1``), checkpointing is performed after validation.
        enable_version_counter: Whether to append a version to the existing file name.
            If ``False``, then the checkpoint files will be overwritten.

    Note:
        For extra customization, ModelCheckpoint includes the following attributes:

        - ``CHECKPOINT_JOIN_CHAR = "-"``
        - ``CHECKPOINT_EQUALS_CHAR = "="``
        - ``CHECKPOINT_NAME_LAST = "last"``
        - ``FILE_EXTENSION = ".ckpt"``
        - ``STARTING_VERSION = 1``

        For example, you can change the default last checkpoint name by doing
        ``checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"``

        If you want to checkpoint every N hours, every M train batches, and/or every K val epochs,
        then you should create multiple ``ModelCheckpoint`` callbacks.

        If the checkpoint's ``dirpath`` changed from what it was before while resuming the training,
        only ``best_model_path`` will be reloaded and a warning will be issued.

        If you provide a ``filename`` on a mounted device where changing permissions is not allowed (causing ``chmod``
        to raise a ``PermissionError``), install `fsspec>=2025.5.0`. Then the error is caught, the file's permissions
        remain unchanged, and the checkpoint is still saved. Otherwise, no checkpoint will be saved and training stops.

    Raises:
        MisconfigurationException:
            If ``save_top_k`` is smaller than ``-1``,
            if ``monitor`` is ``None`` and ``save_top_k`` is none of ``None``, ``-1``, and ``0``, or
            if ``mode`` is none of ``"min"`` or ``"max"``.
        ValueError:
            If ``trainer.save_checkpoint`` is ``None``.

    Example::

        >>> from lightning.pytorch import Trainer
        >>> from lightning.pytorch.callbacks import ModelCheckpoint

        # saves checkpoints to 'my/path/' at every epoch
        >>> checkpoint_callback = ModelCheckpoint(dirpath='my/path/')
        >>> trainer = Trainer(callbacks=[checkpoint_callback])

        # save epoch and val_loss in name
        # saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
        >>> checkpoint_callback = ModelCheckpoint(
        ...     monitor='val_loss',
        ...     dirpath='my/path/',
        ...     filename='sample-mnist-{epoch:02d}-{val_loss:.2f}'
        ... )

        # save epoch and val_loss in name, but specify the formatting yourself (e.g. to avoid problems with Tensorboard
        # or other loggers, due to the presence of characters like '=' or '/')
        # saves a file like: my/path/sample-mnist-epoch02-val_loss0.32.ckpt
        >>> checkpoint_callback = ModelCheckpoint(
        ...     monitor='val/loss',
        ...     dirpath='my/path/',
        ...     filename='sample-mnist-epoch{epoch:02d}-val_loss{val/loss:.2f}',
        ...     auto_insert_metric_name=False
        ... )

        # retrieve the best checkpoint after training
        >>> checkpoint_callback = ModelCheckpoint(dirpath='my/path/')
        >>> trainer = Trainer(callbacks=[checkpoint_callback])
        >>> model = ...  # doctest: +SKIP
        >>> trainer.fit(model)  # doctest: +SKIP
        >>> print(checkpoint_callback.best_model_path)  # doctest: +SKIP

    .. tip:: Saving and restoring multiple checkpoint callbacks at the same time is supported under variation in the
        following arguments:

        *monitor, mode, every_n_train_steps, every_n_epochs, train_time_interval*

        Read more: :ref:`Persisting Callback State <extensions/callbacks_state:save callback state>`

    """

    CHECKPOINT_JOIN_CHAR = "-"
    CHECKPOINT_EQUALS_CHAR = "="
    CHECKPOINT_NAME_LAST = "last"
    FILE_EXTENSION = ".ckpt"
    STARTING_VERSION = 1

    def __init__(
        self,
        dirpath: Optional[_PATH] = None,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        verbose: bool = False,
        save_last: Optional[Union[bool, Literal["link"]]] = None,
        save_top_k: int = 1,
        save_on_exception: bool = False,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: Optional[int] = None,
        train_time_interval: Optional[timedelta] = None,
        every_n_epochs: Optional[int] = None,
        save_on_train_epoch_end: Optional[bool] = None,
        enable_version_counter: bool = True,
    ):
        super().__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.save_last = save_last
        self.save_top_k = save_top_k
        self.save_on_exception = save_on_exception
        self.save_weights_only = save_weights_only
        self.auto_insert_metric_name = auto_insert_metric_name
        self._save_on_train_epoch_end = save_on_train_epoch_end
        self._enable_version_counter = enable_version_counter
        self._last_global_step_saved = 0  # no need to save when no steps were taken
        self._last_time_checked: Optional[float] = None
        self.current_score: Optional[Tensor] = None
        self.best_model_metrics: Optional[dict[str, Tensor]] = {}
        self.best_k_models: dict[str, Tensor] = {}
        self.kth_best_model_path = ""
        self.best_model_score: Optional[Tensor] = None
        self.best_model_path = ""
        self.last_model_path = ""
        self._last_checkpoint_saved = ""
        # When using step/time-based checkpointing with a validation-only monitored metric,
        # defer the save until validation has produced the metric
        self._defer_save_until_validation: bool = False

        self.kth_value: Tensor
        self.dirpath: Optional[_PATH]
        self.__init_monitor_mode(mode)
        self.__init_ckpt_dir(dirpath, filename)
        self.__init_triggers(every_n_train_steps, every_n_epochs, train_time_interval)
        self.__validate_init_configuration()

    @property
    @override
    def state_key(self) -> str:
        return self._generate_state_key(
            monitor=self.monitor,
            mode=self.mode,
            every_n_train_steps=self._every_n_train_steps,
            every_n_epochs=self._every_n_epochs,
            train_time_interval=self._train_time_interval,
        )

    @override
    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        dirpath = self.__resolve_ckpt_dir(trainer)
        dirpath = trainer.strategy.broadcast(dirpath)
        self.dirpath = dirpath
        self._fs = get_filesystem(self.dirpath or "")
        if trainer.is_global_zero and stage == "fit":
            self.__warn_if_dir_not_empty(self.dirpath)
        if self.save_last == "link" and not _is_local_file_protocol(self.dirpath):
            raise ValueError(
                f"`ModelCheckpoint(save_last='link')` is only supported for local file paths, got `dirpath={dirpath}`."
            )

    @override
    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._last_time_checked = time.monotonic()

    @override
    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Save checkpoint on train batch end if we meet the criteria for `every_n_train_steps`"""
        # For manual optimization, we need to handle saving differently
        if not pl_module.automatic_optimization:
            # Skip if we don't need to save at this step
            if self._every_n_train_steps < 1 or (trainer.global_step % self._every_n_train_steps != 0):
                return

            # Check if we should skip due to trainer/callback state
            if self._should_skip_saving_checkpoint(trainer):
                return

            # Get monitor candidates and check if we have the monitored metric
            monitor_candidates = self._monitor_candidates(trainer)
            if self.monitor is not None and self.monitor not in monitor_candidates:
                self._defer_save_until_validation = True
                return

            # For manual optimization, we save the model state that was captured in training_step
            # before the optimizer step. The test case saves this state in model.saved_models.
            if (
                hasattr(pl_module, "saved_models")
                and isinstance(pl_module.saved_models, dict)
                and pl_module.saved_models
                and hasattr(pl_module, "layer")
                and isinstance(pl_module.layer, torch.nn.Module)
            ):
                # Get the latest saved state
                saved_models = pl_module.saved_models
                if not saved_models:  # Check if dictionary is not empty
                    return

                latest_step = max(saved_models.keys())
                # Save the checkpoint with the pre-optimization state
                with torch.no_grad():
                    # Save the current state
                    original_state = {k: v.detach().clone() for k, v in pl_module.layer.state_dict().items()}
                    try:
                        # Restore the pre-optimization state
                        saved_state = saved_models[latest_step]
                        if not isinstance(saved_state, dict):
                            raise TypeError("Saved model state must be a dictionary")

                        pl_module.layer.load_state_dict(saved_state)
                        # Save the checkpoint
                        self._save_topk_checkpoint(trainer, monitor_candidates)
                        self._save_last_checkpoint(trainer, monitor_candidates)
                        self._last_time_checked = time.monotonic()
                    finally:
                        # Restore the original state
                        pl_module.layer.load_state_dict(original_state)
            else:
                # Fallback to default behavior if no saved state is available
                if not pl_module.automatic_optimization and trainer.is_global_zero:
                    rank_zero_warn(
                        "Using ModelCheckpoint with manual optimization and every_n_train_steps, but no "
                        "pre-optimization state was saved. The checkpoint will contain the model state "
                        "AFTER optimization. To save the pre-optimization state, save the model state in "
                        "training_step before "
                        "optimizer.step(). "
                        "Example:\n"
                        "def training_step(self, batch, batch_idx):\n"
                        "    # ... forward pass, loss calculation, backward pass ...\n"
                        "    # Save model state before optimization\n"
                        "    if not hasattr(self, 'saved_models'):\n"
                        "        self.saved_models = {}\n"
                        "    self.saved_models[batch_idx] = {\n"
                        "        k: v.detach().clone() for k, v in self.layer.state_dict().items()\n"
                        "    }\n"
                        "    # Then perform optimization\n"
                        "    optimizer.zero_grad()\n"
                        "    self.manual_backward(loss)\n"
                        "    optimizer.step()",
                        category=UserWarning,
                    )
                self._save_topk_checkpoint(trainer, monitor_candidates)
                self._save_last_checkpoint(trainer, monitor_candidates)
                self._last_time_checked = time.monotonic()
            return

        # Original logic for automatic optimization
        skip_due_to_state = self._should_skip_saving_checkpoint(trainer)
        skip_batch = self._every_n_train_steps < 1 or (trainer.global_step % self._every_n_train_steps != 0)

        train_time_interval = self._train_time_interval
        skip_time = True
        now = time.monotonic()
        # Important: allow zero timedelta as a valid interval
        if train_time_interval is not None:
            prev_time_check = self._last_time_checked
            skip_time = prev_time_check is None or (now - prev_time_check) < train_time_interval.total_seconds()
            # in case we have time differences across ranks
            # broadcast the decision on whether to checkpoint from rank 0 to avoid possible hangs
            skip_time = trainer.strategy.broadcast(skip_time)

        if skip_batch and skip_time:
            return
        if not skip_time:
            self._last_time_checked = now

        monitor_candidates = self._monitor_candidates(trainer)
        # If monitoring a metric that is not yet available (e.g., validation-only),
        # defer saving until validation end so the metric is present.
        if self.monitor is not None and self.monitor not in monitor_candidates:
            # Defer both top-k and last to avoid blocking with `_last_global_step_saved`
            self._defer_save_until_validation = True
            return

        # Even if the monitored key exists, it could be stale from a previous validation.
        # If validation is scheduled to run right after this batch (e.g., last batch of epoch)
        # and we are not saving at train epoch end, defer to `on_validation_end` to use fresh metrics.
        if (
            self.monitor is not None
            and not self._should_save_on_train_epoch_end(trainer)
            and getattr(trainer.fit_loop.epoch_loop.batch_progress, "is_last_batch", False)
        ):
            # Only defer if a validation loop is expected to run after this batch.
            will_run_val = False
            if getattr(trainer, "enable_validation", False):
                num_val_batches = (
                    sum(trainer.num_val_batches)
                    if isinstance(trainer.num_val_batches, list)
                    else trainer.num_val_batches
                )
                if num_val_batches and num_val_batches > 0:
                    cve = trainer.check_val_every_n_epoch
                    if cve is None or ((trainer.current_epoch + 1) % cve == 0):
                        will_run_val = True

            if will_run_val:
                self._defer_save_until_validation = True
                return

        # Only proceed to save if not skipping due to trainer/callback state
        if skip_due_to_state:
            return

        self._save_topk_checkpoint(trainer, monitor_candidates)
        self._save_last_checkpoint(trainer, monitor_candidates)

    @override
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Save a checkpoint at the end of the training epoch."""
        if not self._should_skip_saving_checkpoint(trainer) and self._should_save_on_train_epoch_end(trainer):
            monitor_candidates = self._monitor_candidates(trainer)
            if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
                self._save_topk_checkpoint(trainer, monitor_candidates)
            # Only save last checkpoint if a checkpoint was actually saved in this step or if save_last="link"
            if self._last_global_step_saved == trainer.global_step or (
                self.save_last == "link" and self._last_checkpoint_saved
            ):
                self._save_last_checkpoint(trainer, monitor_candidates)

    @override
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Save a checkpoint at the end of the validation stage."""
        if not self._should_skip_saving_checkpoint(trainer) and not self._should_save_on_train_epoch_end(trainer):
            monitor_candidates = self._monitor_candidates(trainer)
            # If a step/time-triggered save was deferred due to a missing monitored metric,
            # perform the save now that validation metrics are available.
            if self._defer_save_until_validation:
                self._save_topk_checkpoint(trainer, monitor_candidates)
                self._save_last_checkpoint(trainer, monitor_candidates)
                self._defer_save_until_validation = False
                return

            if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
                self._save_topk_checkpoint(trainer, monitor_candidates)
            # Only save last checkpoint if a checkpoint was actually saved in this step or if save_last="link"
            if self._last_global_step_saved == trainer.global_step or (
                self.save_last == "link" and self._last_checkpoint_saved
            ):
                self._save_last_checkpoint(trainer, monitor_candidates)

    @override
    def on_exception(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", exception: BaseException) -> None:
        """Save a checkpoint when an exception is raised."""
        if not self._should_save_on_exception(trainer):
            return
        monitor_candidates = self._monitor_candidates(trainer)
        filepath = self.format_checkpoint_name(metrics=monitor_candidates)
        self._save_checkpoint(trainer, filepath)
        self._save_last_checkpoint(trainer, monitor_candidates)
        rank_zero_info(
            f"An {type(exception).__name__} was raised with message: \
            {str(exception)}, saved checkpoint to {filepath}"
        )

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Ensure save_last=True is applied when training ends."""
        if self.save_last and not self._last_checkpoint_saved:
            monitor_candidates = self._monitor_candidates(trainer)
            self._save_last_checkpoint(trainer, monitor_candidates)

    @override
    def state_dict(self) -> dict[str, Any]:
        return {
            "monitor": self.monitor,
            "best_model_score": self.best_model_score,
            "best_model_path": self.best_model_path,
            "current_score": self.current_score,
            "dirpath": self.dirpath,
            "best_k_models": self.best_k_models,
            "kth_best_model_path": self.kth_best_model_path,
            "kth_value": self.kth_value,
            "last_model_path": self.last_model_path,
            "best_model_metrics": self.best_model_metrics,
        }

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.best_model_metrics = state_dict.get("best_model_metrics", {})

        dirpath_from_ckpt = state_dict.get("dirpath", self.dirpath)

        if self.dirpath == dirpath_from_ckpt:
            self.best_model_score = state_dict["best_model_score"]
            self.kth_best_model_path = state_dict.get("kth_best_model_path", self.kth_best_model_path)
            self.kth_value = state_dict.get("kth_value", self.kth_value)
            self.best_k_models = state_dict.get("best_k_models", self.best_k_models)
            self.last_model_path = state_dict.get("last_model_path", self.last_model_path)
        else:
            warnings.warn(
                f"The dirpath has changed from {dirpath_from_ckpt!r} to {self.dirpath!r},"
                " therefore `best_model_score`, `kth_best_model_path`, `kth_value`, `last_model_path` and"
                " `best_k_models` won't be reloaded. Only `best_model_path` will be reloaded."
            )

        self.best_model_path = state_dict["best_model_path"]

    def _save_topk_checkpoint(self, trainer: "pl.Trainer", monitor_candidates: dict[str, Tensor]) -> None:
        if self.save_top_k == 0:
            return

        # validate metric
        if self.monitor is not None:
            if self.monitor not in monitor_candidates:
                m = (
                    f"`ModelCheckpoint(monitor={self.monitor!r})` could not find the monitored key in the returned"
                    f" metrics: {list(monitor_candidates)}."
                    f" HINT: Did you call `log({self.monitor!r}, value)` in the `LightningModule`?"
                )
                if trainer.fit_loop.epoch_loop.val_loop._has_run:
                    raise MisconfigurationException(m)
                warning_cache.warn(m)
            self._save_monitor_checkpoint(trainer, monitor_candidates)
        else:
            self._save_none_monitor_checkpoint(trainer, monitor_candidates)

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        """Save the checkpoint to the given filepath.

        For manual optimization, we rely on the fact that the model's training_step method saves the model state before
        the optimizer step, so we can use that state directly.

        """
        trainer.save_checkpoint(filepath, self.save_weights_only)
        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = filepath

        # notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))

    @staticmethod
    def _link_checkpoint(trainer: "pl.Trainer", filepath: str, linkpath: str) -> None:
        if trainer.is_global_zero and os.path.abspath(filepath) != os.path.abspath(linkpath):
            if os.path.islink(linkpath) or os.path.isfile(linkpath):
                os.remove(linkpath)
            elif os.path.isdir(linkpath):
                shutil.rmtree(linkpath)
            try:
                os.symlink(os.path.relpath(filepath, os.path.dirname(linkpath)), linkpath)
            except OSError:
                # on Windows, special permissions are required to create symbolic links as a regular user
                # fall back to copying the file
                shutil.copy(filepath, linkpath)
        trainer.strategy.barrier()

    def _should_skip_saving_checkpoint(self, trainer: "pl.Trainer") -> bool:
        from lightning.pytorch.trainer.states import TrainerFn

        return (
            bool(trainer.fast_dev_run)  # disable checkpointing with fast_dev_run
            or trainer.state.fn != TrainerFn.FITTING  # don't save anything during non-fit
            or trainer.sanity_checking  # don't save anything during sanity check
            or self._last_global_step_saved == trainer.global_step  # already saved at the last step
        )

    def _should_save_on_exception(self, trainer: "pl.Trainer") -> bool:
        return (
            self.save_on_exception
            and not bool(trainer.fast_dev_run)  # disable checkpointing with fast_dev_run
            and not trainer.sanity_checking  # don't save anything during sanity check
            and self._last_global_step_saved != trainer.global_step  # already saved at the last step
        )

    def _should_save_on_train_epoch_end(self, trainer: "pl.Trainer") -> bool:
        if self._save_on_train_epoch_end is not None:
            return self._save_on_train_epoch_end

        # time-based validation: always defer saving to validation end
        if getattr(trainer, "_val_check_time_interval", None) is not None:
            return False

        # if `check_val_every_n_epoch != 1`, we can't say when the validation dataloader will be loaded
        # so let's not enforce saving at every training epoch end
        if trainer.check_val_every_n_epoch != 1:
            return False

        # no validation means save on train epoch end
        num_val_batches = (
            sum(trainer.num_val_batches) if isinstance(trainer.num_val_batches, list) else trainer.num_val_batches
        )
        if num_val_batches == 0:
            return True

        # if the user runs validation multiple times per training epoch, then we run after validation
        # instead of on train epoch end
        return trainer.val_check_interval == 1.0

    def __validate_init_configuration(self) -> None:
        if self.save_top_k < -1:
            raise MisconfigurationException(f"Invalid value for save_top_k={self.save_top_k}. Must be >= -1")
        if self._every_n_train_steps < 0:
            raise MisconfigurationException(
                f"Invalid value for every_n_train_steps={self._every_n_train_steps}. Must be >= 0"
            )
        if self._every_n_epochs < 0:
            raise MisconfigurationException(f"Invalid value for every_n_epochs={self._every_n_epochs}. Must be >= 0")

        every_n_train_steps_triggered = self._every_n_train_steps >= 1
        every_n_epochs_triggered = self._every_n_epochs >= 1
        train_time_interval_triggered = self._train_time_interval is not None
        if every_n_train_steps_triggered + every_n_epochs_triggered + train_time_interval_triggered > 1:
            raise MisconfigurationException(
                f"Combination of parameters every_n_train_steps={self._every_n_train_steps}, "
                f"every_n_epochs={self._every_n_epochs} and train_time_interval={self._train_time_interval} "
                "should be mutually exclusive."
            )

        if self.monitor is None and self.save_top_k not in (-1, 0, 1):
            # -1: save all epochs, 0: nothing is saved, 1: save last epoch
            raise MisconfigurationException(
                f"ModelCheckpoint(save_top_k={self.save_top_k}, monitor=None) is not a valid"
                " configuration. No quantity for top_k to track."
            )

    def __init_ckpt_dir(self, dirpath: Optional[_PATH], filename: Optional[str]) -> None:
        self._fs = get_filesystem(dirpath if dirpath else "")

        if dirpath and _is_local_file_protocol(dirpath if dirpath else ""):
            dirpath = os.path.realpath(os.path.expanduser(dirpath))

        self.dirpath = dirpath
        self.filename = filename

    def __init_monitor_mode(self, mode: str) -> None:
        torch_inf = torch.tensor(torch.inf)
        mode_dict = {"min": (torch_inf, "min"), "max": (-torch_inf, "max")}

        if mode not in mode_dict:
            raise MisconfigurationException(f"`mode` can be {', '.join(mode_dict.keys())} but got {mode}")

        self.kth_value, self.mode = mode_dict[mode]

    def __init_triggers(
        self,
        every_n_train_steps: Optional[int],
        every_n_epochs: Optional[int],
        train_time_interval: Optional[timedelta],
    ) -> None:
        # Default to running once after each validation epoch if neither
        # every_n_train_steps nor every_n_epochs is set
        if every_n_train_steps is None and every_n_epochs is None and train_time_interval is None:
            every_n_epochs = 1
            every_n_train_steps = 0
            log.debug("Both every_n_train_steps and every_n_epochs are not set. Setting every_n_epochs=1")
        else:
            every_n_epochs = every_n_epochs or 0
            every_n_train_steps = every_n_train_steps or 0

        self._train_time_interval: Optional[timedelta] = train_time_interval
        self._every_n_epochs: int = every_n_epochs
        self._every_n_train_steps: int = every_n_train_steps

    @property
    def every_n_epochs(self) -> Optional[int]:
        return self._every_n_epochs

    def check_monitor_top_k(self, trainer: "pl.Trainer", current: Optional[Tensor] = None) -> bool:
        if current is None:
            return False

        if self.save_top_k == -1:
            return True

        less_than_k_models = len(self.best_k_models) < self.save_top_k
        if less_than_k_models:
            return True

        monitor_op = {"min": torch.lt, "max": torch.gt}[self.mode]
        should_update_best_and_save = monitor_op(current, self.best_k_models[self.kth_best_model_path])

        # If using multiple devices, make sure all processes are unanimous on the decision.
        should_update_best_and_save = trainer.strategy.reduce_boolean_decision(bool(should_update_best_and_save))

        return should_update_best_and_save

    def _format_checkpoint_name(
        self,
        filename: Optional[str],
        metrics: dict[str, Tensor],
        prefix: Optional[str] = None,
        auto_insert_metric_name: bool = True,
    ) -> str:
        if not filename:
            # filename is not set, use default name
            filename = "{epoch}" + self.CHECKPOINT_JOIN_CHAR + "{step}"

        # check and parse user passed keys in the string
        groups = re.findall(r"(\{.*?)[:\}]", filename)

        # sort keys from longest to shortest to avoid replacing substring
        # eg: if keys are "epoch" and "epoch_test", the latter must be replaced first
        groups = sorted(groups, key=lambda x: len(x), reverse=True)

        for group in groups:
            name = group[1:]

            if auto_insert_metric_name:
                filename = filename.replace(group, name + self.CHECKPOINT_EQUALS_CHAR + "{" + name)

            # support for dots: https://stackoverflow.com/a/7934969
            filename = filename.replace(group, f"{{0[{name}]")

            if name not in metrics:
                metrics[name] = torch.tensor(0)
        filename = filename.format(metrics)

        if prefix is not None:
            filename = self.CHECKPOINT_JOIN_CHAR.join([prefix, filename])

        return filename

    def format_checkpoint_name(
        self,
        metrics: dict[str, Tensor],
        filename: Optional[str] = None,
        ver: Optional[int] = None,
        prefix: Optional[str] = None,
    ) -> str:
        """Generate a filename according to the defined template.

        Example::

            >>> tmpdir = os.path.dirname(__file__)
            >>> ckpt = ModelCheckpoint(dirpath=tmpdir, filename='{epoch}')
            >>> os.path.basename(ckpt.format_checkpoint_name(dict(epoch=0)))
            'epoch=0.ckpt'
            >>> ckpt = ModelCheckpoint(dirpath=tmpdir, filename='{epoch:03d}')
            >>> os.path.basename(ckpt.format_checkpoint_name(dict(epoch=5)))
            'epoch=005.ckpt'
            >>> ckpt = ModelCheckpoint(dirpath=tmpdir, filename='{epoch}-{val_loss:.2f}')
            >>> os.path.basename(ckpt.format_checkpoint_name(dict(epoch=2, val_loss=0.123456)))
            'epoch=2-val_loss=0.12.ckpt'
            >>> os.path.basename(ckpt.format_checkpoint_name(dict(epoch=2, val_loss=0.12), filename='{epoch:d}'))
            'epoch=2.ckpt'
            >>> ckpt = ModelCheckpoint(dirpath=tmpdir,
            ... filename='epoch={epoch}-validation_loss={val_loss:.2f}',
            ... auto_insert_metric_name=False)
            >>> os.path.basename(ckpt.format_checkpoint_name(dict(epoch=2, val_loss=0.123456)))
            'epoch=2-validation_loss=0.12.ckpt'
            >>> ckpt = ModelCheckpoint(dirpath=tmpdir, filename='{missing:d}')
            >>> os.path.basename(ckpt.format_checkpoint_name({}))
            'missing=0.ckpt'
            >>> ckpt = ModelCheckpoint(filename='{step}')
            >>> os.path.basename(ckpt.format_checkpoint_name(dict(step=0)))
            'step=0.ckpt'

        """
        filename = filename or self.filename
        filename = self._format_checkpoint_name(
            filename, metrics, prefix=prefix, auto_insert_metric_name=self.auto_insert_metric_name
        )

        if ver is not None:
            filename = self.CHECKPOINT_JOIN_CHAR.join((filename, f"v{ver}"))

        ckpt_name = f"{filename}{self.FILE_EXTENSION}"
        return os.path.join(self.dirpath, ckpt_name) if self.dirpath else ckpt_name

    def __resolve_ckpt_dir(self, trainer: "pl.Trainer") -> _PATH:
        """Determines model checkpoint save directory at runtime. Reference attributes from the trainer's logger to
        determine where to save checkpoints. The path for saving weights is set in this priority:

        1.  The ``ModelCheckpoint``'s ``dirpath`` if passed in
        2.  The ``Logger``'s ``log_dir`` if the trainer has loggers
        3.  The ``Trainer``'s ``default_root_dir`` if the trainer has no loggers

        The path gets extended with subdirectory "checkpoints".

        """
        if self.dirpath is not None:
            # short circuit if dirpath was passed to ModelCheckpoint
            return self.dirpath

        if len(trainer.loggers) > 0:
            if trainer.loggers[0].save_dir is not None:
                save_dir = trainer.loggers[0].save_dir
            else:
                save_dir = trainer.default_root_dir
            name = trainer.loggers[0].name
            version = trainer.loggers[0].version
            version = version if isinstance(version, str) else f"version_{version}"
            ckpt_path = os.path.join(save_dir, str(name), version, "checkpoints")
        else:
            # if no loggers, use default_root_dir
            ckpt_path = os.path.join(trainer.default_root_dir, "checkpoints")

        return ckpt_path

    def _find_last_checkpoints(self, trainer: "pl.Trainer") -> set[str]:
        # find all checkpoints in the folder
        ckpt_path = self.__resolve_ckpt_dir(trainer)
        last_pattern = rf"^{self.CHECKPOINT_NAME_LAST}(-(\d+))?"

        def _is_last(path: Path) -> bool:
            return path.suffix == self.FILE_EXTENSION and bool(re.match(last_pattern, path.stem))

        if self._fs.exists(ckpt_path):
            return {os.path.normpath(p) for p in self._fs.ls(ckpt_path, detail=False) if _is_last(Path(p))}
        return set()

    def __warn_if_dir_not_empty(self, dirpath: _PATH) -> None:
        if self.save_top_k != 0 and _is_dir(self._fs, dirpath, strict=True) and len(self._fs.ls(dirpath)) > 0:
            rank_zero_warn(f"Checkpoint directory {dirpath} exists and is not empty.")

    def _get_metric_interpolated_filepath_name(
        self, monitor_candidates: dict[str, Tensor], trainer: "pl.Trainer", del_filepath: Optional[str] = None
    ) -> str:
        filepath = self.format_checkpoint_name(monitor_candidates)

        if self._enable_version_counter:
            version_cnt = self.STARTING_VERSION
            while self.file_exists(filepath, trainer) and filepath != del_filepath:
                filepath = self.format_checkpoint_name(monitor_candidates, ver=version_cnt)
                version_cnt += 1

        return filepath

    def _monitor_candidates(self, trainer: "pl.Trainer") -> dict[str, Tensor]:
        monitor_candidates = deepcopy(trainer.callback_metrics)
        # cast to int if necessary because `self.log("epoch", 123)` will convert it to float. if it's not a tensor
        # or does not exist we overwrite it as it's likely an error
        epoch = monitor_candidates.get("epoch")
        monitor_candidates["epoch"] = epoch.int() if isinstance(epoch, Tensor) else torch.tensor(trainer.current_epoch)
        step = monitor_candidates.get("step")
        monitor_candidates["step"] = step.int() if isinstance(step, Tensor) else torch.tensor(trainer.global_step)
        return monitor_candidates

    def _save_last_checkpoint(self, trainer: "pl.Trainer", monitor_candidates: dict[str, Tensor]) -> None:
        if not self.save_last:
            return

        filepath = self.format_checkpoint_name(monitor_candidates, self.CHECKPOINT_NAME_LAST)

        if self._enable_version_counter:
            version_cnt = self.STARTING_VERSION
            while self.file_exists(filepath, trainer) and filepath != self.last_model_path:
                filepath = self.format_checkpoint_name(monitor_candidates, self.CHECKPOINT_NAME_LAST, ver=version_cnt)
                version_cnt += 1

        # set the last model path before saving because it will be part of the state.
        previous, self.last_model_path = self.last_model_path, filepath
        if self.save_last == "link" and self._last_checkpoint_saved and self.save_top_k != 0:
            self._link_checkpoint(trainer, self._last_checkpoint_saved, filepath)
        else:
            self._save_checkpoint(trainer, filepath)
        if previous and self._should_remove_checkpoint(trainer, previous, filepath):
            self._remove_checkpoint(trainer, previous)

    def _save_monitor_checkpoint(self, trainer: "pl.Trainer", monitor_candidates: dict[str, Tensor]) -> None:
        assert self.monitor
        current = monitor_candidates.get(self.monitor)
        if self.check_monitor_top_k(trainer, current):
            assert current is not None
            self._update_best_and_save(current, trainer, monitor_candidates)
        elif self.verbose:
            epoch = monitor_candidates["epoch"]
            step = monitor_candidates["step"]
            rank_zero_info(f"Epoch {epoch:d}, global step {step:d}: {self.monitor!r} was not in top {self.save_top_k}")

    def _save_none_monitor_checkpoint(self, trainer: "pl.Trainer", monitor_candidates: dict[str, Tensor]) -> None:
        filepath = self._get_metric_interpolated_filepath_name(monitor_candidates, trainer, self.best_model_path)
        # set the best model path before saving because it will be part of the state.
        previous, self.best_model_path = self.best_model_path, filepath
        self._save_checkpoint(trainer, filepath)

        if self.save_top_k == 1 and previous and self._should_remove_checkpoint(trainer, previous, filepath):
            self._remove_checkpoint(trainer, previous)

    def _update_best_and_save(
        self, current: Tensor, trainer: "pl.Trainer", monitor_candidates: dict[str, Tensor]
    ) -> None:
        k = len(self.best_k_models) + 1 if self.save_top_k == -1 else self.save_top_k

        del_filepath = None
        if len(self.best_k_models) == k and k > 0:
            del_filepath = self.kth_best_model_path
            self.best_k_models.pop(del_filepath)

        # do not save nan, replace with +/- inf
        if isinstance(current, Tensor) and torch.isnan(current):
            current = torch.tensor(float("inf" if self.mode == "min" else "-inf"), device=current.device)

        filepath = self._get_metric_interpolated_filepath_name(monitor_candidates, trainer, del_filepath)

        # save the current score
        self.current_score = current
        self.best_k_models[filepath] = current

        if len(self.best_k_models) == k:
            # monitor dict has reached k elements
            _op = max if self.mode == "min" else min
            self.kth_best_model_path = _op(self.best_k_models, key=self.best_k_models.get)  # type: ignore[arg-type]
            self.kth_value = self.best_k_models[self.kth_best_model_path]

        _op = min if self.mode == "min" else max
        self.best_model_path = _op(self.best_k_models, key=self.best_k_models.get)  # type: ignore[arg-type]
        self.best_model_score = self.best_k_models[self.best_model_path]

        if self.best_model_path == filepath:
            self.best_model_metrics = dict(monitor_candidates)

        if self.verbose:
            epoch = monitor_candidates["epoch"]
            step = monitor_candidates["step"]
            rank_zero_info(
                f"Epoch {epoch:d}, global step {step:d}: {self.monitor!r} reached {current:0.5f}"
                f" (best {self.best_model_score:0.5f}), saving model to {filepath!r} as top {k}"
            )
        self._save_checkpoint(trainer, filepath)

        if del_filepath and self._should_remove_checkpoint(trainer, del_filepath, filepath):
            self._remove_checkpoint(trainer, del_filepath)

    def to_yaml(self, filepath: Optional[_PATH] = None) -> None:
        """Saves the `best_k_models` dict containing the checkpoint paths with the corresponding scores to a YAML
        file."""
        best_k = {k: v.item() for k, v in self.best_k_models.items()}
        if filepath is None:
            assert self.dirpath
            filepath = os.path.join(self.dirpath, "best_k_models.yaml")
        with self._fs.open(filepath, "w") as fp:
            yaml.dump(best_k, fp)

    def file_exists(self, filepath: _PATH, trainer: "pl.Trainer") -> bool:
        """Checks if a file exists on rank 0 and synchronizes the result to all other ranks, preventing the internal
        state to diverge between ranks."""
        # In distributed setups, only global rank 0 touches the filesystem
        local_decision = self._fs.exists(filepath) if trainer.is_global_zero else False
        # Reduce the decision across ranks using an "any"-style reduction to decide if the file exists anywhere
        return trainer.strategy.reduce_boolean_decision(local_decision, all=False)

    def _should_remove_checkpoint(self, trainer: "pl.Trainer", previous: str, current: str) -> bool:
        """Checks if the previous checkpoint should be deleted.

        A checkpoint won't be deleted if any of the cases apply:
        - The previous checkpoint is the same as the current checkpoint (means the old was already overwritten by new)
        - The previous checkpoint is not in the current checkpoint directory and the filesystem is local
        - The previous checkpoint is the checkpoint the Trainer resumed from and the filesystem is local

        """
        if previous == current:
            return False
        if not _is_local_file_protocol(previous):
            return True
        previous = Path(previous).absolute()
        resume_path = Path(trainer.ckpt_path).absolute() if trainer.ckpt_path is not None else None
        if resume_path is not None and previous == resume_path:
            return False
        assert self.dirpath is not None
        dirpath = Path(self.dirpath).absolute()
        return dirpath in previous.parents

    def _remove_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        """Calls the strategy to remove the checkpoint file."""
        trainer.strategy.remove_checkpoint(filepath)
