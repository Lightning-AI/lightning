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

# THIS FILE MUST READ EASILY, FOR UNDERSTANDING AND DEBUGGING PURPOSES.
# DO NOT OBSCURE THE TRAINING LOOP
# THIS IS A HARD REQUIREMENT TO CONTRIBUTING TO LIGHTNING
# WE FAVOR READABILITY OVER ENGINEERING-CONSTRUCTS BY DESIGN
# DO NOT REMOVE THIS NOTICE
# - WILLIAM FALCON
"""Trainer to automate the training."""

import logging
import math
import os
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from datetime import timedelta
from typing import Any, Optional, Union
from weakref import proxy

import torch
from torch.optim import Optimizer

import lightning.pytorch as pl
from lightning.fabric.utilities.apply_func import convert_tensors_to_scalars
from lightning.fabric.utilities.cloud_io import _is_local_file_protocol
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.accelerators import Accelerator
from lightning.pytorch.callbacks import Callback, Checkpoint, EarlyStopping, ProgressBar
from lightning.pytorch.core.datamodule import LightningDataModule
from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.loggers.utilities import _log_hyperparams
from lightning.pytorch.loops import _PredictionLoop, _TrainingEpochLoop
from lightning.pytorch.loops.evaluation_loop import _EvaluationLoop
from lightning.pytorch.loops.fit_loop import _FitLoop
from lightning.pytorch.loops.utilities import _parse_loop_limits, _reset_progress
from lightning.pytorch.plugins import _PLUGIN_INPUT, Precision
from lightning.pytorch.profilers import Profiler
from lightning.pytorch.strategies import ParallelStrategy, Strategy
from lightning.pytorch.trainer import call, setup
from lightning.pytorch.trainer.configuration_validator import _verify_loop_configurations
from lightning.pytorch.trainer.connectors.accelerator_connector import (
    _LITERAL_WARN,
    _PRECISION_INPUT,
    _PRECISION_INPUT_STR,
    _AcceleratorConnector,
)
from lightning.pytorch.trainer.connectors.callback_connector import _CallbackConnector
from lightning.pytorch.trainer.connectors.checkpoint_connector import _CheckpointConnector
from lightning.pytorch.trainer.connectors.data_connector import _DataConnector
from lightning.pytorch.trainer.connectors.logger_connector import _LoggerConnector
from lightning.pytorch.trainer.connectors.logger_connector.result import _OUT_DICT, _PBAR_DICT, _ResultCollection
from lightning.pytorch.trainer.connectors.signal_connector import _SignalConnector
from lightning.pytorch.trainer.states import RunningStage, TrainerFn, TrainerState, TrainerStatus
from lightning.pytorch.utilities import GradClipAlgorithmType, parsing
from lightning.pytorch.utilities.argparse import _defaults_from_env_vars
from lightning.pytorch.utilities.compile import _maybe_unwrap_optimized, _verify_strategy_supports_compile
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_warn
from lightning.pytorch.utilities.seed import isolate_rng
from lightning.pytorch.utilities.types import (
    _EVALUATE_OUTPUT,
    _PREDICT_OUTPUT,
    EVAL_DATALOADERS,
    TRAIN_DATALOADERS,
    LRSchedulerConfig,
)
from lightning.pytorch.utilities.warnings import PossibleUserWarning

log = logging.getLogger(__name__)


class Trainer:
    @_defaults_from_env_vars
    def __init__(
        self,
        *,
        accelerator: Union[str, Accelerator] = "auto",
        strategy: Union[str, Strategy] = "auto",
        devices: Union[list[int], str, int] = "auto",
        num_nodes: int = 1,
        precision: Optional[_PRECISION_INPUT] = None,
        logger: Optional[Union[Logger, Iterable[Logger], bool]] = None,
        callbacks: Optional[Union[list[Callback], Callback]] = None,
        fast_dev_run: Union[int, bool] = False,
        max_epochs: Optional[int] = None,
        min_epochs: Optional[int] = None,
        max_steps: int = -1,
        min_steps: Optional[int] = None,
        max_time: Optional[Union[str, timedelta, dict[str, int]]] = None,
        limit_train_batches: Optional[Union[int, float]] = None,
        limit_val_batches: Optional[Union[int, float]] = None,
        limit_test_batches: Optional[Union[int, float]] = None,
        limit_predict_batches: Optional[Union[int, float]] = None,
        overfit_batches: Union[int, float] = 0.0,
        val_check_interval: Optional[Union[int, float]] = None,
        check_val_every_n_epoch: Optional[int] = 1,
        num_sanity_val_steps: Optional[int] = None,
        log_every_n_steps: Optional[int] = None,
        enable_checkpointing: Optional[bool] = None,
        enable_progress_bar: Optional[bool] = None,
        enable_model_summary: Optional[bool] = None,
        accumulate_grad_batches: int = 1,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
        deterministic: Optional[Union[bool, _LITERAL_WARN]] = None,
        benchmark: Optional[bool] = None,
        inference_mode: bool = True,
        use_distributed_sampler: bool = True,
        profiler: Optional[Union[Profiler, str]] = None,
        detect_anomaly: bool = False,
        barebones: bool = False,
        plugins: Optional[Union[_PLUGIN_INPUT, list[_PLUGIN_INPUT]]] = None,
        sync_batchnorm: bool = False,
        reload_dataloaders_every_n_epochs: int = 0,
        default_root_dir: Optional[_PATH] = None,
        enable_autolog_hparams: bool = True,
    ) -> None:
        r"""Customize every aspect of training via flags.

        Args:
            accelerator: Supports passing different accelerator types ("cpu", "gpu", "tpu", "hpu", "mps", "auto")
                as well as custom accelerator instances.

            strategy: Supports different training strategies with aliases as well custom strategies.
                Default: ``"auto"``.

            devices: The devices to use. Can be set to a positive number (int or str), a sequence of device indices
                (list or str), the value ``-1`` to indicate all available devices should be used, or ``"auto"`` for
                automatic selection based on the chosen accelerator. Default: ``"auto"``.

            num_nodes: Number of GPU nodes for distributed training.
                Default: ``1``.

            precision: Double precision (64, '64' or '64-true'), full precision (32, '32' or '32-true'),
                16bit mixed precision (16, '16', '16-mixed') or bfloat16 mixed precision ('bf16', 'bf16-mixed').
                Can be used on CPU, GPU, TPUs, or HPUs.
                Default: ``'32-true'``.

            logger: Logger (or iterable collection of loggers) for experiment tracking. A ``True`` value uses
                the default ``TensorBoardLogger`` if it is installed, otherwise ``CSVLogger``.
                ``False`` will disable logging. If multiple loggers are provided, local files
                (checkpoints, profiler traces, etc.) are saved in the ``log_dir`` of the first logger.
                Default: ``True``.

            callbacks: Add a callback or list of callbacks.
                Default: ``None``.

            fast_dev_run: Runs n if set to ``n`` (int) else 1 if set to ``True`` batch(es)
                of train, val and test to find any bugs (ie: a sort of unit test).
                Default: ``False``.

            max_epochs: Stop training once this number of epochs is reached. Disabled by default (None).
                If both max_epochs and max_steps are not specified, defaults to ``max_epochs = 1000``.
                To enable infinite training, set ``max_epochs = -1``.

            min_epochs: Force training for at least these many epochs. Disabled by default (None).

            max_steps: Stop training after this number of steps. Disabled by default (-1). If ``max_steps = -1``
                and ``max_epochs = None``, will default to ``max_epochs = 1000``. To enable infinite training, set
                ``max_epochs`` to ``-1``.

            min_steps: Force training for at least these number of steps. Disabled by default (``None``).

            max_time: Stop training after this amount of time has passed. Disabled by default (``None``).
                The time duration can be specified in the format DD:HH:MM:SS (days, hours, minutes seconds), as a
                :class:`datetime.timedelta`, or a dictionary with keys that will be passed to
                :class:`datetime.timedelta`.

            limit_train_batches: How much of training dataset to check (float = fraction, int = num_batches).
                Default: ``1.0``.

            limit_val_batches: How much of validation dataset to check (float = fraction, int = num_batches).
                Default: ``1.0``.

            limit_test_batches: How much of test dataset to check (float = fraction, int = num_batches).
                Default: ``1.0``.

            limit_predict_batches: How much of prediction dataset to check (float = fraction, int = num_batches).
                Default: ``1.0``.

            overfit_batches: Overfit a fraction of training/validation data (float) or a set number of batches (int).
                Default: ``0.0``.

            val_check_interval: How often to check the validation set. Pass a ``float`` in the range [0.0, 1.0] to check
                after a fraction of the training epoch. Pass an ``int`` to check after a fixed number of training
                batches. An ``int`` value can only be higher than the number of training batches when
                ``check_val_every_n_epoch=None``, which validates after every ``N`` training batches
                across epochs or during iteration-based training.
                Default: ``1.0``.

            check_val_every_n_epoch: Perform a validation loop after every `N` training epochs. If ``None``,
                validation will be done solely based on the number of training batches, requiring ``val_check_interval``
                to be an integer value.
                Default: ``1``.

            num_sanity_val_steps: Sanity check runs n validation batches before starting the training routine.
                Set it to `-1` to run all batches in all validation dataloaders.
                Default: ``2``.

            log_every_n_steps: How often to log within steps.
                Default: ``50``.

            enable_checkpointing: If ``True``, enable checkpointing.
                It will configure a default ModelCheckpoint callback if there is no user-defined ModelCheckpoint in
                :paramref:`~lightning.pytorch.trainer.trainer.Trainer.callbacks`.
                Default: ``True``.

            enable_progress_bar: Whether to enable to progress bar by default.
                Default: ``True``.

            enable_model_summary: Whether to enable model summarization by default.
                Default: ``True``.

            accumulate_grad_batches: Accumulates gradients over k batches before stepping the optimizer.
                Default: 1.

            gradient_clip_val: The value at which to clip gradients. Passing ``gradient_clip_val=None`` disables
                gradient clipping. If using Automatic Mixed Precision (AMP), the gradients will be unscaled before.
                Default: ``None``.

            gradient_clip_algorithm: The gradient clipping algorithm to use. Pass ``gradient_clip_algorithm="value"``
                to clip by value, and ``gradient_clip_algorithm="norm"`` to clip by norm. By default it will
                be set to ``"norm"``.

            deterministic: If ``True``, sets whether PyTorch operations must use deterministic algorithms.
                Set to ``"warn"`` to use deterministic algorithms whenever possible, throwing warnings on operations
                that don't support deterministic mode. If not set, defaults to ``False``. Default: ``None``.

            benchmark: The value (``True`` or ``False``) to set ``torch.backends.cudnn.benchmark`` to.
                The value for ``torch.backends.cudnn.benchmark`` set in the current session will be used
                (``False`` if not manually set). If :paramref:`~lightning.pytorch.trainer.trainer.Trainer.deterministic`
                is set to ``True``, this will default to ``False``. Override to manually set a different value.
                Default: ``None``.

            inference_mode: Whether to use :func:`torch.inference_mode` or :func:`torch.no_grad` during
                evaluation (``validate``/``test``/``predict``).

            use_distributed_sampler: Whether to wrap the DataLoader's sampler with
                :class:`torch.utils.data.DistributedSampler`. If not specified this is toggled automatically for
                strategies that require it. By default, it will add ``shuffle=True`` for the train sampler and
                ``shuffle=False`` for validation/test/predict samplers. If you want to disable this logic, you can pass
                ``False`` and add your own distributed sampler in the dataloader hooks. If ``True`` and a distributed
                sampler was already added, Lightning will not replace the existing one. For iterable-style datasets,
                we don't do this automatically.

            profiler: To profile individual steps during training and assist in identifying bottlenecks.
                Default: ``None``.

            detect_anomaly: Enable anomaly detection for the autograd engine.
                Default: ``False``.

            barebones: Whether to run in "barebones mode", where all features that may impact raw speed are
                disabled. This is meant for analyzing the Trainer overhead and is discouraged during regular training
                runs. The following features are deactivated:
                :paramref:`~lightning.pytorch.trainer.trainer.Trainer.enable_checkpointing`,
                :paramref:`~lightning.pytorch.trainer.trainer.Trainer.logger`,
                :paramref:`~lightning.pytorch.trainer.trainer.Trainer.enable_progress_bar`,
                :paramref:`~lightning.pytorch.trainer.trainer.Trainer.log_every_n_steps`,
                :paramref:`~lightning.pytorch.trainer.trainer.Trainer.enable_model_summary`,
                :paramref:`~lightning.pytorch.trainer.trainer.Trainer.num_sanity_val_steps`,
                :paramref:`~lightning.pytorch.trainer.trainer.Trainer.fast_dev_run`,
                :paramref:`~lightning.pytorch.trainer.trainer.Trainer.detect_anomaly`,
                :paramref:`~lightning.pytorch.trainer.trainer.Trainer.profiler`,
                :meth:`~lightning.pytorch.core.LightningModule.log`,
                :meth:`~lightning.pytorch.core.LightningModule.log_dict`.
            plugins: Plugins allow modification of core behavior like ddp and amp, and enable custom lightning plugins.
                Default: ``None``.

            sync_batchnorm: Synchronize batch norm layers between process groups/whole world.
                Default: ``False``.

            reload_dataloaders_every_n_epochs: Set to a positive integer to reload dataloaders every n epochs.
                Default: ``0``.

            default_root_dir: Default path for logs and weights when no logger/ckpt_callback passed.
                Default: ``os.getcwd()``.
                Can be remote file paths such as `s3://mybucket/path` or 'hdfs://path/'

            enable_autolog_hparams: Whether to log hyperparameters at the start of a run.
                Default: ``True``.

        Raises:
            TypeError:
                If ``gradient_clip_val`` is not an int or float.

            MisconfigurationException:
                If ``gradient_clip_algorithm`` is invalid.

        """
        super().__init__()
        log.debug(f"{self.__class__.__name__}: Initializing trainer with parameters: {locals()}")

        if default_root_dir is not None:
            default_root_dir = os.fspath(default_root_dir)

        self.barebones = barebones
        if barebones:
            # opt-outs
            if enable_checkpointing:
                raise ValueError(
                    f"`Trainer(barebones=True, enable_checkpointing={enable_checkpointing!r})` was passed."
                    " Checkpointing can impact raw speed so it is disabled in barebones mode."
                )
            enable_checkpointing = False
            if logger is not None and logger is not False:
                raise ValueError(
                    f"`Trainer(barebones=True, logger={logger!r})` was passed."
                    " Logging can impact raw speed so it is disabled in barebones mode."
                )
            logger = False
            if enable_progress_bar:
                raise ValueError(
                    f"`Trainer(barebones=True, enable_progress_bar={enable_progress_bar!r})` was passed."
                    " The progress bar can impact raw speed so it is disabled in barebones mode."
                )
            enable_progress_bar = False
            if log_every_n_steps is not None and log_every_n_steps != 0:
                raise ValueError(
                    f"`Trainer(barebones=True, log_every_n_steps={log_every_n_steps!r})` was passed."
                    " Logging can impact raw speed so it is disabled in barebones mode."
                )
            log_every_n_steps = 0
            if enable_model_summary:
                raise ValueError(
                    f"`Trainer(barebones=True, enable_model_summary={enable_model_summary!r})` was passed."
                    " Model summary can impact raw speed so it is disabled in barebones mode."
                )
            enable_model_summary = False
            if num_sanity_val_steps is not None and num_sanity_val_steps != 0:
                raise ValueError(
                    f"`Trainer(barebones=True, num_sanity_val_steps={num_sanity_val_steps!r})` was passed."
                    " Sanity checking can impact raw speed so it is disabled in barebones mode."
                )
            num_sanity_val_steps = 0
            # opt-ins
            if fast_dev_run is not False and fast_dev_run != 0:
                raise ValueError(
                    f"`Trainer(barebones=True, fast_dev_run={fast_dev_run!r})` was passed."
                    " Development run is not meant for raw speed evaluation so it is disabled in barebones mode."
                )
            if detect_anomaly:
                raise ValueError(
                    f"`Trainer(barebones=True, detect_anomaly={detect_anomaly!r})` was passed."
                    " Anomaly detection can impact raw speed so it is disabled in barebones mode."
                )
            if profiler is not None:
                raise ValueError(
                    f"`Trainer(barebones=True, profiler={profiler!r})` was passed."
                    " Profiling can impact raw speed so it is disabled in barebones mode."
                )
            deactivated = (
                " - Checkpointing: `Trainer(enable_checkpointing=True)`",
                " - Progress bar: `Trainer(enable_progress_bar=True)`",
                " - Model summary: `Trainer(enable_model_summary=True)`",
                " - Logging: `Trainer(logger=True)`, `Trainer(log_every_n_steps>0)`,"
                " `LightningModule.log(...)`, `LightningModule.log_dict(...)`",
                " - Sanity checking: `Trainer(num_sanity_val_steps>0)`",
                " - Development run: `Trainer(fast_dev_run=True)`",
                " - Anomaly detection: `Trainer(detect_anomaly=True)`",
                " - Profiling: `Trainer(profiler=...)`",
            )
            rank_zero_info(
                "You are running in `Trainer(barebones=True)` mode. All features that may impact raw speed have been"
                " disabled to facilitate analyzing the Trainer overhead. Specifically, the following features are"
                f" deactivated:{os.linesep}{os.linesep.join(deactivated)}"
            )
        else:
            # set the opt-out defaults
            if enable_checkpointing is None:
                enable_checkpointing = True
            if logger is None:
                logger = True
            if enable_progress_bar is None:
                enable_progress_bar = True
            if log_every_n_steps is None:
                log_every_n_steps = 50
            if enable_model_summary is None:
                enable_model_summary = True
            if num_sanity_val_steps is None:
                num_sanity_val_steps = 2

        # init connectors
        self._data_connector = _DataConnector(self)

        self._accelerator_connector = _AcceleratorConnector(
            devices=devices,
            accelerator=accelerator,
            strategy=strategy,
            num_nodes=num_nodes,
            sync_batchnorm=sync_batchnorm,
            benchmark=benchmark,
            use_distributed_sampler=use_distributed_sampler,
            deterministic=deterministic,
            precision=precision,
            plugins=plugins,
        )
        self._logger_connector = _LoggerConnector(self)
        self._callback_connector = _CallbackConnector(self)
        self._checkpoint_connector = _CheckpointConnector(self)
        self._signal_connector = _SignalConnector(self)

        # init loops
        self.fit_loop = _FitLoop(self, min_epochs=min_epochs, max_epochs=max_epochs)
        self.fit_loop.epoch_loop = _TrainingEpochLoop(self, min_steps=min_steps, max_steps=max_steps)
        self.validate_loop = _EvaluationLoop(
            self, TrainerFn.VALIDATING, RunningStage.VALIDATING, inference_mode=inference_mode
        )
        self.test_loop = _EvaluationLoop(self, TrainerFn.TESTING, RunningStage.TESTING, inference_mode=inference_mode)
        self.predict_loop = _PredictionLoop(self, inference_mode=inference_mode)

        self.accumulate_grad_batches = accumulate_grad_batches

        # init callbacks
        # Declare attributes to be set in _callback_connector on_trainer_init
        self._callback_connector.on_trainer_init(
            callbacks,
            enable_checkpointing,
            enable_progress_bar,
            default_root_dir,
            enable_model_summary,
            max_time,
        )

        # init data flags
        self.check_val_every_n_epoch: Optional[int]
        self._data_connector.on_trainer_init(
            val_check_interval,
            reload_dataloaders_every_n_epochs,
            check_val_every_n_epoch,
        )

        # gradient clipping
        if gradient_clip_val is not None and not isinstance(gradient_clip_val, (int, float)):
            raise TypeError(f"`gradient_clip_val` should be an int or a float. Got {gradient_clip_val}.")

        if gradient_clip_algorithm is not None and not GradClipAlgorithmType.supported_type(
            gradient_clip_algorithm.lower()
        ):
            raise MisconfigurationException(
                f"`gradient_clip_algorithm` {gradient_clip_algorithm} is invalid. "
                f"Allowed algorithms: {GradClipAlgorithmType.supported_types()}."
            )

        self.gradient_clip_val: Optional[Union[int, float]] = gradient_clip_val
        self.gradient_clip_algorithm: Optional[GradClipAlgorithmType] = (
            GradClipAlgorithmType(gradient_clip_algorithm.lower()) if gradient_clip_algorithm is not None else None
        )

        if detect_anomaly:
            rank_zero_info(
                "You have turned on `Trainer(detect_anomaly=True)`. This will significantly slow down compute speed and"
                " is recommended only for model debugging."
            )
        self._detect_anomaly: bool = detect_anomaly

        setup._log_device_info(self)

        self.should_stop = False
        self.state = TrainerState()

        # configure profiler
        setup._init_profiler(self, profiler)

        # init logger flags
        self._loggers: list[Logger]
        self._logger_connector.on_trainer_init(logger, log_every_n_steps)

        # init debugging flags
        self.val_check_batch: Union[int, float]
        self.val_check_interval: Union[int, float]
        self.num_sanity_val_steps: Union[int, float]
        self.limit_train_batches: Union[int, float]
        self.limit_val_batches: Union[int, float]
        self.limit_test_batches: Union[int, float]
        self.limit_predict_batches: Union[int, float]
        setup._init_debugging_flags(
            self,
            limit_train_batches,
            limit_val_batches,
            limit_test_batches,
            limit_predict_batches,
            fast_dev_run,
            overfit_batches,
            val_check_interval,
            num_sanity_val_steps,
        )

        self.enable_autolog_hparams = enable_autolog_hparams

    def fit(
        self,
        model: "pl.LightningModule",
        train_dataloaders: Optional[Union[TRAIN_DATALOADERS, LightningDataModule]] = None,
        val_dataloaders: Optional[EVAL_DATALOADERS] = None,
        datamodule: Optional[LightningDataModule] = None,
        ckpt_path: Optional[_PATH] = None,
    ) -> None:
        r"""Runs the full optimization routine.

        Args:
            model: Model to fit.

            train_dataloaders: An iterable or collection of iterables specifying training samples.
                Alternatively, a :class:`~lightning.pytorch.core.datamodule.LightningDataModule` that defines
                the :class:`~lightning.pytorch.core.hooks.DataHooks.train_dataloader` hook.

            val_dataloaders: An iterable or collection of iterables specifying validation samples.

            datamodule: A :class:`~lightning.pytorch.core.datamodule.LightningDataModule` that defines
                the :class:`~lightning.pytorch.core.hooks.DataHooks.train_dataloader` hook.

            ckpt_path: Path/URL of the checkpoint from which training is resumed. Could also be one of two special
                keywords ``"last"`` and ``"hpc"``. If there is no checkpoint file at the path, an exception is raised.

        Raises:
            TypeError:
                If ``model`` is not :class:`~lightning.pytorch.core.LightningModule` for torch version less than
                2.0.0 and if ``model`` is not :class:`~lightning.pytorch.core.LightningModule` or
                :class:`torch._dynamo.OptimizedModule` for torch versions greater than or equal to 2.0.0 .

        For more information about multiple dataloaders, see this :ref:`section <multiple-dataloaders>`.

        """
        model = _maybe_unwrap_optimized(model)
        self.strategy._lightning_module = model
        _verify_strategy_supports_compile(model, self.strategy)
        self.state.fn = TrainerFn.FITTING
        self.state.status = TrainerStatus.RUNNING
        self.training = True
        call._call_and_handle_interrupt(
            self, self._fit_impl, model, train_dataloaders, val_dataloaders, datamodule, ckpt_path
        )

    def _fit_impl(
        self,
        model: "pl.LightningModule",
        train_dataloaders: Optional[Union[TRAIN_DATALOADERS, LightningDataModule]] = None,
        val_dataloaders: Optional[EVAL_DATALOADERS] = None,
        datamodule: Optional[LightningDataModule] = None,
        ckpt_path: Optional[_PATH] = None,
    ) -> None:
        log.debug(f"{self.__class__.__name__}: trainer fit stage")

        # if a datamodule comes in as the second arg, then fix it for the user
        if isinstance(train_dataloaders, LightningDataModule):
            datamodule = train_dataloaders
            train_dataloaders = None
        # If you supply a datamodule you can't supply train_dataloader or val_dataloaders
        if (train_dataloaders is not None or val_dataloaders is not None) and datamodule is not None:
            raise MisconfigurationException(
                "You cannot pass `train_dataloader` or `val_dataloaders` to `trainer.fit(datamodule=...)`"
            )

        # links data to the trainer
        self._data_connector.attach_data(
            model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders, datamodule=datamodule
        )

        assert self.state.fn is not None
        ckpt_path = self._checkpoint_connector._select_ckpt_path(
            self.state.fn,
            ckpt_path,
            model_provided=True,
            model_connected=self.lightning_module is not None,
        )
        self._run(model, ckpt_path=ckpt_path)

        assert self.state.stopped
        self.training = False
        return

    def validate(
        self,
        model: Optional["pl.LightningModule"] = None,
        dataloaders: Optional[Union[EVAL_DATALOADERS, LightningDataModule]] = None,
        ckpt_path: Optional[_PATH] = None,
        verbose: bool = True,
        datamodule: Optional[LightningDataModule] = None,
    ) -> _EVALUATE_OUTPUT:
        r"""Perform one evaluation epoch over the validation set.

        Args:
            model: The model to validate.

            dataloaders: An iterable or collection of iterables specifying validation samples.
                Alternatively, a :class:`~lightning.pytorch.core.datamodule.LightningDataModule` that defines
                the :class:`~lightning.pytorch.core.hooks.DataHooks.val_dataloader` hook.

            ckpt_path: Either ``"best"``, ``"last"``, ``"hpc"`` or path to the checkpoint you wish to validate.
                If ``None`` and the model instance was passed, use the current weights.
                Otherwise, the best model checkpoint from the previous ``trainer.fit`` call will be loaded
                if a checkpoint callback is configured.

            verbose: If True, prints the validation results.

            datamodule: A :class:`~lightning.pytorch.core.datamodule.LightningDataModule` that defines
                the :class:`~lightning.pytorch.core.hooks.DataHooks.val_dataloader` hook.

        For more information about multiple dataloaders, see this :ref:`section <multiple-dataloaders>`.

        Returns:
            List of dictionaries with metrics logged during the validation phase, e.g., in model- or callback hooks
            like :meth:`~lightning.pytorch.LightningModule.validation_step` etc.
            The length of the list corresponds to the number of validation dataloaders used.

        Raises:
            TypeError:
                If no ``model`` is passed and there was no ``LightningModule`` passed in the previous run.
                If ``model`` passed is not `LightningModule` or `torch._dynamo.OptimizedModule`.

            MisconfigurationException:
                If both ``dataloaders`` and ``datamodule`` are passed. Pass only one of these.

            RuntimeError:
                If a compiled ``model`` is passed and the strategy is not supported.

        """
        if model is None:
            # do we still have a reference from a previous call?
            if self.lightning_module is None:
                raise TypeError(
                    "`Trainer.validate()` requires a `LightningModule` when it hasn't been passed in a previous run"
                )
        else:
            model = _maybe_unwrap_optimized(model)
            self.strategy._lightning_module = model
        _verify_strategy_supports_compile(self.lightning_module, self.strategy)
        self.state.fn = TrainerFn.VALIDATING
        self.state.status = TrainerStatus.RUNNING
        self.validating = True
        return call._call_and_handle_interrupt(
            self, self._validate_impl, model, dataloaders, ckpt_path, verbose, datamodule
        )

    def _validate_impl(
        self,
        model: Optional["pl.LightningModule"] = None,
        dataloaders: Optional[Union[EVAL_DATALOADERS, LightningDataModule]] = None,
        ckpt_path: Optional[_PATH] = None,
        verbose: bool = True,
        datamodule: Optional[LightningDataModule] = None,
    ) -> Optional[Union[_PREDICT_OUTPUT, _EVALUATE_OUTPUT]]:
        # --------------------
        # SETUP HOOK
        # --------------------
        log.debug(f"{self.__class__.__name__}: trainer validate stage")

        # if a datamodule comes in as the second arg, then fix it for the user
        if isinstance(dataloaders, LightningDataModule):
            datamodule = dataloaders
            dataloaders = None
        # If you supply a datamodule you can't supply val_dataloaders
        if dataloaders is not None and datamodule:
            raise MisconfigurationException("You cannot pass both `trainer.validate(dataloaders=..., datamodule=...)`")

        if model is None:
            model = self.lightning_module
            model_provided = False
        else:
            model_provided = True

        self.validate_loop.verbose = verbose

        # links data to the trainer
        self._data_connector.attach_data(model, val_dataloaders=dataloaders, datamodule=datamodule)

        assert self.state.fn is not None
        ckpt_path = self._checkpoint_connector._select_ckpt_path(
            self.state.fn, ckpt_path, model_provided=model_provided, model_connected=self.lightning_module is not None
        )
        results = self._run(model, ckpt_path=ckpt_path)
        # remove the tensors from the validation results
        results = convert_tensors_to_scalars(results)

        assert self.state.stopped
        self.validating = False

        return results

    def test(
        self,
        model: Optional["pl.LightningModule"] = None,
        dataloaders: Optional[Union[EVAL_DATALOADERS, LightningDataModule]] = None,
        ckpt_path: Optional[_PATH] = None,
        verbose: bool = True,
        datamodule: Optional[LightningDataModule] = None,
    ) -> _EVALUATE_OUTPUT:
        r"""Perform one evaluation epoch over the test set. It's separated from fit to make sure you never run on your
        test set until you want to.

        Args:
            model: The model to test.

            dataloaders: An iterable or collection of iterables specifying test samples.
                Alternatively, a :class:`~lightning.pytorch.core.datamodule.LightningDataModule` that defines
                the :class:`~lightning.pytorch.core.hooks.DataHooks.test_dataloader` hook.

            ckpt_path: Either ``"best"``, ``"last"``, ``"hpc"`` or path to the checkpoint you wish to test.
                If ``None`` and the model instance was passed, use the current weights.
                Otherwise, the best model checkpoint from the previous ``trainer.fit`` call will be loaded
                if a checkpoint callback is configured.

            verbose: If True, prints the test results.

            datamodule: A :class:`~lightning.pytorch.core.datamodule.LightningDataModule` that defines
                the :class:`~lightning.pytorch.core.hooks.DataHooks.test_dataloader` hook.

        For more information about multiple dataloaders, see this :ref:`section <multiple-dataloaders>`.

        Returns:
            List of dictionaries with metrics logged during the test phase, e.g., in model- or callback hooks
            like :meth:`~lightning.pytorch.LightningModule.test_step` etc.
            The length of the list corresponds to the number of test dataloaders used.

        Raises:
            TypeError:
                If no ``model`` is passed and there was no ``LightningModule`` passed in the previous run.
                If ``model`` passed is not `LightningModule` or `torch._dynamo.OptimizedModule`.

            MisconfigurationException:
                If both ``dataloaders`` and ``datamodule`` are passed. Pass only one of these.

            RuntimeError:
                If a compiled ``model`` is passed and the strategy is not supported.

        """
        if model is None:
            # do we still have a reference from a previous call?
            if self.lightning_module is None:
                raise TypeError(
                    "`Trainer.test()` requires a `LightningModule` when it hasn't been passed in a previous run"
                )
        else:
            model = _maybe_unwrap_optimized(model)
            self.strategy._lightning_module = model
        _verify_strategy_supports_compile(self.lightning_module, self.strategy)
        self.state.fn = TrainerFn.TESTING
        self.state.status = TrainerStatus.RUNNING
        self.testing = True
        return call._call_and_handle_interrupt(
            self, self._test_impl, model, dataloaders, ckpt_path, verbose, datamodule
        )

    def _test_impl(
        self,
        model: Optional["pl.LightningModule"] = None,
        dataloaders: Optional[Union[EVAL_DATALOADERS, LightningDataModule]] = None,
        ckpt_path: Optional[_PATH] = None,
        verbose: bool = True,
        datamodule: Optional[LightningDataModule] = None,
    ) -> Optional[Union[_PREDICT_OUTPUT, _EVALUATE_OUTPUT]]:
        # --------------------
        # SETUP HOOK
        # --------------------
        log.debug(f"{self.__class__.__name__}: trainer test stage")

        # if a datamodule comes in as the second arg, then fix it for the user
        if isinstance(dataloaders, LightningDataModule):
            datamodule = dataloaders
            dataloaders = None
        # If you supply a datamodule you can't supply test_dataloaders
        if dataloaders is not None and datamodule:
            raise MisconfigurationException("You cannot pass both `trainer.test(dataloaders=..., datamodule=...)`")

        if model is None:
            model = self.lightning_module
            model_provided = False
        else:
            model_provided = True

        self.test_loop.verbose = verbose

        # links data to the trainer
        self._data_connector.attach_data(model, test_dataloaders=dataloaders, datamodule=datamodule)

        assert self.state.fn is not None
        ckpt_path = self._checkpoint_connector._select_ckpt_path(
            self.state.fn, ckpt_path, model_provided=model_provided, model_connected=self.lightning_module is not None
        )
        results = self._run(model, ckpt_path=ckpt_path)
        # remove the tensors from the test results
        results = convert_tensors_to_scalars(results)

        assert self.state.stopped
        self.testing = False

        return results

    def predict(
        self,
        model: Optional["pl.LightningModule"] = None,
        dataloaders: Optional[Union[EVAL_DATALOADERS, LightningDataModule]] = None,
        datamodule: Optional[LightningDataModule] = None,
        return_predictions: Optional[bool] = None,
        ckpt_path: Optional[_PATH] = None,
    ) -> Optional[_PREDICT_OUTPUT]:
        r"""Run inference on your data. This will call the model forward function to compute predictions. Useful to
        perform distributed and batched predictions. Logging is disabled in the predict hooks.

        Args:
            model: The model to predict with.

            dataloaders: An iterable or collection of iterables specifying predict samples.
                Alternatively, a :class:`~lightning.pytorch.core.datamodule.LightningDataModule` that defines
                the :class:`~lightning.pytorch.core.hooks.DataHooks.predict_dataloader` hook.

            datamodule: A :class:`~lightning.pytorch.core.datamodule.LightningDataModule` that defines
                the :class:`~lightning.pytorch.core.hooks.DataHooks.predict_dataloader` hook.

            return_predictions: Whether to return predictions.
                ``True`` by default except when an accelerator that spawns processes is used (not supported).

            ckpt_path: Either ``"best"``, ``"last"``, ``"hpc"`` or path to the checkpoint you wish to predict.
                If ``None`` and the model instance was passed, use the current weights.
                Otherwise, the best model checkpoint from the previous ``trainer.fit`` call will be loaded
                if a checkpoint callback is configured.

        For more information about multiple dataloaders, see this :ref:`section <multiple-dataloaders>`.

        Returns:
            Returns a list of dictionaries, one for each provided dataloader containing their respective predictions.

        Raises:
            TypeError:
                If no ``model`` is passed and there was no ``LightningModule`` passed in the previous run.
                If ``model`` passed is not `LightningModule` or `torch._dynamo.OptimizedModule`.

            MisconfigurationException:
                If both ``dataloaders`` and ``datamodule`` are passed. Pass only one of these.

            RuntimeError:
                If a compiled ``model`` is passed and the strategy is not supported.

        See :ref:`Lightning inference section<deploy/production_basic:Predict step with your LightningModule>` for more.

        """
        if model is None:
            # do we still have a reference from a previous call?
            if self.lightning_module is None:
                raise TypeError(
                    "`Trainer.predict()` requires a `LightningModule` when it hasn't been passed in a previous run"
                )
        else:
            model = _maybe_unwrap_optimized(model)
            self.strategy._lightning_module = model
        _verify_strategy_supports_compile(self.lightning_module, self.strategy)
        self.state.fn = TrainerFn.PREDICTING
        self.state.status = TrainerStatus.RUNNING
        self.predicting = True
        return call._call_and_handle_interrupt(
            self, self._predict_impl, model, dataloaders, datamodule, return_predictions, ckpt_path
        )

    def _predict_impl(
        self,
        model: Optional["pl.LightningModule"] = None,
        dataloaders: Optional[Union[EVAL_DATALOADERS, LightningDataModule]] = None,
        datamodule: Optional[LightningDataModule] = None,
        return_predictions: Optional[bool] = None,
        ckpt_path: Optional[_PATH] = None,
    ) -> Optional[_PREDICT_OUTPUT]:
        # --------------------
        # SETUP HOOK
        # --------------------
        log.debug(f"{self.__class__.__name__}: trainer predict stage")

        self.predict_loop.return_predictions = return_predictions  # type: ignore[assignment]

        # if a datamodule comes in as the second arg, then fix it for the user
        if isinstance(dataloaders, LightningDataModule):
            datamodule = dataloaders
            dataloaders = None
        if dataloaders is not None and datamodule:
            raise MisconfigurationException("You cannot pass both `trainer.predict(dataloaders=..., datamodule=...)`")

        if model is None:
            model = self.lightning_module
            model_provided = False
        else:
            model_provided = True

        # links data to the trainer
        self._data_connector.attach_data(model, predict_dataloaders=dataloaders, datamodule=datamodule)

        assert self.state.fn is not None
        ckpt_path = self._checkpoint_connector._select_ckpt_path(
            self.state.fn, ckpt_path, model_provided=model_provided, model_connected=self.lightning_module is not None
        )
        results = self._run(model, ckpt_path=ckpt_path)

        assert self.state.stopped
        self.predicting = False

        return results

    def _run(
        self, model: "pl.LightningModule", ckpt_path: Optional[_PATH] = None
    ) -> Optional[Union[_EVALUATE_OUTPUT, _PREDICT_OUTPUT]]:
        if self.state.fn == TrainerFn.FITTING:
            min_epochs, max_epochs = _parse_loop_limits(
                self.min_steps, self.max_steps, self.min_epochs, self.max_epochs, self
            )
            self.fit_loop.min_epochs = min_epochs
            self.fit_loop.max_epochs = max_epochs

        if self.barebones:
            # no progress bar in barebones can make it look like the Trainer hung
            rank_zero_info(
                "`Trainer(barebones=True)` started running. The progress bar is disabled so you might want to"
                " manually print the progress in your model."
            )

        # clean hparams
        if hasattr(model, "hparams"):
            parsing.clean_namespace(model.hparams)

        # attach model to the strategy
        self.strategy.connect(model)

        self._callback_connector._attach_model_callbacks()
        self._callback_connector._attach_model_logging_functions()

        _verify_loop_configurations(self)

        # ----------------------------
        # SET UP THE TRAINER
        # ----------------------------
        log.debug(f"{self.__class__.__name__}: setting up strategy environment")
        self.strategy.setup_environment()
        self.__setup_profiler()

        log.debug(f"{self.__class__.__name__}: preparing data")
        self._data_connector.prepare_data()

        call._call_setup_hook(self)  # allow user to set up LightningModule in accelerator environment
        log.debug(f"{self.__class__.__name__}: configuring model")
        call._call_configure_model(self)

        # check if we should delay restoring checkpoint till later
        if not self.strategy.restore_checkpoint_after_setup:
            log.debug(f"{self.__class__.__name__}: restoring module and callbacks from checkpoint path: {ckpt_path}")
            self._checkpoint_connector._restore_modules_and_callbacks(ckpt_path)

        # reset logger connector
        self._logger_connector.reset_results()
        self._logger_connector.reset_metrics()

        # strategy will configure model and move it to the device
        self.strategy.setup(self)

        # hook
        if self.state.fn == TrainerFn.FITTING:
            call._call_callback_hooks(self, "on_fit_start")
            call._call_lightning_module_hook(self, "on_fit_start")

        # only log hparams if enabled
        if self.enable_autolog_hparams:
            _log_hyperparams(self)

        if self.strategy.restore_checkpoint_after_setup:
            log.debug(f"{self.__class__.__name__}: restoring module and callbacks from checkpoint path: {ckpt_path}")
            self._checkpoint_connector._restore_modules_and_callbacks(ckpt_path)

        # restore optimizers, etc.
        log.debug(f"{self.__class__.__name__}: restoring training state")
        self._checkpoint_connector.restore_training_state()

        self._checkpoint_connector.resume_end()

        self._signal_connector.register_signal_handlers()

        # ----------------------------
        # RUN THE TRAINER
        # ----------------------------
        results = self._run_stage()

        # ----------------------------
        # POST-Training CLEAN UP
        # ----------------------------
        log.debug(f"{self.__class__.__name__}: trainer tearing down")
        self._teardown()

        if self.state.fn == TrainerFn.FITTING:
            call._call_callback_hooks(self, "on_fit_end")
            call._call_lightning_module_hook(self, "on_fit_end")

        log.debug(f"{self.__class__.__name__}: calling teardown hooks")
        call._call_teardown_hook(self)

        self.state.status = TrainerStatus.FINISHED
        self.state.stage = None

        return results

    def _teardown(self) -> None:
        """This is the Trainer's internal teardown, unrelated to the `teardown` hooks in LightningModule and Callback;
        those are handled by :meth:`_call_teardown_hook`."""
        self.strategy.teardown()
        loop = self._active_loop
        # loop should never be `None` here but it can because we don't know the trainer stage with `ddp_spawn`
        if loop is not None:
            loop.teardown()
        self._logger_connector.teardown()
        self._signal_connector.teardown()

    def _run_stage(self) -> Optional[Union[_PREDICT_OUTPUT, _EVALUATE_OUTPUT]]:
        # wait for all to join if on distributed
        self.strategy.barrier("run-stage")
        self.lightning_module.zero_grad()

        if self.evaluating:
            return self._evaluation_loop.run()
        if self.predicting:
            return self.predict_loop.run()
        if self.training:
            with isolate_rng():
                self._run_sanity_check()
            with torch.autograd.set_detect_anomaly(self._detect_anomaly):
                self.fit_loop.run()
            return None
        raise RuntimeError(f"Unexpected state {self.state}")

    def _run_sanity_check(self) -> None:
        val_loop = self.fit_loop.epoch_loop.val_loop

        should_sanity_check = (
            self.enable_validation
            and self.num_sanity_val_steps > 0
            # do not sanity check if restarting because it would mess up the loaded state
            and not val_loop.restarting
        )

        # run tiny validation (if validation defined)
        # to make sure program won't crash during val
        if should_sanity_check:
            stage = self.state.stage
            self.sanity_checking = True

            # reset logger connector
            self._logger_connector.reset_results()
            self._logger_connector.reset_metrics()

            call._call_callback_hooks(self, "on_sanity_check_start")

            # run eval step
            val_loop.run()

            call._call_callback_hooks(self, "on_sanity_check_end")

            # reset logger connector
            self._logger_connector.reset_results()
            self._logger_connector.reset_metrics()

            # reset the progress tracking state after sanity checking. we don't need to set the state before
            # because sanity check only runs when we are not restarting
            _reset_progress(val_loop)

            # restore the previous stage when the sanity check if finished
            self.state.stage = stage

    def __setup_profiler(self) -> None:
        assert self.state.fn is not None
        local_rank = self.local_rank if self.world_size > 1 else None
        self.profiler._lightning_module = proxy(self.lightning_module)
        self.profiler.setup(stage=self.state.fn, local_rank=local_rank, log_dir=self.log_dir)

    @contextmanager
    def init_module(self, empty_init: Optional[bool] = None) -> Generator:
        """Tensors that you instantiate under this context manager will be created on the device right away and have
        the right data type depending on the precision setting in the Trainer.

        The parameters and tensors get created on the device and with the right data type right away without wasting
        memory being allocated unnecessarily.

        Args:
            empty_init: Whether to initialize the model with empty weights (uninitialized memory).
                If ``None``, the strategy will decide. Some strategies may not support all options.
                Set this to ``True`` if you are loading a checkpoint into a large model.

        """
        if is_overridden("model_sharded_context", self.strategy, parent=Strategy):
            # warning instead of error so that code changes are not required when changing strategies
            # this is a limitation because processes are not expected to have been launched when this is called
            rank_zero_warn(
                f"`trainer.init_module` cannot fully support proper instantiation of your model with the"
                f" `{type(self.strategy).__name__}` strategy. Please instantiate your model inside the"
                f"`LightningModule.configure_model` hook instead",
                # ideally we would check if `configure_model` is already overridden, but we don't have a reliable
                # reference to the model yet
                category=PossibleUserWarning,
            )
        with self.strategy.tensor_init_context(empty_init=empty_init):
            yield

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Print something only on the first process. If running on multiple machines, it will print from the first
        process in each machine.

        Arguments passed to this method are forwarded to the Python built-in :func:`print` function.

        """
        if self.local_rank == 0:
            print(*args, **kwargs)

    """
    Accelerator properties
    """

    @property
    def accelerator(self) -> Accelerator:
        assert self.strategy.accelerator
        return self.strategy.accelerator

    @property
    def strategy(self) -> Strategy:
        return self._accelerator_connector.strategy

    @property
    def precision_plugin(self) -> Precision:
        return self.strategy.precision_plugin

    @property
    def global_rank(self) -> int:
        return self.strategy.global_rank

    @property
    def local_rank(self) -> int:
        # some strategies define a local rank
        return getattr(self.strategy, "local_rank", 0)

    @property
    def node_rank(self) -> int:
        # some strategies define a node rank
        return getattr(self.strategy, "node_rank", 0)

    @property
    def world_size(self) -> int:
        # some strategies define a world size
        return getattr(self.strategy, "world_size", 1)

    @property
    def num_nodes(self) -> int:
        return getattr(self.strategy, "num_nodes", 1)

    @property
    def device_ids(self) -> list[int]:
        """List of device indexes per node."""
        devices = (
            self.strategy.parallel_devices
            if isinstance(self.strategy, ParallelStrategy)
            else [self.strategy.root_device]
        )
        assert devices is not None
        device_ids = []
        for idx, device in enumerate(devices):
            if isinstance(device, torch.device):
                device_ids.append(device.index or idx)
            elif isinstance(device, int):
                device_ids.append(device)
        return device_ids

    @property
    def num_devices(self) -> int:
        """Number of devices the trainer uses per node."""
        return len(self.device_ids)

    @property
    def lightning_module(self) -> "pl.LightningModule":
        # TODO: this is actually an optional return
        return self.strategy.lightning_module  # type: ignore[return-value]

    @property
    def optimizers(self) -> list[Optimizer]:
        return self.strategy.optimizers

    @optimizers.setter
    def optimizers(self, new_optims: list[Optimizer]) -> None:
        self.strategy.optimizers = new_optims

    @property
    def lr_scheduler_configs(self) -> list[LRSchedulerConfig]:
        return self.strategy.lr_scheduler_configs

    @property
    def precision(self) -> _PRECISION_INPUT_STR:
        return self.strategy.precision_plugin.precision

    @property
    def scaler(self) -> Optional[Any]:
        return getattr(self.precision_plugin, "scaler", None)

    @property
    def model(self) -> Optional[torch.nn.Module]:
        """The LightningModule, but possibly wrapped into DataParallel or DistributedDataParallel.

        To access the pure LightningModule, use
        :meth:`~lightning.pytorch.trainer.trainer.Trainer.lightning_module` instead.

        """
        return self.strategy.model

    """
    General properties
    """

    @property
    def log_dir(self) -> Optional[str]:
        """The directory for the current experiment. Use this to save images to, etc...

        .. note:: You must call this on all processes. Failing to do so will cause your program to stall forever.

         .. code-block:: python

             def training_step(self, batch, batch_idx):
                 img = ...
                 save_img(img, self.trainer.log_dir)

        """
        if len(self.loggers) > 0:
            if not isinstance(self.loggers[0], (TensorBoardLogger, CSVLogger)):
                dirpath = self.loggers[0].save_dir
            else:
                dirpath = self.loggers[0].log_dir
        else:
            dirpath = self.default_root_dir

        dirpath = self.strategy.broadcast(dirpath)
        return dirpath

    @property
    def is_global_zero(self) -> bool:
        """Whether this process is the global zero in multi-node training.

        .. code-block:: python

            def training_step(self, batch, batch_idx):
                if self.trainer.is_global_zero:
                    print("in node 0, accelerator 0")

        """
        return self.strategy.is_global_zero

    @property
    def distributed_sampler_kwargs(self) -> Optional[dict[str, Any]]:
        if isinstance(self.strategy, ParallelStrategy):
            return self.strategy.distributed_sampler_kwargs
        return None

    @property
    def enable_validation(self) -> bool:
        """Check if we should run validation during training."""
        return (
            self.fit_loop.epoch_loop.val_loop._data_source.is_defined()
            and is_overridden("validation_step", self.lightning_module)
            and self.limit_val_batches > 0
        )

    @property
    def default_root_dir(self) -> str:
        """The default location to save artifacts of loggers, checkpoints etc.

        It is used as a fallback if logger or checkpoint callback do not define specific save paths.

        """
        if _is_local_file_protocol(self._default_root_dir):
            return os.path.normpath(os.path.expanduser(self._default_root_dir))
        return self._default_root_dir

    @property
    def early_stopping_callback(self) -> Optional[EarlyStopping]:
        """The first :class:`~lightning.pytorch.callbacks.early_stopping.EarlyStopping` callback in the
        Trainer.callbacks list, or ``None`` if it doesn't exist."""
        callbacks = self.early_stopping_callbacks
        return callbacks[0] if len(callbacks) > 0 else None

    @property
    def early_stopping_callbacks(self) -> list[EarlyStopping]:
        """A list of all instances of :class:`~lightning.pytorch.callbacks.early_stopping.EarlyStopping` found in the
        Trainer.callbacks list."""
        return [c for c in self.callbacks if isinstance(c, EarlyStopping)]

    @property
    def checkpoint_callback(self) -> Optional[Checkpoint]:
        """The first :class:`~lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint` callback in the
        Trainer.callbacks list, or ``None`` if it doesn't exist."""
        callbacks = self.checkpoint_callbacks
        return callbacks[0] if len(callbacks) > 0 else None

    @property
    def checkpoint_callbacks(self) -> list[Checkpoint]:
        """A list of all instances of :class:`~lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint` found in
        the Trainer.callbacks list."""
        return [c for c in self.callbacks if isinstance(c, Checkpoint)]

    @property
    def progress_bar_callback(self) -> Optional[ProgressBar]:
        """An instance of :class:`~lightning.pytorch.callbacks.progress.progress_bar.ProgressBar` found in the
        Trainer.callbacks list, or ``None`` if one doesn't exist."""
        for c in self.callbacks:
            if isinstance(c, ProgressBar):
                return c
        return None

    @property
    def ckpt_path(self) -> Optional[_PATH]:
        """Set to the path/URL of a checkpoint loaded via :meth:`~lightning.pytorch.trainer.trainer.Trainer.fit`,
        :meth:`~lightning.pytorch.trainer.trainer.Trainer.validate`,
        :meth:`~lightning.pytorch.trainer.trainer.Trainer.test`, or
        :meth:`~lightning.pytorch.trainer.trainer.Trainer.predict`.

        ``None`` otherwise.

        """
        return self._checkpoint_connector._ckpt_path

    @ckpt_path.setter
    def ckpt_path(self, ckpt_path: Optional[_PATH]) -> None:
        """Allows you to manage which checkpoint is loaded statefully.

        .. code-block:: python

            trainer = Trainer()
            trainer.ckpt_path = "my/checkpoint/file.ckpt"
            trainer.fit(model)
            ...

            # you will be in charge of resetting this
            trainer.ckpt_path = None
            trainer.test(model)

        """
        self._checkpoint_connector._ckpt_path = ckpt_path
        self._checkpoint_connector._user_managed = bool(ckpt_path)

    def save_checkpoint(
        self, filepath: _PATH, weights_only: bool = False, storage_options: Optional[Any] = None
    ) -> None:
        r"""Runs routine to create a checkpoint.

        This method needs to be called on all processes in case the selected strategy is handling distributed
        checkpointing.

        Args:
            filepath: Path where checkpoint is saved.
            weights_only: If ``True``, will only save the model weights.
            storage_options: parameter for how to save to storage, passed to ``CheckpointIO`` plugin

        Raises:
            AttributeError:
                If the model is not attached to the Trainer before calling this method.

        """
        if self.model is None:
            raise AttributeError(
                "Saving a checkpoint is only possible if a model is attached to the Trainer. Did you call"
                " `Trainer.save_checkpoint()` before calling `Trainer.{fit,validate,test,predict}`?"
            )
        with self.profiler.profile("save_checkpoint"):
            checkpoint = self._checkpoint_connector.dump_checkpoint(weights_only)
            self.strategy.save_checkpoint(checkpoint, filepath, storage_options=storage_options)
            self.strategy.barrier("Trainer.save_checkpoint")

    """
    State properties
    """

    @property
    def interrupted(self) -> bool:
        return self.state.status == TrainerStatus.INTERRUPTED

    @property
    def training(self) -> bool:
        return self.state.stage == RunningStage.TRAINING

    @training.setter
    def training(self, val: bool) -> None:
        if val:
            self.state.stage = RunningStage.TRAINING
        elif self.training:
            self.state.stage = None

    @property
    def testing(self) -> bool:
        return self.state.stage == RunningStage.TESTING

    @testing.setter
    def testing(self, val: bool) -> None:
        if val:
            self.state.stage = RunningStage.TESTING
        elif self.testing:
            self.state.stage = None

    @property
    def predicting(self) -> bool:
        return self.state.stage == RunningStage.PREDICTING

    @predicting.setter
    def predicting(self, val: bool) -> None:
        if val:
            self.state.stage = RunningStage.PREDICTING
        elif self.predicting:
            self.state.stage = None

    @property
    def validating(self) -> bool:
        return self.state.stage == RunningStage.VALIDATING

    @validating.setter
    def validating(self, val: bool) -> None:
        if val:
            self.state.stage = RunningStage.VALIDATING
        elif self.validating:
            self.state.stage = None

    @property
    def evaluating(self) -> bool:
        return self.state.stage is not None and self.state.stage.evaluating

    @property
    def sanity_checking(self) -> bool:
        """Whether sanity checking is running.

        Useful to disable some hooks, logging or callbacks during the sanity checking.

        """
        return self.state.stage == RunningStage.SANITY_CHECKING

    @sanity_checking.setter
    def sanity_checking(self, val: bool) -> None:
        if val:
            self.state.stage = RunningStage.SANITY_CHECKING
        elif self.sanity_checking:
            self.state.stage = None

    @property
    def received_sigterm(self) -> bool:
        """Whether a ``signal.SIGTERM`` signal was received.

        For example, this can be checked to exit gracefully.

        """
        return self._signal_connector.received_sigterm

    """
    Loop properties
    """

    @property
    def global_step(self) -> int:
        """The number of optimizer steps taken (does not reset each epoch).

        This includes multiple optimizers (if enabled).

        """
        return self.fit_loop.epoch_loop.global_step

    @property
    def current_epoch(self) -> int:
        """The current epoch, updated after the epoch end hooks are run."""
        return self.fit_loop.epoch_progress.current.completed

    @property
    def max_epochs(self) -> Optional[int]:
        return self.fit_loop.max_epochs

    @property
    def min_epochs(self) -> Optional[int]:
        return self.fit_loop.min_epochs

    @property
    def max_steps(self) -> int:
        return self.fit_loop.max_steps

    @property
    def min_steps(self) -> Optional[int]:
        return self.fit_loop.min_steps

    @property
    def is_last_batch(self) -> bool:
        """Whether trainer is executing the last batch."""
        return self.fit_loop.epoch_loop.batch_progress.is_last_batch

    @property
    def train_dataloader(self) -> Optional[TRAIN_DATALOADERS]:
        """The training dataloader(s) used during ``trainer.fit()``."""
        if (combined_loader := self.fit_loop._combined_loader) is not None:
            return combined_loader.iterables
        return None

    @property
    def val_dataloaders(self) -> Optional[EVAL_DATALOADERS]:
        """The validation dataloader(s) used during ``trainer.fit()`` or ``trainer.validate()``."""
        if (combined_loader := self.fit_loop.epoch_loop.val_loop._combined_loader) is not None or (
            combined_loader := self.validate_loop._combined_loader
        ) is not None:
            return combined_loader.iterables
        return None

    @property
    def test_dataloaders(self) -> Optional[EVAL_DATALOADERS]:
        """The test dataloader(s) used during ``trainer.test()``."""
        if (combined_loader := self.test_loop._combined_loader) is not None:
            return combined_loader.iterables
        return None

    @property
    def predict_dataloaders(self) -> Optional[EVAL_DATALOADERS]:
        """The prediction dataloader(s) used during ``trainer.predict()``."""
        if (combined_loader := self.predict_loop._combined_loader) is not None:
            return combined_loader.iterables
        return None

    @property
    def num_training_batches(self) -> Union[int, float]:
        """The number of training batches that will be used during ``trainer.fit()``."""
        return self.fit_loop.max_batches

    @property
    def num_sanity_val_batches(self) -> list[Union[int, float]]:
        """The number of validation batches that will be used during the sanity-checking part of ``trainer.fit()``."""
        max_batches = self.fit_loop.epoch_loop.val_loop.max_batches
        # re-compute the `min` in case this is called outside the sanity-checking stage
        return [min(self.num_sanity_val_steps, batches) for batches in max_batches]

    @property
    def num_val_batches(self) -> list[Union[int, float]]:
        """The number of validation batches that will be used during ``trainer.fit()`` or ``trainer.validate()``."""
        if self.state.fn == TrainerFn.VALIDATING:
            return self.validate_loop.max_batches
        # if no trainer.fn is set, assume fit's validation
        # use the protected access, because it shouldn't return the sanity_val batches
        return self.fit_loop.epoch_loop.val_loop._max_batches

    @property
    def num_test_batches(self) -> list[Union[int, float]]:
        """The number of test batches that will be used during ``trainer.test()``."""
        return self.test_loop.max_batches

    @property
    def num_predict_batches(self) -> list[Union[int, float]]:
        """The number of prediction batches that will be used during ``trainer.predict()``."""
        return self.predict_loop.max_batches

    @property
    def _evaluation_loop(self) -> _EvaluationLoop:
        if self.state.fn == TrainerFn.FITTING:
            return self.fit_loop.epoch_loop.val_loop
        if self.state.fn == TrainerFn.VALIDATING:
            return self.validate_loop
        if self.state.fn == TrainerFn.TESTING:
            return self.test_loop
        raise RuntimeError("The `Trainer._evaluation_loop` property isn't defined. Accessed outside of scope")

    @property
    def _active_loop(self) -> Optional[Union[_FitLoop, _EvaluationLoop, _PredictionLoop]]:
        if self.training:
            return self.fit_loop
        if self.sanity_checking or self.evaluating:
            return self._evaluation_loop
        if self.predicting:
            return self.predict_loop
        return None

    """
    Logging properties
    """

    @property
    def logger(self) -> Optional[Logger]:
        """The first :class:`~lightning.pytorch.loggers.logger.Logger` being used."""
        return self.loggers[0] if len(self.loggers) > 0 else None

    @logger.setter
    def logger(self, logger: Optional[Logger]) -> None:
        if not logger:
            self.loggers = []
        else:
            self.loggers = [logger]

    @property
    def loggers(self) -> list[Logger]:
        """The list of :class:`~lightning.pytorch.loggers.logger.Logger` used.

        .. code-block:: python

            for logger in trainer.loggers:
                logger.log_metrics({"foo": 1.0})

        """
        return self._loggers

    @loggers.setter
    def loggers(self, loggers: Optional[list[Logger]]) -> None:
        self._loggers = loggers if loggers else []

    @property
    def callback_metrics(self) -> _OUT_DICT:
        """The metrics available to callbacks.

        .. code-block:: python

            def training_step(self, batch, batch_idx):
                self.log("a_val", 2.0)


            callback_metrics = trainer.callback_metrics
            assert callback_metrics["a_val"] == 2.0

        """
        return self._logger_connector.callback_metrics

    @property
    def logged_metrics(self) -> _OUT_DICT:
        """The metrics sent to the loggers.

        This includes metrics logged via :meth:`~lightning.pytorch.core.LightningModule.log` with the
        :paramref:`~lightning.pytorch.core.LightningModule.log.logger` argument set.

        """
        return self._logger_connector.logged_metrics

    @property
    def progress_bar_metrics(self) -> _PBAR_DICT:
        """The metrics sent to the progress bar.

        This includes metrics logged via :meth:`~lightning.pytorch.core.LightningModule.log` with the
        :paramref:`~lightning.pytorch.core.LightningModule.log.prog_bar` argument set.

        """
        return self._logger_connector.progress_bar_metrics

    @property
    def _results(self) -> Optional[_ResultCollection]:
        active_loop = self._active_loop
        if active_loop is not None:
            return active_loop._results
        return None

    """
    Other
    """

    @property
    def estimated_stepping_batches(self) -> Union[int, float]:
        r"""The estimated number of batches that will ``optimizer.step()`` during training.

        This accounts for gradient accumulation and the current trainer configuration. This might be used when setting
        up your training dataloader, if it hasn't been set up already.

        .. code-block:: python

            def configure_optimizers(self):
                optimizer = ...
                stepping_batches = self.trainer.estimated_stepping_batches
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=stepping_batches)
                return [optimizer], [scheduler]

        Raises:
            MisconfigurationException:
                If estimated stepping batches cannot be computed due to different `accumulate_grad_batches`
                at different epochs.

        """
        # infinite training
        if self.max_epochs == -1:
            return float("inf") if self.max_steps == -1 else self.max_steps

        if self.train_dataloader is None:
            rank_zero_info("Loading `train_dataloader` to estimate number of stepping batches.")
            self.fit_loop.setup_data()

        total_batches = self.num_training_batches

        # iterable dataset
        if total_batches == float("inf"):
            return self.max_steps

        assert self.max_epochs is not None
        max_estimated_steps = math.ceil(total_batches / self.accumulate_grad_batches) * max(self.max_epochs, 1)

        max_estimated_steps = min(max_estimated_steps, self.max_steps) if self.max_steps != -1 else max_estimated_steps
        return max_estimated_steps
