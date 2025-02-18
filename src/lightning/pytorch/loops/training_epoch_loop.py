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
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional, Union

from typing_extensions import override

import lightning.pytorch as pl
from lightning.fabric.utilities.types import _Stateful
from lightning.fabric.utilities.warnings import PossibleUserWarning
from lightning.pytorch import loops  # import as loops to avoid circular imports
from lightning.pytorch.loops.fetchers import _DataFetcher, _DataLoaderIterDataFetcher
from lightning.pytorch.loops.optimization import _AutomaticOptimization, _ManualOptimization
from lightning.pytorch.loops.optimization.automatic import _OUTPUTS_TYPE as _OPTIMIZER_LOOP_OUTPUTS_TYPE
from lightning.pytorch.loops.optimization.manual import _OUTPUTS_TYPE as _MANUAL_LOOP_OUTPUTS_TYPE
from lightning.pytorch.loops.progress import _BatchProgress, _SchedulerProgress
from lightning.pytorch.loops.utilities import _is_max_limit_reached
from lightning.pytorch.trainer import call
from lightning.pytorch.trainer.connectors.logger_connector.result import _ResultCollection
from lightning.pytorch.trainer.states import RunningStage, TrainerFn
from lightning.pytorch.utilities.exceptions import MisconfigurationException, SIGTERMException
from lightning.pytorch.utilities.rank_zero import WarningCache, rank_zero_warn
from lightning.pytorch.utilities.signature_utils import is_param_in_hook_signature

_BATCH_OUTPUTS_TYPE = Optional[Union[_OPTIMIZER_LOOP_OUTPUTS_TYPE, _MANUAL_LOOP_OUTPUTS_TYPE]]


@dataclass
class RestartStage:
    NONE = "none"
    RESTARTED_ON_TRAIN_BATCH_END = "restarted_on_train_batch_end"
    RESTARTED_ON_LAST = "restarted_on_last"


class _TrainingEpochLoop(loops._Loop):
    """Iterates over all batches in the dataloader (one epoch) that the user returns in their
    :meth:`~lightning.pytorch.core.LightningModule.train_dataloader` method.

    Its main responsibilities are calling the ``*_epoch_{start,end}`` hooks, accumulating outputs if the user request
    them in one of these hooks, and running validation at the requested interval.

    The validation is carried out by yet another loop,
    :class:`~lightning.pytorch.loops._EvaluationLoop`.

    In the ``run()`` method, the training epoch loop could in theory simply call the
    ``LightningModule.training_step`` already and perform the optimization.
    However, Lightning has built-in support for automatic optimization with multiple optimizers.
    For this reason there are actually two more loops nested under
    :class:`~lightning.pytorch.loops._TrainingEpochLoop`.

    Args:
        min_steps: The minimum number of steps (batches) to process
        max_steps: The maximum number of steps (batches) to process

    """

    def __init__(self, trainer: "pl.Trainer", min_steps: Optional[int] = None, max_steps: int = -1) -> None:
        super().__init__(trainer)
        if max_steps < -1:
            raise MisconfigurationException(
                f"`max_steps` must be a non-negative integer or -1 (infinite steps). You passed in {max_steps}."
            )
        self.min_steps = min_steps
        self.max_steps = max_steps

        self.batch_progress = _BatchProgress()
        self.scheduler_progress = _SchedulerProgress()

        self.automatic_optimization = _AutomaticOptimization(trainer)
        self.manual_optimization = _ManualOptimization(trainer)

        self.val_loop = loops._EvaluationLoop(
            trainer, TrainerFn.FITTING, RunningStage.VALIDATING, verbose=False, inference_mode=False
        )

        self._results = _ResultCollection(training=True)
        self._warning_cache = WarningCache()
        self._batches_that_stepped: int = 0
        self._restart_stage = RestartStage.NONE
        self._skip_next_val = False

    @property
    def total_batch_idx(self) -> int:
        """Returns the current batch index (across epochs)"""
        # use `ready` instead of `completed` in case this is accessed after `completed` has been increased
        # but before the next `ready` increase
        return self.batch_progress.total.ready - 1

    @property
    def batch_idx(self) -> int:
        """Returns the current batch index (within this epoch)"""
        # use `ready` instead of `completed` in case this is accessed after `completed` has been increased
        # but before the next `ready` increase
        return self.batch_progress.current.ready - 1

    @property
    def global_step(self) -> int:
        lightning_module = self.trainer.lightning_module
        if lightning_module is None or lightning_module.automatic_optimization:
            return self.automatic_optimization.optim_progress.optimizer_steps
        return self.manual_optimization.optim_step_progress.total.completed

    @property
    def _is_training_done(self) -> bool:
        max_steps_reached = _is_max_limit_reached(self.global_step, self.max_steps)
        return max_steps_reached or self._num_ready_batches_reached()

    @property
    def _is_validation_done(self) -> bool:
        # when we are restarting we want to check whether the val loop has finished
        return not self.restarting or self.val_loop._has_run

    @property
    def done(self) -> bool:
        """Evaluates when to leave the loop."""
        if self._is_training_done and self._is_validation_done:
            return True

        if self.trainer.should_stop:
            # early stopping
            min_epochs = self.trainer.fit_loop.min_epochs
            can_stop_early = self.trainer.fit_loop._can_stop_early
            if not can_stop_early:
                self._warning_cache.info(
                    f"Trainer was signaled to stop but the required `min_epochs={min_epochs!r}` or"
                    f" `min_steps={self.min_steps!r}` has not been met. Training will continue..."
                )
            return can_stop_early

        return False

    def run(self, data_fetcher: _DataFetcher) -> None:
        self.reset()
        self.on_run_start(data_fetcher)
        while not self.done:
            try:
                self.advance(data_fetcher)
                self.on_advance_end(data_fetcher)
            except StopIteration:
                break
            finally:
                self.on_iteration_done()

    @property
    def restarted_on_train_batch_end(self) -> bool:
        return self._restart_stage == RestartStage.RESTARTED_ON_TRAIN_BATCH_END

    @property
    def restarted_on_last(self) -> bool:
        return self._restart_stage == RestartStage.RESTARTED_ON_LAST

    def update_restart_stage(self) -> None:
        if (
            self.restarting
            and self.batch_progress.total.started == self.batch_progress.total.ready
            and self.batch_progress.total.processed == self.batch_progress.total.started
            and self.batch_progress.total.completed == self.batch_progress.total.processed - 1
        ):
            self._restart_stage = RestartStage.RESTARTED_ON_TRAIN_BATCH_END
        elif (
            self.restarting
            and self.batch_progress.total.started == self.batch_progress.total.ready
            and self.batch_progress.total.processed == self.batch_progress.total.started
            and self.batch_progress.total.completed == self.batch_progress.total.processed
        ):
            self._restart_stage = RestartStage.RESTARTED_ON_LAST
        else:
            self._restart_stage = RestartStage.NONE

        self.val_loop.update_restart_stage()

    def reset_restart_stage(self) -> None:
        self._restart_stage = RestartStage.NONE

    def reset(self) -> None:
        """Resets the internal state of the loop for a new run."""
        if (
            self.restarting
            and not self._should_accumulate()
            and (self.restarted_on_train_batch_end or not self.restarted_on_last)
        ):
            # batches_that_stepped is never set prior to saving a checkpoint, even when saving
            # happens on_validation_end
            # we could set it in the checkpoint but we prefer to keep checkpoints backward compatible
            self._batches_that_stepped += 1

        if self.restarted_on_train_batch_end:
            self.batch_progress.increment_completed()
            # handle situation in which save happened on_train_batch_end and epoch is at end
            if self.batch_progress.current.completed >= self.trainer.num_training_batches:
                self.batch_progress.reset_on_run()
                self.scheduler_progress.reset_on_run()
                self.automatic_optimization.optim_progress.reset_on_run()
                self.val_loop.batch_progress.total.reset()

        if self.restarting:
            self.batch_progress.reset_on_restart()
            self.scheduler_progress.reset_on_restart()
            self.automatic_optimization.optim_progress.reset_on_restart()

            trainer = self.trainer
            if trainer.num_training_batches != float("inf"):
                expected_steps = math.ceil(trainer.num_training_batches / trainer.accumulate_grad_batches)
                loader = trainer.fit_loop._combined_loader
                assert loader is not None
                is_resumable_loader = all(isinstance(loader, _Stateful) for loader in loader.flattened)
                if self.global_step % expected_steps != 0 and not is_resumable_loader:
                    rank_zero_warn(
                        "You're resuming from a checkpoint that ended before the epoch ended and your dataloader is"
                        " not resumable. This can cause unreliable results if further training is done."
                        " Consider using an end-of-epoch checkpoint or make your dataloader resumable by implementing"
                        " the `state_dict` / `load_state_dict` interface.",
                        category=PossibleUserWarning,
                    )
        else:
            self.batch_progress.reset_on_run()
            self.scheduler_progress.reset_on_run()
            self.automatic_optimization.optim_progress.reset_on_run()
            # when the epoch starts, the total val batch progress should be reset as it's supposed to count the batches
            # seen per epoch, this is useful for tracking when validation is run multiple times per epoch
            self.val_loop.batch_progress.total.reset()

    def on_run_start(self, data_fetcher: _DataFetcher) -> None:
        # `iter()` was called once in `FitLoop.setup_data()` already
        if self.trainer.current_epoch > 0 and not self.restarting:
            iter(data_fetcher)  # creates the iterator inside the fetcher

        # add the previous `fetched` value to properly track `is_last_batch` with no prefetching
        data_fetcher.fetched += self.batch_progress.current.ready
        data_fetcher._start_profiler = self._on_before_fetch
        data_fetcher._stop_profiler = self._on_after_fetch

    def _on_before_fetch(self) -> None:
        self.trainer.profiler.start(f"[{self.__class__.__name__}].train_dataloader_next")

    def _on_after_fetch(self) -> None:
        self.trainer.profiler.stop(f"[{self.__class__.__name__}].train_dataloader_next")

    def advance(self, data_fetcher: _DataFetcher) -> None:
        """Runs a single training batch.

        Raises:
            StopIteration: When the epoch is canceled by the user returning -1

        """
        if self.restarting and self._should_check_val_fx(data_fetcher):
            if self.val_loop.restarted_mid_evaluation:
                # Go back and finish running validation
                return

            if self.restarted_on_last:
                # Avoid running validation again if we saved on last
                self._skip_next_val = True
                return

            # fast forward progress counters to end of validation
            self.val_loop.increment_progress_to_evaluation_end()

        # we are going to train first so the val loop does not need to restart
        self.val_loop.restarting = False

        if using_dataloader_iter := isinstance(data_fetcher, _DataLoaderIterDataFetcher):
            dataloader_iter = next(data_fetcher)
            # hook's batch_idx and dataloader_idx arguments correctness cannot be guaranteed in this setting
            batch = data_fetcher._batch
            batch_idx = data_fetcher._batch_idx
        else:
            dataloader_iter = None
            batch, _, __ = next(data_fetcher)
            # TODO: we should instead use the batch_idx returned by the fetcher, however, that will require saving the
            # fetcher state so that the batch_idx is correct after restarting
            batch_idx = self.batch_idx + 1
        # Note: `is_last_batch` is not yet determined if data fetcher is a `_DataLoaderIterDataFetcher`
        self.batch_progress.is_last_batch = data_fetcher.done

        trainer = self.trainer
        if not using_dataloader_iter:
            batch = trainer.precision_plugin.convert_input(batch)
            batch = trainer.lightning_module._on_before_batch_transfer(batch, dataloader_idx=0)
            batch = call._call_strategy_hook(trainer, "batch_to_device", batch, dataloader_idx=0)

        self.batch_progress.increment_ready()
        trainer._logger_connector.on_batch_start(batch)

        batch_output: _BATCH_OUTPUTS_TYPE = None  # for mypy
        if batch is None and not using_dataloader_iter:
            self._warning_cache.warn("train_dataloader yielded None. If this was on purpose, ignore this warning...")
        else:
            # hook
            call._call_callback_hooks(trainer, "on_train_batch_start", batch, batch_idx)
            response = call._call_lightning_module_hook(trainer, "on_train_batch_start", batch, batch_idx)
            call._call_strategy_hook(trainer, "on_train_batch_start", batch, batch_idx)
            if response == -1:
                self.batch_progress.increment_processed()
                raise StopIteration

            self.batch_progress.increment_started()

            kwargs = (
                self._build_kwargs(OrderedDict(), batch, batch_idx)
                if not using_dataloader_iter
                else OrderedDict(any=dataloader_iter)
            )
            with trainer.profiler.profile("run_training_batch"):
                if trainer.lightning_module.automatic_optimization:
                    # in automatic optimization, there can only be one optimizer
                    batch_output = self.automatic_optimization.run(trainer.optimizers[0], batch_idx, kwargs)
                else:
                    batch_output = self.manual_optimization.run(kwargs)

        self.batch_progress.increment_processed()

        # update non-plateau LR schedulers
        # update epoch-interval ones only when we are at the end of training epoch
        self.update_lr_schedulers("step", update_plateau_schedulers=False)
        if self._num_ready_batches_reached():
            self.update_lr_schedulers("epoch", update_plateau_schedulers=False)

        if using_dataloader_iter:
            # update the hook kwargs now that the step method might have consumed the iterator
            batch = data_fetcher._batch
            batch_idx = data_fetcher._batch_idx
            # update `is_last_batch` again after dataloader_iter was fetched in `training_step()`
            self.batch_progress.is_last_batch = data_fetcher.done

        call._call_callback_hooks(trainer, "on_train_batch_end", batch_output, batch, batch_idx)
        call._call_lightning_module_hook(trainer, "on_train_batch_end", batch_output, batch, batch_idx)
        trainer._logger_connector.on_batch_end()

        self.batch_progress.increment_completed()

        # -----------------------------------------
        # SAVE METRICS TO LOGGERS AND PROGRESS_BAR
        # -----------------------------------------
        trainer._logger_connector.update_train_step_metrics()

    def on_advance_end(self, data_fetcher: _DataFetcher) -> None:
        # -----------------------------------------
        # VALIDATE IF NEEDED
        # -----------------------------------------
        should_check_val = self._should_check_val_fx(data_fetcher)

        if self._skip_next_val:
            should_check_val = False
            self._skip_next_val = False

        if should_check_val:
            # this needs to be set so the correct `trainer._active_loop` is picked
            self.trainer.validating = True
            # save and reset this state in case validation runs inside training loop (val_check_interval<1.0)
            first_loop_iter = self.trainer._logger_connector._first_loop_iter

            if not self._should_accumulate():
                # clear gradients to not leave any unused memory during validation
                call._call_lightning_module_hook(self.trainer, "on_validation_model_zero_grad")

            self.val_loop.run()
            self.trainer.training = True
            self.trainer._logger_connector._first_loop_iter = first_loop_iter

        # update plateau LR scheduler after metrics are logged
        self.update_lr_schedulers("step", update_plateau_schedulers=True)

        if not self._should_accumulate():
            # this is increased once per batch disregarding multiple optimizers on purpose for loggers
            self._batches_that_stepped += 1
        # this will save based on the `batches_that_stepped` value
        self._save_loggers_on_train_batch_end()

        # if training finished, defer exit to the parent. this assumes there will be enough time in between
        # which might not be the case depending on what's in the `*_epoch_end` hooks
        if not self._is_training_done and self.trainer.received_sigterm:
            raise SIGTERMException

    def teardown(self) -> None:
        self._results.cpu()
        self.val_loop.teardown()

    @override
    def on_save_checkpoint(self) -> dict:
        state_dict = super().on_save_checkpoint()
        state_dict["_batches_that_stepped"] = self._batches_that_stepped
        return state_dict

    @override
    def on_load_checkpoint(self, state_dict: dict) -> None:
        self._batches_that_stepped = state_dict.get("_batches_that_stepped", 0)

    def _accumulated_batches_reached(self) -> bool:
        """Determine if accumulation will be finished by the end of the current batch."""
        return self.batch_progress.current.ready % self.trainer.accumulate_grad_batches == 0

    def _num_ready_batches_reached(self) -> bool:
        """Checks if we are in the last batch or if there are more batches to follow."""
        epoch_finished_on_ready = self.batch_progress.current.ready == self.trainer.num_training_batches
        return epoch_finished_on_ready or self.batch_progress.is_last_batch

    def _should_accumulate(self) -> bool:
        """Checks if the optimizer step should be performed or gradients should be accumulated for the current step."""
        accumulation_done = self._accumulated_batches_reached()
        # Lightning steps on the final batch
        is_final_batch = self._num_ready_batches_reached()
        # but the strategy might not
        strategy_accumulates_on_final_batch = self.trainer.strategy.handles_gradient_accumulation or not is_final_batch
        return not accumulation_done and strategy_accumulates_on_final_batch

    def update_lr_schedulers(self, interval: str, update_plateau_schedulers: bool) -> None:
        """Updates the lr schedulers based on the given interval."""
        if interval == "step" and self._should_accumulate():
            return
        self._update_learning_rates(interval=interval, update_plateau_schedulers=update_plateau_schedulers)

    def _update_learning_rates(self, interval: str, update_plateau_schedulers: bool) -> None:
        """Update learning rates.

        Args:
            interval: either 'epoch' or 'step'.
            update_plateau_schedulers: control whether ``ReduceLROnPlateau`` or non-plateau schedulers get updated.
                This is used so non-plateau schedulers can be updated before running validation. Checkpoints are
                commonly saved during validation, however, on-plateau schedulers might monitor a validation metric
                so they have to be updated separately.

        """
        trainer = self.trainer

        if not trainer.lr_scheduler_configs or not trainer.lightning_module.automatic_optimization:
            return

        for config in trainer.lr_scheduler_configs:
            if update_plateau_schedulers ^ config.reduce_on_plateau:
                continue

            current_idx = self.batch_idx if interval == "step" else trainer.current_epoch
            current_idx += 1  # account for both batch and epoch starts from 0
            # Take step if call to update_learning_rates matches the interval key and
            # the current step modulo the schedulers frequency is zero
            if config.interval == interval and current_idx % config.frequency == 0:
                monitor_val = None
                if config.reduce_on_plateau:
                    monitor_key = config.monitor
                    assert monitor_key is not None
                    monitor_val = self._get_monitor_value(monitor_key)
                    if monitor_val is None:
                        if config.strict:
                            avail_metrics = list(trainer.callback_metrics)
                            raise MisconfigurationException(
                                f"ReduceLROnPlateau conditioned on metric {monitor_key}"
                                f" which is not available. Available metrics are: {avail_metrics}."
                                " Condition can be set using `monitor` key in lr scheduler dict"
                            )
                        rank_zero_warn(
                            f"ReduceLROnPlateau conditioned on metric {monitor_key}"
                            " which is not available but strict is set to `False`."
                            " Skipping learning rate update.",
                            category=RuntimeWarning,
                        )
                        continue

                if getattr(self.trainer.optimizers[config.opt_idx], "_skip_next_scheduler_step", False):
                    continue

                self.scheduler_progress.increment_ready()

                # update LR
                call._call_lightning_module_hook(
                    trainer,
                    "lr_scheduler_step",
                    config.scheduler,
                    monitor_val,
                )
                self.scheduler_progress.increment_completed()

    def _get_monitor_value(self, key: str) -> Optional[Any]:
        # this is a separate method to aid in testing
        return self.trainer.callback_metrics.get(key)

    def _should_check_val_epoch(self) -> bool:
        return self.trainer.enable_validation and (
            self.trainer.check_val_every_n_epoch is None
            or (self.trainer.current_epoch + 1) % self.trainer.check_val_every_n_epoch == 0
        )

    def _should_check_val_fx(self, data_fetcher: _DataFetcher) -> bool:
        """Decide if we should run validation."""
        if not self._should_check_val_epoch():
            return False

        # val_check_batch is inf for iterable datasets with no length defined
        is_infinite_dataset = self.trainer.val_check_batch == float("inf")
        is_last_batch = self.batch_progress.is_last_batch
        if is_last_batch and (is_infinite_dataset or isinstance(data_fetcher, _DataLoaderIterDataFetcher)):
            return True

        if self.trainer.should_stop and self.trainer.fit_loop._can_stop_early:
            # allow validation if requesting to stop early through `Trainer.should_stop` (e.g. by early stopping)
            # and when the loop allows to stop (min_epochs/steps met)
            return True

        # TODO: let training/eval loop handle logic around limit_*_batches and val_check_batch
        is_val_check_batch = is_last_batch
        if isinstance(self.trainer.limit_train_batches, int) and is_infinite_dataset:
            is_val_check_batch = (self.batch_idx + 1) % self.trainer.limit_train_batches == 0
        elif self.trainer.val_check_batch != float("inf"):
            # if `check_val_every_n_epoch is `None`, run a validation loop every n training batches
            # else condition it based on the batch_idx of the current epoch
            current_iteration = self.total_batch_idx if self.trainer.check_val_every_n_epoch is None else self.batch_idx
            is_val_check_batch = (current_iteration + 1) % self.trainer.val_check_batch == 0

        return is_val_check_batch

    def _save_loggers_on_train_batch_end(self) -> None:
        """Flushes loggers to disk."""
        if self.trainer.should_stop:
            for logger in self.trainer.loggers:
                logger.save()

    def _build_kwargs(self, kwargs: OrderedDict, batch: Any, batch_idx: int) -> OrderedDict:
        """Helper method to build the arguments for the current step.

        Args:
            kwargs: The kwargs passed down to the hooks.
            batch: The current batch to run through the step.
            batch_idx: the index of the current batch.

        Returns:
            The kwargs passed down to the hooks.

        """
        kwargs["batch"] = batch
        training_step_fx = getattr(self.trainer.lightning_module, "training_step")
        # the `batch_idx` is optional, but its name can be anything
        # as long as there are two arguments after 'self', we assume they are the `batch` and `batch_idx`
        if is_param_in_hook_signature(training_step_fx, "batch_idx", min_args=2):
            kwargs["batch_idx"] = batch_idx
        return kwargs
