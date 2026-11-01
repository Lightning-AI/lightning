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
r"""
BatchSizeFinder
===============

Finds optimal batch size
"""

from typing import Optional

from typing_extensions import override

import lightning.pytorch as pl
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.tuner.batch_size_scaling import _scale_batch_size
from lightning.pytorch.utilities.exceptions import MisconfigurationException, _TunerExitException
from lightning.pytorch.utilities.parsing import lightning_hasattr
from lightning.pytorch.utilities.rank_zero import rank_zero_warn


class BatchSizeFinder(Callback):
    """Finds the largest batch size supported by a given model before encountering an out of memory (OOM) error.

    All you need to do is add it as a callback inside Trainer and call ``trainer.{fit,validate,test,predict}``.
    Internally, it calls the respective step function ``steps_per_trial`` times for each batch size until one
    of the batch sizes generates an OOM error.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    Args:
        mode: search strategy to update the batch size:

            - ``'power'``: Keep multiplying the batch size by 2, until we get an OOM error.
            - ``'binsearch'``: Initially keep multiplying by 2 and after encountering an OOM error
              do a binary search between the last successful batch size and the batch size that failed.

        steps_per_trial: number of steps to run with a given batch size.
            Ideally 1 should be enough to test if an OOM error occurs,
            however in practice a few are needed.

        init_val: initial batch size to start the search with.

        max_trials: max number of increases in batch size done before
            algorithm is terminated

        batch_arg_name: name of the attribute that stores the batch size.
            It is expected that the user has provided a model or datamodule that has a hyperparameter
            with that name. We will look for this attribute name in the following places

            - ``model``
            - ``model.hparams``
            - ``trainer.datamodule`` (the datamodule passed to the tune method)

        margin: Margin to reduce the found batch size by to provide a safety buffer. Only applied when using
            'binsearch' mode. Should be a float between 0 and 1. Defaults to 0.05 (5% reduction).
        max_val: Maximum batch size limit, defaults to 8192.
            Helps prevent testing unrealistically large or inefficient batch sizes (e.g., 2**25)
            when running on CPU or when automatic OOM detection is not available.
        weights_only: Whether to restrict loading the temporary checkpoint to tensors and primitive types.
            Set to ``False`` for checkpoints containing trusted custom objects. Defaults to ``None``, which
            uses the default behavior of :func:`torch.load`.

    Example::

        # 1. Customize the BatchSizeFinder callback to run at different epochs. This feature is
        # useful while fine-tuning models since you can't always use the same batch size after
        # unfreezing the backbone.
        from lightning.pytorch.callbacks import BatchSizeFinder


        class FineTuneBatchSizeFinder(BatchSizeFinder):
            def __init__(self, milestones, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.milestones = milestones

            def on_fit_start(self, *args, **kwargs):
                return

            def on_train_epoch_start(self, trainer, pl_module):
                if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
                    self.scale_batch_size(trainer, pl_module)


        trainer = Trainer(callbacks=[FineTuneBatchSizeFinder(milestones=(5, 10))])
        trainer.fit(...)

    Example::

        # 2. Run batch size finder for validate/test/predict.
        from lightning.pytorch.callbacks import BatchSizeFinder


        class EvalBatchSizeFinder(BatchSizeFinder):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def on_fit_start(self, *args, **kwargs):
                return

            def on_test_start(self, trainer, pl_module):
                self.scale_batch_size(trainer, pl_module)


        trainer = Trainer(callbacks=[EvalBatchSizeFinder()])
        trainer.test(...)

    """

    SUPPORTED_MODES = ("power", "binsearch")

    def __init__(
        self,
        mode: str = "power",
        steps_per_trial: int = 3,
        init_val: int = 2,
        max_trials: int = 25,
        batch_arg_name: str = "batch_size",
        margin: float = 0.05,
        max_val: int = 8192,
        weights_only: Optional[bool] = None,
    ) -> None:
        mode = mode.lower()
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"`mode` should be either of {self.SUPPORTED_MODES}")

        assert 0.0 <= margin < 1.0, f"`margin` should be between 0 and 1. Found {margin=}"

        self.optimal_batch_size: Optional[int] = init_val
        self._mode = mode
        self._steps_per_trial = steps_per_trial
        self._init_val = init_val
        self._max_trials = max_trials
        self._batch_arg_name = batch_arg_name
        self._margin = margin
        self._max_val = max_val
        self._weights_only = weights_only
        self._early_exit = False

    @override
    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        if trainer._accelerator_connector.is_distributed:
            raise MisconfigurationException("The Batch size finder is not supported with distributed strategies.")
        # TODO: check if this can be enabled (#4040)
        if not trainer.fit_loop._data_source.is_module():
            raise MisconfigurationException(
                "The Batch size finder cannot be used with dataloaders passed directly to `.fit()`. Please disable"
                " the feature or incorporate the dataloader into your LightningModule or LightningDataModule."
            )

        # TODO: Add support for multiple eval dataloader
        if stage != "fit":
            loop = trainer._active_loop
            assert loop is not None
            loop.setup_data()
            combined_loader = loop._combined_loader
            assert combined_loader is not None
            if len(combined_loader.flattened) > 1:
                stage = trainer.state.stage
                assert stage is not None
                raise MisconfigurationException(
                    f"The Batch size finder cannot be used with multiple {stage.dataloader_prefix} dataloaders."
                )

        if not lightning_hasattr(pl_module, self._batch_arg_name):
            raise MisconfigurationException(
                f"Field {self._batch_arg_name} not found in `model`, `datamodule`, nor their `hparams` attributes."
            )

        if (
            hasattr(pl_module, self._batch_arg_name)
            and hasattr(pl_module, "hparams")
            and self._batch_arg_name in pl_module.hparams
        ):
            rank_zero_warn(
                f"Field `model.{self._batch_arg_name}` and `model.hparams.{self._batch_arg_name}` are mutually"
                f" exclusive! `model.{self._batch_arg_name}` will be used as the initial batch size for scaling."
                " If this is not the intended behavior, please remove either one."
            )

    def scale_batch_size(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        new_size = _scale_batch_size(
            trainer,
            self._mode,
            self._steps_per_trial,
            self._init_val,
            self._max_trials,
            self._batch_arg_name,
            self._margin,
            self._max_val,
            weights_only=self._weights_only,
        )

        self.optimal_batch_size = new_size
        if self._early_exit:
            raise _TunerExitException()

    @override
    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.scale_batch_size(trainer, pl_module)

    @override
    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.sanity_checking or trainer.state.fn != "validate":
            return

        self.scale_batch_size(trainer, pl_module)

    @override
    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.scale_batch_size(trainer, pl_module)

    @override
    def on_predict_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.scale_batch_size(trainer, pl_module)
