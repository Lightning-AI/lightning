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
from typing import TYPE_CHECKING, Literal, Optional, Union

import lightning.pytorch as pl
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

if TYPE_CHECKING:
    from lightning.pytorch.tuner.lr_finder import _LRFinder


class Tuner:
    """Tuner class to tune your model."""

    def __init__(self, trainer: "pl.Trainer") -> None:
        self._trainer = trainer

    def scale_batch_size(
        self,
        model: "pl.LightningModule",
        train_dataloaders: Optional[Union[TRAIN_DATALOADERS, "pl.LightningDataModule"]] = None,
        val_dataloaders: Optional[EVAL_DATALOADERS] = None,
        dataloaders: Optional[EVAL_DATALOADERS] = None,
        datamodule: Optional["pl.LightningDataModule"] = None,
        method: Literal["fit", "validate", "test", "predict"] = "fit",
        mode: str = "power",
        steps_per_trial: int = 3,
        init_val: int = 2,
        max_trials: int = 25,
        batch_arg_name: str = "batch_size",
        margin: float = 0.05,
        max_val: int = 8192,
    ) -> Optional[int]:
        """Iteratively try to find the largest batch size for a given model that does not give an out of memory (OOM)
        error.

        Args:
            model: Model to tune.
            train_dataloaders: A collection of :class:`torch.utils.data.DataLoader` or a
                :class:`~lightning.pytorch.core.datamodule.LightningDataModule` specifying training samples.
                In the case of multiple dataloaders, please see this :ref:`section <multiple-dataloaders>`.
            val_dataloaders: A :class:`torch.utils.data.DataLoader` or a sequence of them specifying validation samples.
            dataloaders: A :class:`torch.utils.data.DataLoader` or a sequence of them specifying val/test/predict
                samples used for running tuner on validation/testing/prediction.
            datamodule: An instance of :class:`~lightning.pytorch.core.datamodule.LightningDataModule`.
            method: Method to run tuner on. It can be any of ``("fit", "validate", "test", "predict")``.
            mode: Search strategy to update the batch size:

                - ``'power'``: Keep multiplying the batch size by 2, until we get an OOM error.
                - ``'binsearch'``: Initially keep multiplying by 2 and after encountering an OOM error
                    do a binary search between the last successful batch size and the batch size that failed.

            steps_per_trial: number of steps to run with a given batch size.
                Ideally 1 should be enough to test if an OOM error occurs,
                however in practise a few are needed
            init_val: initial batch size to start the search with
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

        """
        _check_tuner_configuration(train_dataloaders, val_dataloaders, dataloaders, method)
        _check_scale_batch_size_configuration(self._trainer)
        assert 0.0 <= margin < 1.0, f"`margin` should be between 0 and 1. Found {margin=}"

        # local import to avoid circular import
        from lightning.pytorch.callbacks.batch_size_finder import BatchSizeFinder

        batch_size_finder: Callback = BatchSizeFinder(
            mode=mode,
            steps_per_trial=steps_per_trial,
            init_val=init_val,
            max_trials=max_trials,
            batch_arg_name=batch_arg_name,
            margin=margin,
            max_val=max_val,
        )
        # do not continue with the loop in case Tuner is used
        batch_size_finder._early_exit = True
        self._trainer.callbacks = [batch_size_finder] + self._trainer.callbacks

        if method == "fit":
            self._trainer.fit(model, train_dataloaders, val_dataloaders, datamodule)
        elif method == "validate":
            self._trainer.validate(model, dataloaders, datamodule=datamodule)
        elif method == "test":
            self._trainer.test(model, dataloaders, datamodule=datamodule)
        elif method == "predict":
            self._trainer.predict(model, dataloaders, datamodule=datamodule)

        self._trainer.callbacks = [cb for cb in self._trainer.callbacks if cb is not batch_size_finder]
        return batch_size_finder.optimal_batch_size

    def lr_find(
        self,
        model: "pl.LightningModule",
        train_dataloaders: Optional[Union[TRAIN_DATALOADERS, "pl.LightningDataModule"]] = None,
        val_dataloaders: Optional[EVAL_DATALOADERS] = None,
        dataloaders: Optional[EVAL_DATALOADERS] = None,
        datamodule: Optional["pl.LightningDataModule"] = None,
        method: Literal["fit", "validate", "test", "predict"] = "fit",
        min_lr: float = 1e-8,
        max_lr: float = 1,
        num_training: int = 100,
        mode: str = "exponential",
        early_stop_threshold: Optional[float] = 4.0,
        update_attr: bool = True,
        attr_name: str = "",
        weights_only: Optional[bool] = None,
    ) -> Optional["_LRFinder"]:
        """Enables the user to do a range test of good initial learning rates, to reduce the amount of guesswork in
        picking a good starting learning rate.

        Args:
            model: Model to tune.
            train_dataloaders: A collection of :class:`torch.utils.data.DataLoader` or a
                :class:`~lightning.pytorch.core.datamodule.LightningDataModule` specifying training samples.
                In the case of multiple dataloaders, please see this :ref:`section <multiple-dataloaders>`.
            val_dataloaders: A :class:`torch.utils.data.DataLoader` or a sequence of them specifying validation samples.
            dataloaders: A :class:`torch.utils.data.DataLoader` or a sequence of them specifying val/test/predict
                samples used for running tuner on validation/testing/prediction.
            datamodule: An instance of :class:`~lightning.pytorch.core.datamodule.LightningDataModule`.
            method: Method to run tuner on. It can be any of ``("fit", "validate", "test", "predict")``.
            min_lr: minimum learning rate to investigate
            max_lr: maximum learning rate to investigate
            num_training: number of learning rates to test
            mode: Search strategy to update learning rate after each batch:

                - ``'exponential'``: Increases the learning rate exponentially.
                - ``'linear'``: Increases the learning rate linearly.

            early_stop_threshold: Threshold for stopping the search. If the
                loss at any point is larger than early_stop_threshold*best_loss
                then the search is stopped. To disable, set to None.
            update_attr: Whether to update the learning rate attribute or not.
            attr_name: Name of the attribute which stores the learning rate. The names 'learning_rate' or 'lr' get
                automatically detected. Otherwise, set the name here.
            weights_only: Defaults to ``None``. If ``True``, restricts loading to ``state_dicts`` of plain
                ``torch.Tensor`` and other primitive types. If loading a checkpoint from a trusted source that contains
                an ``nn.Module``, use ``weights_only=False``. If loading checkpoint from an untrusted source, we
                recommend using ``weights_only=True``. For more information, please refer to the
                `PyTorch Developer Notes on Serialization Semantics <https://docs.pytorch.org/docs/main/notes/serialization.html#id3>`_.

        Raises:
            MisconfigurationException:
                If learning rate/lr in ``model`` or ``model.hparams`` isn't overridden,
                or if you are using more than one optimizer.

        """
        if method != "fit":
            raise MisconfigurationException("method='fit' is the only valid configuration to run lr finder.")

        _check_tuner_configuration(train_dataloaders, val_dataloaders, dataloaders, method)
        _check_lr_find_configuration(self._trainer)

        # local import to avoid circular import
        from lightning.pytorch.callbacks.lr_finder import LearningRateFinder

        lr_finder_callback: Callback = LearningRateFinder(
            min_lr=min_lr,
            max_lr=max_lr,
            num_training_steps=num_training,
            mode=mode,
            early_stop_threshold=early_stop_threshold,
            update_attr=update_attr,
            attr_name=attr_name,
            weights_only=weights_only,
        )

        lr_finder_callback._early_exit = True
        self._trainer.callbacks = [lr_finder_callback] + self._trainer.callbacks

        try:
            self._trainer.fit(model, train_dataloaders, val_dataloaders, datamodule)
        finally:
            self._trainer.callbacks = [cb for cb in self._trainer.callbacks if cb is not lr_finder_callback]

        return lr_finder_callback.optimal_lr


def _check_tuner_configuration(
    train_dataloaders: Optional[Union[TRAIN_DATALOADERS, "pl.LightningDataModule"]] = None,
    val_dataloaders: Optional[EVAL_DATALOADERS] = None,
    dataloaders: Optional[EVAL_DATALOADERS] = None,
    method: Literal["fit", "validate", "test", "predict"] = "fit",
) -> None:
    supported_methods = ("fit", "validate", "test", "predict")
    if method not in supported_methods:
        raise ValueError(f"method {method!r} is invalid. Should be one of {supported_methods}.")

    if method == "fit":
        if dataloaders is not None:
            raise MisconfigurationException(
                f"In tuner with method={method!r}, `dataloaders` argument should be None,"
                " please consider setting `train_dataloaders` and `val_dataloaders` instead."
            )
    else:
        if train_dataloaders is not None or val_dataloaders is not None:
            raise MisconfigurationException(
                f"In tuner with `method`={method!r}, `train_dataloaders` and `val_dataloaders`"
                " arguments should be None, please consider setting `dataloaders` instead."
            )


def _check_lr_find_configuration(trainer: "pl.Trainer") -> None:
    # local import to avoid circular import
    from lightning.pytorch.callbacks.lr_finder import LearningRateFinder

    configured_callbacks = [cb for cb in trainer.callbacks if isinstance(cb, LearningRateFinder)]
    if configured_callbacks:
        raise ValueError(
            "Trainer is already configured with a `LearningRateFinder` callback."
            "Please remove it if you want to use the Tuner."
        )


def _check_scale_batch_size_configuration(trainer: "pl.Trainer") -> None:
    if trainer._accelerator_connector.is_distributed:
        raise ValueError("Tuning the batch size is currently not supported with distributed strategies.")

    # local import to avoid circular import
    from lightning.pytorch.callbacks.batch_size_finder import BatchSizeFinder

    configured_callbacks = [cb for cb in trainer.callbacks if isinstance(cb, BatchSizeFinder)]
    if configured_callbacks:
        raise ValueError(
            "Trainer is already configured with a `BatchSizeFinder` callback."
            "Please remove it if you want to use the Tuner."
        )
