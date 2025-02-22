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
Weight Averaging Callback
^^^^^^^^^^^^^^^^^^^^^^^^^
"""

import itertools
from copy import deepcopy
from typing import Any, Optional, Union

import torch
from torch.optim.swa_utils import AveragedModel
from typing_extensions import override

import lightning.pytorch as pl
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_warn
from lightning.pytorch.utilities.types import STEP_OUTPUT


class WeightAveraging(Callback):
    r"""A callback that updates an averaged model for Stochastic Weight Averaging (SWA) or Exponential Moving Average
    (EMA) after each training step.

    Arguments given to the constructor will be passed to the :class:`AveragedModel` constructor. There are a couple of
    differences to the default values, however. By default, the average model is stored on the CPU. If ``device`` is set
    to ``None``, the device will be inferred from the original model. By default, the callback will compute running
    averages for both the parameters and the buffers of the model. Setting ``use_buffers`` to ``False`` will cause only
    the model parameters to be averaged, leaving updating the batch normalization statistics to the user (using
    ``torch.optim.swa_utils.update_bn()``).

    You can provide a custom averaging function with the ``avg_fn`` or ``multi_avg_fn`` parameter. See the
    :class:`AveragedModel` class for details. If no averaging function is provided, the default is to compute the
    equally-weighted average of the weights (SWA).

    You can customize when the average model is updated by overriding the ``should_update()`` method. The callback calls
    it with either ``step_idx`` or ``epoch_idx`` and the method returns a boolean indicating whether to update after the
    given step or epoch. The default is to update after every step.

    During validation and after the training finishes, the current model parameters will be replaced with the averaged
    values.

    Example::

        from lightning.pytorch.callbacks import WeightAveraging
        from torch.optim.swa_utils import get_ema_avg_fn

        class EMAWeightAveraging(WeightAveraging):
            def __init__(self):
                super().__init__(avg_fn=get_ema_avg_fn())

            def should_update(self, step_idx=None, epoch_idx=None):
                # Start after 100 steps.
                return (step_idx is not None) and (step_idx >= 100)

        trainer = Trainer(callbacks=EMAWeightAveraging(), max_epochs=10)
        trainer.fit(model, dataloader)

    Args:
        device: If provided, the :class:`AveragedModel` will be stored on the ``device``. If ``None`` the device will be
            inferred from the original model.
        use_buffers: If ``False``, the buffers of the model will not be averaged.
        kwargs: Additional keyword arguments to be passed to the :class:`AveragedModel` constructor, such as ``avg_fn``
            or ``multi_avg_fn``.

    """

    def __init__(
        self,
        device: Optional[Union[torch.device, str, int]] = "cpu",
        use_buffers: bool = True,
        **kwargs: Any,
    ) -> None:
        # The default value is a string so that jsonargparse knows how to serialize it.
        if isinstance(device, str):
            self._device: Optional[Union[torch.device, int]] = torch.device(device)
        else:
            self._device = device
        self._use_buffers = use_buffers
        self._kwargs = kwargs

        self._average_model: Optional[AveragedModel] = None

        # Number of optimizer steps taken, when the average model was last updated. Initializing this with zero ensures
        # that self.should_update() will be first called after the first optimizer step, which takes place after N
        # batches when using accumulate_grad_batches=N.
        self._latest_update_step = 0
        # The epoch after which the average model was last updated. The first epoch is 0, so initializing this to a
        # negative value means that if self.should_update(epoch_idx=0) returns True, the first update is after the first
        # epoch.
        self._latest_update_epoch = -1

    def should_update(self, step_idx: Optional[int] = None, epoch_idx: Optional[int] = None) -> bool:
        """Called after every optimizer step and after every training epoch to check whether the average model should
        be updated.

        One of the arguments is set to the zero-based index of the last training step or epoch. The default
        implementation returns ``True`` when any ``step_idx`` is provided. The user can customize when the average model
        gets updated by overriding this method.

        Args:
            step_idx: Index of the last optimizer step, or ``None`` when called at the epoch end.
            epoch_idx: Index of the last epoch, or ``None`` when called after an optimizer step.

        Returns:
            ``True`` if the average model should be updated and ``False`` if not.

        """
        return step_idx is not None

    @override
    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        """Called when fit, validate, test, predict, or tune begins.

        Creates an :class:`AveragedModel` when fit begins.

        Args:
            trainer: The current :class:`~lightning.pytorch.trainer.trainer.Trainer` instance.
            pl_module: The current :class:`~lightning.pytorch.core.LightningModule` instance.
            stage: The :class:`~lightning.pytorch.trainer.trainer.Trainer` state.

        """
        if stage == "fit":
            device = self._device or pl_module.device
            self._average_model = AveragedModel(
                model=pl_module, device=device, use_buffers=self._use_buffers, **self._kwargs
            )

    @override
    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        """Called when a training batch ends.

        Updates the :class:`AveragedModel` parameters, if requested by ``self.should_update()``.

        Args:
            trainer: The current :class:`~lightning.pytorch.trainer.trainer.Trainer` instance.
            pl_module: The current :class:`~lightning.pytorch.core.LightningModule` instance.
            outputs: Outputs from the training batch.
            batch: The training batch.
            batch_idx: Index of the training batch.

        """
        # trainer.global_step is the number of optimizer steps taken so far, i.e. 1 after the first optimizer step. To
        # make step_idx consistent with epoch_idx, we'll pass a zero-based index.
        step_idx = trainer.global_step - 1
        if (trainer.global_step > self._latest_update_step) and self.should_update(step_idx=step_idx):
            assert self._average_model is not None
            self._average_model.update_parameters(pl_module)
            self._latest_update_step = trainer.global_step

    @override
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when a training epoch ends.

        Updates the :class:`AveragedModel` parameters, if requested by ``self.should_update()``.

        Args:
            trainer: The current :class:`~lightning.pytorch.trainer.trainer.Trainer` instance.
            pl_module: The current :class:`~lightning.pytorch.core.LightningModule` instance.

        """
        if (trainer.current_epoch > self._latest_update_epoch) and self.should_update(epoch_idx=trainer.current_epoch):
            assert self._average_model is not None
            self._average_model.update_parameters(pl_module)
            self._latest_update_epoch = trainer.current_epoch

    @override
    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when training ends.

        Transfers parameters from the :class:`AveragedModel` to the current model.

        Args:
            trainer: The current :class:`~lightning.pytorch.trainer.trainer.Trainer` instance.
            pl_module: The current :class:`~lightning.pytorch.core.LightningModule` instance.

        """
        assert self._average_model is not None
        rank_zero_info("Loading the average model parameters to the final model.")
        self._copy_average_to_current(pl_module)

    @override
    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when a validation epoch begins.

        Transfers parameter values from the :class:`AveragedModel` to the current model.

        Args:
            trainer: The current :class:`~lightning.pytorch.trainer.trainer.Trainer` instance.
            pl_module: The current :class:`~lightning.pytorch.core.LightningModule` instance.

        """
        if self._average_model is not None:
            rank_zero_info("Loading the average model parameters for validation.")
            self._swap_models(pl_module)

    @override
    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when a validation epoch ends.

        Recovers the current model parameters from the :class:`AveragedModel`.

        Args:
            trainer: The current :class:`~lightning.pytorch.trainer.trainer.Trainer` instance.
            pl_module: The current :class:`~lightning.pytorch.core.LightningModule` instance.

        """
        if self._average_model is not None:
            rank_zero_info("Recovering the current model parameters after validation.")
            self._swap_models(pl_module)

    @override
    def state_dict(self) -> dict[str, Any]:
        """Called when saving a checkpoint.

        Creates a ``state_dict`` of the callback state.

        Returns:
            A dictionary containing the callback state.

        """
        return {"latest_update_step": self._latest_update_step}

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Called when loading a checkpoint.

        Reloads the callback state given a ``state_dict``.

        Args:
            state_dict: A dictionary containing the callback state.

        """
        self._latest_update_step = state_dict["latest_update_step"]

    @override
    def on_save_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: dict[str, Any]
    ) -> None:
        r"""Called when saving a checkpoint.

        Moves the current model state to the key ``current_model_state``, and places the average model state in
        ``state_dict`` instead. Any other state variables of the ``AveragedModel`` will be saved in
        ``averaging_state``.

        Args:
            trainer: The current :class:`~lightning.pytorch.trainer.trainer.Trainer` instance.
            pl_module: The current :class:`~lightning.pytorch.core.LightningModule` instance.
            checkpoint: The checkpoint dictionary that will be saved.

        """
        if self._average_model is None:
            rank_zero_info(
                "You're using the WeightAveraging callback, but saving a checkpoint outside the 'fit' stage. The state "
                "of the WeightAveraging callback won't be saved in the checkpoint. If training has finished, the "
                "average model parameters will be saved to the state_dict in the checkpoint."
            )
        else:
            rank_zero_info("The average model parameters will be saved to the state_dict in the checkpoint.")
            average_model_state = self._average_model.state_dict()
            checkpoint["current_model_state"] = checkpoint["state_dict"]
            checkpoint["state_dict"] = {
                name[7:]: value for name, value in average_model_state.items() if name.startswith("module.")
            }
            checkpoint["averaging_state"] = {
                name: value for name, value in average_model_state.items() if not name.startswith("module.")
            }

    @override
    def on_load_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: dict[str, Any]
    ) -> None:
        r"""Called when loading a model checkpoint.

        Loads the current model and the :class:`AveragedModel` parameters from the checkpoint.

        Args:
            trainer: The current :class:`~lightning.pytorch.trainer.trainer.Trainer` instance.
            pl_module: The current :class:`~lightning.pytorch.core.LightningModule` instance.
            checkpoint: The full checkpoint dictionary that got loaded by the Trainer.

        """
        if self._average_model is None:
            rank_zero_warn(
                "You're using the WeightAveraging callback, but loading a checkpoint outside the 'fit' stage. The "
                "WeightAveraging state cannot be restored. If you're using the checkpoint for prediction or testing, "
                "you can ignore this warning. To disable the warning, remove the WeightAveraging callback."
            )
        elif ("current_model_state" in checkpoint) and ("averaging_state" in checkpoint):
            rank_zero_info("Found current_model_state in the checkpoint. This will be used to initialize the model.")
            average_model_state = {"module." + name: value for name, value in checkpoint["state_dict"].items()}
            average_model_state |= checkpoint["averaging_state"]
            self._average_model.load_state_dict(average_model_state)
            checkpoint["state_dict"] = checkpoint["current_model_state"]
        else:
            rank_zero_warn(
                "The checkpoint was not created with WeightAveraging. Both the current and the average model will be "
                "initialized with state_dict."
            )
            self._average_model.module.load_state_dict(deepcopy(checkpoint["state_dict"]), strict=False)

    def _swap_models(self, pl_module: "pl.LightningModule") -> None:
        """Swaps the parameter values of the current model and the :class:`AveragedModel`.

        Args:
            pl_module: The current :class:`~lightning.pytorch.core.LightningModule` instance.

        """
        assert self._average_model is not None
        average_params = itertools.chain(self._average_model.module.parameters(), self._average_model.module.buffers())
        current_params = itertools.chain(pl_module.parameters(), pl_module.buffers())
        for average_param, current_param in zip(average_params, current_params):
            tmp = average_param.data.clone()
            average_param.data.copy_(current_param.data)
            current_param.data.copy_(tmp)

    def _copy_average_to_current(self, pl_module: "pl.LightningModule") -> None:
        """Copies the parameter values from the :class:`AveragedModel` to the current model.

        Args:
            pl_module: The current :class:`~lightning.pytorch.core.LightningModule` instance.

        """
        assert self._average_model is not None
        average_params = itertools.chain(self._average_model.module.parameters(), self._average_model.module.buffers())
        current_params = itertools.chain(pl_module.parameters(), pl_module.buffers())
        for average_param, current_param in zip(average_params, current_params):
            current_param.data.copy_(average_param.data)
