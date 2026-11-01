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
Gradient Accumulator
====================

Change gradient accumulation factor according to scheduling.
Trainer also calls ``optimizer.step()`` for the last indivisible step number.

"""

from typing import Any

from typing_extensions import override

import lightning.pytorch as pl
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning.pytorch.utilities.rank_zero import rank_zero_warn


class GradientAccumulationScheduler(Callback):
    r"""Change gradient accumulation factor according to scheduling.

    Args:
        scheduling: scheduling in format {epoch: accumulation_factor}

    Note:
        The argument scheduling is a dictionary. Each key represent an epoch and
        its associated accumulation factor value.
        Warning: Epoch are zero-indexed c.f it means if you want to change
        the accumulation factor after 4 epochs, set ``Trainer(accumulate_grad_batches={4: factor})``
        or ``GradientAccumulationScheduler(scheduling={4: factor})``.
        For more info check the example below.

    Raises:
        TypeError:
            If ``scheduling`` is an empty ``dict``,
            or not all keys and values of ``scheduling`` are integers.
        IndexError:
            If ``minimal_epoch`` is less than 0.

    Example::

        >>> from lightning.pytorch import Trainer
        >>> from lightning.pytorch.callbacks import GradientAccumulationScheduler

        # from epoch 5, it starts accumulating every 2 batches. Here we have 4 instead of 5
        # because epoch (key) should be zero-indexed.
        >>> accumulator = GradientAccumulationScheduler(scheduling={4: 2})
        >>> trainer = Trainer(callbacks=[accumulator])

    """

    def __init__(self, scheduling: dict[int, int]):
        super().__init__()

        if not scheduling:  # empty dict error
            raise TypeError("Empty dict cannot be interpreted correct")

        if any(not isinstance(key, int) or key < 0 for key in scheduling):
            raise MisconfigurationException(
                f"Epoch should be an int greater than or equal to 0. Got {list(scheduling.keys())}."
            )

        if any(not isinstance(value, int) or value < 1 for value in scheduling.values()):
            raise MisconfigurationException(
                f"Accumulation factor should be an int greater than 0. Got {list(scheduling.values())}."
            )

        minimal_epoch = min(scheduling.keys())
        if minimal_epoch < 0:
            raise IndexError(f"Epochs indexing from 1, epoch {minimal_epoch} cannot be interpreted correct")
        if minimal_epoch != 0:  # if user didn't define first epoch accumulation factor
            scheduling.update({0: 1})

        self.scheduling = scheduling
        self.epochs = sorted(scheduling.keys())

    def going_to_accumulate_grad_batches(self) -> bool:
        return any(v > 1 for v in self.scheduling.values())

    def get_accumulate_grad_batches(self, epoch: int) -> int:
        accumulate_grad_batches = 1
        for iter_epoch in reversed(self.epochs):
            if epoch >= iter_epoch:
                accumulate_grad_batches = self.scheduling[iter_epoch]
                break
        return accumulate_grad_batches

    @override
    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Performns a configuration validation before training starts and raises errors for incompatible settings."""

        if not pl_module.automatic_optimization:
            raise RuntimeError(
                """Automatic gradient accumulation and the `GradientAccumulationScheduler` is not supported for
                manual optimization. Please remove the callback or switch to automatic optimization."""
            )

        overridden_optimizer_step = is_overridden("optimizer_step", pl_module)
        overridden_optimizer_zero_grad = is_overridden("optimizer_zero_grad", pl_module)
        going_to_accumulate_grad_batches = self.going_to_accumulate_grad_batches()
        has_overridden_optimization_functions = overridden_optimizer_step or overridden_optimizer_zero_grad
        if has_overridden_optimization_functions and going_to_accumulate_grad_batches:
            rank_zero_warn(
                "When using `Trainer(accumulate_grad_batches != 1)` and overriding"
                " `LightningModule.optimizer_{step,zero_grad}`, the hooks will not be called on every batch"
                " (rather, they are called on every optimization step)."
            )

        # local import to avoid circular import
        from lightning.pytorch.strategies import DeepSpeedStrategy

        if isinstance(trainer.strategy, DeepSpeedStrategy):
            raise RuntimeError(
                f"The `{type(trainer.strategy).__name__}` does not support `accumulate_grad_batches` changing"
                " between epochs."
            )
        previous_assignment = getattr(trainer, "_gradient_accumulation_scheduler_assignment", None)
        if trainer.accumulate_grad_batches != 1 and previous_assignment != (self, trainer.accumulate_grad_batches):
            raise ValueError(
                "You have set `accumulate_grad_batches` and are using the `GradientAccumulationScheduler`"
                " callback. Either remove `accumulate_grad_batches` from the Trainer or remove the callback."
            )

    @override
    def on_train_epoch_start(self, trainer: "pl.Trainer", *_: Any) -> None:
        trainer.accumulate_grad_batches = self.get_accumulate_grad_batches(trainer.current_epoch)

        # Keep provenance on the Trainer so reusing or serializing the callback does not retain a Trainer.
        trainer._gradient_accumulation_scheduler_assignment = (self, trainer.accumulate_grad_batches)
