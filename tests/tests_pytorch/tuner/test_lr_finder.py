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
import glob
import logging
import math
import os
from copy import deepcopy
from typing import Any
from unittest import mock

import pytest
import torch
from lightning_utilities.test.warning import no_warning_call

from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks.finetuning import BackboneFinetuning
from lightning.pytorch.callbacks.lr_finder import LearningRateFinder
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.tuner.lr_finder import _LRFinder
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.types import STEP_OUTPUT
from tests_pytorch.helpers.datamodules import ClassifDataModule
from tests_pytorch.helpers.runif import RunIf
from tests_pytorch.helpers.simple_models import ClassificationModel
from tests_pytorch.helpers.utils import getattr_recursive


def test_error_with_multiple_optimizers(tmp_path):
    """Check that error is thrown when more than 1 optimizer is passed."""

    class CustomBoringModel(BoringModel):
        def __init__(self, lr):
            super().__init__()
            self.save_hyperparameters()
            self.automatic_optimization = False

        def configure_optimizers(self):
            optimizer1 = torch.optim.SGD(self.parameters(), lr=self.hparams.lr)
            optimizer2 = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
            return [optimizer1, optimizer2]

    model = CustomBoringModel(lr=1e-2)

    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1)
    tuner = Tuner(trainer)

    with pytest.raises(MisconfigurationException, match="only works with single optimizer"):
        tuner.lr_find(model)


def test_model_reset_correctly(tmp_path):
    """Check that model weights are correctly reset after _lr_find()"""
    model = BoringModel()
    model.lr = 0.1

    # logger file to get meta
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1)
    tuner = Tuner(trainer)
    before_state_dict = deepcopy(model.state_dict())

    tuner.lr_find(model, num_training=5)

    after_state_dict = model.state_dict()

    for key in before_state_dict:
        assert torch.all(torch.eq(before_state_dict[key], after_state_dict[key])), (
            "Model was not reset correctly after learning rate finder"
        )

    assert not any(f for f in os.listdir(tmp_path) if f.startswith(".lr_find"))


def test_trainer_reset_correctly(tmp_path):
    """Check that all trainer parameters are reset correctly after lr_find()"""
    model = BoringModel()
    model.lr = 0.1

    # logger file to get meta
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1)
    tuner = Tuner(trainer)

    changed_attributes = [
        "accumulate_grad_batches",
        "callbacks",
        "checkpoint_callback",
        "current_epoch",
        "loggers",
        "global_step",
        "max_steps",
        "fit_loop.max_steps",
        "strategy.setup_optimizers",
        "should_stop",
    ]
    expected = {ca: getattr_recursive(trainer, ca) for ca in changed_attributes}

    with no_warning_call(UserWarning, match="Please add the following callbacks"):
        tuner.lr_find(model, num_training=5)

    actual = {ca: getattr_recursive(trainer, ca) for ca in changed_attributes}
    assert actual == expected
    assert model.trainer == trainer


@pytest.mark.parametrize("use_hparams", [False, True])
def test_tuner_lr_find(tmp_path, use_hparams):
    """Test that lr_find updates the learning rate attribute."""
    seed_everything(1)

    class CustomBoringModel(BoringModel):
        def __init__(self, lr):
            super().__init__()
            self.save_hyperparameters()
            self.lr = lr

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=self.hparams.lr if use_hparams else self.lr)

    before_lr = 1e-2
    model = CustomBoringModel(lr=before_lr)
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=2)
    tuner = Tuner(trainer)
    tuner.lr_find(model, update_attr=True)

    after_lr = model.hparams.lr if use_hparams else model.lr

    assert after_lr is not None
    assert before_lr != after_lr, "Learning rate was not altered after running learning rate finder"


@pytest.mark.parametrize("use_hparams", [False, True])
def test_trainer_arg_str(tmp_path, use_hparams):
    """Test that setting trainer arg to string works."""
    seed_everything(1)

    class CustomBoringModel(BoringModel):
        def __init__(self, my_fancy_lr):
            super().__init__()
            self.save_hyperparameters()
            self.my_fancy_lr = my_fancy_lr

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=self.hparams.my_fancy_lr if use_hparams else self.my_fancy_lr)

    before_lr = 1e-2
    model = CustomBoringModel(my_fancy_lr=before_lr)
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=2)
    tuner = Tuner(trainer)
    tuner.lr_find(model, update_attr=True, attr_name="my_fancy_lr")
    after_lr = model.hparams.my_fancy_lr if use_hparams else model.my_fancy_lr

    assert after_lr is not None
    assert before_lr != after_lr, "Learning rate was not altered after running learning rate finder"


@pytest.mark.parametrize("opt", ["Adam", "Adagrad"])
def test_call_to_trainer_method(tmp_path, opt):
    """Test that directly calling the trainer method works."""
    seed_everything(1)

    class CustomBoringModel(BoringModel):
        def __init__(self, lr):
            super().__init__()
            self.save_hyperparameters()

        def configure_optimizers(self):
            return (
                torch.optim.Adagrad(self.parameters(), lr=self.hparams.lr)
                if opt == "Adagrad"
                else torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
            )

    before_lr = 1e-2
    model = CustomBoringModel(1e-2)
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=2)

    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, mode="linear")
    after_lr = lr_finder.suggestion()
    assert after_lr is not None
    model.hparams.lr = after_lr
    tuner.lr_find(model, update_attr=True)

    assert after_lr is not None
    assert before_lr != after_lr, "Learning rate was not altered after running learning rate finder"


@RunIf(sklearn=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_datamodule_parameter(tmp_path):
    """Test that the datamodule parameter works."""
    seed_everything(1)

    dm = ClassifDataModule()
    model = ClassificationModel(lr=1e-3)

    before_lr = model.lr
    # logger file to get meta
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=2)

    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, datamodule=dm)
    after_lr = lr_finder.suggestion()
    model.lr = after_lr

    assert after_lr is not None
    assert before_lr != after_lr, "Learning rate was not altered after running learning rate finder"


def test_accumulation_and_early_stopping(tmp_path):
    """Test that early stopping of learning rate finder works, and that accumulation also works for this feature."""
    seed_everything(1)

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.lr = 1e-3

    model = TestModel()
    trainer = Trainer(default_root_dir=tmp_path, accumulate_grad_batches=2)
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, early_stop_threshold=None)

    assert lr_finder.suggestion() != 1e-3
    assert len(lr_finder.results["lr"]) == len(lr_finder.results["loss"]) == 100
    assert lr_finder._total_batch_idx == 199


def test_suggestion_parameters_work(tmp_path):
    """Test that default skipping does not alter results in basic case."""
    seed_everything(1)

    class CustomBoringModel(BoringModel):
        def __init__(self, lr):
            super().__init__()
            self.lr = lr

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=self.lr)

    # logger file to get meta
    model = CustomBoringModel(lr=1e-2)
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=3)
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model)
    lr1 = lr_finder.suggestion(skip_begin=10)  # default
    lr2 = lr_finder.suggestion(skip_begin=70)  # way too high, should have an impact

    assert lr1 is not None
    assert lr2 is not None
    assert lr1 != lr2, "Skipping parameter did not influence learning rate"


def test_suggestion_with_non_finite_values(tmp_path):
    """Test that non-finite values does not alter results."""
    seed_everything(1)

    class CustomBoringModel(BoringModel):
        def __init__(self, lr):
            super().__init__()
            self.lr = lr

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=self.lr)

    model = CustomBoringModel(lr=1e-2)
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=3)
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model)

    before_lr = lr_finder.suggestion()
    lr_finder.results["loss"][-1] = float("nan")
    after_lr = lr_finder.suggestion()

    assert before_lr is not None
    assert after_lr is not None
    assert before_lr == after_lr, "Learning rate was altered because of non-finite loss values"


def test_lr_finder_fails_fast_on_bad_config(tmp_path):
    """Test that tune fails if the model does not have a lr BEFORE running lr find."""
    trainer = Trainer(default_root_dir=tmp_path, max_steps=2)
    tuner = Tuner(trainer)
    with pytest.raises(AttributeError, match="should have one of these fields"):
        tuner.lr_find(BoringModel(), update_attr=True)


def test_lr_candidates_between_min_and_max(tmp_path):
    """Test that learning rate candidates are between min_lr and max_lr."""
    seed_everything(1)

    class TestModel(BoringModel):
        def __init__(self, learning_rate=0.1):
            super().__init__()
            self.save_hyperparameters()

    model = TestModel()
    trainer = Trainer(default_root_dir=tmp_path)

    lr_min = 1e-8
    lr_max = 1.0
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, max_lr=lr_min, min_lr=lr_max, num_training=3)
    lr_candidates = lr_finder.results["lr"]
    assert all(lr_min <= lr <= lr_max for lr in lr_candidates)


def test_lr_finder_ends_before_num_training(tmp_path):
    """Tests learning rate finder ends before `num_training` steps."""

    class TestModel(BoringModel):
        def __init__(self, learning_rate=0.1):
            super().__init__()
            self.save_hyperparameters()

        def on_before_optimizer_step(self, optimizer):
            assert self.global_step < num_training

    model = TestModel()
    trainer = Trainer(default_root_dir=tmp_path)
    tuner = Tuner(trainer)
    num_training = 3
    tuner.lr_find(model=model, num_training=num_training)


def test_multiple_lr_find_calls_gives_same_results(tmp_path):
    """Tests that lr_finder gives same results if called multiple times."""
    seed_everything(1)
    model = BoringModel()
    model.lr = 0.1

    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=2,
        limit_train_batches=10,
        limit_val_batches=2,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
    )
    tuner = Tuner(trainer)
    all_res = [tuner.lr_find(model).results for _ in range(3)]

    assert all(
        all_res[0][k] == curr_lr_finder[k] and len(curr_lr_finder[k]) > 10
        for curr_lr_finder in all_res[1:]
        for k in all_res[0]
    )


@pytest.mark.parametrize(
    ("skip_begin", "skip_end", "losses", "expected_error"),
    [
        (0, 0, [], True),
        (10, 1, [], True),
        (0, 2, [0, 1, 2], True),
        (0, 1, [0, 1, 2], False),
        (1, 1, [0, 1, 2], True),
        (1, 1, [0, 1, 2, 3], False),
        (0, 1, [float("nan"), float("nan"), 0, float("inf"), 1, 2, 3, float("inf"), 2, float("nan"), 1], False),
        (4, 1, [float("nan"), float("nan"), 0, float("inf"), 1, 2, 3, float("inf"), 2, float("nan"), 1], False),
        (0, 5, [0, 1, 2, 3], True),  # skip_end past the sample count leaves nothing, it must not wrap around
    ],
)
def test_suggestion_not_enough_finite_points(losses, skip_begin, skip_end, expected_error, caplog):
    """Tests the error handling when not enough finite points are available to make a suggestion."""
    caplog.clear()
    lr_finder = _LRFinder(
        mode="exponential",
        lr_min=1e-8,
        lr_max=1,
        num_training=100,
    )
    lrs = list(torch.arange(len(losses)))
    lr_finder.results = {
        "lr": lrs,
        "loss": losses,
    }
    with caplog.at_level(logging.ERROR, logger="root.tuner.lr_finder"):
        lr = lr_finder.suggestion(skip_begin=skip_begin, skip_end=skip_end)

        if expected_error:
            assert lr is None
            assert "Failed to compute suggestion for learning rate" in caplog.text
        else:
            assert lr is not None


def test_suggestion_with_skip_end_zero():
    """`skip_end=0` asks for nothing to be skipped, so the last sample has to stay in the window."""
    lr_finder = _LRFinder(mode="exponential", lr_min=1e-8, lr_max=1, num_training=100)
    # the steepest drop is between the last two points, so it is only reachable when nothing is skipped
    # (every slope differs, so the argmin never comes down to a float tie)
    lr_finder.results = {"lr": [0.1, 0.2, 0.3, 0.4], "loss": [10.0, 8.0, 7.0, 1.0]}

    assert lr_finder.suggestion(skip_begin=0, skip_end=0) == 0.4
    assert lr_finder._optimal_idx == 3

    # the default still drops the last sample
    assert lr_finder.suggestion(skip_begin=0, skip_end=1) == 0.1
    assert lr_finder._optimal_idx == 0


def test_lr_attribute_when_suggestion_invalid(tmp_path):
    """Tests learning rate finder ends before `num_training` steps."""

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.learning_rate = 0.123

    model = TestModel()
    trainer = Trainer(default_root_dir=tmp_path)
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model=model, update_attr=True, num_training=1)  # force insufficient data points
    assert lr_finder.suggestion() is None
    assert model.learning_rate == 0.123  # must remain unchanged because suggestion is not possible


def test_lr_finder_callback_restarting(tmp_path):
    """Test that `LearningRateFinder` does not set restarting=True when loading checkpoint."""
    num_lr_steps = 100

    class MyBoringModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.learning_rate = 0.123

        def on_train_batch_start(self, batch, batch_idx):
            if getattr(self, "_expected_max_steps", None) is not None:
                assert self.trainer.fit_loop.max_steps == self._expected_max_steps

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate)

    class CustomLearningRateFinder(LearningRateFinder):
        milestones = (1,)

        def lr_find(self, trainer, pl_module) -> None:
            pl_module._expected_max_steps = trainer.global_step + self._num_training_steps
            super().lr_find(trainer, pl_module)
            pl_module._expected_max_steps = None
            assert not trainer.fit_loop.restarting
            assert not trainer.fit_loop.epoch_loop.restarting

        def on_train_epoch_start(self, trainer, pl_module):
            if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
                self.lr_find(trainer, pl_module)

    model = MyBoringModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=3,
        callbacks=[
            CustomLearningRateFinder(early_stop_threshold=None, update_attr=True, num_training_steps=num_lr_steps)
        ],
        limit_train_batches=10,
        limit_val_batches=0,
        limit_test_batches=0,
        num_sanity_val_steps=0,
        enable_model_summary=False,
    )

    trainer.fit(model)


@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
@RunIf(standalone=True)
def test_lr_finder_with_ddp(tmp_path):
    seed_everything(7)

    init_lr = 1e-4
    dm = ClassifDataModule()
    model = ClassificationModel(lr=init_lr)

    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        strategy="ddp",
        devices=2,
        accelerator="cpu",
    )

    tuner = Tuner(trainer)
    tuner.lr_find(model, datamodule=dm, update_attr=True, num_training=20)
    lr = trainer.lightning_module.lr
    lr = trainer.strategy.broadcast(lr)
    assert trainer.lightning_module.lr == lr
    assert lr != init_lr


def test_lr_finder_callback_val_batches(tmp_path):
    """Test that `LearningRateFinder` does not limit the number of val batches during training."""

    class CustomBoringModel(BoringModel):
        def __init__(self, lr):
            super().__init__()
            self.lr = lr

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=self.lr)

    num_lr_tuner_training_steps = 5
    model = CustomBoringModel(0.1)
    trainer = Trainer(
        default_root_dir=tmp_path,
        num_sanity_val_steps=0,
        max_epochs=1,
        enable_model_summary=False,
        callbacks=[LearningRateFinder(num_training_steps=num_lr_tuner_training_steps)],
    )
    trainer.fit(model)

    assert trainer.num_val_batches[0] == len(trainer.val_dataloaders)
    assert trainer.num_val_batches[0] != num_lr_tuner_training_steps


def test_lr_finder_training_step_none_output(tmp_path):
    # add some nans into the skipped steps (first 10) but also into the steps used to compute the lr
    none_steps = [5, 12, 17]

    class CustomBoringModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.lr = 0.123

        def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
            if self.trainer.global_step in none_steps:
                return None

            return super().training_step(batch, batch_idx)

    seed_everything(1)
    model = CustomBoringModel()

    trainer = Trainer(default_root_dir=tmp_path)

    tuner = Tuner(trainer)
    # restrict number of steps for faster test execution
    # and disable early stopping to easily check expected number of lrs and losses
    lr_finder = tuner.lr_find(model=model, update_attr=True, num_training=20, early_stop_threshold=None)
    assert len(lr_finder.results["lr"]) == len(lr_finder.results["loss"]) == 20
    assert torch.isnan(torch.tensor(lr_finder.results["loss"])[none_steps]).all()

    suggested_lr = lr_finder.suggestion()
    assert math.isfinite(suggested_lr)
    assert math.isclose(model.lr, suggested_lr)


def test_lr_finder_with_early_stopping(tmp_path):
    class ModelWithValidation(BoringModel):
        def __init__(self):
            super().__init__()
            self.learning_rate = 0.1

        def validation_step(self, batch, batch_idx):
            output = self.step(batch)
            # Log validation loss that EarlyStopping will monitor
            self.log("val_loss", output, on_epoch=True)
            return output

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

            # Add ReduceLROnPlateau scheduler that monitors val_loss (issue #20355)
            plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=2
            )
            scheduler_config = {"scheduler": plateau_scheduler, "interval": "epoch", "monitor": "val_loss"}

            return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

    model = ModelWithValidation()

    # Both callbacks that previously caused issues
    callbacks = [
        LearningRateFinder(num_training_steps=100, update_attr=False),
        EarlyStopping(monitor="val_loss", patience=3),
    ]

    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=10,
        callbacks=callbacks,
        limit_train_batches=5,
        limit_val_batches=3,
        enable_model_summary=False,
        enable_progress_bar=False,
    )

    trainer.fit(model)
    assert trainer.state.finished

    # Verify that both callbacks were active
    lr_finder_callback = None
    early_stopping_callback = None
    for callback in trainer.callbacks:
        if isinstance(callback, LearningRateFinder):
            lr_finder_callback = callback
        elif isinstance(callback, EarlyStopping):
            early_stopping_callback = callback

    assert lr_finder_callback is not None, "LearningRateFinder callback should be present"
    assert early_stopping_callback is not None, "EarlyStopping callback should be present"

    # Verify learning rate finder ran and has results
    assert lr_finder_callback.optimal_lr is not None, "Learning rate finder should have results"
    suggestion = lr_finder_callback.optimal_lr.suggestion()
    if suggestion is not None:
        assert suggestion > 0, "Learning rate suggestion should be positive"


def test_gradient_correctness():
    """Test that torch.gradient uses correct spacing parameter."""
    lr_finder = _LRFinder(mode="exponential", lr_min=1e-6, lr_max=1e-1, num_training=20)

    # Synthetic example
    lrs = torch.linspace(0, 2 * math.pi, steps=1000)
    losses = torch.sin(lrs)
    lr_finder.results = {"lr": lrs.tolist(), "loss": losses.tolist()}

    # Test the suggestion method
    suggestion = lr_finder.suggestion(skip_begin=2, skip_end=2)
    assert suggestion is not None
    assert abs(suggestion - math.pi) < 1e-2, "Suggestion should be close to pi for this synthetic example"


def test_lr_finder_callback_applies_lr_after_restore(tmp_path):
    """LearningRateFinder used as a callback should apply its suggested LR to the optimizer used after state
    restoration."""

    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    from lightning.pytorch.callbacks import LearningRateMonitor

    class RandomDataset(Dataset):
        def __init__(self, n: int = 256, in_dim: int = 28 * 28):
            self.x = torch.randn(n, in_dim)
            self.y = torch.randn(n, in_dim)

        def __len__(self) -> int:
            return len(self.x)

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]

    class TinyAE(BoringModel):
        def __init__(self, lr: float = 1e-5):
            super().__init__()
            self.save_hyperparameters()
            self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
            self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

        def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
            x, y = batch
            z = self.encoder(x)
            x_hat = self.decoder(z)
            return F.mse_loss(x_hat, y)

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    seed_everything(123)

    ds = RandomDataset(n=512)
    train_loader = DataLoader(ds, batch_size=64, shuffle=False)

    model = TinyAE(lr=1e-5)

    lr_finder_cb = LearningRateFinder()  # default update_attr=True should apply suggestion
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=2,
        callbacks=[lr_finder_cb, lr_monitor],
        enable_model_summary=False,
        enable_progress_bar=False,
        log_every_n_steps=1,
    )

    trainer.fit(model, train_loader)
    assert model.hparams.lr is not None
    # Ensure LR Finder produced a suggestion for this setup; if not, the test can't assert application
    assert lr_finder_cb.optimal_lr is not None, "LR Finder should have computed results"
    suggestion = lr_finder_cb.optimal_lr.suggestion()
    assert suggestion is not None, "LR Finder should produce a suggestion for this setup"

    # Verify that the optimizer used for subsequent training has the suggested LR applied
    assert trainer.optimizers, "Trainer should have an optimizer after fit"
    current_lr = trainer.optimizers[0].param_groups[0]["lr"]
    assert current_lr == pytest.approx(suggestion), (
        f"LR Finder suggestion {suggestion} should be applied to optimizer, but got {current_lr}"
    )


def test_exponential_vs_linear_mode_gradient_difference(tmp_path):
    """Test that exponential and linear modes produce different but valid suggestions.

    This verifies that the spacing fix works for both modes and that they behave differently as expected due to their
    different lr progressions.

    """

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.lr = 1e-3

    seed_everything(42)

    # Test both modes with identical parameters
    model_linear = TestModel()
    model_exp = TestModel()

    trainer_linear = Trainer(default_root_dir=tmp_path, max_epochs=1)
    trainer_exp = Trainer(default_root_dir=tmp_path, max_epochs=1)

    tuner_linear = Tuner(trainer_linear)
    tuner_exp = Tuner(trainer_exp)

    lr_finder_linear = tuner_linear.lr_find(model_linear, min_lr=1e-6, max_lr=1e-1, num_training=50, mode="linear")
    lr_finder_exp = tuner_exp.lr_find(model_exp, min_lr=1e-6, max_lr=1e-1, num_training=50, mode="exponential")

    # Both should produce valid suggestions
    suggestion_linear = lr_finder_linear.suggestion()
    suggestion_exp = lr_finder_exp.suggestion()

    assert suggestion_linear is not None
    assert suggestion_exp is not None
    assert suggestion_linear > 0
    assert suggestion_exp > 0

    # Verify that gradient computation uses correct spacing for both modes
    for lr_finder, mode in [(lr_finder_linear, "linear"), (lr_finder_exp, "exponential")]:
        losses = torch.tensor(lr_finder.results["loss"][10:-10])
        lrs = torch.tensor(lr_finder.results["lr"][10:-10])
        is_finite = torch.isfinite(losses)
        losses_filtered = losses[is_finite]
        lrs_filtered = lrs[is_finite]

        if len(losses_filtered) >= 2:
            # Test that gradient computation works and produces finite results
            gradients = torch.gradient(losses_filtered, spacing=[lrs_filtered])[0]
            assert torch.isfinite(gradients).all(), f"Non-finite gradients in {mode} mode"
            assert len(gradients) == len(losses_filtered)

            # Verify gradients with spacing differ from gradients without spacing
            gradients_no_spacing = torch.gradient(losses_filtered)[0]

            # For exponential mode, these should definitely be different, for linear mode, they might be similar
            if mode == "exponential":
                assert not torch.allclose(gradients, gradients_no_spacing, rtol=0.1), (
                    "Gradients should differ significantly in exponential mode when using proper spacing"
                )


def test_lr_finder_checkpoint_cleanup_on_error(tmp_path):
    """Test that temporary checkpoint files are cleaned up even when an error occurs during lr finding."""

    class FailingModel(BoringModel):
        def __init__(self, fail_on_step=2):
            super().__init__()
            self.fail_on_step = fail_on_step
            self.current_step = 0
            self.learning_rate = 1e-3

        def training_step(self, batch, batch_idx):
            self.current_step += 1
            if self.current_step >= self.fail_on_step:
                raise RuntimeError("Intentional failure for testing cleanup")
            return super().training_step(batch, batch_idx)

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            return [optimizer], [lr_scheduler]

    model = FailingModel()
    lr_finder = LearningRateFinder(num_training_steps=5)

    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        callbacks=[lr_finder],
    )

    # Check no lr_find checkpoint files exist initially
    lr_find_checkpoints = glob.glob(os.path.join(tmp_path, ".lr_find_*.ckpt"))
    assert len(lr_find_checkpoints) == 0, "No lr_find checkpoint files should exist initially"

    # Run lr finder and expect it to fail
    with pytest.raises(RuntimeError, match="Intentional failure for testing cleanup"):
        trainer.fit(model)

    # Check that no lr_find checkpoint files are left behind
    lr_find_checkpoints = glob.glob(os.path.join(tmp_path, ".lr_find_*.ckpt"))
    assert len(lr_find_checkpoints) == 0, (
        f"lr_find checkpoint files should be cleaned up, but found: {lr_find_checkpoints}"
    )


def test_lr_finder_with_backbone_finetuning_callback(tmp_path):
    """Test that lr_find works correctly with BackboneFinetuning callback."""

    class ModelWithBackbone(BoringModel):
        def __init__(self):
            super().__init__()
            # Create a simple backbone-head architecture
            self.backbone = torch.nn.Sequential(torch.nn.Linear(32, 16), torch.nn.ReLU(), torch.nn.Linear(16, 8))
            self.head = torch.nn.Linear(8, 2)
            self.learning_rate = 1e-3

        def forward(self, x):
            backbone_features = self.backbone(x)
            return self.head(backbone_features)

        def configure_optimizers(self):
            return torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)

    model = ModelWithBackbone()
    backbone_finetuning = BackboneFinetuning(unfreeze_backbone_at_epoch=1)

    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=3,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        callbacks=[backbone_finetuning],
    )

    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, num_training=5)

    assert lr_finder is not None
    assert hasattr(lr_finder, "results")
    assert len(lr_finder.results) > 0
    trainer.fit(model)

    # Check that backbone was unfrozen at the correct epoch
    for param in model.backbone.parameters():
        assert param.requires_grad, "Backbone parameters should be unfrozen after epoch 1"


def test_lr_finder_respects_weights_only(tmp_path):
    """Test that lr_find works correctly when saving more than the weights."""

    # Simple torch Module
    class TorchCoder(torch.nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.net = torch.nn.Linear(in_features, out_features)

        def forward(self, x):
            return self.net(x)

    # Simple model
    class SimpleModel(LightningModule):
        def __init__(self, coder, loss, lr=1e-3):
            super().__init__()
            self.save_hyperparameters()
            self.layer = coder
            self.loss = loss
            self.lr = lr

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self.layer(x)
            return self.loss(y_hat, y)

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.lr)

    # Dummy data
    x = torch.randn(16, 4)
    y = torch.randn(16, 2)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=4)

    model = SimpleModel(
        TorchCoder(4, 2),
        loss=torch.nn.MSELoss(),
    )

    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        logger=False,
        enable_checkpointing=True,
    )

    # This should NOT raise an exception after the fix
    lr_finder = Tuner(trainer).lr_find(
        model,
        train_dataloaders=loader,
        weights_only=False,  # <-- the key part
    )

    assert lr_finder is not None
    assert hasattr(lr_finder, "results")
