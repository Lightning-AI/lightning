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
import copy
import logging
import math
import os
import pickle
from typing import Optional
from unittest import mock
from unittest.mock import Mock

import cloudpickle
import pytest
import torch

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStoppingReason
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from tests_pytorch.helpers.datamodules import ClassifDataModule
from tests_pytorch.helpers.runif import RunIf
from tests_pytorch.helpers.simple_models import ClassificationModel

_logger = logging.getLogger(__name__)


def test_early_stopping_state_key():
    early_stopping = EarlyStopping(monitor="val_loss")
    assert early_stopping.state_key == "EarlyStopping{'monitor': 'val_loss', 'mode': 'min'}"


class EarlyStoppingTestRestore(EarlyStopping):
    # this class has to be defined outside the test function, otherwise we get pickle error
    def __init__(self, expected_state, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expected_state = expected_state
        # cache the state for each epoch
        self.saved_states = []

    def on_train_start(self, trainer, pl_module):
        if self.expected_state:
            assert self.state_dict() == self.expected_state

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        self.saved_states.append(self.state_dict().copy())


@RunIf(sklearn=True, skip_windows=True)  # Flaky test on Windows for unknown reasons
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_resume_early_stopping_from_checkpoint(tmp_path):
    """Prevent regressions to bugs:

    https://github.com/Lightning-AI/pytorch-lightning/issues/1464
    https://github.com/Lightning-AI/pytorch-lightning/issues/1463

    """
    seed_everything(42)
    model = ClassificationModel()
    dm = ClassifDataModule()
    checkpoint_callback = ModelCheckpoint(dirpath=tmp_path, monitor="train_loss", save_top_k=1)
    early_stop_callback = EarlyStoppingTestRestore(None, monitor="train_loss")
    trainer = Trainer(
        default_root_dir=tmp_path,
        callbacks=[early_stop_callback, checkpoint_callback],
        num_sanity_val_steps=0,
        max_epochs=4,
    )
    trainer.fit(model, datamodule=dm)

    assert len(early_stop_callback.saved_states) == 4

    checkpoint_filepath = checkpoint_callback.kth_best_model_path
    # ensure state is persisted properly
    checkpoint = torch.load(checkpoint_filepath, weights_only=True)
    # the checkpoint saves "epoch + 1"
    early_stop_callback_state = early_stop_callback.saved_states[checkpoint["epoch"]]
    assert len(early_stop_callback.saved_states) == 4
    es_name = "EarlyStoppingTestRestore{'monitor': 'train_loss', 'mode': 'min'}"
    assert checkpoint["callbacks"][es_name] == early_stop_callback_state

    # ensure state is reloaded properly (assertion in the callback)
    early_stop_callback = EarlyStoppingTestRestore(early_stop_callback_state, monitor="train_loss")
    new_trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        callbacks=[early_stop_callback],
    )

    with pytest.raises(MisconfigurationException, match=r"You restored a checkpoint with current_epoch"):
        new_trainer.fit(model, datamodule=dm, ckpt_path=checkpoint_filepath)


@RunIf(sklearn=True)
def test_early_stopping_no_extraneous_invocations(tmp_path):
    """Test to ensure that callback methods aren't being invoked outside of the callback handler."""
    model = ClassificationModel()
    dm = ClassifDataModule()
    early_stop_callback = EarlyStopping(monitor="train_loss")
    early_stop_callback._run_early_stopping_check = Mock()
    expected_count = 4
    trainer = Trainer(
        default_root_dir=tmp_path,
        callbacks=[early_stop_callback],
        limit_train_batches=4,
        limit_val_batches=4,
        max_epochs=expected_count,
        enable_checkpointing=False,
    )
    trainer.fit(model, datamodule=dm)

    assert trainer.early_stopping_callback == early_stop_callback
    assert trainer.early_stopping_callbacks == [early_stop_callback]
    assert early_stop_callback._run_early_stopping_check.call_count == expected_count


@pytest.mark.parametrize(
    ("loss_values", "patience", "expected_stop_epoch"),
    [([6, 5, 5, 5, 5, 5], 3, 4), ([6, 5, 4, 4, 3, 3], 1, 3), ([6, 5, 6, 5, 5, 5], 3, 4)],
)
def test_early_stopping_patience(tmp_path, loss_values: list, patience: int, expected_stop_epoch: int):
    """Test to ensure that early stopping is not triggered before patience is exhausted."""

    class ModelOverrideValidationReturn(BoringModel):
        validation_return_values = torch.tensor(loss_values)

        def on_validation_epoch_end(self):
            loss = self.validation_return_values[self.current_epoch]
            self.log("test_val_loss", loss)

    model = ModelOverrideValidationReturn()
    early_stop_callback = EarlyStopping(monitor="test_val_loss", patience=patience, verbose=True)
    trainer = Trainer(
        default_root_dir=tmp_path,
        callbacks=[early_stop_callback],
        num_sanity_val_steps=0,
        max_epochs=10,
        enable_progress_bar=False,
    )
    trainer.fit(model)
    assert trainer.current_epoch - 1 == expected_stop_epoch


@pytest.mark.parametrize("validation_step_none", [True, False])
@pytest.mark.parametrize(
    ("loss_values", "patience", "expected_stop_epoch"),
    [([6, 5, 5, 5, 5, 5], 3, 4), ([6, 5, 4, 4, 3, 3], 1, 3), ([6, 5, 6, 5, 5, 5], 3, 4)],
)
def test_early_stopping_patience_train(
    tmp_path, validation_step_none: bool, loss_values: list, patience: int, expected_stop_epoch: int
):
    """Test to ensure that early stopping is not triggered before patience is exhausted."""

    class ModelOverrideTrainReturn(BoringModel):
        train_return_values = torch.tensor(loss_values)

        def on_train_epoch_end(self):
            loss = self.train_return_values[self.current_epoch]
            self.log("train_loss", loss)

    model = ModelOverrideTrainReturn()

    if validation_step_none:
        model.validation_step = None

    early_stop_callback = EarlyStopping(
        monitor="train_loss", patience=patience, verbose=True, check_on_train_epoch_end=True
    )
    trainer = Trainer(
        default_root_dir=tmp_path,
        callbacks=[early_stop_callback],
        num_sanity_val_steps=0,
        max_epochs=10,
        enable_progress_bar=False,
    )
    trainer.fit(model)
    assert trainer.current_epoch - 1 == expected_stop_epoch


def test_pickling():
    early_stopping = EarlyStopping(monitor="foo")

    early_stopping_pickled = pickle.dumps(early_stopping)
    early_stopping_loaded = pickle.loads(early_stopping_pickled)
    assert vars(early_stopping) == vars(early_stopping_loaded)

    early_stopping_pickled = cloudpickle.dumps(early_stopping)
    early_stopping_loaded = cloudpickle.loads(early_stopping_pickled)
    assert vars(early_stopping) == vars(early_stopping_loaded)


@RunIf(sklearn=True)
def test_early_stopping_no_val_step(tmp_path):
    """Test that early stopping callback falls back to training metrics when no validation defined."""
    model = ClassificationModel()
    dm = ClassifDataModule()
    model.validation_step = None
    model.val_dataloader = None

    stopping = EarlyStopping(monitor="train_loss", min_delta=0.1, patience=0, check_on_train_epoch_end=True)
    trainer = Trainer(default_root_dir=tmp_path, callbacks=[stopping], overfit_batches=0.20, max_epochs=10)
    trainer.fit(model, datamodule=dm)

    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.current_epoch < trainer.max_epochs - 1


@pytest.mark.parametrize(
    ("stopping_threshold", "divergence_threshold", "losses", "expected_epoch"),
    [
        (None, None, [8, 4, 2, 3, 4, 5, 8, 10], 5),
        (2.9, None, [9, 8, 7, 6, 5, 6, 4, 3, 2, 1], 8),
        (None, 15.9, [9, 4, 2, 16, 32, 64], 3),
    ],
)
def test_early_stopping_thresholds(tmp_path, stopping_threshold, divergence_threshold, losses, expected_epoch):
    class CurrentModel(BoringModel):
        def on_validation_epoch_end(self):
            val_loss = losses[self.current_epoch]
            self.log("abc", val_loss)

    model = CurrentModel()
    early_stopping = EarlyStopping(
        monitor="abc", stopping_threshold=stopping_threshold, divergence_threshold=divergence_threshold
    )
    trainer = Trainer(
        default_root_dir=tmp_path,
        callbacks=[early_stopping],
        limit_train_batches=0.2,
        limit_val_batches=0.2,
        max_epochs=20,
    )
    trainer.fit(model)
    assert trainer.current_epoch - 1 == expected_epoch, "early_stopping failed"


@pytest.mark.parametrize("stop_value", [torch.tensor(torch.inf), torch.tensor(torch.nan)])
def test_early_stopping_on_non_finite_monitor(tmp_path, stop_value):
    losses = [4, 3, stop_value, 2, 1]
    expected_stop_epoch = 2

    class CurrentModel(BoringModel):
        def on_validation_epoch_end(self):
            val_loss = losses[self.current_epoch]
            self.log("val_loss", val_loss)

    model = CurrentModel()
    early_stopping = EarlyStopping(monitor="val_loss", check_finite=True)
    trainer = Trainer(
        default_root_dir=tmp_path,
        callbacks=[early_stopping],
        limit_train_batches=0.2,
        limit_val_batches=0.2,
        max_epochs=10,
    )
    trainer.fit(model)
    assert trainer.current_epoch - 1 == expected_stop_epoch
    assert early_stopping.stopped_epoch == expected_stop_epoch


@pytest.mark.parametrize(
    ("limit_train_batches", "min_epochs", "min_steps", "stop_step"),
    [
        # IF `min_steps` was set to a higher value than the `trainer.global_step` when `early_stopping` is being
        # triggered, THEN the trainer should continue until reaching `trainer.global_step == min_steps` and stop
        (3, 0, 10, 10),
        (5, 0, 10, 10),
        # IF `min_epochs` resulted in a higher number of steps than the `trainer.global_step` when `early_stopping` is
        # being triggered, THEN the trainer should continue until reaching
        # `trainer.global_step` == `min_epochs * len(train_dataloader)`
        (3, 2, 0, 6),
        (5, 2, 0, 10),
        # IF both `min_epochs` and `min_steps` are provided and higher than the `trainer.global_step` when
        # `early_stopping` is being triggered, THEN the highest between `min_epochs * len(train_dataloader)` and
        # `min_steps` would be reached
        (3, 1, 10, 10),
        (5, 1, 10, 10),
        (3, 3, 10, 10),
        (5, 3, 10, 15),
    ],
)
def test_min_epochs_min_steps_global_step(tmp_path, limit_train_batches, min_epochs, min_steps, stop_step):
    if min_steps:
        assert limit_train_batches < min_steps

    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            self.log("foo", batch_idx)
            return super().training_step(batch, batch_idx)

    es_callback = EarlyStopping("foo")
    trainer = Trainer(
        default_root_dir=tmp_path,
        callbacks=es_callback,
        limit_val_batches=0,
        limit_train_batches=limit_train_batches,
        min_epochs=min_epochs,
        min_steps=min_steps,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    model = TestModel()

    expected_epochs = max(math.ceil(min_steps / limit_train_batches), min_epochs)
    # trigger early stopping directly after the first epoch
    side_effect = [(True, "")] * expected_epochs
    with mock.patch.object(es_callback, "_evaluate_stopping_criteria", side_effect=side_effect):
        trainer.fit(model)

    assert trainer.should_stop
    # epochs continue until min steps are reached
    assert trainer.current_epoch == expected_epochs
    # steps continue until min steps are reached AND the epoch is exhausted
    assert trainer.global_step == stop_step


def test_early_stopping_mode_options():
    with pytest.raises(MisconfigurationException, match="`mode` can be .* got unknown_option"):
        EarlyStopping(monitor="foo", mode="unknown_option")


class EarlyStoppingModel(BoringModel):
    def __init__(self, expected_end_epoch: int, early_stop_on_train: bool, dist_diverge_epoch: Optional[int] = None):
        super().__init__()
        self.expected_end_epoch = expected_end_epoch
        self.early_stop_on_train = early_stop_on_train
        self.dist_diverge_epoch = dist_diverge_epoch

    def _dist_diverge(self):
        should_diverge = (
            self.dist_diverge_epoch and self.current_epoch >= self.dist_diverge_epoch and self.trainer.global_rank == 0
        )
        return 10 if should_diverge else None

    def _epoch_end(self) -> None:
        losses = [8, 4, 2, 3, 4, 5, 8, 10]
        loss = self._dist_diverge() or losses[self.current_epoch]
        self.log("abc", torch.tensor(loss))
        self.log("cba", torch.tensor(0))

    def on_train_epoch_end(self):
        if not self.early_stop_on_train:
            return
        self._epoch_end()

    def on_validation_epoch_end(self):
        if self.early_stop_on_train:
            return
        self._epoch_end()

    def on_train_end(self) -> None:
        assert self.trainer.current_epoch - 1 == self.expected_end_epoch, "Early Stopping Failed"


_ES_CHECK = {"check_on_train_epoch_end": True}
_ES_CHECK_P3 = {"patience": 3, "check_on_train_epoch_end": True}
_SPAWN_MARK = {"marks": RunIf(skip_windows=True)}


@pytest.mark.parametrize(
    ("callbacks", "expected_stop_epoch", "check_on_train_epoch_end", "strategy", "devices", "dist_diverge_epoch"),
    [
        ([EarlyStopping("abc"), EarlyStopping("cba", patience=3)], 3, False, "auto", 1, None),
        ([EarlyStopping("cba", patience=3), EarlyStopping("abc")], 3, False, "auto", 1, None),
        pytest.param(
            [EarlyStopping("abc", patience=1), EarlyStopping("cba")], 2, False, "ddp_spawn", 2, 2, **_SPAWN_MARK
        ),
        pytest.param(
            [EarlyStopping("abc"), EarlyStopping("cba", patience=3)], 3, False, "ddp_spawn", 2, None, **_SPAWN_MARK
        ),
        pytest.param(
            [EarlyStopping("cba", patience=3), EarlyStopping("abc")], 3, False, "ddp_spawn", 2, None, **_SPAWN_MARK
        ),
        ([EarlyStopping("abc", **_ES_CHECK), EarlyStopping("cba", **_ES_CHECK_P3)], 3, True, "auto", 1, None),
        ([EarlyStopping("cba", **_ES_CHECK_P3), EarlyStopping("abc", **_ES_CHECK)], 3, True, "auto", 1, None),
        pytest.param(
            [EarlyStopping("abc", **_ES_CHECK), EarlyStopping("cba", **_ES_CHECK_P3)],
            3,
            True,
            "ddp_spawn",
            2,
            None,
            **_SPAWN_MARK,
        ),
        pytest.param(
            [EarlyStopping("cba", **_ES_CHECK_P3), EarlyStopping("abc", **_ES_CHECK)],
            3,
            True,
            "ddp_spawn",
            2,
            None,
            **_SPAWN_MARK,
        ),
    ],
)
def test_multiple_early_stopping_callbacks(
    tmp_path,
    callbacks: list[EarlyStopping],
    expected_stop_epoch: int,
    check_on_train_epoch_end: bool,
    strategy: str,
    devices: int,
    dist_diverge_epoch: Optional[int],
):
    """Ensure when using multiple early stopping callbacks we stop if any signals we should stop."""

    model = EarlyStoppingModel(expected_stop_epoch, check_on_train_epoch_end, dist_diverge_epoch=dist_diverge_epoch)

    trainer = Trainer(
        default_root_dir=tmp_path,
        callbacks=callbacks,
        limit_train_batches=0.1,
        limit_val_batches=0.1,
        max_epochs=20,
        strategy=strategy,
        accelerator="cpu",
        devices=devices,
    )
    trainer.fit(model)


@pytest.mark.parametrize(
    "case",
    {
        "val_check_interval": {"val_check_interval": 0.3, "limit_train_batches": 10, "max_epochs": 10},
        "check_val_every_n_epoch": {"check_val_every_n_epoch": 2, "max_epochs": 5},
    }.items(),
)
def test_check_on_train_epoch_end_smart_handling(tmp_path, case):
    class TestModel(BoringModel):
        def validation_step(self, batch, batch_idx):
            self.log("foo", 1)
            return super().validation_step(batch, batch_idx)

    case, kwargs = case
    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        limit_val_batches=1,
        callbacks=EarlyStopping(monitor="foo"),
        enable_progress_bar=False,
        **kwargs,
    )

    side_effect = [(False, "A"), (True, "B")]
    with mock.patch(
        "lightning.pytorch.callbacks.EarlyStopping._evaluate_stopping_criteria", side_effect=side_effect
    ) as es_mock:
        trainer.fit(model)

    assert es_mock.call_count == len(side_effect)
    if case == "val_check_interval":
        assert trainer.global_step == len(side_effect) * int(trainer.limit_train_batches * trainer.val_check_interval)
    else:
        assert trainer.current_epoch == len(side_effect) * trainer.check_val_every_n_epoch


def test_early_stopping_squeezes():
    early_stopping = EarlyStopping(monitor="foo")
    trainer = Trainer()
    trainer.callback_metrics["foo"] = torch.tensor([[[0]]])

    with mock.patch(
        "lightning.pytorch.callbacks.EarlyStopping._evaluate_stopping_criteria", return_value=(False, "")
    ) as es_mock:
        early_stopping._run_early_stopping_check(trainer)

    es_mock.assert_called_once_with(torch.tensor(0))


@pytest.mark.parametrize(
    ("log_rank_zero_only", "world_size", "global_rank", "expected_log"),
    [
        (False, 1, 0, "bar"),
        (False, 2, 0, "[rank: 0] bar"),
        (False, 2, 1, "[rank: 1] bar"),
        (True, 1, 0, "bar"),
        (True, 2, 0, "[rank: 0] bar"),
        (True, 2, 1, None),
    ],
)
def test_early_stopping_log_info(log_rank_zero_only, world_size, global_rank, expected_log):
    """Checks if log.info() gets called with expected message when used within EarlyStopping."""
    # set the global_rank and world_size if trainer is not None
    # or else always expect the simple logging message
    trainer = Mock(global_rank=global_rank, world_size=world_size)

    with mock.patch("lightning.pytorch.callbacks.early_stopping.log.info") as log_mock:
        EarlyStopping._log_info(trainer, "bar", log_rank_zero_only)

    # check log.info() was called or not with expected arg
    if expected_log:
        log_mock.assert_called_once_with(expected_log)
    else:
        log_mock.assert_not_called()


class ModelWithHighLoss(BoringModel):
    def on_validation_epoch_end(self):
        self.log("val_loss", 10.0)


class ModelWithDecreasingLoss(BoringModel):
    def __init__(self):
        super().__init__()
        self.epoch_losses = [5.0, 3.0, 1.0, 0.5]

    def on_validation_epoch_end(self):
        loss = self.epoch_losses[self.current_epoch] if self.current_epoch < len(self.epoch_losses) else 0.1
        self.log("val_loss", loss)


class ModelWithIncreasingLoss(BoringModel):
    def __init__(self):
        super().__init__()
        self.epoch_losses = [1.0, 2.0, 5.0, 10.0]

    def on_validation_epoch_end(self):
        loss = self.epoch_losses[self.current_epoch] if self.current_epoch < len(self.epoch_losses) else 15.0
        self.log("val_loss", loss)


class ModelWithNaNLoss(BoringModel):
    def __init__(self):
        super().__init__()
        self.epoch_losses = [1.0, 0.5, float("nan")]

    def on_validation_epoch_end(self):
        loss = self.epoch_losses[self.current_epoch] if self.current_epoch < len(self.epoch_losses) else float("nan")
        self.log("val_loss", loss)


class ModelWithImprovingLoss(BoringModel):
    def __init__(self):
        super().__init__()
        self.epoch_losses = [5.0, 4.0, 3.0, 2.0, 1.0]

    def on_validation_epoch_end(self):
        loss = self.epoch_losses[self.current_epoch] if self.current_epoch < len(self.epoch_losses) else 0.1
        self.log("val_loss", loss)


@pytest.mark.parametrize(
    (
        "model_cls",
        "early_stopping_kwargs",
        "trainer_kwargs",
        "expected_reason",
        "reason_message_substr",
        "should_stop",
        "state_dict_override",
    ),
    [
        # Patience exhausted
        (
            ModelWithHighLoss,
            {"monitor": "val_loss", "patience": 2, "verbose": True},
            {"max_epochs": 10, "enable_progress_bar": False},
            EarlyStoppingReason.PATIENCE_EXHAUSTED,
            "did not improve",
            True,
            None,
        ),
        # Stopping threshold
        (
            ModelWithDecreasingLoss,
            {"monitor": "val_loss", "stopping_threshold": 0.6, "mode": "min", "verbose": True},
            {"max_epochs": 10, "enable_progress_bar": False},
            EarlyStoppingReason.STOPPING_THRESHOLD,
            "Stopping threshold reached",
            True,
            None,
        ),
        # Divergence threshold
        (
            ModelWithIncreasingLoss,
            {"monitor": "val_loss", "divergence_threshold": 8.0, "mode": "min", "verbose": True},
            {"max_epochs": 10, "enable_progress_bar": False},
            EarlyStoppingReason.DIVERGENCE_THRESHOLD,
            "Divergence threshold reached",
            True,
            None,
        ),
        # Non-finite metric
        (
            ModelWithNaNLoss,
            {"monitor": "val_loss", "check_finite": True, "verbose": True},
            {"max_epochs": 10, "enable_progress_bar": False},
            EarlyStoppingReason.NON_FINITE_METRIC,
            "is not finite",
            True,
            None,
        ),
        # Not stopped (normal completion)
        (
            ModelWithImprovingLoss,
            {"monitor": "val_loss", "patience": 3, "verbose": True},
            {"max_epochs": 3, "enable_progress_bar": False},
            EarlyStoppingReason.NOT_STOPPED,
            None,
            False,
            None,
        ),
        # State persistence
        (
            None,
            {"monitor": "val_loss", "patience": 3},
            {},
            EarlyStoppingReason.PATIENCE_EXHAUSTED,
            "Test message",
            None,
            {"stopping_reason": EarlyStoppingReason.PATIENCE_EXHAUSTED, "stopping_reason_message": "Test message"},
        ),
        # Backward compatibility (old state dict)
        (
            None,
            {"monitor": "val_loss", "patience": 3},
            {},
            EarlyStoppingReason.NOT_STOPPED,
            None,
            None,
            {
                "wait_count": 2,
                "stopped_epoch": 5,
                "best_score": torch.tensor(0.5),
                "patience": 3,
            },
        ),
    ],
)
def test_early_stopping_reasons(
    tmp_path,
    model_cls,
    early_stopping_kwargs,
    trainer_kwargs,
    expected_reason,
    reason_message_substr,
    should_stop,
    state_dict_override,
):
    """Test all early stopping reasons in a single parametrized test."""
    if state_dict_override is not None:
        early_stopping = EarlyStopping(**early_stopping_kwargs)
        if "stopping_reason" in state_dict_override:
            # State persistence test
            early_stopping.stopping_reason = state_dict_override["stopping_reason"]
            early_stopping.stopping_reason_message = state_dict_override["stopping_reason_message"]
            state_dict = early_stopping.state_dict()
            new_early_stopping = EarlyStopping(**early_stopping_kwargs)
            new_early_stopping.load_state_dict(state_dict)
            assert new_early_stopping.stopping_reason == expected_reason
            assert new_early_stopping.stopping_reason_message == reason_message_substr
        else:
            # Backward compatibility test
            early_stopping.load_state_dict(copy.deepcopy(state_dict_override))
            assert early_stopping.stopping_reason == expected_reason
            assert early_stopping.stopping_reason_message is None
            assert early_stopping.wait_count == state_dict_override["wait_count"]
            assert early_stopping.stopped_epoch == state_dict_override["stopped_epoch"]
        return

    # All other tests
    model = model_cls()
    early_stopping = EarlyStopping(**early_stopping_kwargs)
    trainer = Trainer(
        default_root_dir=tmp_path,
        callbacks=[early_stopping],
        **trainer_kwargs,
    )
    trainer.fit(model)

    assert early_stopping.stopping_reason == expected_reason
    if reason_message_substr is not None:
        assert early_stopping.stopping_reason_message is not None
        assert reason_message_substr in early_stopping.stopping_reason_message
    else:
        assert early_stopping.stopping_reason_message is None
    if should_stop is not None:
        if should_stop:
            assert early_stopping.stopped_epoch > 0
        else:
            assert early_stopping.stopped_epoch == 0


@pytest.mark.parametrize(
    ("mode", "divergence_threshold", "current", "expected_op"),
    [
        ("min", 8.0, torch.tensor(10.0), ">"),
        ("max", 0.5, torch.tensor(0.2), "<"),
    ],
)
def test_early_stopping_divergence_threshold_reason_message(mode, divergence_threshold, current, expected_op):
    early_stopping = EarlyStopping(monitor="val_metric", mode=mode, divergence_threshold=divergence_threshold)
    should_stop, reason = early_stopping._evaluate_stopping_criteria(current)
    assert should_stop is True
    assert early_stopping.stopping_reason == EarlyStoppingReason.DIVERGENCE_THRESHOLD
    assert reason == (
        f"Divergence threshold reached: val_metric = {current} {expected_op} {divergence_threshold}."
        " Signaling Trainer to stop."
    )

