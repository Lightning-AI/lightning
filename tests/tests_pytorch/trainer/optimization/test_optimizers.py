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
from unittest import mock
from unittest.mock import call, patch

import pytest
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.core.optimizer import (
    _configure_optimizers,
    _configure_schedulers_automatic_opt,
    _init_optimizers_and_lr_schedulers,
)
from lightning.pytorch.demos.boring_classes import BoringDataModule, BoringModel
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.types import LRSchedulerConfig
from tests_pytorch.helpers.runif import RunIf


def test_optimizer_with_scheduling(tmp_path):
    """Verify that learning rate scheduling is working."""
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmp_path, max_epochs=1, limit_val_batches=0.1, limit_train_batches=0.2, val_check_interval=0.5
    )
    trainer.fit(model)

    init_lr = 0.1
    adjusted_lr = [pg["lr"] for pg in trainer.optimizers[0].param_groups]

    assert len(trainer.lr_scheduler_configs) == 1
    assert all(a == adjusted_lr[0] for a in adjusted_lr)
    assert init_lr * 0.1 == adjusted_lr[0]


def test_multi_optimizer_with_scheduling(tmp_path):
    """Verify that learning rate scheduling is working."""

    class Model(BoringModel):
        init_lr = 5e-4

        def training_step(self, batch, batch_idx):
            opt1, opt2 = self.optimizers()
            loss = self.loss(self.step(batch))
            opt1.zero_grad()
            opt2.zero_grad()
            self.manual_backward(loss)
            opt1.step()
            opt2.step()

        def on_train_epoch_end(self):
            scheduler1, scheduler2 = self.lr_schedulers()
            scheduler1.step()
            scheduler2.step()

        def configure_optimizers(self):
            optimizer1 = optim.Adam(self.parameters(), lr=self.init_lr)
            optimizer2 = optim.Adam(self.parameters(), lr=self.init_lr)
            lr_scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=1)
            lr_scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=1)
            return [optimizer1, optimizer2], [lr_scheduler1, lr_scheduler2]

    model = Model()
    model.automatic_optimization = False
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1, limit_val_batches=0.1, limit_train_batches=0.2)
    trainer.fit(model)

    adjusted_lr1 = [pg["lr"] for pg in trainer.optimizers[0].param_groups]
    adjusted_lr2 = [pg["lr"] for pg in trainer.optimizers[1].param_groups]

    assert len(trainer.lr_scheduler_configs) == 2
    assert all(a == adjusted_lr1[0] for a in adjusted_lr1)
    assert all(a == adjusted_lr2[0] for a in adjusted_lr2)
    assert model.init_lr * 0.1 == adjusted_lr1[0]
    assert model.init_lr * 0.1 == adjusted_lr2[0]


def test_reducelronplateau_with_no_monitor_raises(tmp_path):
    """Test exception when a ReduceLROnPlateau is used with no monitor."""
    model = BoringModel()
    optimizer = optim.Adam(model.parameters())
    model.configure_optimizers = lambda: ([optimizer], [optim.lr_scheduler.ReduceLROnPlateau(optimizer)])
    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=True)
    with pytest.raises(
        MisconfigurationException, match="`configure_optimizers` must include a monitor when a `ReduceLROnPlateau`"
    ):
        trainer.fit(model)


def test_reducelronplateau_with_no_monitor_in_lr_scheduler_dict_raises(tmp_path):
    """Test exception when lr_scheduler dict has a ReduceLROnPlateau with no monitor."""
    model = BoringModel()
    optimizer = optim.Adam(model.parameters())
    model.configure_optimizers = lambda: {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer)},
    }
    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=True)
    with pytest.raises(MisconfigurationException, match="must include a monitor when a `ReduceLROnPlateau`"):
        trainer.fit(model)


def test_onecyclelr_with_epoch_interval_warns():
    """Test warning when a OneCycleLR is used and interval is epoch."""
    model = BoringModel()
    optimizer = optim.Adam(model.parameters())
    lr_scheduler = {"scheduler": optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, total_steps=3)}
    with pytest.warns(RuntimeWarning, match="Are you sure you didn't mean 'interval': 'step'?"):
        _configure_schedulers_automatic_opt([lr_scheduler], None)


def test_scheduler_initialized_with_custom_reduceonplateau():
    """Test for initialize custom scheduler with `reduce_on_plateau` argument."""

    class CustomReduceLROnPlateau:
        pass

    lr_scheduler = {"reduce_on_plateau": True, "scheduler": CustomReduceLROnPlateau(), "monitor": "my_loss"}
    config = _configure_schedulers_automatic_opt([lr_scheduler], None)
    assert isinstance(config[0].scheduler, CustomReduceLROnPlateau)
    assert config[0].reduce_on_plateau


def test_reducelronplateau_scheduling(tmp_path):
    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            self.log("foo", batch_idx)
            return super().training_step(batch, batch_idx)

        def configure_optimizers(self):
            optimizer = optim.Adam(self.parameters())
            return {
                "optimizer": optimizer,
                "lr_scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer),
                "monitor": "foo",
            }

    model = TestModel()
    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=True)
    trainer.fit(model)

    lr_scheduler = trainer.lr_scheduler_configs[0]
    assert lr_scheduler == LRSchedulerConfig(
        scheduler=lr_scheduler.scheduler,
        monitor="foo",
        interval="epoch",
        frequency=1,
        reduce_on_plateau=True,
        strict=True,
        name=None,
    )


def test_optimizer_return_options(tmp_path):
    trainer = Trainer(default_root_dir=tmp_path)
    model = BoringModel()
    trainer.strategy.connect(model)
    trainer.lightning_module.trainer = trainer

    # single optimizer
    opt_a = optim.Adam(model.parameters(), lr=0.002)
    opt_b = optim.SGD(model.parameters(), lr=0.002)
    scheduler_a = optim.lr_scheduler.StepLR(opt_a, 10)
    optim.lr_scheduler.StepLR(opt_b, 10)

    # single optimizer
    model.configure_optimizers = lambda: opt_a
    opt, lr_sched = _init_optimizers_and_lr_schedulers(model)
    assert len(opt) == 1
    assert len(lr_sched) == 0

    # opt tuple
    model.automatic_optimization = False
    model.configure_optimizers = lambda: (opt_a, opt_b)
    opt, lr_sched = _init_optimizers_and_lr_schedulers(model)
    assert opt == [opt_a, opt_b]
    assert len(lr_sched) == 0

    # opt list
    model.automatic_optimization = False
    model.configure_optimizers = lambda: [opt_a, opt_b]
    opt, lr_sched = _init_optimizers_and_lr_schedulers(model)
    assert opt == [opt_a, opt_b]
    assert len(lr_sched) == 0

    ref_lr_sched = LRSchedulerConfig(
        scheduler=scheduler_a,
        interval="epoch",
        frequency=1,
        reduce_on_plateau=False,
        monitor=None,
        strict=True,
        name=None,
    )

    # opt tuple of 2 lists
    model.automatic_optimization = True
    model.configure_optimizers = lambda: ([opt_a], [scheduler_a])
    opt, lr_sched = _init_optimizers_and_lr_schedulers(model)
    assert len(opt) == len(lr_sched) == 1
    assert opt[0] == opt_a
    assert lr_sched[0] == ref_lr_sched

    # opt tuple of 1 list
    model.automatic_optimization = True
    model.configure_optimizers = lambda: ([opt_a], scheduler_a)
    opt, lr_sched = _init_optimizers_and_lr_schedulers(model)
    assert len(opt) == len(lr_sched) == 1
    assert opt[0] == opt_a
    assert lr_sched[0] == ref_lr_sched

    # opt single dictionary
    model.automatic_optimization = True
    model.configure_optimizers = lambda: {"optimizer": opt_a, "lr_scheduler": scheduler_a}
    opt, lr_sched = _init_optimizers_and_lr_schedulers(model)
    assert len(opt) == len(lr_sched) == 1
    assert opt[0] == opt_a
    assert lr_sched[0] == ref_lr_sched

    # opt list of dictionaries
    model.automatic_optimization = False
    model.configure_optimizers = lambda: [
        {"optimizer": opt_a, "lr_scheduler": scheduler_a},
        {"optimizer": opt_b, "lr_scheduler": scheduler_a},
    ]
    opt, lr_sched = _init_optimizers_and_lr_schedulers(model)
    assert len(opt) == len(lr_sched) == 2
    assert opt == [opt_a, opt_b]
    assert lr_sched == [ref_lr_sched, ref_lr_sched]


def test_none_optimizer(tmp_path):
    model = BoringModel()
    model.configure_optimizers = lambda: None
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1, limit_val_batches=0.1, limit_train_batches=0.2)
    with pytest.warns(UserWarning, match="will run with no optimizer"):
        trainer.fit(model)


def test_configure_optimizer_from_dict(tmp_path):
    """Tests if `configure_optimizer` method could return a dictionary with `optimizer` field only."""

    class TestModel(BoringModel):
        def configure_optimizers(self):
            return {"optimizer": optim.SGD(params=self.parameters(), lr=1e-03)}

    model = TestModel()
    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=True)
    trainer.fit(model)


@pytest.mark.parametrize("fn", ["validate", "test", "predict"])
def test_init_optimizers_during_evaluation_and_prediction(tmp_path, fn):
    """Test that optimizers is an empty list during evaluation and prediction."""

    class TestModel(BoringModel):
        def configure_optimizers(self):
            optimizer1 = optim.Adam(self.parameters(), lr=0.1)
            optimizer2 = optim.Adam(self.parameters(), lr=0.1)
            lr_scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=1)
            lr_scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=1)
            return [optimizer1, optimizer2], [lr_scheduler1, lr_scheduler2]

    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=2)
    train_fn = getattr(trainer, fn)
    train_fn(TestModel(), datamodule=BoringDataModule(), ckpt_path=None)

    assert len(trainer.lr_scheduler_configs) == 0
    assert len(trainer.optimizers) == 0


@pytest.mark.parametrize("complete_epoch", [True, False])
@mock.patch("torch.optim.lr_scheduler.ReduceLROnPlateau.step")
def test_lr_scheduler_strict(step_mock, tmp_path, complete_epoch):
    """Test "strict" support in lr_scheduler dict."""
    model = BoringModel()
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    max_epochs = 1 if complete_epoch else None
    max_steps = -1 if complete_epoch else 1
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=max_epochs, max_steps=max_steps)

    model.configure_optimizers = lambda: {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "monitor": "giraffe", "strict": True},
    }

    if complete_epoch:
        with pytest.raises(
            MisconfigurationException,
            match=r"ReduceLROnPlateau conditioned on metric .* which is not available\. Available metrics are:",
        ):
            trainer.fit(model)
    else:
        trainer.fit(model)

    step_mock.assert_not_called()

    model.configure_optimizers = lambda: {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "monitor": "giraffe", "strict": False},
    }

    if complete_epoch:
        trainer = Trainer(default_root_dir=tmp_path, max_epochs=max_epochs, max_steps=max_steps)
        with pytest.warns(
            RuntimeWarning, match=r"ReduceLROnPlateau conditioned on metric .* which is not available but strict"
        ):
            trainer.fit(model)

    step_mock.assert_not_called()


def test_unknown_configure_optimizers_raises(tmp_path):
    """Test exception with an unsupported configure_optimizers return."""
    model = BoringModel()
    model.configure_optimizers = lambda: 1
    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=True)
    with pytest.raises(MisconfigurationException, match="Unknown configuration for model optimizers"):
        trainer.fit(model)


def test_optimizer_config_dict_with_extra_keys_warns(tmp_path):
    """Test exception when optimizer configuration dict has extra keys."""
    model = BoringModel()
    optimizer = optim.Adam(model.parameters())
    optim_conf = {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": optim.lr_scheduler.StepLR(optimizer, 1)},
        "foo": 1,
        "bar": 2,
    }
    with pytest.warns(RuntimeWarning, match=r"Found unsupported keys in the optimizer configuration: \{.+\}"):
        _configure_optimizers(optim_conf)


def test_multiple_optimizer_config_dicts_with_extra_keys_warns(tmp_path):
    """Test exception when multiple optimizer configuration dicts have extra keys."""
    model = BoringModel()
    optimizer1 = optim.Adam(model.parameters(), lr=0.01)
    optimizer2 = optim.Adam(model.parameters(), lr=0.01)
    lr_scheduler_config_1 = {"scheduler": optim.lr_scheduler.StepLR(optimizer1, 1)}
    lr_scheduler_config_2 = {"scheduler": optim.lr_scheduler.StepLR(optimizer2, 1)}
    optim_conf = [
        {"optimizer": optimizer1, "lr_scheduler": lr_scheduler_config_1, "foo": 1, "bar": 2},
        {"optimizer": optimizer2, "lr_scheduler": lr_scheduler_config_2, "foo": 1, "bar": 2},
    ]
    with pytest.warns(RuntimeWarning, match=r"Found unsupported keys in the optimizer configuration: \{.+\}"):
        _configure_optimizers(optim_conf)


def test_lr_scheduler_with_unknown_interval_raises(tmp_path):
    """Test exception when lr_scheduler dict has unknown interval param value."""
    model = BoringModel()
    optimizer = optim.Adam(model.parameters())
    model.configure_optimizers = lambda: {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": optim.lr_scheduler.StepLR(optimizer, 1), "interval": "incorrect_unknown_value"},
    }
    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=True)
    with pytest.raises(MisconfigurationException, match=r'The "interval" key in lr scheduler dict must be'):
        trainer.fit(model)


def test_lr_scheduler_with_extra_keys_warns(tmp_path):
    """Test warning when lr_scheduler dict has extra keys."""
    model = BoringModel()
    optimizer = optim.Adam(model.parameters())
    model.configure_optimizers = lambda: {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": optim.lr_scheduler.StepLR(optimizer, 1), "foo": 1, "bar": 2},
    }
    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=True)
    with pytest.warns(RuntimeWarning, match=r"Found unsupported keys in the lr scheduler dict: \{.+\}"):
        trainer.fit(model)


def test_lr_scheduler_with_no_actual_scheduler_raises(tmp_path):
    """Test exception when lr_scheduler dict has no scheduler."""
    model = BoringModel()
    model.configure_optimizers = lambda: {"optimizer": optim.Adam(model.parameters()), "lr_scheduler": {}}
    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=True)
    with pytest.raises(MisconfigurationException, match='The lr scheduler dict must have the key "scheduler"'):
        trainer.fit(model)


def test_invalid_optimizer_in_scheduler(tmp_path):
    """Test exception when optimizer attached to lr_schedulers wasn't returned."""

    class InvalidOptimizerModel(BoringModel):
        def configure_optimizers(self):
            opt1 = optim.SGD(self.layer.parameters(), lr=0.1)
            opt2 = optim.SGD(self.layer.parameters(), lr=0.1)
            lr_scheduler = optim.lr_scheduler.StepLR(opt2, step_size=1)
            return [opt1], [lr_scheduler]

    model = InvalidOptimizerModel()
    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=True)
    with pytest.raises(MisconfigurationException, match="attached with an optimizer that wasn't returned"):
        trainer.fit(model)


def test_invalid_optimizer_dict_raises(tmp_path):
    """Test exception when lr_scheduler dict has no scheduler."""

    class DummyModel(BoringModel):
        def configure_optimizers(self):
            return [{"optimizer": optim.Adam(self.parameters())}, optim.Adam(self.parameters())]

    model = DummyModel()
    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=True)
    with pytest.raises(MisconfigurationException, match="Unknown configuration for model optimizers"):
        trainer.fit(model)


@RunIf(min_cuda_gpus=2, standalone=True)
def test_optimizer_state_on_device(tmp_path):
    """Test that optimizers that create state initially at instantiation still end up with the state on the GPU."""

    class TestModel(BoringModel):
        def configure_optimizers(self):
            # Adagrad creates state tensors immediately, model is not yet on GPU.
            return optim.Adagrad(self.parameters())

        def on_train_start(self, *args, **kwargs):
            opt = self.optimizers()
            _, state = next(iter(opt.state.items()))
            assert state["sum"].device == torch.device("cuda", self.local_rank) == self.device

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator="gpu",
        devices=2,
        strategy="ddp",
        fast_dev_run=True,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model)


@pytest.mark.parametrize("check_val_every_n_epoch", [1, 2])
@mock.patch("torch.optim.lr_scheduler.StepLR.step")
def test_lr_scheduler_epoch_step_frequency(mocked_sched, check_val_every_n_epoch, tmp_path):
    epochs = 4
    expected_steps = epochs + 1  # every LRScheduler gets called once at init

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        limit_train_batches=2,
        limit_val_batches=2,
        check_val_every_n_epoch=check_val_every_n_epoch,
        max_epochs=epochs,
    )
    trainer.fit(model)
    assert mocked_sched.call_count == expected_steps


@pytest.mark.parametrize(("every_n_train_steps", "epoch_interval"), [(None, True), (2, False), (2, True)])
def test_lr_scheduler_state_updated_before_saving(tmp_path, every_n_train_steps, epoch_interval):
    batches = 2
    max_epochs = 1
    lr, gamma = 1, 10
    trainer = Trainer(
        default_root_dir=tmp_path,
        enable_progress_bar=False,
        logger=False,
        max_epochs=max_epochs,
        limit_train_batches=batches,
        limit_val_batches=1,
        callbacks=[ModelCheckpoint(dirpath=tmp_path, every_n_train_steps=every_n_train_steps)],
    )

    class TestModel(BoringModel):
        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.parameters(), lr=lr)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
            lr_scheduler_config = {"scheduler": lr_scheduler}
            if not epoch_interval:
                lr_scheduler_config["interval"] = "step"
            return [optimizer], [lr_scheduler_config]

        def on_save_checkpoint(self, checkpoint):
            lr_scheduler_config = checkpoint["lr_schedulers"][0]
            # 2 batches ran. since the lr_scheduler_config interval is `step`, the step count should be 2
            assert self.trainer.global_step == batches
            compare_to = max_epochs if epoch_interval else batches
            assert lr_scheduler_config["_step_count"] - 1 == compare_to  # step count starts at 1
            assert lr_scheduler_config["_last_lr"] == [lr * gamma**compare_to]
            self.on_save_checkpoint_called = True

    model = TestModel()
    trainer.fit(model)
    assert model.on_save_checkpoint_called


@pytest.mark.parametrize("save_on_train_epoch_end", [False, True])
def test_plateau_scheduler_lr_step_interval_updated_after_saving(tmp_path, save_on_train_epoch_end):
    batches = 4
    trainer = Trainer(
        default_root_dir=tmp_path,
        enable_progress_bar=False,
        logger=False,
        max_epochs=1,
        limit_train_batches=batches,
        limit_val_batches=1,
        callbacks=[ModelCheckpoint(dirpath=tmp_path, save_on_train_epoch_end=save_on_train_epoch_end)],
    )

    class Model(BoringModel):
        def training_step(self, batch, batch_idx):
            self.log("foo", batch_idx)
            return super().training_step(batch, batch_idx)

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters())

            lr_scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            lr_scheduler_config_1 = {"scheduler": lr_scheduler1, "interval": "step", "monitor": "foo"}

            lr_scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            lr_scheduler_config_2 = {"scheduler": lr_scheduler2, "interval": "step"}
            return [optimizer], [lr_scheduler_config_1, lr_scheduler_config_2]

        def on_save_checkpoint(self, checkpoint):
            lr_scheduler_config_1 = checkpoint["lr_schedulers"][0]
            last_epoch = lr_scheduler_config_1["last_epoch"]
            assert last_epoch == batches - (not save_on_train_epoch_end)  # last epoch starts at 0

            lr_scheduler_config_2 = checkpoint["lr_schedulers"][1]
            assert lr_scheduler_config_2["_step_count"] - 1 == batches  # step count starts at 1

            self.on_save_checkpoint_called = True

    model = Model()
    trainer.fit(model)
    assert model.on_save_checkpoint_called


def test_lr_scheduler_step_hook(tmp_path):
    """Test that custom lr scheduler works and `lr_scheduler_step` is called at appropriate time."""

    class CustomEpochScheduler:
        def __init__(self, optimizer):
            self.optimizer = optimizer

        def step(self, epoch): ...

        def state_dict(self): ...

        def load_state_dict(self, state_dict): ...

    class CustomBoringModel(BoringModel):
        def lr_scheduler_step(self, scheduler: int, metric):
            # step-level
            if isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
                super().lr_scheduler_step(scheduler, metric)
            # epoch-level, custom scheduler
            elif isinstance(scheduler, CustomEpochScheduler):
                scheduler.step(epoch=self.current_epoch)

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=1e-2)
            lr_scheduler1 = {"scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=1), "interval": "step"}
            lr_scheduler2 = CustomEpochScheduler(optimizer)
            return [optimizer], [lr_scheduler1, lr_scheduler2]

    model = CustomBoringModel()
    max_epochs = 3
    limit_train_batches = 2
    trainer = Trainer(
        default_root_dir=tmp_path,
        enable_checkpointing=False,
        logger=False,
        max_epochs=max_epochs,
        limit_train_batches=limit_train_batches,
        limit_val_batches=0,
    )
    with (
        mock.patch.object(CustomEpochScheduler, "step") as mock_method_epoch,
        mock.patch.object(torch.optim.lr_scheduler.StepLR, "step") as mock_method_step,
    ):
        trainer.fit(model)

    assert mock_method_epoch.mock_calls == [call(epoch=e) for e in range(max_epochs)]
    # first step is called by PyTorch LRScheduler
    assert mock_method_step.call_count == max_epochs * limit_train_batches + 1


def test_invalid_scheduler_missing_state_dict():
    """Test that custom lr scheduler raises an error if it's missing the state dict."""

    class CustomScheduler:
        def __init__(self, optimizer):
            self.optimizer = optimizer

        def step(self): ...

    class CustomBoringModel(BoringModel):
        def configure_optimizers(self):
            opt = torch.optim.SGD(self.parameters(), lr=1e-2)
            lr_scheduler = CustomScheduler(opt)
            return {"optimizer": opt, "lr_scheduler": lr_scheduler}

    model = CustomBoringModel()
    model.trainer = Trainer()
    with pytest.raises(TypeError, match="provided lr scheduler `CustomScheduler` is invalid"):
        _init_optimizers_and_lr_schedulers(model)


@pytest.mark.parametrize("override", [False, True])
def test_invalid_lr_scheduler_with_custom_step_method(override):
    """Test that custom lr scheduler raises an error if it doesn't follow PyTorch LR Scheduler API."""

    class CustomScheduler:
        def __init__(self, optimizer):
            self.optimizer = optimizer

        def step(self, foobar):  # breaks the API, forces user to override `lr_scheduler_step`
            ...

        def state_dict(self): ...

        def load_state_dict(self, state_dict): ...

    class CustomBoringModel(BoringModel):
        def configure_optimizers(self):
            opt = torch.optim.SGD(self.parameters(), lr=1e-2)
            lr_scheduler = CustomScheduler(opt)
            return {"optimizer": opt, "lr_scheduler": lr_scheduler}

    model = CustomBoringModel()
    model.trainer = Trainer()
    if override:

        def lr_scheduler_step(*_): ...

        # the user did override the hook, no error
        model.lr_scheduler_step = lr_scheduler_step
        _init_optimizers_and_lr_schedulers(model)
    else:
        with pytest.raises(MisconfigurationException, match="CustomScheduler` doesn't follow"):
            _init_optimizers_and_lr_schedulers(model)


@patch("torch.optim.lr_scheduler.StepLR.step")
def test_lr_scheduler_step_across_epoch_boundaries(mocked_sched, tmp_path):
    class StepAcrossEpochsModel(LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(32, 2)

        def forward(self, x):
            return self.layer(x)

        def training_step(self, batch, batch_idx):
            # Add print statement to track batch index and global step
            if hasattr(self, "trainer"):
                print(f"Batch idx: {batch_idx}, Global step: {self.trainer.global_step}")
            return {"loss": torch.tensor(0.1, requires_grad=True)}

        def train_dataloader(self):
            x = torch.randn(21, 32)
            y = torch.randn(21, 2)
            return DataLoader(TensorDataset(x, y), batch_size=3)

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 5,  # Scheduler steps every 5 iterations
                },
            }

    model = StepAcrossEpochsModel()

    # Trainer configuration for cross-epoch testing
    trainer = Trainer(
        default_root_dir=tmp_path,
        limit_train_batches=7,  # More than `frequency` iterations per epoch
        max_epochs=3,  # Test across multiple epochs
    )

    # Fit the model
    trainer.fit(model)

    # Debug print statements
    print(f"Mocked scheduler step calls: {mocked_sched.call_count}")
    print(f"Mocked scheduler call history: {mocked_sched.call_args_list}")

    # Calculate the total number of steps (iterations) and expected scheduler calls
    total_steps = 7 * 3  # Total iterations (7 batches per epoch * 3 epochs)
    expected_steps = (total_steps - 1) // 5  # Scheduler steps every 5 iterations

    print(f"Total steps: {total_steps}")
    print(f"Expected steps: {expected_steps}")

    # Assert that the scheduler was called the expected number of times
    # Allow for a small difference due to environment or rounding discrepancies
    assert abs(mocked_sched.call_count - expected_steps) <= 1, (
        f"Scheduler was called {mocked_sched.call_count} times, but expected {expected_steps} calls."
    )
