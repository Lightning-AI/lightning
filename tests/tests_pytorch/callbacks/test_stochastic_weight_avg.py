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
import logging
import os
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Optional
from unittest import mock

import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import SWALR
from torch.utils.data import DataLoader

from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_6
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset, RandomIterableDataset
from lightning.pytorch.strategies import Strategy
from lightning.pytorch.strategies.launchers import _MultiProcessingLauncher
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from tests_pytorch.helpers.runif import RunIf


def test_swa_callback_initial_state():
    swa = StochasticWeightAveraging(
        swa_lrs=0.01,
        swa_epoch_start=0.1,
        annealing_epochs=1,
        annealing_strategy="linear",
        avg_fn=sum,
    )
    assert swa._swa_lrs == 0.01
    assert swa._swa_epoch_start == 0.1
    assert swa._annealing_epochs == 1
    assert swa._annealing_strategy == "linear"
    assert swa._avg_fn == sum
    assert swa._average_model is None


class SwaTestModel(BoringModel):
    def __init__(
        self, batchnorm: bool = True, interval: str = "epoch", iterable_dataset: bool = False, crash_on_epoch=None
    ):
        super().__init__()
        layers = [nn.Linear(32, 32)]
        if batchnorm:
            layers.append(nn.BatchNorm1d(32))
        layers += [nn.ReLU(), nn.Linear(32, 2)]
        self.layer = nn.Sequential(*layers)
        self.interval = interval
        self.iterable_dataset = iterable_dataset
        self.crash_on_epoch = crash_on_epoch

    def training_step(self, batch, batch_idx):
        if self.crash_on_epoch and self.trainer.current_epoch >= self.crash_on_epoch:
            raise Exception("SWA crash test")
        return super().training_step(batch, batch_idx)

    def train_dataloader(self):
        dset_cls = RandomIterableDataset if self.iterable_dataset else RandomDataset
        dset = dset_cls(32, 64)
        return DataLoader(dset, batch_size=2)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=1),
                "interval": self.interval,
            },
        }


class SwaTestCallback(StochasticWeightAveraging):
    update_parameters_calls: int = 0
    transfer_weights_calls: int = 0
    # Record the first epoch, as if we are resuming from a checkpoint this may not be equal to 0
    first_epoch: Optional[int] = None

    def update_parameters(self, *args, **kwargs):
        self.update_parameters_calls += 1
        return StochasticWeightAveraging.update_parameters(*args, **kwargs)

    def transfer_weights(self, *args, **kwargs):
        self.transfer_weights_calls += 1
        return StochasticWeightAveraging.transfer_weights(*args, **kwargs)

    def on_train_epoch_start(self, trainer, *args):
        super().on_train_epoch_start(trainer, *args)
        if self.first_epoch is None and not trainer.fit_loop.restarting:
            # since the checkpoint loaded was saved `on_train_epoch_end`, the first `FitLoop` iteration will
            # not update the model and just call the epoch-level hooks, for that reason, we check that we are not
            # restarting before choosing the first epoch
            self.first_epoch = trainer.current_epoch
        assert trainer.fit_loop._skip_backward == (trainer.current_epoch > self.swa_end)
        if self.swa_start <= trainer.current_epoch:
            assert isinstance(trainer.lr_scheduler_configs[0].scheduler, SWALR)
            assert trainer.lr_scheduler_configs[0].interval == "epoch"
            assert trainer.lr_scheduler_configs[0].frequency == 1

    def on_train_epoch_end(self, trainer, *args):
        super().on_train_epoch_end(trainer, *args)
        if self.swa_start <= trainer.current_epoch <= self.swa_end:
            swa_epoch = trainer.current_epoch - self.swa_start
            assert self.n_averaged == swa_epoch + 1
            assert self._swa_scheduler is not None
            # Scheduler is stepped once on initialization and then at the end of each epoch
            assert self._swa_scheduler._step_count == swa_epoch + 2
        elif trainer.current_epoch > self.swa_end:
            assert self.n_averaged == self._max_epochs - self.swa_start

    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)

        # make sure these are correctly set again
        assert not trainer.fit_loop._skip_backward
        assert trainer.accumulate_grad_batches == 2
        assert trainer.num_training_batches == 5

        if not isinstance(trainer.strategy.launcher, _MultiProcessingLauncher):
            # check backward call count. the batchnorm update epoch should not backward
            assert trainer.strategy.backward.call_count == (
                (trainer.max_epochs - self.first_epoch) * trainer.limit_train_batches
            )

        # check call counts
        first_swa_epoch = max(self.first_epoch, self.swa_start)
        assert self.update_parameters_calls == trainer.max_epochs - first_swa_epoch
        assert self.transfer_weights_calls == 1


def train_with_swa(
    tmp_path,
    batchnorm=True,
    strategy="auto",
    accelerator="cpu",
    devices=1,
    interval="epoch",
    iterable_dataset=False,
):
    model = SwaTestModel(batchnorm=batchnorm, interval=interval, iterable_dataset=iterable_dataset)
    swa_start = 2
    max_epochs = 5
    swa_callback = SwaTestCallback(swa_epoch_start=swa_start, swa_lrs=0.1)
    assert swa_callback.update_parameters_calls == 0
    assert swa_callback.transfer_weights_calls == 0

    trainer = Trainer(
        default_root_dir=tmp_path,
        enable_progress_bar=False,
        enable_model_summary=False,
        max_epochs=max_epochs,
        limit_train_batches=5,
        limit_val_batches=0,
        callbacks=[swa_callback],
        accumulate_grad_batches=2,
        strategy=strategy,
        accelerator=accelerator,
        devices=devices,
    )

    weights_only = False if _TORCH_GREATER_EQUAL_2_6 else None
    with _backward_patch(trainer):
        trainer.fit(model, weights_only=weights_only)

    # check the model is the expected
    assert trainer.lightning_module == model


@RunIf(min_cuda_gpus=2, standalone=True)
def test_swa_callback_ddp(tmp_path):
    train_with_swa(tmp_path, strategy="ddp", accelerator="gpu", devices=2)


@RunIf(min_cuda_gpus=2)
def test_swa_callback_ddp_spawn(tmp_path):
    train_with_swa(tmp_path, strategy="ddp_spawn", accelerator="gpu", devices=2)


@RunIf(skip_windows=True)
def test_swa_callback_ddp_cpu(tmp_path):
    train_with_swa(tmp_path, strategy="ddp_spawn", accelerator="cpu", devices=2)


@pytest.mark.parametrize(
    "accelerator", [pytest.param("gpu", marks=RunIf(min_cuda_gpus=1)), pytest.param("mps", marks=RunIf(mps=True))]
)
def test_swa_callback_1_gpu(tmp_path, accelerator):
    train_with_swa(tmp_path, accelerator=accelerator, devices=1)


@pytest.mark.parametrize("batchnorm", [True, False])
@pytest.mark.parametrize("iterable_dataset", [True, False])
def test_swa_callback(tmp_path, batchnorm: bool, iterable_dataset: bool):
    train_with_swa(tmp_path, batchnorm=batchnorm, iterable_dataset=iterable_dataset)


@pytest.mark.parametrize("interval", ["epoch", "step"])
def test_swa_callback_scheduler_step(tmp_path, interval: str):
    train_with_swa(tmp_path, interval=interval)


def test_swa_warns(tmp_path, caplog):
    model = SwaTestModel(interval="step")
    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=True, callbacks=StochasticWeightAveraging(swa_lrs=1e-2))
    with caplog.at_level(level=logging.INFO), pytest.warns(UserWarning, match="SWA is currently only supported"):
        trainer.fit(model)
    assert "Swapping scheduler `StepLR` for `SWALR`" in caplog.text


def test_swa_raises():
    with pytest.raises(MisconfigurationException, match=">0 integer or a float between 0 and 1"):
        StochasticWeightAveraging(swa_epoch_start=0, swa_lrs=0.1)
    with pytest.raises(MisconfigurationException, match=">0 integer or a float between 0 and 1"):
        StochasticWeightAveraging(swa_epoch_start=1.5, swa_lrs=0.1)
    with pytest.raises(MisconfigurationException, match=">0 integer or a float between 0 and 1"):
        StochasticWeightAveraging(swa_epoch_start=-1, swa_lrs=0.1)
    with pytest.raises(MisconfigurationException, match="positive float, or a list of positive floats"):
        StochasticWeightAveraging(swa_epoch_start=5, swa_lrs=[0.2, 1])


def test_swa_deepcopy(tmp_path):
    """Test to ensure SWA Callback doesn't deepcopy dataloaders and datamodule potentially leading to OOM."""

    class TestSWA(StochasticWeightAveraging):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.setup_called = False

        def setup(self, trainer, pl_module, stage) -> None:
            super().setup(trainer, pl_module, stage)
            assert self._average_model.train_dataloader is not pl_module.train_dataloader
            assert self._average_model.train_dataloader.__self__ == self._average_model
            assert self._average_model._trainer is None
            self.setup_called = True

    model = BoringModel()
    swa = TestSWA(swa_lrs=1e-2)
    trainer = Trainer(default_root_dir=tmp_path, callbacks=swa, fast_dev_run=True)
    trainer.fit(model, train_dataloaders=DataLoader(RandomDataset(32, 2)))
    assert swa.setup_called


def test_swa_multiple_lrs(tmp_path):
    swa_lrs = [0.123, 0.321]

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.layer1 = torch.nn.Linear(32, 32)
            self.layer2 = torch.nn.Linear(32, 2)
            self.on_train_epoch_start_called = False

        def forward(self, x):
            x = self.layer1(x)
            return self.layer2(x)

        def configure_optimizers(self):
            params = [{"params": self.layer1.parameters(), "lr": 0.1}, {"params": self.layer2.parameters(), "lr": 0.2}]
            return torch.optim.Adam(params)

        def on_train_epoch_start(self):
            optimizer = trainer.optimizers[0]
            assert [pg["lr"] for pg in optimizer.param_groups] == [0.1, 0.2]
            assert [pg["initial_lr"] for pg in optimizer.param_groups] == swa_lrs
            assert [pg["swa_lr"] for pg in optimizer.param_groups] == swa_lrs
            self.on_train_epoch_start_called = True

    model = TestModel()
    swa_callback = StochasticWeightAveraging(swa_lrs=swa_lrs)
    trainer = Trainer(
        default_root_dir=tmp_path,
        callbacks=swa_callback,
        fast_dev_run=1,
    )
    trainer.fit(model)
    assert model.on_train_epoch_start_called


def _swa_resume_training_from_checkpoint(tmp_path, model, resume_model, ddp=False):
    swa_start = 3
    trainer_kwargs = {
        "default_root_dir": tmp_path,
        "max_epochs": 5,
        "accelerator": "cpu",
        "strategy": "ddp_spawn" if ddp else "auto",
        "devices": 2 if ddp else 1,
        "limit_train_batches": 5,
        "limit_val_batches": 0,
        "accumulate_grad_batches": 2,
        "enable_progress_bar": False,
        "logger": False,
    }
    trainer = Trainer(callbacks=SwaTestCallback(swa_epoch_start=swa_start, swa_lrs=0.1), **trainer_kwargs)

    weights_only = False if _TORCH_GREATER_EQUAL_2_6 else None
    with _backward_patch(trainer), pytest.raises(Exception, match="SWA crash test"):
        trainer.fit(model, weights_only=weights_only)

    checkpoint_dir = Path(tmp_path) / "checkpoints"
    checkpoint_files = os.listdir(checkpoint_dir)
    assert len(checkpoint_files) == 1
    ckpt_path = str(checkpoint_dir / checkpoint_files[0])

    trainer = Trainer(callbacks=SwaTestCallback(swa_epoch_start=swa_start, swa_lrs=0.1), **trainer_kwargs)

    with _backward_patch(trainer):
        trainer.fit(resume_model, ckpt_path=ckpt_path, weights_only=weights_only)


class CustomSchedulerModel(SwaTestModel):
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)

        def lr_lambda(current_step: int):
            return 0.1

        scheduler = LambdaLR(optimizer, lr_lambda, -1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.interval,
            },
        }


@pytest.mark.parametrize("crash_on_epoch", [1, 3])
def test_swa_resume_training_from_checkpoint(tmp_path, crash_on_epoch):
    model = SwaTestModel(crash_on_epoch=crash_on_epoch)
    resume_model = SwaTestModel()
    _swa_resume_training_from_checkpoint(tmp_path, model, resume_model)


@pytest.mark.parametrize("crash_on_epoch", [1, 3])
def test_swa_resume_training_from_checkpoint_custom_scheduler(tmp_path, crash_on_epoch):
    # Reproduces the bug reported in https://github.com/Lightning-AI/pytorch-lightning/issues/11665
    model = CustomSchedulerModel(crash_on_epoch=crash_on_epoch)
    resume_model = CustomSchedulerModel()
    _swa_resume_training_from_checkpoint(tmp_path, model, resume_model)


@RunIf(skip_windows=True)
# flaky with "torch.multiprocessing.spawn.ProcessExitedException: process 0 terminated with signal SIGABRT" (GLOO)
@pytest.mark.flaky(reruns=3)
def test_swa_resume_training_from_checkpoint_ddp(tmp_path):
    model = SwaTestModel(crash_on_epoch=3)
    resume_model = SwaTestModel()
    _swa_resume_training_from_checkpoint(tmp_path, model, resume_model, ddp=True)


@pytest.mark.parametrize(
    "strategy",
    [
        pytest.param("deepspeed", marks=RunIf(deepspeed=True, min_cuda_gpus=1)),
        pytest.param("fsdp", marks=RunIf(min_cuda_gpus=1, skip_windows=True)),
    ],
)
def test_misconfiguration_error_with_sharded_model(tmp_path, strategy: str):
    model = SwaTestModel()
    swa_callback = SwaTestCallback(swa_epoch_start=2, swa_lrs=0.1)
    trainer = Trainer(
        default_root_dir=tmp_path,
        enable_progress_bar=False,
        max_epochs=5,
        callbacks=[swa_callback],
        strategy=strategy,
        accelerator="gpu",
        devices=1,
    )
    with pytest.raises(MisconfigurationException, match="SWA does not currently support sharded models"):
        trainer.fit(model)


def test_swa_with_infinite_epochs_and_batchnorm(tmp_path):
    """Test that SWA works correctly with max_epochs=-1 (infinite training) and BatchNorm."""
    model = SwaTestModel(batchnorm=True)
    swa_callback = StochasticWeightAveraging(swa_lrs=0.1, swa_epoch_start=2)

    trainer = Trainer(
        default_root_dir=tmp_path,
        enable_progress_bar=False,
        enable_model_summary=False,
        max_epochs=-1,
        max_steps=30,  # Use max_steps as stopping condition
        limit_train_batches=5,
        limit_val_batches=0,
        callbacks=[swa_callback],
        logger=False,
    )
    assert trainer.max_epochs == -1
    assert trainer.fit_loop.max_epochs == -1

    trainer.fit(model)
    assert trainer.current_epoch >= 5
    assert trainer.global_step == 30
    assert trainer.max_epochs == -1

    # Verify SWA was actually applied (update_parameters should have been called)
    # SWA starts at epoch 2, so with 6 epochs (0-5), we should have 4 updates (epochs 2, 3, 4, 5)
    assert swa_callback.n_averaged is not None
    assert swa_callback.n_averaged > 0, "SWA should have updated parameters"


def _backward_patch(trainer: Trainer) -> AbstractContextManager:
    return mock.patch.object(Strategy, "backward", wraps=trainer.strategy.backward)


def test_swa_update_parameters_checks_n_averaged_once():
    """Verify update_parameters evaluates `n_averaged == 0` once per call instead of once per parameter tensor."""

    class _CountingScalarTensor(torch.Tensor):
        bool_calls = 0

        def __bool__(self):
            type(self).bool_calls += 1
            return super().__bool__()

    avg_model = BoringModel()
    model = BoringModel()
    for p in avg_model.parameters():
        p.data.fill_(1.0)
    for p in model.parameters():
        p.data.fill_(3.0)

    num_params = sum(1 for _ in model.parameters())
    assert num_params > 1

    n_averaged = torch.tensor(0, dtype=torch.long).as_subclass(_CountingScalarTensor)
    _CountingScalarTensor.bool_calls = 0
    StochasticWeightAveraging.update_parameters(avg_model, model, n_averaged, StochasticWeightAveraging.avg_fn)
    assert _CountingScalarTensor.bool_calls == 1
    assert int(n_averaged) == 1
    for p in avg_model.parameters():
        assert torch.allclose(p, torch.full_like(p, 3.0))

    for p in model.parameters():
        p.data.fill_(5.0)
    _CountingScalarTensor.bool_calls = 0
    StochasticWeightAveraging.update_parameters(avg_model, model, n_averaged, StochasticWeightAveraging.avg_fn)
    assert _CountingScalarTensor.bool_calls == 1
    assert int(n_averaged) == 2
    for p in avg_model.parameters():
        assert torch.allclose(p, torch.full_like(p, 4.0))
