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
import os
from pathlib import Path
from typing import Any, Optional

import pytest
import torch
from torch import Tensor, nn
from torch.optim.swa_utils import get_swa_avg_fn
from torch.utils.data import DataLoader

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import WeightAveraging
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset, RandomIterableDataset
from tests_pytorch.helpers.runif import RunIf


class WeightAveragingTestModel(BoringModel):
    def __init__(
        self, batch_norm: bool = True, iterable_dataset: bool = False, crash_on_epoch: Optional[int] = None
    ) -> None:
        super().__init__()
        layers = [nn.Linear(32, 32)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(32))
        layers += [nn.ReLU(), nn.Linear(32, 2)]
        self.layer = nn.Sequential(*layers)
        self.iterable_dataset = iterable_dataset
        self.crash_on_epoch = crash_on_epoch

    def training_step(self, batch: Tensor, batch_idx: int) -> None:
        if self.crash_on_epoch and self.trainer.current_epoch >= self.crash_on_epoch:
            raise Exception("CRASH TEST")
        return super().training_step(batch, batch_idx)

    def train_dataloader(self) -> None:
        dataset_class = RandomIterableDataset if self.iterable_dataset else RandomDataset
        return DataLoader(dataset_class(32, 32), batch_size=4)

    def configure_optimizers(self) -> None:
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)


class EMAAveragingFunction:
    """EMA averaging function.

    Functionally equivalent to the closure that ``get_ema_avg_fn()`` would return. This class is needed because we
    cannot use a closure with ddp_spawn. (``Popen(process_obj)`` would fail with
    ``Can't get local object 'get_ema_avg_fn.<locals>.ema_update'``).

    """

    def __init__(self, decay: float = 0.999) -> None:
        self.decay = decay

    @torch.no_grad()
    def __call__(self, ema_param: Tensor, current_param: Tensor, num_averaged: Tensor) -> Tensor:
        return self.decay * ema_param + (1 - self.decay) * current_param


class EMATestCallback(WeightAveraging):
    def __init__(self, devices: int = 1, **kwargs: Any) -> None:
        super().__init__(avg_fn=EMAAveragingFunction(), **kwargs)
        self.devices = devices
        self.swap_calls = 0
        self.copy_calls = 0
        # Record the first epoch, as if we are resuming from a checkpoint this may not be equal to 0.
        self.first_epoch: Optional[int] = None

    def _swap_models(self, *args: Any, **kwargs: Any):
        self.swap_calls += 1
        return super()._swap_models(*args, **kwargs)

    def _copy_average_to_current(self, *args: Any, **kwargs: Any):
        self.copy_calls += 1
        return super()._copy_average_to_current(*args, **kwargs)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_train_start(trainer, pl_module)
        assert self.swap_calls == 0
        assert self.copy_calls == 0

    def on_train_epoch_start(self, trainer: Trainer, *args: Any) -> None:
        super().on_train_epoch_start(trainer, *args)
        # Since the checkpoint loaded was saved `on_train_epoch_end`, the first `FitLoop` iteration will not update the
        # model and will just call the epoch-level hooks. For that reason, we check that we are not restarting before
        # choosing the first epoch.
        if self.first_epoch is None and not trainer.fit_loop.restarting:
            self.first_epoch = trainer.current_epoch

    def on_train_epoch_end(self, trainer: Trainer, *args: Any) -> None:
        super().on_train_epoch_end(trainer, *args)
        assert self._average_model.n_averaged == trainer.global_step
        assert self.swap_calls == (trainer.current_epoch + 1 - self.first_epoch) * 2
        assert self.copy_calls == 0

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_train_end(trainer, pl_module)
        # length=32, batch_size=4, accumulate_grad_batches=2
        # => Using one process we have 4 optimizer steps per epoch.
        # => Using two processes we have 2 optimizer steps per epoch.
        steps_per_epoch = 4 // self.devices
        assert self._average_model.n_averaged == trainer.max_epochs * steps_per_epoch
        assert self.swap_calls == (trainer.max_epochs - self.first_epoch) * 2
        assert self.copy_calls == 1


class SWATestCallback(WeightAveraging):
    def __init__(self, **kwargs: Any) -> None:
        avg_fn = get_swa_avg_fn()
        update_on_epoch = lambda x: x in (3, 5, 7)
        super().__init__(avg_fn=avg_fn, update_on_epoch=update_on_epoch, **kwargs)

        self.swap_calls = 0
        self.copy_calls = 0
        # Record the first epoch, as if we are resuming from a checkpoint this may not be equal to 0.
        self.first_epoch: Optional[int] = None

    def _swap_models(self, *args: Any, **kwargs: Any):
        self.swap_calls += 1
        return super()._swap_models(*args, **kwargs)

    def _copy_average_to_current(self, *args: Any, **kwargs: Any):
        self.copy_calls += 1
        return super()._copy_average_to_current(*args, **kwargs)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_train_start(trainer, pl_module)
        assert self.swap_calls == 0
        assert self.copy_calls == 0

    def on_train_epoch_start(self, trainer: Trainer, *args: Any) -> None:
        super().on_train_epoch_start(trainer, *args)
        # Since the checkpoint loaded was saved `on_train_epoch_end`, the first `FitLoop` iteration will not update the
        # model and will just call the epoch-level hooks. For that reason, we check that we are not restarting before
        # choosing the first epoch.
        if self.first_epoch is None and not trainer.fit_loop.restarting:
            self.first_epoch = trainer.current_epoch

    def on_train_epoch_end(self, trainer: Trainer, *args: Any) -> None:
        super().on_train_epoch_end(trainer, *args)
        if trainer.current_epoch < 3:
            assert self._average_model.n_averaged == 0
        elif trainer.current_epoch < 5:
            assert self._average_model.n_averaged == 1
        elif trainer.current_epoch < 7:
            assert self._average_model.n_averaged == 2
        else:
            assert self._average_model.n_averaged == 3
        assert self.swap_calls == (trainer.current_epoch + 1 - self.first_epoch) * 2
        assert self.copy_calls == 0

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_train_end(trainer, pl_module)
        assert self._average_model.n_averaged == 3
        assert self.swap_calls == (trainer.max_epochs - self.first_epoch) * 2
        assert self.copy_calls == 1


def test_weight_averaging_deepcopy(tmp_path):
    """Ensure that WeightAveraging callback doesn't deepcopy the data loaders or the data module and consume memory
    more than necessary."""

    class TestCallback(WeightAveraging):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.setup_called = False

        def setup(self, trainer, pl_module, stage) -> None:
            super().setup(trainer, pl_module, stage)
            assert self._average_model.module.train_dataloader is not pl_module.train_dataloader
            assert self._average_model.module.train_dataloader.__self__ == self._average_model.module
            assert self._average_model.module._trainer is None
            self.setup_called = True

    callback = TestCallback()
    trainer = Trainer(default_root_dir=tmp_path, callbacks=callback, fast_dev_run=True)
    trainer.fit(BoringModel(), train_dataloaders=DataLoader(RandomDataset(32, 2)))
    assert callback.setup_called


@pytest.mark.parametrize("batch_norm", [True, False])
@pytest.mark.parametrize("iterable_dataset", [True, False])
def test_ema(tmp_path, batch_norm: bool, iterable_dataset: bool):
    _train(tmp_path, EMATestCallback(), batch_norm=batch_norm, iterable_dataset=iterable_dataset)


@pytest.mark.parametrize(
    "accelerator", [pytest.param("gpu", marks=RunIf(min_cuda_gpus=1)), pytest.param("mps", marks=RunIf(mps=True))]
)
def test_ema_accelerator(tmp_path, accelerator):
    _train(tmp_path, EMATestCallback(), accelerator=accelerator, devices=1)


@RunIf(min_cuda_gpus=2, standalone=True)
def test_ema_ddp(tmp_path):
    _train(tmp_path, EMATestCallback(devices=2), strategy="ddp", accelerator="gpu", devices=2)


@RunIf(min_cuda_gpus=2)
def test_ema_ddp_spawn(tmp_path):
    _train(tmp_path, EMATestCallback(devices=2), strategy="ddp_spawn", accelerator="gpu", devices=2)


@RunIf(skip_windows=True)
def test_ema_ddp_spawn_cpu(tmp_path):
    _train(tmp_path, EMATestCallback(devices=2), strategy="ddp_spawn", accelerator="cpu", devices=2)


@pytest.mark.parametrize("crash_on_epoch", [1, 3])
def test_ema_resume(tmp_path, crash_on_epoch):
    _train_and_resume(tmp_path, crash_on_epoch=crash_on_epoch)


@RunIf(skip_windows=True)
def test_ema_resume_ddp(tmp_path):
    _train_and_resume(tmp_path, crash_on_epoch=3, use_ddp=True)


def test_swa(tmp_path):
    _train(tmp_path, SWATestCallback())


def _train(
    tmp_path: str,
    callback: WeightAveraging,
    batch_norm: bool = True,
    strategy: str = "auto",
    accelerator: str = "cpu",
    devices: int = 1,
    iterable_dataset: bool = False,
    checkpoint_path: Optional[str] = None,
    crash_on_epoch: Optional[int] = None,
) -> None:
    trainer = Trainer(
        default_root_dir=tmp_path,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        max_epochs=8,
        num_sanity_val_steps=0,
        callbacks=callback,
        accumulate_grad_batches=2,
        strategy=strategy,
        accelerator=accelerator,
        devices=devices,
    )
    model = WeightAveragingTestModel(
        batch_norm=batch_norm, iterable_dataset=iterable_dataset, crash_on_epoch=crash_on_epoch
    )

    if crash_on_epoch is None:
        trainer.fit(model, ckpt_path=checkpoint_path)
    else:
        with pytest.raises(Exception, match="CRASH TEST"):
            trainer.fit(model, ckpt_path=checkpoint_path)

    assert trainer.lightning_module == model


def _train_and_resume(tmp_path: str, crash_on_epoch: int, use_ddp: bool = False) -> None:
    strategy = "ddp_spawn" if use_ddp else "auto"
    devices = 2 if use_ddp else 1

    _train(
        tmp_path, EMATestCallback(devices=devices), strategy=strategy, devices=devices, crash_on_epoch=crash_on_epoch
    )

    checkpoint_dir = Path(tmp_path) / "checkpoints"
    checkpoint_names = os.listdir(checkpoint_dir)
    assert len(checkpoint_names) == 1
    checkpoint_path = str(checkpoint_dir / checkpoint_names[0])

    _train(
        tmp_path, EMATestCallback(devices=devices), strategy=strategy, devices=devices, checkpoint_path=checkpoint_path
    )
