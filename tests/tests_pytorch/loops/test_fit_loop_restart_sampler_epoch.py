from typing import Optional

import pytest
import torch
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset


class MockSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.epoch = 0
        self.observed_epochs = []

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        self.observed_epochs.append(self.epoch)
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class CustomBoringModel(BoringModel):
    def __init__(self):
        super().__init__()
        self.last_sampler = None

    def train_dataloader(self):
        dataset = RandomDataset(32, 64)
        sampler = MockSampler(dataset)
        self.last_sampler = sampler
        return DataLoader(dataset, sampler=sampler, batch_size=2)


def test_fit_loop_resume_sampler_epoch(tmp_path):
    """Test that the sampler epoch is restored before the iterator is instantiated on resume."""
    model = CustomBoringModel()

    # Train for 1 epoch and save checkpoint
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1, enable_progress_bar=False, logger=False)
    trainer.fit(model)

    ckpt_path = trainer.checkpoint_callback.best_model_path

    # Resume training for epoch 2
    model2 = CustomBoringModel()
    trainer2 = Trainer(default_root_dir=tmp_path, max_epochs=2, enable_progress_bar=False, logger=False)
    trainer2.fit(model2, ckpt_path=ckpt_path)

    # The dataloader is instantiated and __iter__ is called.
    # The first observed epoch for the resumed run should be 1.
    assert model2.last_sampler.observed_epochs[0] == 1


class _StopOnEpoch(Callback):
    def __init__(self, stop_epoch: int) -> None:
        self.stop_epoch = stop_epoch

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.current_epoch == self.stop_epoch:
            raise StopIteration


class _DeterministicDistributedModel(LightningModule):
    def __init__(self, num_samples: int = 7, num_workers: int = 0) -> None:
        super().__init__()
        self.num_workers = num_workers
        self.layer = torch.nn.Linear(1, 1)
        self.data = torch.arange(num_samples, dtype=torch.float32).reshape(-1, 1) / 10
        self.seen_batches = []

    def train_dataloader(self) -> DataLoader:
        sampler = DistributedSampler(self.data, num_replicas=1, rank=0, seed=41)
        return DataLoader(self.data, sampler=sampler, batch_size=2, num_workers=self.num_workers)

    def training_step(self, batch, batch_idx: int):
        self.seen_batches.append((self.current_epoch, batch_idx, batch.tolist()))
        return (self.layer(batch) - 1.0).square().mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)


@pytest.mark.parametrize("num_workers", [0, 1])
def test_fit_loop_resume_sampler_multi_worker_determinism(tmp_path, num_workers):
    """Verify that resuming from checkpoint with multi-worker prefetching produces the exact same batch sequence,
    model parameters, and optimizer momentum as uninterrupted training."""

    def run_training(dir_name: str, stop_epoch: Optional[int] = None, ckpt_path: Optional[str] = None):
        model = _DeterministicDistributedModel(num_samples=7, num_workers=num_workers)
        checkpoint_cb = ModelCheckpoint(
            dirpath=tmp_path / dir_name,
            save_last=True,
            save_on_train_epoch_end=True,
        )
        callbacks = [checkpoint_cb]
        if stop_epoch is not None:
            callbacks.append(_StopOnEpoch(stop_epoch))

        trainer = Trainer(
            default_root_dir=tmp_path / dir_name,
            max_epochs=4,
            callbacks=callbacks,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator="cpu",
            devices=1,
        )
        try:
            trainer.fit(model, ckpt_path=ckpt_path)
        except StopIteration:
            pass
        return model, trainer, checkpoint_cb

    # 1. Reference uninterrupted run (epochs 0 to 3)
    seed_everything(61)
    full_model, full_trainer, _ = run_training("full")

    # 2. Interrupted run stopped at epoch 2
    seed_everything(61)
    split_model, _, split_cb = run_training("split", stop_epoch=2)

    # 3. Resumed run from epoch 2 checkpoint
    resumed_model, resumed_trainer, _ = run_training("resumed", ckpt_path=split_cb.last_model_path)

    # Assert batch sequence equality across workers
    assert full_model.seen_batches == split_model.seen_batches + resumed_model.seen_batches

    # Assert exact model weights equality
    for name, param in full_model.state_dict().items():
        torch.testing.assert_close(param, resumed_model.state_dict()[name], rtol=1e-6, atol=1e-7)

    # Assert exact optimizer momentum state equality
    full_opt_state = full_trainer.optimizers[0].state_dict()["state"]
    resumed_opt_state = resumed_trainer.optimizers[0].state_dict()["state"]
    for param_id in full_opt_state:
        torch.testing.assert_close(
            full_opt_state[param_id]["momentum_buffer"],
            resumed_opt_state[param_id]["momentum_buffer"],
            rtol=1e-6,
            atol=1e-7,
        )
