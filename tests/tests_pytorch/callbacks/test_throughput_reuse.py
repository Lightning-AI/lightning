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
import warnings
from unittest.mock import Mock

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ThroughputMonitor


class CapabilityModel(LightningModule):
    def __init__(self, has_flops):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        if has_flops:
            self.flops_per_batch = 10

    def forward(self, batch):
        return self.weight * batch[0]

    def training_step(self, batch, batch_idx):
        return self(batch).square().mean()

    def validation_step(self, batch, batch_idx):
        return self(batch)

    test_step = validation_step

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def train_dataloader(self):
        return DataLoader(TensorDataset(torch.ones(4, 1)), batch_size=1)

    val_dataloader = train_dataloader
    test_dataloader = train_dataloader
    predict_dataloader = train_dataloader


def make_trainer(tmp_path, monitor, logger):
    return Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=logger,
        callbacks=[monitor],
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        limit_val_batches=0,
    )


def run_stage(trainer, model, stage):
    trainer.limit_val_batches = 1.0 if stage == "validate" else 0
    if stage == "fit":
        # A completed Trainer needs remaining work for a real second fit.
        trainer.fit_loop.max_epochs = trainer.current_epoch + 1
    getattr(trainer, stage)(model)


def check_metrics(logger, stage, has_flops):
    prefix = {"fit": "train", "validate": "validate", "test": "test", "predict": "predict"}[stage]
    metrics = [
        call.kwargs.get("metrics", call.args[0] if call.args else {}) for call in logger.log_metrics.call_args_list
    ]
    throughput = [value for value in metrics if f"{prefix}/device/samples_per_sec" in value]
    assert throughput
    assert all((f"{prefix}/device/flops_per_sec" in value) is has_flops for value in throughput)
    assert all(value[f"{prefix}/device/samples_per_sec"] > 0 for value in throughput)
    if has_flops:
        for value in throughput:
            assert value[f"{prefix}/device/flops_per_sec"] == pytest.approx(
                10 * value[f"{prefix}/device/samples_per_sec"]
            )


@pytest.mark.parametrize("stage", ["fit", "validate", "test", "predict"])
@pytest.mark.parametrize("flags", [(True, False), (False, True), (True, True), (False, False)])
@pytest.mark.parametrize("reuse_trainer", [False, True])
def test_capability_recomputed_each_invocation(tmp_path, stage, flags, reuse_trainer):
    monitor = ThroughputMonitor(batch_size_fn=lambda batch: len(batch[0]), window_size=2)
    trainer = None
    for index, flag in enumerate(flags):
        logger = Mock()
        logger.save_dir = tmp_path
        if trainer is None or not reuse_trainer:
            trainer = make_trainer(tmp_path / str(index), monitor, logger)
        else:
            trainer.logger = logger
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run_stage(trainer, CapabilityModel(flag), stage)
        flops_warnings = [w for w in caught if "need to define a `flops_per_batch`" in str(w.message)]
        assert len(flops_warnings) == int(not flag)
        assert monitor._module_has_flops is flag
        check_metrics(logger, stage, flag)


@pytest.mark.parametrize("stages", [("fit", "validate", "predict"), ("test", "fit", "validate")])
@pytest.mark.parametrize("same_model", [False, True])
def test_capability_changes_across_stages(tmp_path, stages, same_model):
    monitor = ThroughputMonitor(batch_size_fn=lambda batch: len(batch[0]), window_size=2)
    logger = Mock()
    logger.save_dir = tmp_path
    trainer = make_trainer(tmp_path, monitor, logger)
    model = CapabilityModel(True)
    for stage, flag in zip(stages, [True, False, True]):
        if not same_model:
            model = CapabilityModel(flag)
        elif flag:
            model.flops_per_batch = 10
        else:
            del model.flops_per_batch
        logger.reset_mock()
        run_stage(trainer, model, stage)
        assert monitor._module_has_flops is flag
        check_metrics(logger, stage, flag)


@pytest.mark.parametrize("has_flops", [False, True])
def test_fit_validation_share_capability_and_warn_once(tmp_path, has_flops):
    monitor = ThroughputMonitor(batch_size_fn=lambda batch: len(batch[0]), window_size=2)
    logger = Mock()
    logger.save_dir = tmp_path
    trainer = make_trainer(tmp_path, monitor, logger)
    trainer.limit_val_batches = 1.0
    trainer.num_sanity_val_steps = 2
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trainer.fit(CapabilityModel(has_flops))
    assert len([w for w in caught if "need to define a `flops_per_batch`" in str(w.message)]) == int(not has_flops)
    check_metrics(logger, "fit", has_flops)
    check_metrics(logger, "validate", has_flops)


def test_model_setup_can_add_flops_after_callback_setup(tmp_path):
    class SetupModel(CapabilityModel):
        def setup(self, stage):
            self.flops_per_batch = 10

    monitor = ThroughputMonitor(batch_size_fn=lambda batch: len(batch[0]), window_size=2)
    logger = Mock()
    logger.save_dir = tmp_path
    trainer = make_trainer(tmp_path, monitor, logger)
    trainer.fit(CapabilityModel(False))
    logger.reset_mock()
    run_stage(trainer, SetupModel(False), "fit")
    assert monitor._module_has_flops is True
    check_metrics(logger, "fit", True)
