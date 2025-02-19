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
import contextlib
import json
import os
from re import escape
from typing import Any
from unittest import mock
from unittest.mock import ANY, Mock

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.accelerators import CUDAAccelerator
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset, RandomIterableDataset
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.plugins import DeepSpeedPrecision
from lightning.pytorch.strategies.deepspeed import DeepSpeedStrategy
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.imports import _TORCHMETRICS_GREATER_EQUAL_0_11 as _TM_GE_0_11
from tests_pytorch.helpers.datamodules import ClassifDataModule
from tests_pytorch.helpers.runif import RunIf


class ModelParallelBoringModel(BoringModel):
    def __init__(self):
        super().__init__()
        self.layer = None

    def configure_model(self) -> None:
        if self.layer is None:
            self.layer = torch.nn.Linear(32, 2)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.configure_model()


class ModelParallelBoringModelNoSchedulers(ModelParallelBoringModel):
    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)


class ModelParallelBoringModelManualOptim(BoringModel):
    def __init__(self):
        super().__init__()
        self.layer = None

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        loss = self.step(batch)
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

    def configure_model(self) -> None:
        if self.layer is None:
            self.layer = torch.nn.Linear(32, 2)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.configure_model()

    @property
    def automatic_optimization(self) -> bool:
        return False


@pytest.fixture
def deepspeed_config():
    return {
        "optimizer": {"type": "SGD", "params": {"lr": 3e-5}},
        "scheduler": {
            "type": "WarmupLR",
            "params": {"last_batch_iteration": -1, "warmup_min_lr": 0, "warmup_max_lr": 3e-5, "warmup_num_steps": 100},
        },
    }


@pytest.fixture
def deepspeed_zero_config(deepspeed_config):
    return {**deepspeed_config, "zero_allow_untested_optimizer": True, "zero_optimization": {"stage": 2}}


@RunIf(deepspeed=True)
@pytest.mark.parametrize("strategy", ["deepspeed", DeepSpeedStrategy])
def test_deepspeed_strategy_string(tmp_path, strategy):
    """Test to ensure that the strategy can be passed via string or instance, and parallel devices is correctly set."""

    trainer = Trainer(
        accelerator="cpu",
        fast_dev_run=True,
        default_root_dir=tmp_path,
        strategy=strategy if isinstance(strategy, str) else strategy(),
    )

    assert isinstance(trainer.strategy, DeepSpeedStrategy)
    assert trainer.strategy.parallel_devices == [torch.device("cpu")]


@RunIf(deepspeed=True)
def test_deepspeed_strategy_env(tmp_path, monkeypatch, deepspeed_config):
    """Test to ensure that the strategy can be passed via a string with an environment variable."""
    config_path = os.path.join(tmp_path, "temp.json")
    with open(config_path, "w") as f:
        f.write(json.dumps(deepspeed_config))
    monkeypatch.setenv("PL_DEEPSPEED_CONFIG_PATH", config_path)

    trainer = Trainer(accelerator="cpu", fast_dev_run=True, default_root_dir=tmp_path, strategy="deepspeed")

    strategy = trainer.strategy
    assert isinstance(strategy, DeepSpeedStrategy)
    assert strategy.parallel_devices == [torch.device("cpu")]
    assert strategy.config == deepspeed_config


@RunIf(deepspeed=True, mps=False)
def test_deepspeed_precision_choice(cuda_count_1, tmp_path):
    """Test to ensure precision plugin is also correctly chosen.

    DeepSpeed handles precision via Custom DeepSpeedPrecision

    """
    trainer = Trainer(
        fast_dev_run=True,
        default_root_dir=tmp_path,
        accelerator="gpu",
        strategy="deepspeed",
        precision="16-mixed",
    )

    assert isinstance(trainer.strategy, DeepSpeedStrategy)
    assert isinstance(trainer.strategy.precision_plugin, DeepSpeedPrecision)
    assert trainer.strategy.precision_plugin.precision == "16-mixed"


@RunIf(deepspeed=True)
def test_deepspeed_with_invalid_config_path():
    """Test to ensure if we pass an invalid config path we throw an exception."""
    with pytest.raises(
        MisconfigurationException, match="You passed in a path to a DeepSpeed config but the path does not exist"
    ):
        DeepSpeedStrategy(config="invalid_path.json")


@RunIf(deepspeed=True)
def test_deepspeed_with_env_path(tmp_path, monkeypatch, deepspeed_config):
    """Test to ensure if we pass an env variable, we load the config from the path."""
    config_path = os.path.join(tmp_path, "temp.json")
    with open(config_path, "w") as f:
        f.write(json.dumps(deepspeed_config))
    monkeypatch.setenv("PL_DEEPSPEED_CONFIG_PATH", config_path)
    strategy = DeepSpeedStrategy()
    assert strategy.config == deepspeed_config


@RunIf(deepspeed=True)
def test_deepspeed_defaults():
    """Ensure that defaults are correctly set as a config for DeepSpeed if no arguments are passed."""
    strategy = DeepSpeedStrategy()
    assert strategy.config is not None
    assert isinstance(strategy.config["zero_optimization"], dict)


@RunIf(min_cuda_gpus=1, standalone=True, deepspeed=True)
def test_warn_deepspeed_ignored(tmp_path):
    class TestModel(BoringModel):
        def backward(self, loss: Tensor, *args, **kwargs) -> None:
            return loss.backward()

    model = TestModel()
    trainer = Trainer(
        fast_dev_run=True,
        default_root_dir=tmp_path,
        strategy=DeepSpeedStrategy(),
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    with pytest.warns(UserWarning, match="will be ignored since DeepSpeed handles the backward"):
        trainer.fit(model)


@RunIf(min_cuda_gpus=1, deepspeed=True)
@pytest.mark.parametrize(
    ("dataset_cls", "value"),
    [(RandomDataset, "auto"), (RandomDataset, 10), (RandomIterableDataset, "auto"), (RandomIterableDataset, 10)],
)
@mock.patch("deepspeed.init_distributed", autospec=True)
@mock.patch("lightning.pytorch.Trainer.log_dir", new_callable=mock.PropertyMock, return_value="abc")
def test_deepspeed_auto_batch_size_config_select(_, __, tmp_path, dataset_cls, value):
    """Test to ensure that the batch size is correctly set as expected for deepspeed logging purposes."""

    class TestModel(BoringModel):
        def train_dataloader(self):
            return DataLoader(dataset_cls(32, 64))

        def configure_model(self) -> None:
            assert isinstance(self.trainer.strategy, DeepSpeedStrategy)
            config = self.trainer.strategy.config

            # int value overrides auto mode
            expected_value = value if isinstance(value, int) else 1
            if dataset_cls == RandomDataset:
                expected_value = self.train_dataloader().batch_size if value == "auto" else value

            assert config["train_micro_batch_size_per_gpu"] == expected_value
            raise SystemExit

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        fast_dev_run=True,
        accelerator="cuda",
        devices=1,
        strategy=DeepSpeedStrategy(logging_batch_size_per_gpu=value, zero_optimization=False),
    )
    with pytest.raises(SystemExit):
        trainer.fit(model)


@RunIf(min_cuda_gpus=1, standalone=True, deepspeed=True)
def test_deepspeed_run_configure_optimizers(tmp_path):
    """Test end to end that deepspeed works with defaults (without ZeRO as that requires compilation), whilst using
    configure_optimizers for optimizers and schedulers."""
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

    class TestCB(Callback):
        def on_train_start(self, trainer, pl_module) -> None:
            assert isinstance(trainer.optimizers[0], DeepSpeedZeroOptimizer)
            assert isinstance(trainer.optimizers[0].optimizer, torch.optim.SGD)
            assert isinstance(trainer.lr_scheduler_configs[0].scheduler, torch.optim.lr_scheduler.StepLR)
            # check that the lr_scheduler config was preserved
            assert trainer.lr_scheduler_configs[0].name == "Sean"

    class TestModel(BoringModel):
        def configure_optimizers(self):
            [optimizer], [scheduler] = super().configure_optimizers()
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "name": "Sean"}}

    model = TestModel()
    lr_monitor = LearningRateMonitor()
    trainer = Trainer(
        strategy=DeepSpeedStrategy(),  # disable ZeRO so our optimizers are not wrapped
        default_root_dir=tmp_path,
        accelerator="gpu",
        devices=1,
        fast_dev_run=True,
        precision="16-mixed",
        callbacks=[TestCB(), lr_monitor],
        logger=CSVLogger(tmp_path),
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model)

    assert lr_monitor.lrs == {"Sean": [0.1]}

    _assert_save_model_is_equal(model, tmp_path, trainer)


@RunIf(min_cuda_gpus=1, standalone=True, deepspeed=True)
def test_deepspeed_config(tmp_path, deepspeed_zero_config):
    """Test to ensure deepspeed works correctly when passed a DeepSpeed config object including optimizers/schedulers
    and saves the model weights to load correctly."""
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

    class TestCB(Callback):
        def on_train_start(self, trainer, pl_module) -> None:
            from deepspeed.runtime.lr_schedules import WarmupLR

            assert isinstance(trainer.optimizers[0], DeepSpeedZeroOptimizer)
            assert isinstance(trainer.optimizers[0].optimizer, torch.optim.SGD)
            assert isinstance(trainer.lr_scheduler_configs[0].scheduler, WarmupLR)
            assert trainer.lr_scheduler_configs[0].interval == "step"

    model = BoringModel()
    lr_monitor = LearningRateMonitor()
    trainer = Trainer(
        strategy=DeepSpeedStrategy(config=deepspeed_zero_config),
        default_root_dir=tmp_path,
        accelerator="gpu",
        devices=1,
        log_every_n_steps=1,
        limit_train_batches=4,
        limit_val_batches=4,
        limit_test_batches=4,
        max_epochs=2,
        precision="16-mixed",
        callbacks=[TestCB(), lr_monitor],
        logger=CSVLogger(tmp_path),
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    trainer.fit(model)
    trainer.test(model)
    assert list(lr_monitor.lrs) == ["lr-SGD"]
    assert len(set(lr_monitor.lrs["lr-SGD"])) == 8


@RunIf(min_cuda_gpus=1, standalone=True, deepspeed=True)
def test_deepspeed_custom_precision_params(tmp_path):
    """Ensure if we modify the FP16 parameters via the DeepSpeedStrategy, the deepspeed config contains these
    changes."""

    class TestCB(Callback):
        def on_train_start(self, trainer, pl_module) -> None:
            assert trainer.strategy.config["fp16"]["loss_scale"] == 10
            assert trainer.strategy.config["fp16"]["initial_scale_power"] == 11
            assert trainer.strategy.config["fp16"]["loss_scale_window"] == 12
            assert trainer.strategy.config["fp16"]["hysteresis"] == 13
            assert trainer.strategy.config["fp16"]["min_loss_scale"] == 14
            raise SystemExit()

    model = BoringModel()
    ds = DeepSpeedStrategy(
        loss_scale=10, initial_scale_power=11, loss_scale_window=12, hysteresis=13, min_loss_scale=14
    )
    trainer = Trainer(
        default_root_dir=tmp_path,
        strategy=ds,
        precision="16-mixed",
        accelerator="gpu",
        devices=1,
        callbacks=[TestCB()],
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    with pytest.raises(SystemExit):
        trainer.fit(model)


@RunIf(min_cuda_gpus=1, standalone=True, deepspeed=True)
@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_deepspeed_inference_precision_during_inference(precision, tmp_path):
    """Ensure if we modify the precision for deepspeed and execute inference-only, the deepspeed config contains these
    changes."""

    class TestCB(Callback):
        def on_validation_start(self, trainer, pl_module) -> None:
            assert trainer.strategy.config[precision]
            raise SystemExit()

    model = BoringModel()
    strategy = DeepSpeedStrategy(config={precision: {"enabled": True}})

    trainer = Trainer(
        default_root_dir=tmp_path,
        strategy=strategy,
        accelerator="cuda",
        devices=1,
        callbacks=[TestCB()],
        barebones=True,
    )
    with pytest.raises(SystemExit):
        trainer.validate(model)


@RunIf(deepspeed=True)
def test_deepspeed_custom_activation_checkpointing_params():
    """Ensure if we modify the activation checkpointing parameters, the deepspeed config contains these changes."""
    ds = DeepSpeedStrategy(
        partition_activations=True,
        cpu_checkpointing=True,
        contiguous_memory_optimization=True,
        synchronize_checkpoint_boundary=True,
    )
    checkpoint_config = ds.config["activation_checkpointing"]
    assert checkpoint_config["partition_activations"]
    assert checkpoint_config["cpu_checkpointing"]
    assert checkpoint_config["contiguous_memory_optimization"]
    assert checkpoint_config["synchronize_checkpoint_boundary"]


@RunIf(min_cuda_gpus=1, standalone=True, deepspeed=True)
def test_deepspeed_custom_activation_checkpointing_params_forwarded(tmp_path):
    """Ensure if we modify the activation checkpointing parameters, we pass these to deepspeed.checkpointing.configure
    correctly."""
    import deepspeed

    ds = DeepSpeedStrategy(
        partition_activations=True,
        cpu_checkpointing=True,
        contiguous_memory_optimization=True,
        synchronize_checkpoint_boundary=True,
    )

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        fast_dev_run=1,
        strategy=ds,
        precision="16-mixed",
        accelerator="gpu",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    with mock.patch(
        "deepspeed.checkpointing.configure", wraps=deepspeed.checkpointing.configure
    ) as deepspeed_checkpointing_configure:
        trainer.fit(model)

    deepspeed_checkpointing_configure.assert_called_with(
        mpu_=None, partition_activations=True, contiguous_checkpointing=True, checkpoint_in_cpu=True, profile=None
    )


@RunIf(min_cuda_gpus=1, deepspeed=True)
def test_deepspeed_assert_config_zero_offload_disabled(tmp_path, deepspeed_zero_config):
    """Ensure if we use a config and turn off offload_optimizer, that this is set to False within the config."""
    deepspeed_zero_config["zero_optimization"]["offload_optimizer"] = False

    class TestCallback(Callback):
        def setup(self, trainer, pl_module, stage=None) -> None:
            assert trainer.strategy.config["zero_optimization"]["offload_optimizer"] is False
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        enable_progress_bar=False,
        max_epochs=1,
        strategy=DeepSpeedStrategy(config=deepspeed_zero_config),
        precision="16-mixed",
        accelerator="gpu",
        devices=1,
        callbacks=[TestCallback()],
    )
    with pytest.raises(SystemExit):
        trainer.fit(model)


@RunIf(min_cuda_gpus=2, standalone=True, deepspeed=True)
def test_deepspeed_multigpu(tmp_path):
    """Test to ensure that DeepSpeed with multiple GPUs works and deepspeed distributed is initialized correctly."""
    import deepspeed

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        strategy=DeepSpeedStrategy(stage=3),
        accelerator="gpu",
        devices=2,
        fast_dev_run=True,
        precision="16-mixed",
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    with mock.patch.object(
        model, "configure_optimizers", wraps=model.configure_optimizers
    ) as mock_configure_optimizers:
        trainer.test(model)
    assert mock_configure_optimizers.call_count == 0

    with mock.patch("deepspeed.init_distributed", wraps=deepspeed.init_distributed) as mock_deepspeed_distributed:
        trainer.fit(model)
    mock_deepspeed_distributed.assert_called_once()

    _assert_save_model_is_equal(model, tmp_path, trainer)


@RunIf(min_cuda_gpus=1, standalone=True, deepspeed=True)
def test_deepspeed_fp32_works(tmp_path):
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator="gpu",
        devices=1,
        strategy="deepspeed_stage_3",
        fast_dev_run=True,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model)


@RunIf(min_cuda_gpus=2, standalone=True, deepspeed=True)
def test_deepspeed_stage_3_save_warning(tmp_path):
    """Test to ensure that DeepSpeed Stage 3 gives a warning when saving on rank zero."""
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        strategy=DeepSpeedStrategy(stage=3),
        accelerator="gpu",
        devices=2,
        fast_dev_run=True,
        precision="16-mixed",
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model)
    checkpoint_path = os.path.join(tmp_path, "model.pt")

    # both ranks need to call save checkpoint, however only rank 0 needs to check the warning
    context_manager = (
        pytest.warns(UserWarning, match="each worker will save a shard of the checkpoint within a directory.")
        if trainer.is_global_zero
        else contextlib.suppress()
    )
    with context_manager:
        trainer.save_checkpoint(checkpoint_path)


@RunIf(min_cuda_gpus=1, standalone=True, deepspeed=True)
def test_deepspeed_multigpu_single_file(tmp_path):
    """Test to ensure that DeepSpeed loads from a single file checkpoint."""
    model = BoringModel()
    checkpoint_path = os.path.join(tmp_path, "model.pt")
    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=True, accelerator="cpu", devices=1)
    trainer.fit(model)
    trainer.save_checkpoint(checkpoint_path)

    trainer = Trainer(
        default_root_dir=tmp_path,
        strategy=DeepSpeedStrategy(stage=3),
        accelerator="gpu",
        devices=1,
        fast_dev_run=True,
        precision="16-mixed",
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    strategy = trainer.strategy
    assert isinstance(strategy, DeepSpeedStrategy)
    assert not strategy.load_full_weights
    with pytest.raises(FileNotFoundError, match="The provided path is not a valid DeepSpeed checkpoint"):
        trainer.test(model, ckpt_path=checkpoint_path)

    trainer = Trainer(
        default_root_dir=tmp_path,
        strategy=DeepSpeedStrategy(stage=3, load_full_weights=True),
        accelerator="gpu",
        devices=1,
        fast_dev_run=True,
        precision="16-mixed",
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    strategy = trainer.strategy
    assert isinstance(strategy, DeepSpeedStrategy)
    assert strategy.load_full_weights
    trainer.test(model, ckpt_path=checkpoint_path)


class ModelParallelClassificationModel(LightningModule):
    def __init__(self, lr: float = 0.01, num_blocks: int = 5):
        super().__init__()
        self.lr = lr
        self.num_blocks = num_blocks
        self.prepare_data_per_node = True
        self.train_acc = self.valid_acc = self.test_acc = None
        self.model = None

    def make_block(self):
        return nn.Sequential(nn.Linear(32, 32, bias=False), nn.ReLU())

    def configure_model(self) -> None:
        # As of deepspeed v0.9.3, in ZeRO stage 3 all submodules need to be created within this hook,
        # including the metrics. Otherwise, modules that aren't affected by `deepspeed.zero.Init()`
        # won't be moved to the GPU. See https://github.com/microsoft/DeepSpeed/pull/3611
        if self.model is None:
            metric = Accuracy(task="multiclass", num_classes=3) if _TM_GE_0_11 else Accuracy()
            self.train_acc = metric.clone()
            self.valid_acc = metric.clone()
            self.test_acc = metric.clone()
            self.model = nn.Sequential(*(self.make_block() for x in range(self.num_blocks)), nn.Linear(32, 3))

    def forward(self, x):
        x = self.model(x)
        # Ensure output is in float32 for softmax operation
        x = x.float()
        return F.softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc(logits, y), prog_bar=True, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        self.log("val_loss", F.cross_entropy(logits, y), prog_bar=False, sync_dist=True)
        self.log("val_acc", self.valid_acc(logits, y), prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        self.log("test_loss", F.cross_entropy(logits, y), prog_bar=False, sync_dist=True)
        self.log("test_acc", self.test_acc(logits, y), prog_bar=True, sync_dist=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        logits = self.forward(x)
        self.test_acc(logits, y)
        return self.test_acc.compute()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if not hasattr(self, "model"):
            self.configure_model()

        # Lightning saves the lr schedulers, but DeepSpeed saves the optimizer states separately
        assert len(checkpoint["lr_schedulers"]) == 1
        assert "optimizer_states" not in checkpoint


class ManualModelParallelClassificationModel(ModelParallelClassificationModel):
    @property
    def automatic_optimization(self) -> bool:
        return False

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        opt = self.optimizers()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc(logits, y), prog_bar=True, sync_dist=True)
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()


@RunIf(min_cuda_gpus=2, standalone=True, deepspeed=True)
def test_deepspeed_multigpu_stage_3(tmp_path):
    """Test to ensure ZeRO Stage 3 works with a parallel model."""
    model = ModelParallelBoringModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        strategy=DeepSpeedStrategy(stage=3),
        accelerator="gpu",
        devices=2,
        fast_dev_run=True,
        precision="16-mixed",
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.test(model)
    trainer.fit(model)

    _assert_save_model_is_equal(model, tmp_path, trainer)


@RunIf(min_cuda_gpus=2, standalone=True, deepspeed=True)
def test_deepspeed_multigpu_stage_3_manual_optimization(tmp_path, deepspeed_config):
    """Test to ensure ZeRO Stage 3 works with a parallel model."""
    model = ModelParallelBoringModelManualOptim()
    trainer = Trainer(
        default_root_dir=tmp_path,
        strategy=DeepSpeedStrategy(stage=3),
        accelerator="gpu",
        devices=2,
        fast_dev_run=True,
        precision="16-mixed",
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.test(model)
    trainer.fit(model)

    _assert_save_model_is_equal(model, tmp_path, trainer)


@pytest.mark.xfail(strict=False, reason="skipped due to deepspeed/#2449, keep track @rohitgr7")
@pytest.mark.parametrize(("accumulate_grad_batches", "automatic_optimization"), [(1, False), (2, True)])
@RunIf(min_cuda_gpus=2, standalone=True, deepspeed=True, sklearn=True)
def test_deepspeed_multigpu_stage_3_checkpointing(tmp_path, automatic_optimization, accumulate_grad_batches):
    model = ModelParallelClassificationModel() if automatic_optimization else ManualModelParallelClassificationModel()
    dm = ClassifDataModule()
    ck = ModelCheckpoint(monitor="val_acc", mode="max", save_last=True, save_top_k=-1)
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=10,
        strategy=DeepSpeedStrategy(stage=3),
        accelerator="gpu",
        devices=2,
        precision="16-mixed",
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=[ck],
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule=dm)

    results = trainer.test(datamodule=dm)
    saved_results = trainer.test(ckpt_path=ck.best_model_path, datamodule=dm)
    assert saved_results == results

    model = ModelParallelClassificationModel() if automatic_optimization else ManualModelParallelClassificationModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator="gpu",
        devices=2,
        strategy=DeepSpeedStrategy(stage=3),
        precision="16-mixed",
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.test(model, datamodule=dm, ckpt_path=ck.best_model_path)


@RunIf(min_cuda_gpus=1, standalone=True, deepspeed=True, sklearn=True)
def test_deepspeed_multigpu_stage_3_warns_resume_training(tmp_path):
    """Test to ensure with Stage 3 and multiple GPUs that we can resume from training, throwing a warning that the
    optimizer state and scheduler states cannot be restored."""
    dm = ClassifDataModule()
    model = BoringModel()
    checkpoint_path = os.path.join(tmp_path, "model.pt")
    trainer = Trainer(
        default_root_dir=tmp_path,
        fast_dev_run=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
    )
    trainer.fit(model)
    trainer.save_checkpoint(checkpoint_path)

    trainer = Trainer(
        default_root_dir=tmp_path,
        fast_dev_run=True,
        strategy=DeepSpeedStrategy(stage=3, load_full_weights=True),
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    with pytest.warns(
        UserWarning,
        match="A single checkpoint file has been given. This means optimizer states cannot be restored. "
        "If you'd like to restore these states, you must "
        "provide a path to the originally saved DeepSpeed checkpoint.",
    ):
        trainer.fit(model, datamodule=dm, ckpt_path=checkpoint_path)


@RunIf(min_cuda_gpus=1, standalone=True, deepspeed=True, sklearn=True)
def test_deepspeed_multigpu_stage_3_resume_training(tmp_path):
    """Test to ensure with Stage 3 and single GPU that we can resume training."""
    initial_model = ModelParallelClassificationModel()
    dm = ClassifDataModule()

    ck = ModelCheckpoint(monitor="val_acc", mode="max", save_last=True, save_top_k=-1)
    initial_trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        strategy=DeepSpeedStrategy(stage=3),
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        callbacks=[ck],
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    initial_trainer.fit(initial_model, datamodule=dm)

    class TestCallback(Callback):
        def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
            original_deepspeed_strategy = initial_trainer.strategy
            current_deepspeed_strategy = trainer.strategy

            assert isinstance(original_deepspeed_strategy, DeepSpeedStrategy)
            assert isinstance(current_deepspeed_strategy, DeepSpeedStrategy)
            # assert optimizer states are the correctly loaded
            original_optimizer_dict = original_deepspeed_strategy.deepspeed_engine.optimizer.state_dict()
            current_optimizer_dict = current_deepspeed_strategy.deepspeed_engine.optimizer.state_dict()
            for orig_tensor, current_tensor in zip(
                original_optimizer_dict["fp32_flat_groups"], current_optimizer_dict["fp32_flat_groups"]
            ):
                assert torch.all(orig_tensor.eq(current_tensor))
            # assert model state is loaded correctly
            for current_param, initial_param in zip(pl_module.parameters(), initial_model.parameters()):
                assert torch.equal(current_param.cpu(), initial_param.cpu())
            # assert epoch has correctly been restored
            assert trainer.current_epoch == 1

            # assert lr-scheduler states are loaded correctly
            original_lr_scheduler = initial_trainer.lr_scheduler_configs[0].scheduler
            current_lr_scheduler = trainer.lr_scheduler_configs[0].scheduler
            assert original_lr_scheduler.state_dict() == current_lr_scheduler.state_dict()

    model = ModelParallelClassificationModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        strategy=DeepSpeedStrategy(stage=3),
        accelerator="gpu",
        devices=1,
        max_epochs=2,
        limit_train_batches=1,
        limit_val_batches=0,
        precision="16-mixed",
        callbacks=TestCallback(),
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule=dm, ckpt_path=ck.best_model_path)


@pytest.mark.parametrize("offload_optimizer", [False, True])
@RunIf(min_cuda_gpus=2, standalone=True, deepspeed=True, sklearn=True)
def test_deepspeed_multigpu_stage_2_accumulated_grad_batches(tmp_path, offload_optimizer):
    """Test to ensure with Stage 2 and multiple GPUs, accumulated grad batches works."""

    class VerificationCallback(Callback):
        def __init__(self):
            self.on_train_batch_start_called = False

        def on_train_batch_start(self, trainer, pl_module: LightningModule, batch: Any, batch_idx: int) -> None:
            deepspeed_engine = trainer.strategy.model
            assert trainer.global_step == deepspeed_engine.global_steps
            self.on_train_batch_start_called = True

    model = ModelParallelClassificationModel()
    dm = ClassifDataModule()
    verification_callback = VerificationCallback()
    strategy = DeepSpeedStrategy(stage=2, offload_optimizer=offload_optimizer)
    strategy.config["zero_force_ds_cpu_optimizer"] = False
    trainer = Trainer(
        default_root_dir=tmp_path,
        # TODO: this test fails with max_epochs >1 as there are leftover batches per epoch.
        # there's divergence in how Lightning handles the last batch of the epoch with how DeepSpeed does it.
        # we step the optimizers on the last batch but DeepSpeed keeps the accumulation for the next epoch
        max_epochs=1,
        strategy=strategy,
        accelerator="gpu",
        devices=2,
        limit_train_batches=5,
        limit_val_batches=2,
        precision="16-mixed",
        accumulate_grad_batches=2,
        callbacks=[verification_callback],
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    assert trainer.limit_train_batches % trainer.accumulate_grad_batches != 0, "leftover batches should be tested"
    trainer.fit(model, datamodule=dm)
    assert verification_callback.on_train_batch_start_called


@RunIf(min_cuda_gpus=2, standalone=True, deepspeed=True)
def test_deepspeed_multigpu_test(tmp_path):
    """Test to ensure we can use DeepSpeed with just test using ZeRO Stage 3."""
    model = ModelParallelBoringModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        strategy=DeepSpeedStrategy(stage=3),
        accelerator="gpu",
        devices=2,
        fast_dev_run=True,
        precision="16-mixed",
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.test(model)


# TODO(Sean): Once partial parameter partitioning is supported this test should be re-enabled
@pytest.mark.xfail(strict=False, reason="Partial parameter partitioning for DeepSpeed is currently broken.")
@RunIf(min_cuda_gpus=1, standalone=True, deepspeed=True)
def test_deepspeed_multigpu_partial_partition_parameters(tmp_path):
    """Test to ensure that a module that defines a layer inside the ``__init__`` and ``configure_model`` correctly
    converts all parameters to float16 when ``precision=16`` and runs successfully."""

    class TestModel(ModelParallelBoringModel):
        def __init__(self):
            super().__init__()
            self.layer_2 = torch.nn.Linear(32, 32)

        def configure_model(self) -> None:
            if self.layer is None:
                self.layer = torch.nn.Linear(32, 2)

        def forward(self, x):
            x = self.layer_2(x)
            return self.layer(x)

        def on_train_epoch_start(self) -> None:
            assert all(x.dtype == torch.float16 for x in self.parameters())

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        strategy=DeepSpeedStrategy(stage=3),
        accelerator="gpu",
        devices=1,
        fast_dev_run=True,
        precision="16-mixed",
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model)


@RunIf(min_cuda_gpus=1, standalone=True, deepspeed=True)
def test_deepspeed_multigpu_test_rnn(tmp_path):
    """Test to ensure that turning off explicit partitioning of the entire module for ZeRO Stage 3 works when training
    with certain layers which will crash with explicit partitioning."""

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.rnn = torch.nn.GRU(32, 32)

        def on_train_epoch_start(self) -> None:
            assert all(x.dtype == torch.float16 for x in self.parameters())

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        strategy=DeepSpeedStrategy(stage=3),
        accelerator="gpu",
        devices=1,
        fast_dev_run=True,
        precision="16-mixed",
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model)


@RunIf(deepspeed=True, mps=False)
@mock.patch("deepspeed.init_distributed", autospec=True)
@pytest.mark.parametrize("platform", ["Linux", "Windows"])
def test_deepspeed_strategy_env_variables(mock_deepspeed_distributed, tmp_path, platform):
    """Test to ensure that we setup distributed communication using correctly.

    When using windows, ranks environment variables should not be set, and deepspeed should handle this.

    """
    trainer = Trainer(default_root_dir=tmp_path, strategy=DeepSpeedStrategy(stage=3))
    strategy = trainer.strategy
    assert isinstance(strategy, DeepSpeedStrategy)
    with mock.patch("platform.system", return_value=platform) as mock_platform:
        strategy._init_deepspeed_distributed()
    mock_deepspeed_distributed.assert_called()
    mock_platform.assert_called()
    if platform == "Windows":
        # assert no env variables have been set within the DeepSpeedStrategy
        assert all(k not in os.environ for k in ("MASTER_PORT", "MASTER_ADDR", "RANK", "WORLD_SIZE", "LOCAL_RANK"))
    else:
        assert os.environ["MASTER_ADDR"] == str(trainer.strategy.cluster_environment.main_address)
        assert os.environ["MASTER_PORT"] == str(trainer.strategy.cluster_environment.main_port)
        assert os.environ["RANK"] == str(trainer.strategy.global_rank)
        assert os.environ["WORLD_SIZE"] == str(trainer.strategy.world_size)
        assert os.environ["LOCAL_RANK"] == str(trainer.strategy.local_rank)


def _assert_save_model_is_equal(model, tmp_path, trainer):
    from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

    checkpoint_path = os.path.join(tmp_path, "model.pt")
    checkpoint_path = trainer.strategy.broadcast(checkpoint_path)
    trainer.save_checkpoint(checkpoint_path)

    # carry out the check only on rank 0
    if trainer.is_global_zero:
        single_ckpt_path = os.path.join(tmp_path, "single_model.pt")
        convert_zero_checkpoint_to_fp32_state_dict(checkpoint_path, single_ckpt_path)
        state_dict = torch.load(single_ckpt_path, weights_only=False)

        model = model.cpu()
        # Assert model parameters are identical after loading
        for orig_param, saved_model_param in zip(model.parameters(), state_dict.values()):
            if model.dtype == torch.half:
                # moved model to float32 for comparison with single fp32 saved weights
                saved_model_param = saved_model_param.half()
            assert torch.equal(orig_param, saved_model_param)


@RunIf(min_cuda_gpus=2, standalone=True, deepspeed=True)
def test_deepspeed_multigpu_no_schedulers(tmp_path):
    """Test to ensure ZeRO Stage 3 works with a parallel model and no schedulers."""
    model = ModelParallelBoringModelNoSchedulers()
    trainer = Trainer(
        default_root_dir=tmp_path,
        strategy=DeepSpeedStrategy(stage=3),
        accelerator="gpu",
        devices=2,
        fast_dev_run=True,
        precision="16-mixed",
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model)

    _assert_save_model_is_equal(model, tmp_path, trainer)


@RunIf(min_cuda_gpus=1, standalone=True, deepspeed=True)
def test_deepspeed_skip_backward_raises(tmp_path):
    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            return None

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        strategy=DeepSpeedStrategy(),
        accelerator="gpu",
        devices=1,
        fast_dev_run=True,
        precision="16-mixed",
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    with pytest.raises(MisconfigurationException, match="returning `None` .* is not supported"):
        trainer.fit(model)


@mock.patch("torch.optim.lr_scheduler.StepLR.step", autospec=True)
@pytest.mark.parametrize("interval", ["step", "epoch"])
@pytest.mark.parametrize("max_epoch", [2])
@pytest.mark.parametrize("limit_train_batches", [2])
@RunIf(min_cuda_gpus=1, standalone=True, deepspeed=True)
def test_scheduler_step_count(mock_step, tmp_path, max_epoch, limit_train_batches, interval):
    """Test to ensure that the scheduler is called the correct amount of times during training when scheduler is set to
    step or epoch."""

    class TestModel(BoringModel):
        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": interval},
            }

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        limit_train_batches=limit_train_batches,
        limit_val_batches=0,
        max_epochs=max_epoch,
        accelerator="gpu",
        devices=1,
        strategy="deepspeed",
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model)
    if interval == "epoch":
        # assert called once at init and once during training
        assert mock_step.call_count == 1 + max_epoch
    else:
        # assert called once at init and once during training
        assert mock_step.call_count == 1 + (max_epoch * limit_train_batches)


@RunIf(min_cuda_gpus=1, standalone=True, deepspeed=True)
def test_deepspeed_configure_gradient_clipping(tmp_path):
    """Test to ensure that a warning is raised when `LightningModule.configure_gradient_clipping` is overridden in case
    of deepspeed."""

    class TestModel(BoringModel):
        def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
            self.clip_gradients(optimizer, gradient_clip_val, gradient_clip_algorithm)

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator="gpu",
        devices=1,
        strategy="deepspeed",
        fast_dev_run=True,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    with pytest.warns(UserWarning, match="handles gradient clipping internally"):
        trainer.fit(model)


@RunIf(min_cuda_gpus=1, standalone=True, deepspeed=True)
def test_deepspeed_gradient_clip_by_value(tmp_path):
    """Test to ensure that an exception is raised when using `gradient_clip_algorithm='value'`."""
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator="gpu",
        devices=1,
        strategy="deepspeed",
        gradient_clip_algorithm="value",
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    with pytest.raises(MisconfigurationException, match="does not support clipping gradients by value"):
        trainer.fit(model)


@RunIf(min_cuda_gpus=2, standalone=True, deepspeed=True)
def test_deepspeed_multi_save_same_filepath(tmp_path):
    """Test that verifies that deepspeed saves only latest checkpoint in the specified path and deletes the old sharded
    checkpoints."""

    class CustomModel(BoringModel):
        def training_step(self, *args, **kwargs):
            self.log("grank", self.global_rank)
            return super().training_step(*args, **kwargs)

    model = CustomModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        strategy="deepspeed",
        accelerator="gpu",
        devices=2,
        callbacks=[ModelCheckpoint(filename="{epoch}_{step}_{grank}", save_top_k=1)],
        limit_train_batches=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        max_epochs=2,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model)

    filepath = "epoch=1_step=2_grank=0.0.ckpt"
    expected = {filepath}
    assert expected == set(os.listdir(trainer.checkpoint_callback.dirpath))

    ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filepath)
    expected = {"latest", "zero_to_fp32.py", "checkpoint"}
    assert expected == set(os.listdir(ckpt_path))


@RunIf(min_cuda_gpus=2, standalone=True, deepspeed=True)
def test_deepspeed_with_bfloat16_precision(tmp_path):
    """Test that deepspeed works with bfloat16 precision."""
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        strategy="deepspeed_stage_3",
        accelerator="gpu",
        devices=2,
        fast_dev_run=True,
        precision="bf16-mixed",
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    trainer.fit(model)
    assert isinstance(trainer.strategy.precision_plugin, DeepSpeedPrecision)
    assert trainer.strategy.precision_plugin.precision == "bf16-mixed"
    assert trainer.strategy.config["zero_optimization"]["stage"] == 3
    assert trainer.strategy.config["bf16"]["enabled"]
    assert model.layer.weight.dtype == torch.bfloat16


@RunIf(deepspeed=True)
def test_error_with_invalid_accelerator(tmp_path):
    """Test DeepSpeedStrategy raises an exception if an invalid accelerator is used."""
    trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        strategy="deepspeed",
        fast_dev_run=True,
    )
    model = BoringModel()
    with pytest.raises(RuntimeError, match="DeepSpeed strategy is only supported on CUDA"):
        trainer.fit(model)


@RunIf(min_cuda_gpus=2, deepspeed=True, standalone=True)
def test_deepspeed_configure_optimizer_device_set(tmp_path):
    """Test to ensure that the LM has access to the device within the ``configure_optimizer`` function, and
    estimated_stepping_batches works correctly as a result."""

    class TestModel(BoringModel):
        def configure_optimizers(self):
            assert self.trainer.estimated_stepping_batches == 1
            assert self.device.type == "cuda"
            raise SystemExit

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        fast_dev_run=True,
        accelerator="gpu",
        devices=2,
        strategy=DeepSpeedStrategy(),
    )
    with pytest.raises(SystemExit):
        trainer.fit(model)


@RunIf(deepspeed=True)
@pytest.mark.parametrize("device_indices", [[1], [1, 0], [0, 2], [3, 2, 1]])
def test_validate_parallel_devices_indices(device_indices):
    """Test that the strategy validates that it doesn't support selecting specific devices by index.

    DeepSpeed doesn't support it and needs the index to match to the local rank of the process.

    """
    accelerator = Mock(spec=CUDAAccelerator)
    strategy = DeepSpeedStrategy(
        accelerator=accelerator, parallel_devices=[torch.device("cuda", i) for i in device_indices]
    )
    with pytest.raises(
        RuntimeError, match=escape(f"device indices {device_indices!r} don't match the local rank values of processes")
    ):
        strategy.setup_environment()
    accelerator.setup_device.assert_called_once_with(torch.device("cuda", device_indices[0]))


@RunIf(min_cuda_gpus=2, standalone=True, deepspeed=True, bf16_cuda=True)
def test_deepspeed_init_module_with_stage_3():
    """Tests how `.init_module()` behaves with ZeRO stage 3."""
    trainer = Trainer(
        accelerator="cuda", devices=2, strategy="deepspeed_stage_3", precision="bf16-mixed", fast_dev_run=1
    )
    model = ModelParallelBoringModel()
    with mock.patch("deepspeed.zero.Init") as zero_init_mock:
        trainer.fit(model)

    zero_init_mock.assert_called_once_with(enabled=True, remote_device=None, config_dict_or_path=ANY)


@RunIf(min_cuda_gpus=2, standalone=True, deepspeed=True, bf16_cuda=True)
@pytest.mark.parametrize("stage", [1, 2])
def test_deepspeed_init_module_with_stages_1_2(stage):
    """Tests how `.init_module()` behaves with ZeRO stages 1 and 2."""
    strategy = DeepSpeedStrategy(stage=stage)
    trainer = Trainer(accelerator="cuda", devices=2, strategy=strategy, precision="bf16-mixed", fast_dev_run=1)
    model = ModelParallelBoringModel()
    with mock.patch("deepspeed.zero.Init") as zero_init_mock:
        trainer.fit(model)

    zero_init_mock.assert_called_once_with(enabled=False, remote_device=None, config_dict_or_path=ANY)
    assert model.layer.weight.dtype == torch.bfloat16


@RunIf(deepspeed=True)
def test_deepspeed_load_checkpoint_validate_path(tmp_path):
    """Test that we validate the checkpoint path for a DeepSpeed checkpoint and give suggestions for user error."""
    strategy = DeepSpeedStrategy()
    with pytest.raises(FileNotFoundError, match="The provided path is not a valid DeepSpeed checkpoint"):
        strategy.load_checkpoint(checkpoint_path=tmp_path)

    # User tries to pass the subfolder as the path
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    with pytest.raises(FileNotFoundError, match=f"Try to load using this parent directory instead: {tmp_path}"):
        strategy.load_checkpoint(checkpoint_path=checkpoint_path)

    # User tries to pass an individual file inside the checkpoint folder
    checkpoint_path = checkpoint_path / "zero_pp_rank_0_mp_rank_00_model_states.pt"
    checkpoint_path.touch()
    with pytest.raises(FileNotFoundError, match=f"Try to load using this parent directory instead: {tmp_path}"):
        strategy.load_checkpoint(checkpoint_path=checkpoint_path)


@RunIf(min_cuda_gpus=2, standalone=True, deepspeed=True)
def test_deepspeed_multigpu_stage_3_MiCS_support(tmp_path):
    """Test to ensure we can use DeepSpeed with basic ZeRO Stage 3 MiCS Support."""
    model = ModelParallelBoringModel()
    strategy = DeepSpeedStrategy(stage=3)
    strategy.config["zero_optimization"]["stage"] = 3
    strategy.config["zero_optimization"]["mics_shard_size"] = 1
    strategy.config["zero_optimization"]["mics_hierarchical_params_gather"] = False

    trainer = Trainer(
        default_root_dir=tmp_path,
        strategy=strategy,
        accelerator="gpu",
        devices=2,
        fast_dev_run=True,
        precision="16-mixed",
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.test(model)
    trainer.fit(model)

    _assert_save_model_is_equal(model, tmp_path, trainer)
    assert isinstance(trainer.strategy, DeepSpeedStrategy)
    assert "zero_optimization" in trainer.strategy.config
    assert trainer.strategy.config["zero_optimization"]["mics_hierarchical_params_gather"] is False
    assert trainer.strategy.config["zero_optimization"]["mics_shard_size"] == 1
    assert trainer.strategy.config["zero_optimization"]["stage"] == 3


@RunIf(min_cuda_gpus=2, standalone=True, deepspeed=True)
def test_deepspeed_multigpu_stage_3_MiCS_offload_param_support(tmp_path):
    """Test to ensure we can use DeepSpeed with ZeRO Stage param offload 3 MiCS Support \
        However, in some past pratice, offload param + mics + torchrun will cause inner exception in multi-node environment. \
        Probably this exception is caused by torchrun, not deepspeed. """
    model = ModelParallelBoringModel()
    strategy = DeepSpeedStrategy(stage=3, offload_params_device="cpu")
    strategy.config["zero_optimization"]["stage"] = 3
    strategy.config["zero_optimization"]["mics_shard_size"] = 1
    strategy.config["zero_optimization"]["mics_hierarchical_params_gather"] = False
    trainer = Trainer(
        default_root_dir=tmp_path,
        strategy=strategy,
        accelerator="gpu",
        devices=2,
        fast_dev_run=True,
        precision="16-mixed",
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.test(model)
    trainer.fit(model)

    _assert_save_model_is_equal(model, tmp_path, trainer)
    assert isinstance(trainer.strategy, DeepSpeedStrategy)
    assert "zero_optimization" in trainer.strategy.config
    assert trainer.strategy.config["zero_optimization"]["mics_hierarchical_params_gather"] is False
    assert trainer.strategy.config["zero_optimization"]["mics_shard_size"] == 1
    assert trainer.strategy.config["zero_optimization"]["stage"] == 3


@RunIf(min_cuda_gpus=2, standalone=True, deepspeed=True)
def test_deepspeed_multigpu_stage_3_MiCS_offload_param_optimizer_support(tmp_path):
    """Test to ensure we can use DeepSpeed with ZeRO Stage param & optimizer offload 3 MiCS Support."""
    model = ModelParallelBoringModel()
    strategy = DeepSpeedStrategy(stage=3, offload_params_device="cpu", offload_optimizer_device="cpu")
    strategy.config["zero_optimization"]["stage"] = 3
    strategy.config["zero_optimization"]["mics_shard_size"] = 1
    strategy.config["zero_optimization"]["mics_hierarchical_params_gather"] = False
    trainer = Trainer(
        default_root_dir=tmp_path,
        strategy=strategy,
        accelerator="gpu",
        devices=2,
        fast_dev_run=True,
        precision="16-mixed",
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.test(model)
    trainer.fit(model)

    _assert_save_model_is_equal(model, tmp_path, trainer)
    assert isinstance(trainer.strategy, DeepSpeedStrategy)
    assert "zero_optimization" in trainer.strategy.config
    assert trainer.strategy.config["zero_optimization"]["mics_hierarchical_params_gather"] is False
    assert trainer.strategy.config["zero_optimization"]["mics_shard_size"] == 1
    assert trainer.strategy.config["zero_optimization"]["stage"] == 3


@RunIf(min_cuda_gpus=4, standalone=True, deepspeed=True)
def test_deepspeed_multigpu_stage_3_hierarchical_MiCS_support(tmp_path):
    """Test to ensure we can use DeepSpeed with ZeRO Stage 3 MiCS Support ('mics_hierarchical_params_gather' =
    True)."""
    model = ModelParallelBoringModel()
    strategy = DeepSpeedStrategy(stage=3)
    strategy.config["zero_optimization"]["stage"] = 3
    strategy.config["zero_optimization"]["mics_shard_size"] = 2
    strategy.config["zero_optimization"]["offload_param"] = {}
    strategy.config["zero_optimization"]["offload_optimizer"] = {}
    strategy.config["zero_optimization"]["mics_hierarchical_params_gather"] = True
    # Forming a 2 x 2 hierarchy
    trainer = Trainer(
        default_root_dir=tmp_path,
        strategy=strategy,
        accelerator="gpu",
        devices=4,
        fast_dev_run=True,
        precision="16-mixed",
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.test(model)
    trainer.fit(model)

    _assert_save_model_is_equal(model, tmp_path, trainer)
    assert isinstance(trainer.strategy, DeepSpeedStrategy)
    assert "zero_optimization" in trainer.strategy.config
    assert trainer.strategy.config["zero_optimization"]["mics_hierarchical_params_gather"] is True
    assert trainer.strategy.config["zero_optimization"]["mics_shard_size"] == 2
    assert trainer.strategy.config["zero_optimization"]["stage"] == 3
