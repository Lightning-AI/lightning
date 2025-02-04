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
from collections import OrderedDict

import pytest
import torch
from torch import nn
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader

from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_3
from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import BackboneFinetuning, BaseFinetuning, ModelCheckpoint
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset
from tests_pytorch.helpers.runif import RunIf


class TestBackboneFinetuningCallback(BackboneFinetuning):
    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        epoch = trainer.current_epoch
        if self.unfreeze_backbone_at_epoch <= epoch:
            optimizer = trainer.optimizers[0]
            current_lr = optimizer.param_groups[0]["lr"]
            backbone_lr = self.previous_backbone_lr
            if epoch < 6:
                assert backbone_lr <= current_lr
            else:
                assert backbone_lr == current_lr


def test_finetuning_callback(tmp_path):
    """Test finetuning callbacks works as expected."""
    seed_everything(42)

    class FinetuningBoringModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(nn.Linear(32, 32, bias=False), nn.BatchNorm1d(32), nn.ReLU())
            self.layer = torch.nn.Linear(32, 2)
            self.backbone.has_been_used = False

        def forward(self, x):
            self.backbone.has_been_used = True
            x = self.backbone(x)
            return self.layer(x)

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
            return [optimizer], [lr_scheduler]

        def train_dataloader(self):
            return DataLoader(RandomDataset(32, 64), batch_size=2)

    model = FinetuningBoringModel()
    callback = TestBackboneFinetuningCallback(unfreeze_backbone_at_epoch=3, verbose=False)

    trainer = Trainer(limit_train_batches=4, default_root_dir=tmp_path, callbacks=[callback], max_epochs=8)
    trainer.fit(model)

    assert model.backbone.has_been_used


class TestBackboneFinetuningWarningCallback(BackboneFinetuning):
    def finetune_function(self, pl_module, epoch: int, optimizer):
        """Called when the epoch begins."""
        if epoch == 0:
            self.unfreeze_and_add_param_group(
                pl_module.backbone, optimizer, 0.1, train_bn=self.train_bn, initial_denom_lr=self.initial_denom_lr
            )


def test_finetuning_callback_warning(tmp_path):
    """Test finetuning callbacks works as expected."""
    seed_everything(42)

    class FinetuningBoringModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(32, 2, bias=False)
            self.layer = None
            self.backbone.has_been_used = False

        def forward(self, x):
            self.backbone.has_been_used = True
            return self.backbone(x)

        def train_dataloader(self):
            return DataLoader(RandomDataset(32, 64), batch_size=2)

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

    chk = ModelCheckpoint(dirpath=tmp_path, save_last=True)

    model = FinetuningBoringModel()
    model.validation_step = None
    callback = TestBackboneFinetuningWarningCallback(unfreeze_backbone_at_epoch=3, verbose=False)

    with pytest.warns(UserWarning, match="Did you init your optimizer in"):
        trainer = Trainer(limit_train_batches=1, default_root_dir=tmp_path, callbacks=[callback, chk], max_epochs=2)
        trainer.fit(model)

    assert model.backbone.has_been_used
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=3)
    trainer.fit(model, ckpt_path=chk.last_model_path)


def test_freeze_unfreeze_function(tmp_path):
    """Test freeze properly sets requires_grad on the modules."""
    seed_everything(42)

    class FreezeModel(LightningModule):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(nn.Linear(32, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Linear(32, 2))

    model = FreezeModel()
    assert model.backbone[1].track_running_stats
    BaseFinetuning.freeze(model, train_bn=True)
    assert not model.backbone[0].weight.requires_grad
    assert model.backbone[1].weight.requires_grad
    assert model.backbone[1].track_running_stats
    assert not model.backbone[3].weight.requires_grad

    BaseFinetuning.freeze(model, train_bn=False)
    assert not model.backbone[0].weight.requires_grad
    assert not model.backbone[1].weight.requires_grad
    assert not model.backbone[1].track_running_stats
    assert not model.backbone[3].weight.requires_grad

    BaseFinetuning.make_trainable(model)
    assert model.backbone[0].weight.requires_grad
    assert model.backbone[1].weight.requires_grad
    assert model.backbone[1].track_running_stats
    assert model.backbone[3].weight.requires_grad

    BaseFinetuning.freeze(model.backbone[0], train_bn=False)
    assert not model.backbone[0].weight.requires_grad

    BaseFinetuning.freeze(([(model.backbone[1]), [model.backbone[3]]]), train_bn=True)
    assert model.backbone[1].weight.requires_grad
    assert model.backbone[1].track_running_stats
    assert not model.backbone[3].weight.requires_grad


def test_unfreeze_and_add_param_group_function(tmp_path):
    """Test unfreeze_and_add_param_group properly unfreeze parameters and add to the correct param_group."""
    seed_everything(42)

    class FreezeModel(LightningModule):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(32, 32, bias=False),
                nn.Linear(32, 32, bias=False),
                nn.Linear(32, 32, bias=False),
                nn.Linear(32, 32, bias=False),
                nn.Linear(32, 32, bias=False),
                nn.BatchNorm1d(32),
            )

    model = FreezeModel()
    optimizer = SGD(model.backbone[0].parameters(), lr=0.01)

    with pytest.warns(UserWarning, match="The provided params to be frozen already"):
        BaseFinetuning.unfreeze_and_add_param_group(model.backbone[0], optimizer=optimizer)
    assert optimizer.param_groups[0]["lr"] == 0.01

    model.backbone[1].weight.requires_grad = False
    BaseFinetuning.unfreeze_and_add_param_group(model.backbone[1], optimizer=optimizer)
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[1]["lr"] == 0.001
    assert torch.equal(optimizer.param_groups[1]["params"][0], model.backbone[1].weight)
    assert model.backbone[1].weight.requires_grad

    with pytest.warns(UserWarning, match="The provided params to be frozen already"):
        BaseFinetuning.unfreeze_and_add_param_group(model, optimizer=optimizer, lr=100, train_bn=False)
    assert len(optimizer.param_groups) == 3
    assert optimizer.param_groups[2]["lr"] == 100
    assert len(optimizer.param_groups[2]["params"]) == 3
    for group_idx, group in enumerate(optimizer.param_groups):
        if group_idx == 0:
            assert torch.equal(optimizer.param_groups[0]["params"][0], model.backbone[0].weight)
        if group_idx == 2:
            assert torch.equal(optimizer.param_groups[2]["params"][0], model.backbone[2].weight)
            assert torch.equal(optimizer.param_groups[2]["params"][1], model.backbone[3].weight)
            assert torch.equal(optimizer.param_groups[2]["params"][2], model.backbone[4].weight)


class OnEpochLayerFinetuning(BaseFinetuning):
    def freeze_before_training(self, pl_module: LightningModule):
        self.freeze(pl_module.layer)

    def finetune_function(self, pl_module: LightningModule, epoch: int, optimizer: Optimizer):
        self.unfreeze_and_add_param_group(pl_module.layer[epoch + 1], optimizer)


def test_base_finetuning_internal_optimizer_metadata(tmp_path):
    """Test the param_groups updates are properly saved within the internal state of the BaseFinetuning Callbacks."""

    seed_everything(42)

    class FreezeModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.layer = nn.Sequential(
                nn.Linear(32, 32, bias=False),
                nn.Linear(32, 32, bias=True),
                nn.Linear(32, 32, bias=False),
                nn.Linear(32, 32, bias=True),
                nn.Linear(32, 32, bias=False),
                nn.Linear(32, 2, bias=True),
            )

        def forward(self, x):
            return self.layer(x)

        def configure_optimizers(self):
            return torch.optim.SGD(self.layer[0].parameters(), lr=0.1)

    cb = OnEpochLayerFinetuning()
    chk = ModelCheckpoint(dirpath=tmp_path, save_last=True)
    model = FreezeModel()
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=5, limit_train_batches=1, callbacks=[cb, chk])
    trainer.fit(model)
    assert len(cb._internal_optimizer_metadata[0]) == 6
    assert cb._internal_optimizer_metadata[0][0]["params"] == ["layer.0.weight"]
    assert cb._internal_optimizer_metadata[0][1]["params"] == ["layer.1.weight", "layer.1.bias"]
    assert cb._internal_optimizer_metadata[0][2]["params"] == ["layer.2.weight"]
    assert cb._internal_optimizer_metadata[0][3]["params"] == ["layer.3.weight", "layer.3.bias"]
    assert cb._internal_optimizer_metadata[0][4]["params"] == ["layer.4.weight"]
    assert cb._internal_optimizer_metadata[0][5]["params"] == ["layer.5.weight", "layer.5.bias"]

    model = FreezeModel()
    cb = OnEpochLayerFinetuning()
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=10, callbacks=[cb])
    with pytest.raises(IndexError, match="index 6 is out of range"):
        trainer.fit(model, ckpt_path=chk.last_model_path)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3)
        self.act = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return self.bn(x)


class ConvBlockParam(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.module_dict = nn.ModuleDict({"conv": nn.Conv2d(in_channels, out_channels, 3), "act": nn.ReLU()})
        # add trivial test parameter to convblock to validate parent (non-leaf) module parameter handling
        self.parent_param = nn.Parameter(torch.zeros((1), dtype=torch.float))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.module_dict["conv"](x)
        x = self.module_dict["act"](x)
        return self.bn(x)


def test_complex_nested_model():
    """Test flattening, freezing, and thawing of models which contain parent (non-leaf) modules with parameters
    directly themselves rather than exclusively their submodules containing parameters."""

    model = nn.Sequential(
        OrderedDict([
            ("encoder", nn.Sequential(ConvBlockParam(3, 64), ConvBlock(64, 128))),
            ("decoder", ConvBlock(128, 10)),
        ])
    )

    # There are 10 leaf modules or parent modules w/ parameters in the test model
    assert len(BaseFinetuning.flatten_modules(model)) == 10

    BaseFinetuning.freeze(model.encoder, train_bn=True)
    assert not model.encoder[0].module_dict["conv"].weight.requires_grad  # Validate a leaf module parameter is frozen
    assert not model.encoder[0].parent_param.requires_grad  # Validate the parent module parameter is frozen
    assert model.encoder[0].bn.weight.requires_grad

    BaseFinetuning.make_trainable(model)
    encoder_params = list(BaseFinetuning.filter_params(model.encoder, train_bn=True))
    # The 9 parameters of the encoder are:
    # conv0.weight, conv0.bias, bn0.weight, bn0.bias, parent_param
    # conv1.weight, conv1.bias, bn1.weight, bn1.bias
    assert len(encoder_params) == 9


class TestCallbacksRestoreCallback(BaseFinetuning):
    def freeze_before_training(self, pl_module):
        self.freeze(pl_module.layer[:3])

    def finetune_function(self, pl_module, epoch, optimizer):
        if epoch >= 1:
            self.unfreeze_and_add_param_group(pl_module.layer[epoch - 1], optimizer)


class FinetuningBoringModel(BoringModel):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(32, 32), nn.Linear(32, 32), nn.Linear(32, 32), nn.Linear(32, 2))

    def configure_optimizers(self):
        parameters = filter(lambda x: x.requires_grad, self.parameters())
        return torch.optim.SGD(parameters, lr=0.1)


def test_callbacks_restore(tmp_path):
    """Test callbacks restore is called after optimizers have been re-created but before optimizer states reload."""
    chk = ModelCheckpoint(dirpath=tmp_path, save_last=True)

    model = FinetuningBoringModel()
    callback = TestCallbacksRestoreCallback()

    trainer_kwargs = {
        "default_root_dir": tmp_path,
        "limit_train_batches": 1,
        "limit_val_batches": 1,
        "callbacks": [callback, chk],
        "max_epochs": 2,
    }

    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model)

    # only 1 optimizer
    assert len(callback._internal_optimizer_metadata) == 1

    # only 2 param groups
    assert len(callback._internal_optimizer_metadata[0]) == 2

    # original parameters
    expected = {
        "lr": 0.1,
        "momentum": 0,
        "dampening": 0,
        "weight_decay": 0,
        "nesterov": False,
        "params": ["layer.3.weight", "layer.3.bias"],
        "maximize": False,
        "foreach": None,
        "differentiable": False,
    }
    if _TORCH_GREATER_EQUAL_2_3:
        expected["fused"] = None

    assert callback._internal_optimizer_metadata[0][0] == expected

    # new param group
    expected = {
        "lr": 0.01,
        "momentum": 0,
        "dampening": 0,
        "weight_decay": 0,
        "nesterov": False,
        "params": ["layer.0.weight", "layer.0.bias"],
        "maximize": False,
        "foreach": None,
        "differentiable": False,
    }
    if _TORCH_GREATER_EQUAL_2_3:
        expected["fused"] = None

    assert callback._internal_optimizer_metadata[0][1] == expected

    trainer_kwargs["max_epochs"] = 3

    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model, ckpt_path=chk.last_model_path)


class BackboneBoringModel(BoringModel):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(32, 2)
        self.backbone = nn.Linear(32, 32)

    def forward(self, x):
        return self.layer(self.backbone(x))


def test_callbacks_restore_backbone(tmp_path):
    """Test callbacks restore is called after optimizers have been re-created but before optimizer states reload."""

    ckpt = ModelCheckpoint(dirpath=tmp_path, save_last=True)
    trainer = Trainer(
        default_root_dir=tmp_path,
        limit_train_batches=1,
        limit_val_batches=1,
        max_epochs=2,
        enable_progress_bar=False,
        callbacks=[ckpt, BackboneFinetuning(unfreeze_backbone_at_epoch=1)],
    )
    trainer.fit(BackboneBoringModel())

    # initialize a trainer that continues the previous training
    trainer = Trainer(
        default_root_dir=tmp_path,
        limit_train_batches=1,
        limit_val_batches=1,
        max_epochs=3,
        enable_progress_bar=False,
        callbacks=BackboneFinetuning(unfreeze_backbone_at_epoch=1),
    )
    trainer.fit(BackboneBoringModel(), ckpt_path=ckpt.last_model_path)


@RunIf(deepspeed=True)
def test_unsupported_strategies(tmp_path):
    model = BackboneBoringModel()
    callback = BackboneFinetuning()
    trainer = Trainer(accelerator="cpu", strategy="deepspeed", callbacks=[callback])
    with pytest.raises(NotImplementedError, match="does not support running with the DeepSpeed strategy"):
        callback.setup(trainer, model, stage=None)


def test_finetuning_with_configure_model(tmp_path):
    """Test that BaseFinetuning works correctly with configure_model by ensuring freeze_before_training is called after
    configure_model but before training starts."""

    class TrackingFinetuningCallback(BaseFinetuning):
        def __init__(self):
            super().__init__()

        def freeze_before_training(self, pl_module):
            assert hasattr(pl_module, "backbone"), "backbone should be configured before freezing"
            self.freeze(pl_module.backbone)

        def finetune_function(self, pl_module, epoch, optimizer):
            pass

    class TestModel(LightningModule):
        def __init__(self):
            super().__init__()
            self.configure_model_called_count = 0

        def configure_model(self):
            self.backbone = nn.Linear(32, 32)
            self.classifier = nn.Linear(32, 2)
            self.configure_model_called_count += 1

        def forward(self, x):
            x = self.backbone(x)
            return self.classifier(x)

        def training_step(self, batch, batch_idx):
            return self.forward(batch).sum()

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

    model = TestModel()
    callback = TrackingFinetuningCallback()
    trainer = Trainer(
        default_root_dir=tmp_path,
        callbacks=[callback],
        max_epochs=1,
        limit_train_batches=1,
    )

    trainer.fit(model, torch.randn(10, 32))
    assert model.configure_model_called_count == 1
