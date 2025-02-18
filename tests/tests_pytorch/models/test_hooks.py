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
from functools import partial, update_wrapper
from inspect import getmembers, isfunction
from unittest import mock
from unittest.mock import ANY, PropertyMock

import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from lightning.pytorch import Callback, LightningDataModule, LightningModule, Trainer, __version__
from lightning.pytorch.demos.boring_classes import BoringDataModule, BoringModel, RandomDataset
from lightning.pytorch.utilities.model_helpers import is_overridden
from tests_pytorch.helpers.runif import RunIf


class HookedDataModule(BoringDataModule):
    def __init__(self, called):
        super().__init__()

        def call(hook, fn, *args, **kwargs):
            out = fn(*args, **kwargs)
            d = {"name": hook}
            if args:
                d["args"] = args
            if kwargs:
                d["kwargs"] = kwargs
            called.append(d)
            return out

        for h in get_members(LightningDataModule):
            attr = getattr(self, h)
            partial_h = partial(call, h, attr)
            update_wrapper(partial_h, attr)
            setattr(self, h, partial_h)

    # override so that it gets called
    def prepare_data(self): ...


@pytest.mark.parametrize("max_steps", [1, 2, 3])
def test_on_before_zero_grad_called(tmp_path, max_steps):
    class CurrentTestModel(BoringModel):
        on_before_zero_grad_called = 0

        def on_before_zero_grad(self, optimizer):
            self.on_before_zero_grad_called += 1

    model = CurrentTestModel()

    trainer = Trainer(devices=1, default_root_dir=tmp_path, max_steps=max_steps, max_epochs=2)
    assert model.on_before_zero_grad_called == 0
    trainer.fit(model)
    assert max_steps == model.on_before_zero_grad_called

    model.on_before_zero_grad_called = 0
    trainer.test(model)
    assert model.on_before_zero_grad_called == 0


def test_on_train_epoch_end_metrics_collection(tmp_path):
    """Test that progress bar metrics also get collected at the end of an epoch."""
    num_epochs = 3

    class CurrentModel(BoringModel):
        def training_step(self, *args, **kwargs):
            output = super().training_step(*args, **kwargs)
            self.log_dict({"step_metric": torch.tensor(-1), "shared_metric": 100}, logger=False, prog_bar=True)
            return output

        def on_train_epoch_end(self):
            epoch = self.current_epoch
            # both scalar tensors and Python numbers are accepted
            self.log_dict(
                {f"epoch_metric_{epoch}": torch.tensor(epoch), "shared_metric": 111}, logger=False, prog_bar=True
            )

    model = CurrentModel()
    trainer = Trainer(max_epochs=num_epochs, default_root_dir=tmp_path, overfit_batches=2)
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    metrics = trainer.progress_bar_callback.get_metrics(trainer, model)

    # metrics added in training step should be unchanged by epoch end method
    assert metrics["step_metric"] == -1
    # a metric shared in both methods gets overwritten by epoch_end
    assert metrics["shared_metric"] == 111
    # metrics are kept after each epoch
    for i in range(num_epochs):
        assert metrics[f"epoch_metric_{i}"] == i


@pytest.mark.parametrize(
    ("accelerator", "expected_device_str"),
    [
        pytest.param("gpu", "cuda:0", marks=RunIf(min_cuda_gpus=1)),
        pytest.param("mps", "mps:0", marks=RunIf(mps=True)),
    ],
)
@mock.patch(
    "lightning.pytorch.strategies.Strategy.lightning_module",
    new_callable=PropertyMock,
)
def test_apply_batch_transfer_handler(model_getter_mock, accelerator, expected_device_str):
    expected_device = torch.device(expected_device_str)

    class CustomBatch:
        def __init__(self, data):
            self.samples = data[0]
            self.targets = data[1]

    class CurrentTestModel(BoringModel):
        rank = 0
        transfer_batch_to_device_hook_rank = None
        on_after_batch_transfer_hook_rank = None

        def on_after_batch_transfer(self, batch, dataloader_idx):
            assert dataloader_idx == 0
            assert batch.samples.device == batch.targets.device == expected_device
            self.on_after_batch_transfer_hook_rank = self.rank
            self.rank += 1
            batch.targets *= 2
            return batch

        def transfer_batch_to_device(self, batch, device, dataloader_idx):
            assert dataloader_idx == 0
            self.transfer_batch_to_device_hook_rank = self.rank
            self.rank += 1
            batch.samples = batch.samples.to(device)
            batch.targets = batch.targets.to(device)
            return batch

    model = CurrentTestModel()
    batch = CustomBatch((torch.zeros(5, 32), torch.ones(5, 1, dtype=torch.long)))

    trainer = Trainer(accelerator=accelerator, devices=1)
    # running .fit() would require us to implement custom data loaders, we mock the model reference instead

    model_getter_mock.return_value = model
    batch_gpu = trainer.strategy.batch_to_device(batch, expected_device)

    assert model.transfer_batch_to_device_hook_rank == 0
    assert model.on_after_batch_transfer_hook_rank == 1
    assert batch_gpu.samples.device == batch_gpu.targets.device == expected_device
    assert torch.allclose(batch_gpu.samples.cpu(), torch.zeros(5, 32))
    assert torch.allclose(batch_gpu.targets.cpu(), torch.ones(5, 1, dtype=torch.long) * 2)


@RunIf(min_cuda_gpus=2, standalone=True)
def test_transfer_batch_hook_ddp(tmp_path):
    """Test custom data are properly moved to the right device using ddp."""

    class CustomBatch:
        def __init__(self, data):
            self.samples = data[0]

        def to(self, device, **kwargs):
            self.samples = self.samples.to(device, **kwargs)
            return self

    def collate_fn(batch):
        return CustomBatch(batch)

    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            assert batch.samples.device == self.device
            assert isinstance(batch_idx, int)
            # the actual training step is not needed for the assertions
            return super().training_step(torch.rand(1, 32, device=self.device), batch_idx)

        def train_dataloader(self):
            return torch.utils.data.DataLoader(RandomDataset(32, 64), collate_fn=collate_fn)

    model = TestModel()
    model.validation_step = None
    trainer = Trainer(
        default_root_dir=tmp_path,
        limit_train_batches=2,
        limit_val_batches=0,
        max_epochs=1,
        strategy="ddp",
        accelerator="gpu",
        devices=2,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model)


def get_members(cls):
    return {h for h, _ in getmembers(cls, predicate=isfunction) if not h.startswith("_")}


class HookedCallback(Callback):
    def __init__(self, called):
        def call(hook, fn, *args, **kwargs):
            out = fn(*args, **kwargs)
            d = {"name": f"Callback.{hook}"}
            if args:
                d["args"] = args
            if kwargs:
                d["kwargs"] = kwargs
            called.append(d)
            return out

        for h in get_members(Callback):
            attr = getattr(self, h)
            partial_h = partial(call, h, attr)
            update_wrapper(partial_h, attr)
            setattr(self, h, partial_h)

    def state_dict(*args, **kwargs):
        return {"foo": True}


class HookedModel(BoringModel):
    def __init__(self, called):
        super().__init__()
        pl_module_hooks = get_members(LightningModule)
        # remove non-hooks
        pl_module_hooks.difference_update({"optimizers", "log", "log_dict"})
        # remove most `nn.Module` hooks
        module_hooks = get_members(torch.nn.Module)
        module_hooks.difference_update({"forward", "zero_grad", "train"})
        pl_module_hooks.difference_update(module_hooks)

        def call(hook, fn, *args, **kwargs):
            out = fn(*args, **kwargs)
            d = {"name": hook}
            if args:
                d["args"] = args
            elif hook == "train":
                # DeepSpeed calls `train(mode)` but we do not. Standardize
                # https://github.com/microsoft/DeepSpeed/pull/571
                d["args"] = (True,)
            if kwargs:
                d["kwargs"] = kwargs
            called.append(d)
            return out

        for h in pl_module_hooks:
            attr = getattr(self, h)
            partial_h = partial(call, h, attr)
            update_wrapper(partial_h, attr)
            setattr(self, h, partial_h)

    def _train_batch(self, *args, **kwargs):
        if self.automatic_optimization:
            return self._auto_train_batch(*args, **kwargs)
        return self._manual_train_batch(*args, **kwargs)

    @staticmethod
    def _auto_train_batch(trainer, model, batches, device, current_epoch=0, current_batch=0, **kwargs):
        using_deepspeed = kwargs.get("strategy") == "deepspeed"
        out = []
        for i in range(current_batch, batches):
            out.extend([
                {"name": "on_before_batch_transfer", "args": (ANY, 0)},
                {"name": "transfer_batch_to_device", "args": (ANY, device, 0)},
                {"name": "on_after_batch_transfer", "args": (ANY, 0)},
                {"name": "Callback.on_train_batch_start", "args": (trainer, model, ANY, i)},
                {"name": "on_train_batch_start", "args": (ANY, i)},
                {"name": "forward", "args": (ANY,)},
                {"name": "training_step", "args": (ANY, i)},
                {"name": "Callback.on_before_zero_grad", "args": (trainer, model, ANY)},
                {"name": "on_before_zero_grad", "args": (ANY,)},
                {"name": "optimizer_zero_grad", "args": (current_epoch, i, ANY)},
                {"name": "Callback.on_before_backward", "args": (trainer, model, ANY)},
                {"name": "on_before_backward", "args": (ANY,)},
                # DeepSpeed handles backward internally
                *([{"name": "backward", "args": (ANY,)}] if not using_deepspeed else []),
                {"name": "Callback.on_after_backward", "args": (trainer, model)},
                {"name": "on_after_backward"},
                # note: unscaling happens here in the case of AMP
                {"name": "Callback.on_before_optimizer_step", "args": (trainer, model, ANY)},
                {"name": "on_before_optimizer_step", "args": (ANY,)},
                {
                    "name": "clip_gradients",
                    "args": (ANY,),
                    "kwargs": {"gradient_clip_val": None, "gradient_clip_algorithm": None},
                },
                {
                    "name": "configure_gradient_clipping",
                    "args": (ANY,),
                    "kwargs": {"gradient_clip_val": None, "gradient_clip_algorithm": None},
                },
                # this is after because it refers to the `LightningModule.optimizer_step` hook which encapsulates
                # the actual call to `Precision.optimizer_step`
                {
                    "name": "optimizer_step",
                    "args": (current_epoch, i, ANY, ANY),
                },
                *([{"name": "lr_scheduler_step", "args": ANY}] if i == (trainer.num_training_batches - 1) else []),
                {"name": "Callback.on_train_batch_end", "args": (trainer, model, {"loss": ANY}, ANY, i)},
                {"name": "on_train_batch_end", "args": ({"loss": ANY}, ANY, i)},
            ])
        return out

    @staticmethod
    def _manual_train_batch(trainer, model, batches, device, **kwargs):
        using_deepspeed = kwargs.get("strategy") == "deepspeed"
        out = []
        for i in range(batches):
            out.extend([
                {"name": "on_before_batch_transfer", "args": (ANY, 0)},
                {"name": "transfer_batch_to_device", "args": (ANY, device, 0)},
                {"name": "on_after_batch_transfer", "args": (ANY, 0)},
                {"name": "Callback.on_train_batch_start", "args": (trainer, model, ANY, i)},
                {"name": "on_train_batch_start", "args": (ANY, i)},
                {"name": "forward", "args": (ANY,)},
                {"name": "Callback.on_before_backward", "args": (trainer, model, ANY)},
                {"name": "on_before_backward", "args": (ANY,)},
                # DeepSpeed handles backward internally
                *([{"name": "backward", "args": (ANY,)}] if not using_deepspeed else []),
                {"name": "Callback.on_after_backward", "args": (trainer, model)},
                {"name": "on_after_backward"},
                # `manual_backward` calls the previous 3
                {"name": "manual_backward", "args": (ANY,)},
                {"name": "closure"},
                {"name": "Callback.on_before_optimizer_step", "args": (trainer, model, ANY)},
                {"name": "on_before_optimizer_step", "args": (ANY,)},
                {"name": "training_step", "args": (ANY, i)},
                {"name": "Callback.on_train_batch_end", "args": (trainer, model, {"loss": ANY}, ANY, i)},
                {"name": "on_train_batch_end", "args": ({"loss": ANY}, ANY, i)},
            ])
        return out

    @staticmethod
    def _eval_epoch(fn, trainer, model, batches, key, device):
        return [
            {"name": f"Callback.on_{fn}_epoch_start", "args": (trainer, model)},
            {"name": f"on_{fn}_epoch_start"},
            *HookedModel._eval_batch(fn, trainer, model, batches, key, device=device),
            {"name": f"Callback.on_{fn}_epoch_end", "args": (trainer, model)},
            {"name": f"on_{fn}_epoch_end"},
        ]

    @staticmethod
    def _eval_batch(fn, trainer, model, batches, key, device):
        out = []
        outputs = {key: ANY}
        for i in range(batches):
            out.extend([
                {"name": "on_before_batch_transfer", "args": (ANY, 0)},
                {"name": "transfer_batch_to_device", "args": (ANY, device, 0)},
                {"name": "on_after_batch_transfer", "args": (ANY, 0)},
                {"name": f"Callback.on_{fn}_batch_start", "args": (trainer, model, ANY, i)},
                {"name": f"on_{fn}_batch_start", "args": (ANY, i)},
                {"name": "forward", "args": (ANY,)},
                {"name": f"{fn}_step", "args": (ANY, i)},
                {"name": f"Callback.on_{fn}_batch_end", "args": (trainer, model, outputs, ANY, i)},
                {"name": f"on_{fn}_batch_end", "args": (outputs, ANY, i)},
            ])
        return out

    @staticmethod
    def _predict_batch(trainer, model, batches, device):
        out = []
        for i in range(batches):
            out.extend([
                {"name": "on_before_batch_transfer", "args": (ANY, 0)},
                {"name": "transfer_batch_to_device", "args": (ANY, device, 0)},
                {"name": "on_after_batch_transfer", "args": (ANY, 0)},
                {"name": "Callback.on_predict_batch_start", "args": (trainer, model, ANY, i)},
                {"name": "on_predict_batch_start", "args": (ANY, i)},
                {"name": "forward", "args": (ANY,)},
                {"name": "predict_step", "args": (ANY, i)},
                {"name": "Callback.on_predict_batch_end", "args": (trainer, model, ANY, ANY, i)},
                {"name": "on_predict_batch_end", "args": (ANY, ANY, i)},
            ])
        return out

    # override so that it gets called
    def configure_model(self): ...

    # override so that it gets called
    def on_validation_model_train(self): ...

    # override so that it gets called
    def on_test_model_train(self): ...

    # override so that it gets called
    def on_predict_model_train(self): ...

    # override so that it gets called
    def prepare_data(self): ...


@pytest.mark.parametrize(
    "kwargs",
    [
        {"devices": 1},
        # these precision plugins modify the optimization flow, so testing them explicitly
        pytest.param({"accelerator": "gpu", "devices": 1, "precision": "16-mixed"}, marks=RunIf(min_cuda_gpus=1)),
        pytest.param(
            {"accelerator": "gpu", "devices": 1, "precision": "16-mixed", "strategy": "deepspeed"},
            marks=RunIf(min_cuda_gpus=1, standalone=True, deepspeed=True),
        ),
    ],
)
@pytest.mark.parametrize("automatic_optimization", [True, False])
@pytest.mark.parametrize("override_on_validation_model_train", [True, False])
def test_trainer_model_hook_system_fit(override_on_validation_model_train, automatic_optimization, kwargs, tmp_path):
    called = []

    class TestModel(HookedModel):
        def __init__(self, *args):
            super().__init__(*args)
            self.automatic_optimization = automatic_optimization

        def training_step(self, batch, batch_idx):
            if self.automatic_optimization:
                return super().training_step(batch, batch_idx)
            loss = self.step(batch[0])
            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step(lambda: called.append({"name": "closure"}))
            return {"loss": loss}

    model = TestModel(called)

    if not override_on_validation_model_train:
        model.on_validation_model_train = None
    assert is_overridden("on_validation_model_train", model) == override_on_validation_model_train

    callback = HookedCallback(called)
    train_batches = 2
    val_batches = 2
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        limit_train_batches=train_batches,
        limit_val_batches=val_batches,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[callback],
        **kwargs,
    )
    trainer.fit(model)
    saved_ckpt = {
        "callbacks": ANY,
        "epoch": 0,
        "global_step": train_batches,
        "lr_schedulers": ANY,
        "optimizer_states": ANY,
        "pytorch-lightning_version": __version__,
        "state_dict": ANY,
        "loops": ANY,
    }
    using_deepspeed = kwargs.get("strategy") == "deepspeed"
    if kwargs.get("precision") == "16-mixed" and not using_deepspeed:
        saved_ckpt[trainer.precision_plugin.__class__.__qualname__] = ANY
    device = trainer.strategy.root_device
    expected = [
        {"name": "configure_callbacks"},
        {"name": "prepare_data"},
        {"name": "Callback.setup", "args": (trainer, model), "kwargs": {"stage": "fit"}},
        {"name": "setup", "kwargs": {"stage": "fit"}},
        # DeepSpeed needs the batch size to figure out throughput logging
        *([{"name": "train_dataloader"}] if using_deepspeed else []),
        {"name": "configure_model"},
        {"name": "configure_optimizers"},
        {"name": "Callback.on_fit_start", "args": (trainer, model)},
        {"name": "on_fit_start"},
        {"name": "zero_grad"},
        {"name": "Callback.on_sanity_check_start", "args": (trainer, model)},
        {"name": "val_dataloader"},
        {"name": "train", "args": (False,)},
        {"name": "on_validation_model_eval"},
        {"name": "Callback.on_validation_start", "args": (trainer, model)},
        {"name": "on_validation_start"},
        *model._eval_epoch("validation", trainer, model, val_batches, "x", device=device),
        {"name": "Callback.on_validation_end", "args": (trainer, model)},
        {"name": "on_validation_end"},
        *([{"name": "on_validation_model_train"}] if override_on_validation_model_train else []),
        {"name": "Callback.on_sanity_check_end", "args": (trainer, model)},
        {"name": "train_dataloader"},
        {"name": "Callback.on_train_start", "args": (trainer, model)},
        {"name": "on_train_start"},
        {"name": "Callback.on_train_epoch_start", "args": (trainer, model)},
        {"name": "on_train_epoch_start"},
        *model._train_batch(trainer, model, train_batches, device=device, **kwargs),
        {"name": "zero_grad"},
        {"name": "on_validation_model_zero_grad"},
        {"name": "train", "args": (False,)},
        {"name": "on_validation_model_eval"},
        {"name": "Callback.on_validation_start", "args": (trainer, model)},
        {"name": "on_validation_start"},
        *model._eval_epoch("validation", trainer, model, val_batches, "x", device=device),
        {"name": "Callback.on_validation_end", "args": (trainer, model)},
        {"name": "on_validation_end"},
        *([{"name": "on_validation_model_train"}] if override_on_validation_model_train else []),
        {"name": "Callback.on_train_epoch_end", "args": (trainer, model)},
        {"name": "on_train_epoch_end"},  # before ModelCheckpoint because it's a "monitoring callback"
        # `ModelCheckpoint.save_checkpoint` is called here
        {"name": "Callback.state_dict"},
        {"name": "Callback.on_save_checkpoint", "args": (trainer, model, saved_ckpt)},
        {"name": "on_save_checkpoint", "args": (saved_ckpt,)},
        {"name": "Callback.on_train_end", "args": (trainer, model)},
        {"name": "on_train_end"},
        {"name": "Callback.on_fit_end", "args": (trainer, model)},
        {"name": "on_fit_end"},
        {"name": "Callback.teardown", "args": (trainer, model), "kwargs": {"stage": "fit"}},
        {"name": "teardown", "kwargs": {"stage": "fit"}},
    ]
    assert called == expected


def test_trainer_model_hook_system_fit_no_val_and_resume_max_epochs(tmp_path):
    # initial training to get a checkpoint
    model = BoringModel()
    trainer = Trainer(
        devices=1,
        default_root_dir=tmp_path,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=0,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[HookedCallback([])],
    )
    trainer.fit(model)
    best_model_path = trainer.checkpoint_callback.best_model_path

    called = []
    callback = HookedCallback(called)
    # already performed 1 step, resume and do 2 more
    trainer = Trainer(
        devices=1,
        default_root_dir=tmp_path,
        max_epochs=2,
        limit_train_batches=2,
        limit_val_batches=0,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[callback],
    )

    # resume from checkpoint with HookedModel
    model = HookedModel(called)
    trainer.fit(model, ckpt_path=best_model_path)
    loaded_ckpt = {
        "callbacks": ANY,
        "epoch": 0,
        "global_step": 2,
        "lr_schedulers": ANY,
        "optimizer_states": ANY,
        "pytorch-lightning_version": __version__,
        "state_dict": ANY,
        "loops": ANY,
    }
    saved_ckpt = {**loaded_ckpt, "global_step": 4, "epoch": 1}
    expected = [
        {"name": "configure_callbacks"},
        {"name": "prepare_data"},
        {"name": "Callback.setup", "args": (trainer, model), "kwargs": {"stage": "fit"}},
        {"name": "setup", "kwargs": {"stage": "fit"}},
        {"name": "configure_model"},
        {"name": "on_load_checkpoint", "args": (loaded_ckpt,)},
        {"name": "Callback.on_load_checkpoint", "args": (trainer, model, loaded_ckpt)},
        {"name": "Callback.load_state_dict", "args": ({"foo": True},)},
        {"name": "configure_optimizers"},
        {"name": "Callback.on_fit_start", "args": (trainer, model)},
        {"name": "on_fit_start"},
        {"name": "zero_grad"},
        {"name": "train_dataloader"},
        {"name": "Callback.on_train_start", "args": (trainer, model)},
        {"name": "on_train_start"},
        {"name": "Callback.on_train_epoch_start", "args": (trainer, model)},
        {"name": "on_train_epoch_start"},
        *model._train_batch(trainer, model, 2, trainer.strategy.root_device, current_epoch=1, current_batch=0),
        {"name": "Callback.on_train_epoch_end", "args": (trainer, model)},
        {"name": "on_train_epoch_end"},  # before ModelCheckpoint because it's a "monitoring callback"
        # `ModelCheckpoint.save_checkpoint` is called here
        {"name": "Callback.state_dict"},
        {"name": "Callback.on_save_checkpoint", "args": (trainer, model, saved_ckpt)},
        {"name": "on_save_checkpoint", "args": (saved_ckpt,)},
        {"name": "Callback.on_train_end", "args": (trainer, model)},
        {"name": "on_train_end"},
        {"name": "Callback.on_fit_end", "args": (trainer, model)},
        {"name": "on_fit_end"},
        {"name": "Callback.teardown", "args": (trainer, model), "kwargs": {"stage": "fit"}},
        {"name": "teardown", "kwargs": {"stage": "fit"}},
    ]
    assert called == expected


def test_trainer_model_hook_system_fit_no_val_and_resume_max_steps(tmp_path):
    # initial training to get a checkpoint
    model = BoringModel()
    trainer = Trainer(
        devices=1,
        default_root_dir=tmp_path,
        max_steps=1,
        limit_val_batches=0,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[HookedCallback([])],
    )
    trainer.fit(model)
    best_model_path = trainer.checkpoint_callback.best_model_path

    # resume from checkpoint with HookedModel
    called = []
    model = HookedModel(called)
    callback = HookedCallback(called)

    # already performed 1 step, resume and do 2 more
    train_batches = 2
    steps_after_reload = 1 + train_batches
    trainer = Trainer(
        devices=1,
        default_root_dir=tmp_path,
        max_steps=steps_after_reload,
        limit_val_batches=0,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[callback],
    )

    trainer.fit(model, ckpt_path=best_model_path)
    loaded_ckpt = {
        "callbacks": ANY,
        "epoch": 0,
        "global_step": 1,
        "lr_schedulers": ANY,
        "optimizer_states": ANY,
        "pytorch-lightning_version": __version__,
        "state_dict": ANY,
        "loops": ANY,
    }
    saved_ckpt = {**loaded_ckpt, "global_step": steps_after_reload}
    expected = [
        {"name": "configure_callbacks"},
        {"name": "prepare_data"},
        {"name": "Callback.setup", "args": (trainer, model), "kwargs": {"stage": "fit"}},
        {"name": "setup", "kwargs": {"stage": "fit"}},
        {"name": "configure_model"},
        {"name": "on_load_checkpoint", "args": (loaded_ckpt,)},
        {"name": "Callback.on_load_checkpoint", "args": (trainer, model, loaded_ckpt)},
        {"name": "Callback.load_state_dict", "args": ({"foo": True},)},
        {"name": "configure_optimizers"},
        {"name": "Callback.on_fit_start", "args": (trainer, model)},
        {"name": "on_fit_start"},
        {"name": "zero_grad"},
        {"name": "train_dataloader"},
        {"name": "Callback.on_train_start", "args": (trainer, model)},
        {"name": "on_train_start"},
        *model._train_batch(trainer, model, steps_after_reload, trainer.strategy.root_device, current_batch=1),
        {"name": "Callback.on_train_epoch_end", "args": (trainer, model)},
        {"name": "on_train_epoch_end"},  # before ModelCheckpoint because it's a "monitoring callback"
        # `ModelCheckpoint.save_checkpoint` is called here
        {"name": "Callback.state_dict"},
        {"name": "Callback.on_save_checkpoint", "args": (trainer, model, saved_ckpt)},
        {"name": "on_save_checkpoint", "args": (saved_ckpt,)},
        {"name": "Callback.on_train_end", "args": (trainer, model)},
        {"name": "on_train_end"},
        {"name": "Callback.on_fit_end", "args": (trainer, model)},
        {"name": "on_fit_end"},
        {"name": "Callback.teardown", "args": (trainer, model), "kwargs": {"stage": "fit"}},
        {"name": "teardown", "kwargs": {"stage": "fit"}},
    ]
    assert called == expected


@pytest.mark.parametrize("batches", [0, 2])
@pytest.mark.parametrize(
    ("verb", "noun", "dataloader", "key"), [("validate", "validation", "val", "x"), ("test", "test", "test", "y")]
)
@pytest.mark.parametrize("override_on_x_model_train", [True, False])
def test_trainer_model_hook_system_eval(tmp_path, override_on_x_model_train, batches, verb, noun, dataloader, key):
    called = []
    model = HookedModel(called)
    if not override_on_x_model_train:
        setattr(model, f"on_{noun}_model_train", None)
    assert is_overridden(f"on_{noun}_model_train", model) == override_on_x_model_train
    callback = HookedCallback(called)
    trainer = Trainer(
        devices=1,
        default_root_dir=tmp_path,
        max_epochs=1,
        limit_val_batches=batches,
        limit_test_batches=batches,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[callback],
    )
    fn = getattr(trainer, verb)
    fn(model, verbose=False)
    hooks = [
        {"name": f"{dataloader}_dataloader"},
        {"name": "train", "args": (False,)},
        {"name": f"on_{noun}_model_eval"},
        {"name": f"Callback.on_{noun}_start", "args": (trainer, model)},
        {"name": f"on_{noun}_start"},
        *model._eval_epoch(noun, trainer, model, batches, key, trainer.strategy.root_device),
        {"name": f"Callback.on_{noun}_end", "args": (trainer, model)},
        {"name": f"on_{noun}_end"},
        *([{"name": f"on_{noun}_model_train"}] if override_on_x_model_train else []),
    ]
    expected = [
        {"name": "configure_callbacks"},
        {"name": "prepare_data"},
        {"name": "Callback.setup", "args": (trainer, model), "kwargs": {"stage": verb}},
        {"name": "setup", "kwargs": {"stage": verb}},
        {"name": "configure_model"},
        {"name": "zero_grad"},
        *(hooks if batches else []),
        {"name": "Callback.teardown", "args": (trainer, model), "kwargs": {"stage": verb}},
        {"name": "teardown", "kwargs": {"stage": verb}},
    ]
    assert called == expected


def test_trainer_model_hook_system_predict(tmp_path):
    called = []
    model = HookedModel(called)
    callback = HookedCallback(called)
    batches = 2
    trainer = Trainer(
        devices=1,
        default_root_dir=tmp_path,
        limit_predict_batches=batches,
        enable_progress_bar=False,
        callbacks=[callback],
    )
    trainer.predict(model)
    expected = [
        {"name": "configure_callbacks"},
        {"name": "prepare_data"},
        {"name": "Callback.setup", "args": (trainer, model), "kwargs": {"stage": "predict"}},
        {"name": "setup", "kwargs": {"stage": "predict"}},
        {"name": "configure_model"},
        {"name": "zero_grad"},
        {"name": "predict_dataloader"},
        {"name": "train", "args": (False,)},
        {"name": "on_predict_model_eval"},
        {"name": "Callback.on_predict_start", "args": (trainer, model)},
        {"name": "on_predict_start"},
        {"name": "Callback.on_predict_epoch_start", "args": (trainer, model)},
        {"name": "on_predict_epoch_start"},
        *model._predict_batch(trainer, model, batches, trainer.strategy.root_device),
        {"name": "Callback.on_predict_epoch_end", "args": (trainer, model)},
        {"name": "on_predict_epoch_end"},
        {"name": "Callback.on_predict_end", "args": (trainer, model)},
        {"name": "on_predict_end"},
        # TODO: `on_predict_model_train`
        {"name": "Callback.teardown", "args": (trainer, model), "kwargs": {"stage": "predict"}},
        {"name": "teardown", "kwargs": {"stage": "predict"}},
    ]
    assert called == expected


def test_hooks_with_different_argument_names(tmp_path):
    """Test that argument names can be anything in the hooks."""

    class CustomBoringModel(BoringModel):
        def assert_args(self, x, batch_nb):
            assert isinstance(x, Tensor)
            assert x.size() == (1, 32)
            assert isinstance(batch_nb, int)

        def training_step(self, x1, batch_nb1):
            self.assert_args(x1, batch_nb1)
            return super().training_step(x1, batch_nb1)

        def validation_step(self, x2, batch_nb2):
            self.assert_args(x2, batch_nb2)
            return super().validation_step(x2, batch_nb2)

        # we don't support a different name for `dataloader_idx`
        def test_step(self, x3, batch_nb3, dataloader_idx):
            self.assert_args(x3, batch_nb3)
            assert isinstance(dataloader_idx, int)
            return super().test_step(x3, batch_nb3)

        # we don't support a different name for `dataloader_idx`
        def predict_step(self, x4, batch_nb4, dataloader_idx):
            self.assert_args(x4, batch_nb4)
            assert isinstance(dataloader_idx, int)
            return super().predict_step(x4, batch_nb4, dataloader_idx)

        def test_dataloader(self):
            return [DataLoader(RandomDataset(32, 64)), DataLoader(RandomDataset(32, 64))]

        def predict_dataloader(self):
            return [DataLoader(RandomDataset(32, 64)), DataLoader(RandomDataset(32, 64))]

    model = CustomBoringModel()

    trainer = Trainer(devices=1, default_root_dir=tmp_path, fast_dev_run=5)

    trainer.fit(model)
    trainer.test(model)

    preds = trainer.predict(model)
    assert len(preds) == 2
    assert all(len(x) == 5 for x in preds)


def test_trainer_datamodule_hook_system(tmp_path):
    """Test the LightningDataModule hook system."""
    model = BoringModel()
    batches = 2
    trainer = Trainer(
        devices=1,
        default_root_dir=tmp_path,
        max_epochs=1,
        limit_train_batches=batches,
        limit_val_batches=batches,
        limit_test_batches=batches,
        limit_predict_batches=batches,
        enable_progress_bar=False,
        enable_model_summary=False,
        reload_dataloaders_every_n_epochs=1,
    )

    called = []
    dm = HookedDataModule(called)
    trainer.fit(model, datamodule=dm)
    expected = [
        {"name": "prepare_data"},
        {"name": "setup", "kwargs": {"stage": "fit"}},
        {"name": "val_dataloader"},
        {"name": "train_dataloader"},
        {"name": "state_dict"},
        {"name": "teardown", "kwargs": {"stage": "fit"}},
    ]
    assert called == expected

    called = []
    dm = HookedDataModule(called)
    trainer.validate(model, datamodule=dm, verbose=False)
    expected = [
        {"name": "prepare_data"},
        {"name": "setup", "kwargs": {"stage": "validate"}},
        {"name": "val_dataloader"},
        {"name": "teardown", "kwargs": {"stage": "validate"}},
    ]
    assert called == expected

    called = []
    dm = HookedDataModule(called)
    trainer.test(model, datamodule=dm, verbose=False)
    expected = [
        {"name": "prepare_data"},
        {"name": "setup", "kwargs": {"stage": "test"}},
        {"name": "test_dataloader"},
        {"name": "teardown", "kwargs": {"stage": "test"}},
    ]
    assert called == expected

    called = []
    dm = HookedDataModule(called)
    trainer.predict(model, datamodule=dm)
    expected = [
        {"name": "prepare_data"},
        {"name": "setup", "kwargs": {"stage": "predict"}},
        {"name": "predict_dataloader"},
        {"name": "teardown", "kwargs": {"stage": "predict"}},
    ]
    assert called == expected


@pytest.mark.parametrize("override_configure_model", [True, False])
def test_load_from_checkpoint_hook_calls(override_configure_model, tmp_path):
    class CustomHookedDataModule(HookedDataModule):
        def state_dict(self):
            return {"foo": "bar"}

    class CustomHookedModel(HookedModel):
        pass

    if not override_configure_model:
        CustomHookedModel.configure_model = None

    lm_called, ldm_called = [], []
    model = CustomHookedModel(lm_called)
    assert is_overridden("configure_model", model) == override_configure_model

    datamodule = CustomHookedDataModule(ldm_called)
    trainer = Trainer(devices=1)
    trainer.strategy.connect(model)
    trainer._data_connector.attach_data(model, datamodule=datamodule)
    ckpt_path = str(tmp_path / "file.ckpt")
    trainer.save_checkpoint(ckpt_path)

    datamodule_state_dict_key = datamodule.__class__.__qualname__
    saved_ckpt = {
        "callbacks": ANY,
        "epoch": 0,
        "global_step": 0,
        "lr_schedulers": ANY,
        "optimizer_states": ANY,
        "pytorch-lightning_version": __version__,
        "state_dict": ANY,
        "loops": ANY,
        datamodule_state_dict_key: {"foo": "bar"},
    }

    assert lm_called == [{"name": "on_save_checkpoint", "args": (saved_ckpt,)}]
    assert ldm_called == [{"name": "state_dict"}]

    lm_called, ldm_called = [], []
    _ = CustomHookedModel.load_from_checkpoint(ckpt_path, called=lm_called)
    _ = CustomHookedDataModule.load_from_checkpoint(ckpt_path, called=ldm_called)

    expected_lm_called = [{"name": "configure_model"}] if override_configure_model else []
    expected_lm_called += [{"name": "on_load_checkpoint", "args": ({**saved_ckpt, "hyper_parameters": ANY},)}]
    assert lm_called == expected_lm_called
    assert ldm_called == [{"name": "load_state_dict", "args": (saved_ckpt[datamodule_state_dict_key],)}]


def test_train_eval_mode_restored(tmp_path):
    """Test that the trainer restores the `training` mode of all submodules to what it was before entering the loop."""

    class MixedTrainModeModule(BoringModel):
        def __init__(self):
            super().__init__()
            # A frozen submodule should keep its mode, regardless of whether we're training or not
            self.frozen = torch.nn.Linear(2, 2)
            self.frozen.eval()
            self.frozen.requires_grad_(False)

        def training_step(self, *args, **kwargs):
            assert self.layer.weight.requires_grad
            assert self.layer.training
            assert not self.frozen.training
            assert not self.frozen.weight.requires_grad
            return super().training_step(*args, **kwargs)

        def validation_step(self, *args, **kwargs):
            assert self.layer.weight.requires_grad
            assert not self.layer.training
            assert not self.frozen.training
            assert not self.frozen.weight.requires_grad
            return super().validation_step(*args, **kwargs)

        def test_step(self, *args, **kwargs):
            assert self.layer.weight.requires_grad
            assert not self.layer.training
            assert not self.frozen.training
            assert not self.frozen.weight.requires_grad
            return super().test_step(*args, **kwargs)

        def predict_step(self, *args, **kwargs):
            assert self.layer.weight.requires_grad
            assert not self.layer.training
            assert not self.frozen.training
            assert not self.frozen.weight.requires_grad
            return super().predict_step(*args, **kwargs)

    model = MixedTrainModeModule()
    trainer = Trainer(
        devices=1,
        default_root_dir=tmp_path,
        max_epochs=1,
        val_check_interval=1,
        limit_train_batches=3,
        limit_val_batches=2,
        limit_test_batches=2,
        limit_predict_batches=2,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
    )
    trainer.fit(model)
    trainer.validate(model)
    trainer.test(model)
    trainer.predict(model)
