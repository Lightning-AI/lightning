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
from unittest import mock
from unittest.mock import Mock

import pytest
import torch
from torch._dynamo import OptimizedModule
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.multiprocessing import ProcessRaisedException
from torch.nn.parallel.distributed import DistributedDataParallel

import lightning.pytorch as pl
import tests_pytorch.helpers.pipelines as tpipes
from lightning.fabric.plugins.environments import ClusterEnvironment, LightningEnvironment
from lightning.fabric.utilities.distributed import _distributed_is_initialized
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, EarlyStopping
from lightning.pytorch.demos.boring_classes import BoringDataModule, BoringModel
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.strategies.launchers import _SubprocessScriptLauncher
from lightning.pytorch.strategies.launchers.multiprocessing import _MultiProcessingLauncher
from lightning.pytorch.trainer import seed_everything
from tests_pytorch.helpers.datamodules import ClassifDataModule
from tests_pytorch.helpers.runif import RunIf
from tests_pytorch.helpers.simple_models import ClassificationModel


@RunIf(min_cuda_gpus=2, standalone=True, sklearn=True)
def test_multi_gpu_model_ddp_fit_only(tmp_path):
    dm = ClassifDataModule()
    model = ClassificationModel()
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1, accelerator="gpu", devices=2, strategy="ddp")
    trainer.fit(model, datamodule=dm)


@RunIf(min_cuda_gpus=2, standalone=True, sklearn=True)
def test_multi_gpu_model_ddp_test_only(tmp_path):
    dm = ClassifDataModule()
    model = ClassificationModel()
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1, accelerator="gpu", devices=2, strategy="ddp")
    trainer.test(model, datamodule=dm)


@RunIf(min_cuda_gpus=2, standalone=True, sklearn=True)
def test_multi_gpu_model_ddp_fit_test(tmp_path):
    seed_everything(4321)
    dm = ClassifDataModule()
    model = ClassificationModel()
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1, accelerator="gpu", devices=2, strategy="ddp")
    trainer.fit(model, datamodule=dm)
    result = trainer.test(model, datamodule=dm)

    for out in result:
        assert out["test_acc"] > 0.7


@RunIf(skip_windows=True)
@mock.patch("torch.cuda.set_device")
@mock.patch("lightning.pytorch.accelerators.cuda._check_cuda_matmul_precision")
@mock.patch("lightning.pytorch.accelerators.cuda._clear_cuda_memory")
def test_ddp_torch_dist_is_available_in_setup(_, __, ___, cuda_count_1, mps_count_0, tmp_path):
    """Test to ensure torch distributed is available within the setup hook using ddp."""

    class TestModel(BoringModel):
        def setup(self, stage: str) -> None:
            assert _distributed_is_initialized()
            raise SystemExit()

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        fast_dev_run=True,
        strategy=DDPStrategy(process_group_backend="gloo"),
        accelerator="gpu",
        devices=1,
    )
    with pytest.raises(SystemExit):
        trainer.fit(model)


@RunIf(min_cuda_gpus=2, standalone=True)
@pytest.mark.parametrize("precision", ["16-mixed", "32-true"])
def test_ddp_wrapper(tmp_path, precision):
    """Test parameters to ignore are carried over for DDP."""

    class WeirdModule(torch.nn.Module):
        def _save_to_state_dict(self, destination, prefix, keep_vars):
            return {"something": "something"}

    class CustomModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.weird_module = WeirdModule()

            # should be skipped
            self._ddp_params_and_buffers_to_ignore = ["something"]

    class CustomCallback(Callback):
        def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
            assert isinstance(trainer.strategy.model, DistributedDataParallel)
            expected = ["something"]
            assert trainer.strategy.model.parameters_to_ignore == set(expected)
            assert trainer.strategy.model.module._ddp_params_and_buffers_to_ignore == expected

    model = CustomModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        fast_dev_run=True,
        precision=precision,
        strategy="ddp",
        accelerator="gpu",
        devices=2,
        callbacks=CustomCallback(),
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model)


@RunIf(min_cuda_gpus=2, sklearn=True)
def test_multi_gpu_early_stop_ddp_spawn(tmp_path):
    seed_everything(42)

    trainer_options = {
        "default_root_dir": tmp_path,
        "callbacks": [EarlyStopping(monitor="train_acc")],
        "max_epochs": 50,
        "limit_train_batches": 10,
        "limit_val_batches": 10,
        "accelerator": "gpu",
        "devices": [0, 1],
        "strategy": "ddp_spawn",
    }

    dm = ClassifDataModule()
    model = ClassificationModel()
    tpipes.run_model_test(trainer_options, model, dm)


@RunIf(min_cuda_gpus=2)
def test_multi_gpu_model_ddp_spawn(tmp_path):
    seed_everything(42)

    trainer_options = {
        "default_root_dir": tmp_path,
        "max_epochs": 1,
        "limit_train_batches": 10,
        "limit_val_batches": 10,
        "accelerator": "gpu",
        "devices": [0, 1],
        "strategy": "ddp_spawn",
        "enable_progress_bar": False,
    }

    model = BoringModel()

    tpipes.run_model_test(trainer_options, model)


@RunIf(min_cuda_gpus=2)
def test_ddp_all_dataloaders_passed_to_fit(tmp_path):
    """Make sure DDP works with dataloaders passed to fit()"""
    model = BoringModel()

    trainer = Trainer(
        default_root_dir=tmp_path,
        enable_progress_bar=False,
        max_epochs=1,
        limit_train_batches=0.2,
        limit_val_batches=0.2,
        accelerator="gpu",
        devices=[0, 1],
        strategy="ddp_spawn",
    )
    trainer.fit(model, train_dataloaders=model.train_dataloader(), val_dataloaders=model.val_dataloader())


class UnusedParametersModel(BoringModel):
    def __init__(self):
        super().__init__()
        self.intermediate_layer = torch.nn.Linear(32, 32)

    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            batch = self.intermediate_layer(batch)
        return super().training_step(batch, batch_idx)


@RunIf(standalone=True)
def test_find_unused_parameters_ddp_spawn_raises():
    """Test that the DDP strategy can change PyTorch's error message so that it's more useful for Lightning users."""
    trainer = Trainer(accelerator="cpu", devices=1, strategy="ddp_spawn", max_steps=2, logger=False)
    with pytest.raises(
        ProcessRaisedException, match="It looks like your LightningModule has parameters that were not used in"
    ):
        trainer.fit(UnusedParametersModel())


@RunIf(standalone=True)
def test_find_unused_parameters_ddp_exception():
    """Test that the DDP strategy can change PyTorch's error message so that it's more useful for Lightning users."""
    trainer = Trainer(accelerator="cpu", devices=1, strategy="ddp", max_steps=2, logger=False)
    with pytest.raises(RuntimeError, match="It looks like your LightningModule has parameters that were not used in"):
        trainer.fit(UnusedParametersModel())


class BoringCallbackDDPSpawnModel(BoringModel):
    def __init__(self, name: str, val: float):
        super().__init__()
        self.name = name
        self.val = val

    def validation_step(self, batch, batch_idx):
        self.log(self.name, self.val)
        return super().validation_step(batch, batch_idx)


class CustomMultiProcessingLauncher(_MultiProcessingLauncher):
    def get_extra_results(self, trainer):
        extra = super().get_extra_results(trainer)
        extra["test_val"] = "test_val"
        return extra

    def update_main_process_results(self, trainer, extra) -> None:
        trainer.strategy.test_val = extra.pop("test_val")
        return super().update_main_process_results(trainer, extra)


class TestDDPSpawnStrategy(DDPStrategy):
    def _configure_launcher(self):
        self._launcher = CustomMultiProcessingLauncher(self)


@RunIf(skip_windows=True)
def test_ddp_spawn_add_get_queue(tmp_path):
    """Tests get_extra_results/update_main_process_results with DDPSpawnStrategy."""
    ddp_spawn_strategy = TestDDPSpawnStrategy()
    trainer = Trainer(
        default_root_dir=tmp_path, fast_dev_run=True, accelerator="cpu", devices=2, strategy=ddp_spawn_strategy
    )

    val: float = 1.0
    val_name: str = "val_acc"
    model = BoringCallbackDDPSpawnModel(val_name, val)
    dm = BoringDataModule()
    trainer.fit(model, datamodule=dm)
    assert trainer.callback_metrics[val_name] == torch.tensor(val)
    assert ddp_spawn_strategy.test_val == "test_val"


class BoringModelDDPCPU(BoringModel):
    def on_train_start(self) -> None:
        # make sure that the model is on CPU when training
        assert self.device == torch.device("cpu")


@RunIf(skip_windows=True)
def test_ddp_cpu():
    """Tests if device is set correctly when training for DDPStrategy."""
    trainer = Trainer(devices=2, strategy="ddp_spawn", accelerator="cpu", fast_dev_run=True)
    # assert strategy attributes for device setting
    assert isinstance(trainer.strategy, DDPStrategy)
    assert trainer.strategy.root_device == torch.device("cpu")
    model = BoringModelDDPCPU()
    trainer.fit(model)


class BoringZeroRedundancyOptimizerModel(BoringModel):
    def configure_optimizers(self):
        return ZeroRedundancyOptimizer(self.layer.parameters(), optimizer_class=torch.optim.Adam, lr=0.1)


# ZeroRedundancyOptimizer internally calls `torch.load` with `weights_only` not set, triggering the FutureWarning
@pytest.mark.filterwarnings("ignore::FutureWarning")
@RunIf(min_cuda_gpus=2, skip_windows=True)
@pytest.mark.parametrize("strategy", [pytest.param("ddp", marks=RunIf(standalone=True)), "ddp_spawn"])
def test_ddp_strategy_checkpoint_zero_redundancy_optimizer(strategy, tmp_path):
    """Test to ensure that checkpoint is saved correctly when using zero redundancy optimizer."""
    model = BoringZeroRedundancyOptimizerModel()
    trainer = Trainer(default_root_dir=tmp_path, accelerator="gpu", devices=2, strategy=strategy, max_steps=1)

    trainer.fit(model)

    checkpoint_path = os.path.join(tmp_path, "model.pt")
    # need to broadcast because tmp_path is different on each process
    checkpoint_path = trainer.strategy.broadcast(checkpoint_path)
    trainer.save_checkpoint(checkpoint_path)
    saved_model = BoringModel.load_from_checkpoint(checkpoint_path)

    # Assert model parameters are identical after loading
    for trained_param, loaded_param in zip(model.parameters(), saved_model.parameters()):
        assert torch.equal(trained_param.to("cpu"), loaded_param)


def test_configure_launcher_create_processes_externally():
    class MyClusterEnvironment(ClusterEnvironment):
        @property
        def creates_processes_externally(self):
            return True

        @property
        def main_address(self):
            return ""

        @property
        def main_port(self):
            return 8080

        @staticmethod
        def detect():
            return True

        def world_size(self):
            return 1

        def set_world_size(self):
            pass

        def global_rank(self):
            return 0

        def set_global_rank(self):
            pass

        def local_rank(self):
            return 0

        def node_rank(self):
            return 0

    ddp_strategy = DDPStrategy(cluster_environment=MyClusterEnvironment(), parallel_devices=[torch.device("cpu")])
    assert ddp_strategy.launcher is None
    ddp_strategy._configure_launcher()
    assert isinstance(ddp_strategy.launcher, _SubprocessScriptLauncher)

    ddp_strategy.launcher._call_children_scripts = Mock()
    launch_fn = Mock()
    ddp_strategy.launcher.launch(launch_fn)
    ddp_strategy.launcher._call_children_scripts.assert_not_called()
    launch_fn.assert_called_once()


class CheckOptimizerDeviceModel(BoringModel):
    def configure_optimizers(self):
        assert all(param.device.type == "cuda" for param in self.parameters())
        super().configure_optimizers()


@RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize("strategy", ["ddp", "ddp_spawn"])
def test_model_parameters_on_device_for_optimizer(strategy):
    """Test that the strategy has moved the parameters to the device by the time the optimizer gets created."""
    model = CheckOptimizerDeviceModel()
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        fast_dev_run=1,
        accelerator="gpu",
        devices=1,
        strategy=strategy,
    )
    trainer.fit(model)


class BoringModelGPU(BoringModel):
    def on_train_start(self) -> None:
        # make sure that the model is on GPU when training
        assert self.device == torch.device(f"cuda:{self.trainer.strategy.local_rank}")
        self.start_cuda_memory = torch.cuda.memory_allocated()


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True)
def test_ddp_with_2_gpus():
    """Tests if device is set correctly when training and after teardown for DDPStrategy."""
    trainer = Trainer(
        accelerator="gpu",
        devices=2,
        strategy="ddp",
        fast_dev_run=True,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    # assert strategy attributes for device setting
    assert isinstance(trainer.strategy, DDPStrategy)
    local_rank = trainer.strategy.local_rank
    assert trainer.strategy.root_device == torch.device(f"cuda:{local_rank}")

    model = BoringModelGPU()

    trainer.fit(model)

    # assert after training, model is moved to CPU and memory is deallocated
    assert model.device == torch.device("cpu")
    cuda_memory = torch.cuda.memory_allocated()
    assert cuda_memory < model.start_cuda_memory


@RunIf(min_cuda_gpus=4, standalone=True)
@mock.patch("torch.distributed.barrier")
def test_ddp_barrier_non_consecutive_device_ids(barrier_mock, tmp_path):
    """Test correct usage of barriers when device ids do not start at 0 or are not consecutive."""
    model = BoringModel()
    gpus = [1, 3]
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_steps=1,
        accelerator="gpu",
        devices=gpus,
        strategy="ddp",
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model)
    barrier_mock.assert_any_call(device_ids=[gpus[trainer.local_rank]])


@mock.patch.dict(os.environ, {"LOCAL_RANK": "1"})
def test_incorrect_ddp_script_spawning(tmp_path):
    """Test an error message when user accidentally instructs Lightning to spawn children processes on rank > 0."""

    class WronglyImplementedEnvironment(LightningEnvironment):
        @property
        def creates_processes_externally(self):
            # returning false no matter what means Lightning would spawn also on ranks > 0 new processes
            return False

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        strategy="ddp",
        accelerator="cpu",
        devices=2,
        plugins=[WronglyImplementedEnvironment()],
        barebones=True,
    )
    with pytest.raises(
        RuntimeError, match="Lightning attempted to launch new distributed processes with `local_rank > 0`."
    ):
        trainer.fit(model)


@RunIf(min_cuda_gpus=2, standalone=True, dynamo=True)
@mock.patch("lightning.fabric.wrappers.torch.compile", Mock(wraps=torch.compile))
@mock.patch.dict(os.environ, {})
def test_reapply_compile():
    """Test that Trainer can rewrap a compiled module such that compilation happens over the DDP-wrapper."""
    trainer = Trainer(accelerator="gpu", devices=2, strategy="ddp", max_steps=2, logger=False)

    model = BoringModel()
    compile_kwargs = {"mode": "reduce-overhead"}
    compiled_model = torch.compile(model, **compile_kwargs)
    torch.compile.reset_mock()

    trainer.fit(compiled_model)
    trainer_model = trainer.strategy.model

    assert isinstance(trainer_model, OptimizedModule)
    assert isinstance(trainer_model._orig_mod, DistributedDataParallel)
    # Assert we called compile again with the same arguments, but on the DDP-wrapped module
    torch.compile.assert_called_with(trainer_model._orig_mod, **compile_kwargs)

    assert trainer_model._orig_mod.module == model

    # Smoke-testing forward to ensure we don't get compilation errors
    for _ in range(3):
        trainer_model(torch.randn(2, 32, device="gpu")).sum().backward()
