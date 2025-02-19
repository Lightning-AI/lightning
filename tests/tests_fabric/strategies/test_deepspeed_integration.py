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
from copy import deepcopy
from unittest import mock
from unittest.mock import ANY

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lightning.fabric import Fabric
from lightning.fabric.plugins import DeepSpeedPrecision
from lightning.fabric.strategies import DeepSpeedStrategy
from tests_fabric.helpers.datasets import RandomDataset, RandomIterableDataset
from tests_fabric.helpers.runif import RunIf
from tests_fabric.test_fabric import BoringModel


@RunIf(min_cuda_gpus=2, standalone=True, deepspeed=True)
def test_deepspeed_multiple_models():
    fabric = Fabric(strategy=DeepSpeedStrategy(stage=3, logging_batch_size_per_gpu=1), devices=2, accelerator="gpu")
    fabric.launch()

    with fabric.init_module():
        model = BoringModel()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    model, optimizer = fabric.setup(model, optimizer)

    for i in range(2):
        optimizer.zero_grad()
        x = model(torch.randn(1, 32).to(fabric.device))
        loss = x.sum()
        if i == 0:
            # the weights are not initialized with stage 3 until backward is run once
            assert all(w.nelement() == 0 for w in model.state_dict().values())
        fabric.backward(loss, model=model)
        if i == 0:
            # save for later to check that the weights were updated
            state_dict = deepcopy(model.state_dict())
        optimizer.step()

    # check that the model trained, the weights from step 1 do not match the weights from step 2
    for mw_b, mw_a in zip(state_dict.values(), model.state_dict().values()):
        assert not torch.allclose(mw_b, mw_a)

    fabric.seed_everything(42)
    model_1 = BoringModel()
    optimizer_1 = torch.optim.SGD(model_1.parameters(), lr=0.0001)

    fabric.seed_everything(42)
    model_2 = BoringModel()
    optimizer_2 = torch.optim.SGD(model_2.parameters(), lr=0.0001)

    for mw_1, mw_2 in zip(model_1.state_dict().values(), model_2.state_dict().values()):
        assert torch.allclose(mw_1, mw_2)

    model_1, optimizer_1 = fabric.setup(model_1, optimizer_1)
    model_2, optimizer_2 = fabric.setup(model_2, optimizer_2)

    # train model_1 first
    fabric.seed_everything(42)
    data_list = []
    for _ in range(2):
        optimizer_1.zero_grad()
        data = torch.randn(1, 32).to(fabric.device)
        data_list.append(data)
        x = model_1(data)
        loss = x.sum()
        fabric.backward(loss, model=model_1)
        optimizer_1.step()

    # the weights do not match
    assert all(w.nelement() > 1 for w in model_1.state_dict().values())
    assert all(w.nelement() == 0 for w in model_2.state_dict().values())

    # now train model_2 with the same data
    for data in data_list:
        optimizer_2.zero_grad()
        x = model_2(data)
        loss = x.sum()
        fabric.backward(loss, model=model_2)
        optimizer_2.step()

    # the weights should match
    for mw_1, mw_2 in zip(model_1.state_dict().values(), model_2.state_dict().values()):
        assert torch.allclose(mw_1, mw_2)

    # Verify collectives works as expected
    ranks = fabric.all_gather(torch.tensor([fabric.local_rank]).to(fabric.device))
    assert torch.allclose(ranks.cpu(), torch.tensor([[0], [1]]))
    assert fabric.broadcast(True)
    assert fabric.is_global_zero == (fabric.local_rank == 0)


@RunIf(min_cuda_gpus=1, deepspeed=True)
@pytest.mark.parametrize(
    ("dataset_cls", "logging_batch_size_per_gpu", "expected_batch_size"),
    [
        (RandomDataset, None, 1),
        (RandomDataset, 10, 10),
        (RandomIterableDataset, None, 1),
        (RandomIterableDataset, 10, 10),
    ],
)
def test_deepspeed_auto_batch_size_config_select(dataset_cls, logging_batch_size_per_gpu, expected_batch_size):
    """Test to ensure that the batch size is correctly set as expected for deepspeed logging purposes."""
    fabric = Fabric(
        accelerator="cuda",
        devices=1,
        strategy=DeepSpeedStrategy(logging_batch_size_per_gpu=logging_batch_size_per_gpu, zero_optimization=False),
    )
    fabric.launch()
    assert isinstance(fabric._strategy, DeepSpeedStrategy)
    _ = fabric.setup_dataloaders(DataLoader(dataset_cls(32, 64)))
    config = fabric._strategy.config
    assert config["train_micro_batch_size_per_gpu"] == expected_batch_size


@RunIf(min_cuda_gpus=1, standalone=True, deepspeed=True)
def test_deepspeed_configure_optimizers():
    """Test that the deepspeed strategy with default initialization wraps the optimizer correctly."""
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

    fabric = Fabric(
        strategy=DeepSpeedStrategy(),
        accelerator="cuda",
        devices=1,
        precision="16-mixed",
    )
    fabric.launch()
    model = nn.Linear(3, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model, optimizer = fabric.setup(model, optimizer)
    assert isinstance(optimizer.optimizer, DeepSpeedZeroOptimizer)
    assert isinstance(optimizer.optimizer.optimizer, torch.optim.SGD)


@RunIf(min_cuda_gpus=1, deepspeed=True)
def test_deepspeed_custom_precision_params():
    """Test that if the FP16 parameters are set via the DeepSpeedStrategy, the deepspeed config contains these
    changes."""
    strategy = DeepSpeedStrategy(
        loss_scale=10, initial_scale_power=11, loss_scale_window=12, hysteresis=13, min_loss_scale=14
    )
    fabric = Fabric(
        strategy=strategy,
        precision="16-mixed",
        accelerator="cuda",
        devices=1,
    )
    fabric.launch()
    assert fabric._strategy._config_initialized
    assert fabric._strategy.config["fp16"]["loss_scale"] == 10
    assert fabric._strategy.config["fp16"]["initial_scale_power"] == 11
    assert fabric._strategy.config["fp16"]["loss_scale_window"] == 12
    assert fabric._strategy.config["fp16"]["hysteresis"] == 13
    assert fabric._strategy.config["fp16"]["min_loss_scale"] == 14


@RunIf(min_cuda_gpus=1, standalone=True, deepspeed=True)
def test_deepspeed_custom_activation_checkpointing_params_forwarded():
    """Test that the activation checkpointing parameters get passed to `deepspeed.checkpointing.configure`
    correctly."""
    import deepspeed

    strategy = DeepSpeedStrategy(
        partition_activations=True,
        cpu_checkpointing=True,
        contiguous_memory_optimization=True,
        synchronize_checkpoint_boundary=True,
    )
    fabric = Fabric(
        strategy=strategy,
        precision="16-mixed",
        accelerator="cuda",
        devices=1,
    )
    fabric.launch()
    model = nn.Linear(3, 3)
    optimizer = torch.optim.Adam(model.parameters())

    with mock.patch("deepspeed.checkpointing.configure", wraps=deepspeed.checkpointing.configure) as configure:
        fabric.setup(model, optimizer)

    configure.assert_called_with(
        mpu_=None,
        partition_activations=True,
        contiguous_checkpointing=True,
        checkpoint_in_cpu=True,
        profile=None,
    )


@RunIf(min_cuda_gpus=2, standalone=True, deepspeed=True)
def test_deepspeed_multigpu_stage_3():
    """Test to ensure ZeRO Stage 3 works with a parallel model."""
    fabric = Fabric(
        strategy=DeepSpeedStrategy(stage=3),
        accelerator="cuda",
        devices=2,
        precision="16-mixed",
    )
    fabric.launch()

    def _make_block():
        return nn.Sequential(nn.Linear(32, 32, bias=False), nn.ReLU())

    with fabric.init_module():
        model = nn.Sequential(*(_make_block() for _ in range(5)), nn.Linear(32, 3))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    model, optimizer = fabric.setup(model, optimizer)

    x = torch.rand(2, 32, device=fabric.device)
    y = torch.ones(x.size(0), device=x.device, dtype=torch.long)
    x = model(x)
    x = x.float()  # Ensure output is in float32 for softmax operation
    logits = F.softmax(x, dim=1)
    loss = F.cross_entropy(logits, y)
    fabric.backward(loss)
    optimizer.step()
    optimizer.zero_grad()


@RunIf(deepspeed=True)
@mock.patch("deepspeed.init_distributed", autospec=True)
@mock.patch("lightning.fabric.accelerators.mps.MPSAccelerator.is_available", return_value=False)
@pytest.mark.parametrize("platform", ["Linux", "Windows"])
def test_deepspeed_env_variables_on_platforms(_, deepspeed_dist_mock, platform):
    """Test to ensure that we set up distributed communication correctly.

    When using Windows, ranks environment variables should not be set, and DeepSpeed should handle this.

    """
    fabric = Fabric(strategy=DeepSpeedStrategy(stage=3))
    strategy = fabric._strategy
    assert isinstance(strategy, DeepSpeedStrategy)
    with mock.patch("platform.system", return_value=platform) as platform_mock:
        strategy._init_deepspeed_distributed()
    deepspeed_dist_mock.assert_called()
    platform_mock.assert_called()
    if platform == "Windows":
        # assert no env variables have been set within the DeepSpeedStrategy
        assert all(k not in os.environ for k in ("MASTER_PORT", "MASTER_ADDR", "RANK", "WORLD_SIZE", "LOCAL_RANK"))
    else:
        assert os.environ["MASTER_ADDR"] == str(strategy.cluster_environment.main_address)
        assert os.environ["MASTER_PORT"] == str(strategy.cluster_environment.main_port)
        assert os.environ["RANK"] == str(strategy.global_rank)
        assert os.environ["WORLD_SIZE"] == str(strategy.world_size)
        assert os.environ["LOCAL_RANK"] == str(strategy.local_rank)


@RunIf(min_cuda_gpus=2, standalone=True, deepspeed=True, bf16_cuda=True)
def test_deepspeed_with_bfloat16_precision():
    """Test that the DeepSpeed strategy works with bfloat16 precision."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(32, 2)

        def forward(self, x):
            assert x.dtype == torch.bfloat16
            return self.layer(x)

    fabric = Fabric(accelerator="cuda", devices=2, strategy="deepspeed_stage_3", precision="bf16-mixed")
    assert isinstance(fabric._strategy.precision, DeepSpeedPrecision)
    assert fabric._strategy.precision.precision == "bf16-mixed"
    assert fabric._strategy.config["zero_optimization"]["stage"] == 3
    fabric.launch()

    with fabric.init_module():
        model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    model, optimizer = fabric.setup(model, optimizer)
    assert fabric._strategy.config["bf16"]["enabled"]
    assert model.layer.weight.dtype == torch.bfloat16

    batch = torch.rand(2, 32, device=fabric.device)
    assert batch.dtype == torch.float32
    loss = model(batch).sum()
    fabric.backward(loss)
    optimizer.step()
    optimizer.zero_grad()


def _assert_saved_model_is_equal(fabric, model, checkpoint_path):
    """Convert the saved checkpoint to a single file with the model weights consolidated to easily verify the full
    weights in float32 precision."""
    from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

    assert isinstance(fabric.strategy, DeepSpeedStrategy)

    # carry out the check only on rank 0
    if fabric.is_global_zero:
        if fabric.strategy.config["zero_optimization"]["stage"] in (2, 3):
            single_ckpt_path = checkpoint_path / "single_model.pt"
            # the tag is hardcoded in DeepSpeedStrategy
            convert_zero_checkpoint_to_fp32_state_dict(checkpoint_path, single_ckpt_path, tag="checkpoint")
            state_dict = torch.load(single_ckpt_path, weights_only=False)
        else:
            # 'checkpoint' is the tag, hardcoded in DeepSpeedStrategy
            single_ckpt_path = checkpoint_path / "checkpoint" / "mp_rank_00_model_states.pt"
            state_dict = torch.load(single_ckpt_path, weights_only=False)["module"]

        model = model.cpu()

        # assert model parameters are identical after loading
        for orig_param, saved_model_param in zip(model.parameters(), state_dict.values()):
            # perform the equality check in the same precision
            saved_model_param = saved_model_param.cpu().to(orig_param.dtype)
            assert torch.equal(orig_param, saved_model_param)

    fabric.barrier()


@RunIf(min_cuda_gpus=2, standalone=True, deepspeed=True, bf16_cuda=True)
@pytest.mark.parametrize("stage", [1, 2, 3])
def test_deepspeed_save_load_checkpoint_zero_3(stage, tmp_path):
    """Test that DeepSpeed stage 1, 2, and 3 model checkpoints can be saved and loaded successfully."""
    from deepspeed import DeepSpeedEngine

    fabric = Fabric(accelerator="cuda", devices=2, strategy=DeepSpeedStrategy(stage=stage), precision="bf16-mixed")
    fabric.launch()

    checkpoint_path = fabric.broadcast(tmp_path / "deepspeed-checkpoint")

    with fabric.init_module():
        model = BoringModel()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    model, optimizer = fabric.setup(model, optimizer)
    assert isinstance(model._forward_module, DeepSpeedEngine)

    # TODO(fabric): The dtype on the model is not correct, should be torch.bfloat16
    assert model.dtype == torch.float32
    assert next(model.parameters()).dtype == torch.bfloat16

    # dummy training step
    output = model(torch.randn(1, 32).to(fabric.device))
    loss = output.sum()
    fabric.backward(loss)
    optimizer.step()
    optimizer.zero_grad()

    state = {"model": model, "optimizer": optimizer, "steps": 1}
    fabric.save(checkpoint_path, state)

    # re-init all objects and resume
    fabric = Fabric(accelerator="cuda", devices=2, strategy=DeepSpeedStrategy(stage=stage), precision="bf16")
    fabric.launch()
    with fabric.init_module():
        model = BoringModel()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    model, optimizer = fabric.setup(model, optimizer)
    state = {"model": model, "optimizer": optimizer, "steps": 0}

    metadata = fabric.load(checkpoint_path, state)

    # check user data in state reloaded
    assert state["steps"] == 1
    # the remainder of the deepspeed checkpoint contains metadata
    assert "ds_version" in metadata

    _assert_saved_model_is_equal(fabric, model, checkpoint_path)


@RunIf(min_cuda_gpus=2, standalone=True, deepspeed=True, bf16_cuda=True)
@pytest.mark.parametrize("empty_init", [None, True])
def test_deepspeed_init_module_with_stage_3(empty_init):
    """Tests how `.init_module()` behaves with ZeRO stage 3."""
    strategy = DeepSpeedStrategy(stage=3)
    fabric = Fabric(accelerator="cuda", devices=2, strategy=strategy, precision="bf16-true")
    fabric.launch()

    with mock.patch("deepspeed.zero.Init") as zero_init_mock, fabric.init_module(empty_init=empty_init):
        BoringModel()
    fabric.barrier()
    zero_init_mock.assert_called_once_with(enabled=True, remote_device=None, config_dict_or_path=ANY)


@RunIf(min_cuda_gpus=2, standalone=True, deepspeed=True, bf16_cuda=True)
@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.parametrize("empty_init", [None, False, True])
def test_deepspeed_init_module_with_stages_1_2(stage, empty_init):
    """Tests how `.init_module()` behaves with ZeRO stages 1 and 2."""
    strategy = DeepSpeedStrategy(stage=stage)
    fabric = Fabric(accelerator="cuda", devices=2, strategy=strategy, precision="bf16-true")
    fabric.launch()

    with (
        mock.patch("deepspeed.zero.Init") as zero_init_mock,
        mock.patch("torch.Tensor.uniform_") as init_mock,
        fabric.init_module(empty_init=empty_init),
    ):
        model = BoringModel()

    zero_init_mock.assert_called_with(enabled=False, remote_device=None, config_dict_or_path=ANY)
    assert init_mock.call_count == int(not empty_init)
    assert model.layer.weight.dtype == torch.bfloat16


@RunIf(min_cuda_gpus=2, standalone=True, deepspeed=True)
def test_deepspeed_multigpu_stage_3_MiCS_support():
    """Test to ensure ZeRO Stage 3 MiCS works with a parallel model."""
    strategy = DeepSpeedStrategy(stage=3)
    strategy.config["zero_optimization"]["stage"] = 3
    strategy.config["zero_optimization"]["mics_shard_size"] = 1
    strategy.config["zero_optimization"]["mics_hierarchical_params_gather"] = False

    fabric = Fabric(
        strategy=strategy,
        accelerator="cuda",
        devices=2,
        precision="16-mixed",
    )
    fabric.launch()

    def _make_block():
        return nn.Sequential(nn.Linear(32, 32, bias=False), nn.ReLU())

    with fabric.init_module():
        model = nn.Sequential(*(_make_block() for _ in range(5)), nn.Linear(32, 3))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    model, optimizer = fabric.setup(model, optimizer)

    x = torch.rand(2, 32, device=fabric.device)
    y = torch.ones(x.size(0), device=x.device, dtype=torch.long)
    x = model(x)
    x = x.float()  # Ensure output is in float32 for softmax operation
    logits = F.softmax(x, dim=1)
    loss = F.cross_entropy(logits, y)
    fabric.backward(loss)
    optimizer.step()
    optimizer.zero_grad()


@RunIf(min_cuda_gpus=2, standalone=True, deepspeed=True)
def test_deepspeed_multigpu_stage_3_MiCS_offload_param_support():
    """Test to ensure we can use DeepSpeed with ZeRO Stage param offload 3 MiCS Support."""
    strategy = DeepSpeedStrategy(stage=3, offload_params_device="cpu")
    strategy.config["zero_optimization"]["stage"] = 3
    strategy.config["zero_optimization"]["mics_shard_size"] = 1
    strategy.config["zero_optimization"]["mics_hierarchical_params_gather"] = False

    fabric = Fabric(
        strategy=strategy,
        accelerator="cuda",
        devices=2,
        precision="16-mixed",
    )
    fabric.launch()

    def _make_block():
        return nn.Sequential(nn.Linear(32, 32, bias=False), nn.ReLU())

    with fabric.init_module():
        model = nn.Sequential(*(_make_block() for _ in range(5)), nn.Linear(32, 3))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    model, optimizer = fabric.setup(model, optimizer)

    x = torch.rand(2, 32, device=fabric.device)
    y = torch.ones(x.size(0), device=x.device, dtype=torch.long)
    x = model(x)
    x = x.float()  # Ensure output is in float32 for softmax operation
    logits = F.softmax(x, dim=1)
    loss = F.cross_entropy(logits, y)
    fabric.backward(loss)
    optimizer.step()
    optimizer.zero_grad()


@RunIf(min_cuda_gpus=2, standalone=True, deepspeed=True)
def test_deepspeed_multigpu_stage_3_MiCS_offload_param_optimizer_support():
    """Test to ensure we can use DeepSpeed with ZeRO Stage param & optimizer offload 3 MiCS Support."""
    strategy = DeepSpeedStrategy(stage=3, offload_params_device="cpu", offload_optimizer_device="cpu")
    strategy.config["zero_optimization"]["stage"] = 3
    strategy.config["zero_optimization"]["mics_shard_size"] = 1
    strategy.config["zero_optimization"]["mics_hierarchical_params_gather"] = False

    fabric = Fabric(
        strategy=strategy,
        accelerator="cuda",
        devices=2,
        precision="16-mixed",
    )
    fabric.launch()

    def _make_block():
        return nn.Sequential(nn.Linear(32, 32, bias=False), nn.ReLU())

    with fabric.init_module():
        model = nn.Sequential(*(_make_block() for _ in range(5)), nn.Linear(32, 3))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    model, optimizer = fabric.setup(model, optimizer)

    x = torch.rand(2, 32, device=fabric.device)
    y = torch.ones(x.size(0), device=x.device, dtype=torch.long)
    x = model(x)
    x = x.float()  # Ensure output is in float32 for softmax operation
    logits = F.softmax(x, dim=1)
    loss = F.cross_entropy(logits, y)
    fabric.backward(loss)
    optimizer.step()
    optimizer.zero_grad()


@RunIf(min_cuda_gpus=4, standalone=True, deepspeed=True)
def test_deepspeed_multigpu_stage_3_hierarchical_MiCS_support():
    """Test to ensure we can use DeepSpeed with ZeRO Stage 3 MiCS Support ('mics_hierarchical_params_gather' =
    True)."""
    strategy = DeepSpeedStrategy(stage=3)
    strategy.config["zero_optimization"]["stage"] = 3
    strategy.config["zero_optimization"]["mics_shard_size"] = 2
    strategy.config["zero_optimization"]["offload_param"] = {}
    strategy.config["zero_optimization"]["offload_optimizer"] = {}
    strategy.config["zero_optimization"]["mics_hierarchical_params_gather"] = True

    fabric = Fabric(
        strategy=strategy,
        accelerator="cuda",
        devices=2,
        precision="16-mixed",
    )
    fabric.launch()

    def _make_block():
        return nn.Sequential(nn.Linear(32, 32, bias=False), nn.ReLU())

    with fabric.init_module():
        model = nn.Sequential(*(_make_block() for _ in range(5)), nn.Linear(32, 3))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    model, optimizer = fabric.setup(model, optimizer)

    x = torch.rand(2, 32, device=fabric.device)
    y = torch.ones(x.size(0), device=x.device, dtype=torch.long)
    x = model(x)
    x = x.float()  # Ensure output is in float32 for softmax operation
    logits = F.softmax(x, dim=1)
    loss = F.cross_entropy(logits, y)
    fabric.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
