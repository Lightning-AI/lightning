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
import json
from re import escape
from unittest import mock
from unittest.mock import ANY, Mock

import pytest
import torch
from torch.optim import Optimizer

from lightning.fabric.accelerators import CPUAccelerator, CUDAAccelerator
from lightning.fabric.strategies import DeepSpeedStrategy
from lightning.fabric.utilities.exceptions import MisconfigurationException
from tests_fabric.helpers.runif import RunIf


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
def test_deepspeed_only_compatible_with_cuda():
    """Test that the DeepSpeed strategy raises an exception if an invalid accelerator is used."""
    strategy = DeepSpeedStrategy(accelerator=CPUAccelerator())
    with pytest.raises(RuntimeError, match="The DeepSpeed strategy is only supported on CUDA GPUs"):
        strategy.setup_environment()


@RunIf(deepspeed=True)
def test_deepspeed_with_invalid_config_path():
    """Test to ensure if we pass an invalid config path we throw an exception."""
    with pytest.raises(
        FileNotFoundError, match="You passed in a path to a DeepSpeed config but the path does not exist"
    ):
        DeepSpeedStrategy(config="invalid_path.json")


@RunIf(deepspeed=True)
def test_deepspeed_with_env_path(tmp_path, monkeypatch, deepspeed_config):
    """Test to ensure if we pass an env variable, we load the config from the path."""
    config_path = tmp_path / "temp.json"
    with open(config_path, "w") as f:
        f.write(json.dumps(deepspeed_config))
    monkeypatch.setenv("PL_DEEPSPEED_CONFIG_PATH", str(config_path))
    strategy = DeepSpeedStrategy()
    assert strategy.config == deepspeed_config


@RunIf(deepspeed=True)
def test_deepspeed_defaults():
    """Ensure that defaults are correctly set as a config for DeepSpeed if no arguments are passed."""
    strategy = DeepSpeedStrategy()
    assert strategy.config is not None
    assert isinstance(strategy.config["zero_optimization"], dict)
    assert strategy._backward_sync_control is None


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


@RunIf(deepspeed=True)
def test_deepspeed_config_zero_offload(deepspeed_zero_config):
    """Test the various ways optimizer-offloading can be configured."""
    # default config
    strategy = DeepSpeedStrategy(config=deepspeed_zero_config)
    assert "offload_optimizer" not in strategy.config["zero_optimization"]

    # default config
    strategy = DeepSpeedStrategy()
    assert "offload_optimizer" not in strategy.config["zero_optimization"]

    # default config with `offload_optimizer` argument override
    strategy = DeepSpeedStrategy(offload_optimizer=True)
    assert strategy.config["zero_optimization"]["offload_optimizer"] == {
        "buffer_count": 4,
        "device": "cpu",
        "nvme_path": "/local_nvme",
        "pin_memory": False,
    }

    # externally configured through config
    deepspeed_zero_config["zero_optimization"]["offload_optimizer"] = False
    strategy = DeepSpeedStrategy(config=deepspeed_zero_config)
    assert strategy.config["zero_optimization"]["offload_optimizer"] is False


@RunIf(deepspeed=True)
@mock.patch("deepspeed.initialize")
def test_deepspeed_setup_module(init_mock):
    """Test that the DeepSpeed strategy can set up the model for inference (no optimizer required)."""
    model = Mock()
    model.parameters.return_value = []
    strategy = DeepSpeedStrategy()
    strategy.parallel_devices = [torch.device("cuda", 1)]
    init_mock.return_value = [Mock()] * 4  # mock to make tuple unpacking work

    strategy.setup_module(model)
    init_mock.assert_called_with(
        args=ANY,
        config=strategy.config,
        model=model,
        model_parameters=ANY,
        optimizer=None,
        lr_scheduler=None,
        dist_init_required=False,
    )


@RunIf(deepspeed=True)
def test_deepspeed_requires_joint_setup():
    """Test that the DeepSpeed strategy does not support setting up model and optimizer independently."""
    strategy = DeepSpeedStrategy()
    with pytest.raises(
        NotImplementedError, match=escape("does not support setting up the module and optimizer(s) independently")
    ):
        strategy.setup_optimizer(Mock())


@RunIf(deepspeed=True)
def test_deepspeed_save_checkpoint_storage_options(tmp_path):
    """Test that the DeepSpeed strategy does not accept storage options for saving checkpoints."""
    strategy = DeepSpeedStrategy()
    with pytest.raises(TypeError, match=escape("DeepSpeedStrategy.save_checkpoint(..., storage_options=...)` is not")):
        strategy.save_checkpoint(path=tmp_path, state=Mock(), storage_options=Mock())


@RunIf(deepspeed=True)
def test_deepspeed_save_checkpoint_one_deepspeed_engine_required(tmp_path):
    """Test that the DeepSpeed strategy can only save one DeepSpeedEngine per checkpoint."""
    from deepspeed import DeepSpeedEngine

    strategy = DeepSpeedStrategy()

    # missing DeepSpeedEngine
    with pytest.raises(ValueError, match="Could not find a DeepSpeed model in the provided checkpoint state."):
        strategy.save_checkpoint(path=tmp_path, state={})
    with pytest.raises(ValueError, match="Could not find a DeepSpeed model in the provided checkpoint state."):
        strategy.save_checkpoint(path=tmp_path, state={"model": torch.nn.Linear(3, 3)})

    # multiple DeepSpeedEngine
    model1 = Mock(spec=torch.nn.Module)
    model1.modules.return_value = [Mock(spec=DeepSpeedEngine)]
    model2 = Mock(spec=torch.nn.Module)
    model2.modules.return_value = [Mock(spec=DeepSpeedEngine)]
    with pytest.raises(ValueError, match="Found multiple DeepSpeed engine modules in the given state."):
        strategy.save_checkpoint(path=tmp_path, state={"model1": model1, "model2": model2})


@RunIf(deepspeed=True)
def test_deepspeed_save_checkpoint_client_state_separation(tmp_path):
    """Test that the DeepSpeed engine and optimizer get separated from the client state."""
    from deepspeed import DeepSpeedEngine

    strategy = DeepSpeedStrategy()

    # Model only
    model = Mock(spec=DeepSpeedEngine, optimizer=None)
    model.modules.return_value = [model]
    strategy.save_checkpoint(path=tmp_path, state={"model": model, "test": "data"})
    # the client_state should not contain any deepspeed engine or deepspeed optimizer
    model.save_checkpoint.assert_called_with(
        tmp_path, client_state={"test": "data"}, tag="checkpoint", exclude_frozen_parameters=False
    )

    # Model and optimizer
    optimizer = Mock()
    model = Mock(spec=DeepSpeedEngine, optimizer=optimizer)
    model.modules.return_value = [model]
    strategy.save_checkpoint(path=tmp_path, state={"model": model, "optimizer": optimizer, "test": "data"})
    # the client_state should not contain any deepspeed engine or deepspeed optimizer
    model.save_checkpoint.assert_called_with(
        tmp_path, client_state={"test": "data"}, tag="checkpoint", exclude_frozen_parameters=False
    )


@RunIf(deepspeed=True)
def test_deepspeed_save_checkpoint_warn_colliding_keys(tmp_path):
    """Test that the strategy warns if there are keys in the user dict that collide internally with DeepSpeed."""
    from deepspeed import DeepSpeedEngine

    strategy = DeepSpeedStrategy()
    optimizer = Mock()
    model = Mock(spec=DeepSpeedEngine, optimizer=optimizer)
    model.modules.return_value = [model]
    # `mp_world_size` is an internal key
    with pytest.warns(UserWarning, match="Your state has keys that collide with DeepSpeed's internal"):
        strategy.save_checkpoint(path=tmp_path, state={"model": model, "optimizer": optimizer, "mp_world_size": 2})


@RunIf(deepspeed=True)
@pytest.mark.parametrize("exclude_frozen_parameters", [True, False])
def test_deepspeed_save_checkpoint_exclude_frozen_parameters(exclude_frozen_parameters):
    """Test that the DeepSpeed strategy can save checkpoints with the `exclude_frozen_parameters` argument."""
    from deepspeed import DeepSpeedEngine

    strategy = DeepSpeedStrategy(exclude_frozen_parameters=exclude_frozen_parameters)
    assert strategy.exclude_frozen_parameters is exclude_frozen_parameters

    model = Mock(spec=DeepSpeedEngine, optimizer=None)
    model.modules.return_value = [model]
    strategy.save_checkpoint(path="test_path", state={"model": model, "extra": "data"})

    model.save_checkpoint.assert_called_with(
        "test_path",
        client_state={"extra": "data"},
        tag="checkpoint",
        exclude_frozen_parameters=exclude_frozen_parameters,
    )


@RunIf(deepspeed=True)
def test_deepspeed_load_checkpoint_validate_path(tmp_path):
    """Test that we validate the checkpoint path for a DeepSpeed checkpoint and give suggestions for user error."""
    strategy = DeepSpeedStrategy()
    with pytest.raises(FileNotFoundError, match="The provided path is not a valid DeepSpeed checkpoint"):
        strategy.load_checkpoint(path=tmp_path, state={"model": Mock()})

    # User tries to pass the subfolder as the path
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    with pytest.raises(FileNotFoundError, match=f"Try to load using this parent directory instead: {tmp_path}"):
        strategy.load_checkpoint(path=checkpoint_path, state={"model": Mock()})

    # User tries to pass an individual file inside the checkpoint folder
    checkpoint_path = checkpoint_path / "zero_pp_rank_0_mp_rank_00_model_states.pt"
    checkpoint_path.touch()
    with pytest.raises(FileNotFoundError, match=f"Try to load using this parent directory instead: {tmp_path}"):
        strategy.load_checkpoint(path=checkpoint_path, state={"model": Mock()})


def test_validate_checkpoint_directory_remote_uri():
    """Test that the checkpoint path validation supports remote URIs (e.g., S3, HDFS) without mangling the scheme."""
    from lightning.fabric.strategies.deepspeed import _validate_checkpoint_directory

    fs = Mock()
    fs.isdir = Mock(side_effect=lambda p: p in {"s3://bucket/ckpt", "s3://bucket/ckpt/checkpoint"})
    fs.isfile = Mock(side_effect=lambda p: p == "s3://bucket/ckpt/checkpoint/model_states.pt")

    with mock.patch("lightning.fabric.strategies.deepspeed.get_filesystem", return_value=fs):
        # A valid remote checkpoint passes validation
        _validate_checkpoint_directory("s3://bucket/ckpt")

        # User tries to pass the subfolder as the path
        with pytest.raises(FileNotFoundError, match="parent directory instead: s3://bucket/ckpt$"):
            _validate_checkpoint_directory("s3://bucket/ckpt/checkpoint")

        # User tries to pass an individual file inside the checkpoint folder
        with pytest.raises(FileNotFoundError, match="parent directory instead: s3://bucket/ckpt$"):
            _validate_checkpoint_directory("s3://bucket/ckpt/checkpoint/model_states.pt")

    # The scheme must not get mangled (pathlib.Path would collapse "s3://" to "s3:/")
    fs.isdir.assert_any_call("s3://bucket/ckpt")
    fs.isdir.assert_any_call("s3://bucket/ckpt/checkpoint")


def test_validate_checkpoint_directory_local_path(tmp_path):
    """Test that the fsspec-based checkpoint path validation still handles local paths."""
    from lightning.fabric.strategies.deepspeed import _validate_checkpoint_directory

    with pytest.raises(FileNotFoundError, match="The provided path is not a valid DeepSpeed checkpoint"):
        _validate_checkpoint_directory(tmp_path)

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    _validate_checkpoint_directory(tmp_path)  # now a valid DeepSpeed checkpoint

    with pytest.raises(
        FileNotFoundError, match=f"Try to load using this parent directory instead: {escape(str(tmp_path))}"
    ):
        _validate_checkpoint_directory(checkpoint_dir)

    checkpoint_file = checkpoint_dir / "zero_pp_rank_0_mp_rank_00_model_states.pt"
    checkpoint_file.touch()
    with pytest.raises(
        FileNotFoundError, match=f"Try to load using this parent directory instead: {escape(str(tmp_path))}"
    ):
        _validate_checkpoint_directory(checkpoint_file)


@RunIf(deepspeed=True)
def test_deepspeed_load_checkpoint_no_state(tmp_path):
    """Test that DeepSpeed can't load the full state without access to a model instance from the user."""
    strategy = DeepSpeedStrategy()
    with pytest.raises(ValueError, match=escape("Got DeepSpeedStrategy.load_checkpoint(..., state=None")):
        strategy.load_checkpoint(path=tmp_path, state=None)
    with pytest.raises(ValueError, match=escape("Got DeepSpeedStrategy.load_checkpoint(..., state={})")):
        strategy.load_checkpoint(path=tmp_path, state={})


@RunIf(deepspeed=True)
@mock.patch("lightning.fabric.strategies.deepspeed._is_deepspeed_checkpoint", return_value=True)
def test_deepspeed_load_checkpoint_one_deepspeed_engine_required(_, tmp_path):
    """Test that the DeepSpeed strategy can only load one DeepSpeedEngine per checkpoint."""
    from deepspeed import DeepSpeedEngine

    strategy = DeepSpeedStrategy()

    # missing DeepSpeedEngine
    with pytest.raises(ValueError, match="Could not find a DeepSpeed model in the provided checkpoint state."):
        strategy.load_checkpoint(path=tmp_path, state={"other": "data"})
    with pytest.raises(ValueError, match="Could not find a DeepSpeed model in the provided checkpoint state."):
        strategy.load_checkpoint(path=tmp_path, state={"model": torch.nn.Linear(3, 3)})

    # multiple DeepSpeedEngine
    model1 = Mock(spec=torch.nn.Module)
    model1.modules.return_value = [Mock(spec=DeepSpeedEngine)]
    model2 = Mock(spec=torch.nn.Module)
    model2.modules.return_value = [Mock(spec=DeepSpeedEngine)]
    with pytest.raises(ValueError, match="Found multiple DeepSpeed engine modules in the given state."):
        strategy.load_checkpoint(path=tmp_path, state={"model1": model1, "model2": model2})


@RunIf(deepspeed=True)
def test_deepspeed_load_checkpoint_client_state_missing(tmp_path):
    """Test that the DeepSpeed strategy raises a custom error when client state couldn't be loaded by DeepSpeed."""
    from deepspeed import DeepSpeedEngine

    strategy = DeepSpeedStrategy()
    optimizer = Mock()
    model = Mock(spec=DeepSpeedEngine, optimizer=optimizer)
    model.modules.return_value = [model]

    # If the DeepSpeed engine fails to load the checkpoint file (e.g., file not found), it prints a warning and
    # returns None from its function call
    model.load_checkpoint.return_value = [None, None]

    # Check for our custom user error
    with pytest.raises(FileNotFoundError, match="The provided path is not a valid DeepSpeed checkpoint"):
        strategy.load_checkpoint(path=tmp_path, state={"model": model, "optimizer": optimizer, "test": "data"})


@RunIf(deepspeed=True)
@mock.patch("lightning.fabric.strategies.deepspeed._is_deepspeed_checkpoint", return_value=True)
def test_deepspeed_load_checkpoint_state_updated_with_client_state(_, tmp_path):
    """Test that the DeepSpeed strategy properly updates the state variables and returns additional metadata."""
    from deepspeed import DeepSpeedEngine

    strategy = DeepSpeedStrategy()
    optimizer = Mock()
    model = Mock(spec=DeepSpeedEngine, optimizer=optimizer)
    model.modules.return_value = [model]

    # the client state contains the additional user data that was proveded when saving, plus some deepspeed metadata
    loaded_client_state = {"user_data": {"iteration": 5}, "deepspeed_metadata": "data"}
    model.load_checkpoint.return_value = [None, loaded_client_state]

    state = {"model": model, "user_data": {"iteration": 0}}
    metadata = strategy.load_checkpoint(path=tmp_path, state=state)

    # the user's state gets updated with the loaded value
    assert state == {"model": model, "user_data": {"iteration": 5}}
    # additional metadata gets separated from client state
    assert metadata == {"deepspeed_metadata": "data"}


@RunIf(deepspeed=True)
@pytest.mark.parametrize("optimzer_state_requested", [True, False])
@mock.patch("lightning.fabric.strategies.deepspeed._is_deepspeed_checkpoint", return_value=True)
def test_deepspeed_load_checkpoint_optimzer_state_requested(_, optimzer_state_requested, tmp_path):
    """Test that the DeepSpeed strategy loads the optimizer state only when requested."""
    from deepspeed import DeepSpeedEngine

    strategy = DeepSpeedStrategy()
    optimizer = Mock(spec=Optimizer)
    model = Mock(spec=DeepSpeedEngine, optimizer=optimizer)
    model.modules.return_value = [model]

    # required, otherwise mock cannot be unpacked
    model.load_checkpoint.return_value = [None, {}]

    state = {"model": model}
    if optimzer_state_requested:
        state["optimizer"] = optimizer

    strategy.load_checkpoint(path=tmp_path, state=state)
    model.load_checkpoint.assert_called_with(
        tmp_path,
        tag="checkpoint",
        load_optimizer_states=optimzer_state_requested,
        load_lr_scheduler_states=False,
        load_module_strict=True,
    )


@RunIf(deepspeed=True)
@pytest.mark.parametrize("stage", [1, 2, 3])
def test_deepspeed_load_checkpoint_raw_state_dict(stage, tmp_path):
    """Test that the `load_checkpoint` can load raw state dict checkpoints too."""
    strategy = DeepSpeedStrategy(stage=stage)

    model = torch.nn.Linear(3, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    torch.save(model.state_dict(), tmp_path / "model.ckpt")
    torch.save(optimizer.state_dict(), tmp_path / "optimizer.ckpt")

    new_model = torch.nn.Linear(3, 3)
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=2.0)

    strategy.load_checkpoint(tmp_path / "model.ckpt", state=new_model, strict=False)
    assert torch.equal(new_model.weight, model.weight)
    strategy.load_checkpoint(tmp_path / "optimizer.ckpt", state=new_optimizer, strict=False)
    assert new_optimizer.state_dict()["param_groups"][0]["lr"] == 1.0


@RunIf(deepspeed=True)
def test_errors_grad_clipping():
    strategy = DeepSpeedStrategy()
    with pytest.raises(
        NotImplementedError,
        match=(
            "DeepSpeed handles gradient clipping automatically within the optimizer. "
            "Make sure to set the `gradient_clipping` value in your Config."
        ),
    ):
        strategy.clip_gradients_norm(Mock(), Mock(), Mock(), Mock(), Mock())

    with pytest.raises(
        NotImplementedError,
        match=(
            "DeepSpeed handles gradient clipping automatically within the optimizer. "
            "Make sure to set the `gradient_clipping` value in your Config."
        ),
    ):
        strategy.clip_gradients_value(Mock(), Mock(), Mock())


@RunIf(deepspeed=True, mps=False)
def test_deepspeed_save_filter(tmp_path):
    strategy = DeepSpeedStrategy()
    with pytest.raises(TypeError, match="manages the state serialization internally"):
        strategy.save_checkpoint(path=tmp_path, state={}, filter={})


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


@RunIf(deepspeed=True)
def test_deepspeed_universal_checkpointing_config():
    """Ensure universal checkpointing config is set when load_universal_checkpoint=True and clean by default."""
    strategy = DeepSpeedStrategy(load_universal_checkpoint=True)
    assert strategy.load_universal_checkpoint is True
    assert strategy.config.get("checkpoint") == {"load_universal": True}

    default_strategy = DeepSpeedStrategy()
    assert default_strategy.load_universal_checkpoint is False
    assert "checkpoint" not in default_strategy.config


@RunIf(deepspeed=True)
def test_deepspeed_universal_checkpointing_custom_config():
    """Ensure universal checkpointing interacts cleanly with custom DeepSpeed configurations."""
    custom_cfg = {"checkpoint": {"load_universal": True, "tag_validation": "Ignore"}}
    strategy = DeepSpeedStrategy(config=custom_cfg)
    assert strategy.load_universal_checkpoint is True
    assert strategy.config["checkpoint"] == {"load_universal": True, "tag_validation": "Ignore"}

    custom_cfg2 = {"train_batch_size": 16}
    strategy2 = DeepSpeedStrategy(config=custom_cfg2, load_universal_checkpoint=True)
    assert strategy2.load_universal_checkpoint is True
    assert strategy2.config["checkpoint"] == {"load_universal": True}
    assert strategy2.config["train_batch_size"] == 16


@RunIf(deepspeed=True)
def test_deepspeed_universal_checkpointing_conflicts_load_full_weights():
    """Ensure load_full_weights and load_universal_checkpoint cannot both be enabled."""
    with pytest.raises(MisconfigurationException, match="cannot both be set to True"):
        DeepSpeedStrategy(load_full_weights=True, load_universal_checkpoint=True)

    with pytest.raises(MisconfigurationException, match="cannot both be set to True"):
        DeepSpeedStrategy(load_full_weights=True, config={"checkpoint": {"load_universal": True}})


@RunIf(deepspeed=True)
def test_deepspeed_universal_checkpointing_unsupported_version():
    """Ensure an exception is raised when DeepSpeed version is < 0.14.4."""
    with mock.patch("lightning.fabric.strategies.deepspeed._DEEPSPEED_GREATER_EQUAL_0_14_4", False):
        with pytest.raises(MisconfigurationException, match="requires DeepSpeed >= 0.14.4"):
            DeepSpeedStrategy(load_universal_checkpoint=True)


def test_validate_checkpoint_directory_universal_path(tmp_path):
    """Ensure _validate_checkpoint_directory accepts directories with latest_universal."""
    from lightning.fabric.strategies.deepspeed import _validate_checkpoint_directory

    ckpt_dir = tmp_path / "universal_ckpt"
    ckpt_dir.mkdir()
    step_dir = ckpt_dir / "step_1000"
    step_dir.mkdir()
    (ckpt_dir / "latest_universal").write_text("step_1000")

    # Valid parent directory passes
    _validate_checkpoint_directory(ckpt_dir)

    # Subfolder suggests parent directory
    expected_msg = f"Try to load using this parent directory instead: {escape(str(ckpt_dir))}"
    with pytest.raises(FileNotFoundError, match=expected_msg):
        _validate_checkpoint_directory(step_dir)


@RunIf(deepspeed=True)
def test_fabric_deepspeed_load_universal_checkpoint(tmp_path):
    """Ensure Fabric DeepSpeedStrategy passes tag=None when load_universal_checkpoint=True."""
    from deepspeed import DeepSpeedEngine

    ckpt_dir = tmp_path / "universal_ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "latest_universal").write_text("checkpoint")
    (ckpt_dir / "checkpoint").mkdir()

    engine = Mock(spec=DeepSpeedEngine)
    fake_client_state = {"model_state": 123}
    engine.load_checkpoint.return_value = (str(ckpt_dir / "checkpoint"), fake_client_state)

    state = {"model": engine}

    # When load_universal_checkpoint=True, tag=None is passed
    strategy_universal = DeepSpeedStrategy(load_universal_checkpoint=True)
    with mock.patch("lightning.fabric.strategies.deepspeed._get_deepspeed_engines_from_state", return_value=[engine]):
        res = strategy_universal.load_checkpoint(ckpt_dir, state=state)
        assert res == fake_client_state
        engine.load_checkpoint.assert_called_with(
            ckpt_dir,
            tag=None,
            load_optimizer_states=False,
            load_lr_scheduler_states=False,
            load_module_strict=True,
        )

    # When load_universal_checkpoint=False, tag='checkpoint' is passed
    strategy_default = DeepSpeedStrategy()
    with mock.patch("lightning.fabric.strategies.deepspeed._get_deepspeed_engines_from_state", return_value=[engine]):
        res = strategy_default.load_checkpoint(ckpt_dir, state=state)
        assert res == fake_client_state
        engine.load_checkpoint.assert_called_with(
            ckpt_dir,
            tag="checkpoint",
            load_optimizer_states=False,
            load_lr_scheduler_states=False,
            load_module_strict=True,
        )
