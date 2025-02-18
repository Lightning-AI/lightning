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
# limitations under the License
import inspect
import os
import sys
from contextlib import nullcontext
from typing import Any
from unittest import mock
from unittest.mock import Mock

import pytest
import torch
import torch.distributed
from lightning_utilities.test.warning import no_warning_call

import lightning.fabric
from lightning.fabric import Fabric
from lightning.fabric.accelerators import XLAAccelerator
from lightning.fabric.accelerators.accelerator import Accelerator
from lightning.fabric.accelerators.cpu import CPUAccelerator
from lightning.fabric.accelerators.cuda import CUDAAccelerator
from lightning.fabric.accelerators.mps import MPSAccelerator
from lightning.fabric.connector import _Connector
from lightning.fabric.plugins import (
    BitsandbytesPrecision,
    DeepSpeedPrecision,
    DoublePrecision,
    FSDPPrecision,
    HalfPrecision,
    MixedPrecision,
    Precision,
    XLAPrecision,
)
from lightning.fabric.plugins.environments import (
    KubeflowEnvironment,
    LightningEnvironment,
    LSFEnvironment,
    SLURMEnvironment,
    TorchElasticEnvironment,
    XLAEnvironment,
)
from lightning.fabric.plugins.io import TorchCheckpointIO
from lightning.fabric.strategies import (
    DataParallelStrategy,
    DDPStrategy,
    DeepSpeedStrategy,
    FSDPStrategy,
    ModelParallelStrategy,
    SingleDeviceStrategy,
    SingleDeviceXLAStrategy,
    XLAFSDPStrategy,
    XLAStrategy,
)
from lightning.fabric.strategies.ddp import _DDP_FORK_ALIASES
from lightning.fabric.strategies.launchers.subprocess_script import _SubprocessScriptLauncher
from lightning.fabric.utilities.imports import _IS_WINDOWS
from tests_fabric.conftest import mock_tpu_available
from tests_fabric.helpers.runif import RunIf


class DeviceMock(Mock):
    def __instancecheck__(self, instance):
        return True


@pytest.mark.parametrize(
    ("accelerator", "devices"), [("tpu", "auto"), ("tpu", 1), ("tpu", [1]), ("tpu", 8), ("auto", 1), ("auto", 8)]
)
@RunIf(min_python="3.9")  # mocking issue
def test_accelerator_choice_tpu(accelerator, devices, tpu_available, monkeypatch):
    monkeypatch.setattr(torch, "device", DeviceMock())

    connector = _Connector(accelerator=accelerator, devices=devices)
    assert isinstance(connector.accelerator, XLAAccelerator)
    if devices == "auto" or (isinstance(devices, int) and devices > 1):
        assert isinstance(connector.strategy, XLAStrategy)
        assert isinstance(connector.strategy.cluster_environment, XLAEnvironment)
        assert isinstance(connector.cluster_environment, XLAEnvironment)
    else:
        assert isinstance(connector.strategy, SingleDeviceXLAStrategy)


@RunIf(skip_windows=True, standalone=True)
def test_strategy_choice_ddp_on_cpu():
    """Test that selecting DDPStrategy on CPU works."""
    _test_strategy_choice_ddp_and_cpu(ddp_strategy_class=DDPStrategy)


def _test_strategy_choice_ddp_and_cpu(ddp_strategy_class):
    connector = _Connector(
        strategy=ddp_strategy_class(),
        accelerator="cpu",
        devices=2,
    )
    assert isinstance(connector.strategy, ddp_strategy_class)
    assert isinstance(connector.accelerator, CPUAccelerator)
    assert connector.strategy.num_processes == 2
    assert connector.strategy.parallel_devices == [torch.device("cpu")] * 2


@mock.patch.dict(
    os.environ,
    {
        "SLURM_NTASKS": "2",
        "SLURM_JOB_NAME": "SOME_NAME",
        "SLURM_NODEID": "0",
        "LOCAL_RANK": "0",
        "SLURM_PROCID": "0",
        "SLURM_LOCALID": "0",
    },
)
@mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=0)
def test_custom_cluster_environment_in_slurm_environment(_):
    """Test that we choose the custom cluster even when SLURM or TE flags are around."""

    class CustomCluster(LightningEnvironment):
        @property
        def main_address(self):
            return "asdf"

        @property
        def creates_processes_externally(self) -> bool:
            return True

    connector = _Connector(
        plugins=[CustomCluster()],
        accelerator="cpu",
        strategy="ddp",
        devices=2,
    )
    assert isinstance(connector.accelerator, CPUAccelerator)
    assert isinstance(connector.strategy, DDPStrategy)
    assert isinstance(connector.strategy.cluster_environment, CustomCluster)
    # this checks that `strategy._set_world_ranks` was called by the connector
    assert connector.strategy.world_size == 2


@RunIf(mps=False)
@mock.patch.dict(
    os.environ,
    {
        "SLURM_NTASKS": "2",
        "SLURM_NTASKS_PER_NODE": "1",
        "SLURM_JOB_NAME": "SOME_NAME",
        "SLURM_NODEID": "0",
        "LOCAL_RANK": "0",
        "SLURM_PROCID": "0",
        "SLURM_LOCALID": "0",
    },
)
@mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=0)
def test_custom_accelerator(*_):
    class Accel(Accelerator):
        def setup_device(self, device: torch.device) -> None:
            pass

        def get_device_stats(self, device: torch.device) -> dict[str, Any]:
            pass

        def teardown(self) -> None:
            pass

        @staticmethod
        def parse_devices(devices):
            return devices

        @staticmethod
        def get_parallel_devices(devices):
            return [torch.device("cpu")] * devices

        @staticmethod
        def get_device_type() -> str:
            return "cpu"

        @staticmethod
        def auto_device_count() -> int:
            return 1

        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def name() -> str:
            return "custom_acc_name"

    class Prec(Precision):
        pass

    class Strat(SingleDeviceStrategy):
        pass

    strategy = Strat(device=torch.device("cpu"), accelerator=Accel(), precision=Prec())
    connector = _Connector(strategy=strategy, devices=2)
    assert isinstance(connector.accelerator, Accel)
    assert isinstance(connector.strategy, Strat)
    assert isinstance(connector.precision, Prec)
    assert connector.strategy is strategy

    class Strat(DDPStrategy):
        pass

    strategy = Strat(accelerator=Accel(), precision=Prec())
    connector = _Connector(strategy=strategy, devices=2)
    assert isinstance(connector.accelerator, Accel)
    assert isinstance(connector.strategy, Strat)
    assert isinstance(connector.precision, Prec)
    assert connector.strategy is strategy


@pytest.mark.parametrize(
    ("env_vars", "expected_environment"),
    [
        (
            {
                "SLURM_NTASKS": "2",
                "SLURM_NTASKS_PER_NODE": "1",
                "SLURM_JOB_NAME": "SOME_NAME",
                "SLURM_NODEID": "0",
                "LOCAL_RANK": "0",
                "SLURM_PROCID": "0",
                "SLURM_LOCALID": "0",
            },
            SLURMEnvironment,
        ),
        (
            {
                "LSB_JOBID": "1",
                "LSB_DJOB_RANKFILE": "SOME_RANK_FILE",
                "JSM_NAMESPACE_LOCAL_RANK": "1",
                "JSM_NAMESPACE_SIZE": "20",
                "JSM_NAMESPACE_RANK": "1",
            },
            LSFEnvironment,
        ),
    ],
)
@mock.patch("lightning.fabric.plugins.environments.lsf.LSFEnvironment._read_hosts", return_value=["node0", "node1"])
@mock.patch("lightning.fabric.plugins.environments.lsf.LSFEnvironment._get_node_rank", return_value=0)
def test_fallback_from_ddp_spawn_to_ddp_on_cluster(_, __, env_vars, expected_environment):
    with mock.patch.dict(os.environ, env_vars, clear=True):
        connector = _Connector(strategy="ddp_spawn", accelerator="cpu", devices=2)
    assert isinstance(connector.accelerator, CPUAccelerator)
    assert isinstance(connector.strategy, DDPStrategy)
    assert isinstance(connector.strategy.cluster_environment, expected_environment)


@RunIf(mps=False)
@mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=2)
def test_interactive_incompatible_backend_error(_, monkeypatch):
    monkeypatch.setattr(lightning.fabric.connector, "_IS_INTERACTIVE", True)
    with pytest.raises(RuntimeError, match=r"strategy='ddp'\)`.*is not compatible"):
        _Connector(strategy="ddp", accelerator="gpu", devices=2)

    with pytest.raises(RuntimeError, match=r"strategy='ddp_spawn'\)`.*is not compatible"):
        _Connector(strategy="ddp_spawn", accelerator="gpu", devices=2)

    with pytest.raises(RuntimeError, match=r"strategy='ddp'\)`.*is not compatible"):
        # Edge case: _Connector maps dp to ddp if accelerator != gpu
        _Connector(strategy="dp", accelerator="cpu")


def test_precision_and_precision_plugin_raises():
    with pytest.raises(ValueError, match="both `precision=16-true` and `plugins"):
        _Connector(precision="16-true", plugins=Precision())


@mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=2)
@mock.patch("lightning.fabric.accelerators.mps.MPSAccelerator.is_available", return_value=False)
def test_interactive_compatible_dp_strategy_gpu(_, __, monkeypatch):
    monkeypatch.setattr(lightning.fabric.utilities.imports, "_IS_INTERACTIVE", True)
    connector = _Connector(strategy="dp", accelerator="gpu")
    assert connector.strategy.launcher is None


@RunIf(skip_windows=True)
def test_interactive_compatible_strategy_ddp_fork(monkeypatch):
    monkeypatch.setattr(lightning.fabric.utilities.imports, "_IS_INTERACTIVE", True)
    connector = _Connector(strategy="ddp_fork", accelerator="cpu")
    assert connector.strategy.launcher.is_interactive_compatible


@RunIf(mps=True)
@pytest.mark.parametrize(
    ("strategy", "strategy_class"),
    [
        ("ddp", DDPStrategy),
        ("dp", DataParallelStrategy),
        pytest.param("deepspeed", DeepSpeedStrategy, marks=RunIf(deepspeed=True)),
    ],
)
@pytest.mark.parametrize("accelerator", ["mps", "auto", "gpu", MPSAccelerator()])
def test_invalid_ddp_strategy_with_mps(accelerator, strategy, strategy_class):
    with pytest.raises(ValueError, match="strategies from the DDP family are not supported"):
        _Connector(accelerator=accelerator, strategy=strategy)

    with pytest.raises(ValueError, match="strategies from the DDP family are not supported"):
        _Connector(accelerator="mps", strategy=strategy_class())


@RunIf(mps=False)
@pytest.mark.parametrize(
    ("strategy", "strategy_class"),
    [
        ("ddp", DDPStrategy),
        ("ddp_spawn", DDPStrategy),
        pytest.param("deepspeed", DeepSpeedStrategy, marks=RunIf(deepspeed=True)),
    ],
)
@pytest.mark.parametrize("devices", [1, 2])
@mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=2)
def test_strategy_choice_multi_node_gpu(_, strategy, strategy_class, devices):
    connector = _Connector(num_nodes=2, accelerator="gpu", strategy=strategy, devices=devices)
    assert isinstance(connector.strategy, strategy_class)


def test_num_nodes_input_validation():
    with pytest.raises(ValueError, match="`num_nodes` must be a positive integer"):
        _Connector(num_nodes=0)
    with pytest.raises(ValueError, match="`num_nodes` must be a positive integer"):
        _Connector(num_nodes=-1)


@mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=0)
def test_cuda_accelerator_can_not_run_on_system(_):
    connector = _Connector(accelerator="cpu")
    assert isinstance(connector.accelerator, CPUAccelerator)

    with pytest.raises(
        RuntimeError,
        match="CUDAAccelerator` can not run on your system since the accelerator is not available.",
    ):
        _Connector(accelerator="cuda", devices=1)


@pytest.mark.skipif(XLAAccelerator.is_available(), reason="test requires missing TPU")
@mock.patch("lightning.fabric.accelerators.xla._XLA_AVAILABLE", True)
@mock.patch("lightning.fabric.accelerators.xla._using_pjrt", return_value=True)
def test_tpu_accelerator_can_not_run_on_system(_):
    with pytest.raises(RuntimeError, match="XLAAccelerator` can not run on your system"):
        _Connector(accelerator="tpu", devices=8)


@mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=2)
@pytest.mark.parametrize("device_count", [["0"], [0, "1"], ["GPU"], [["0", "1"], [0, 1]], [False]])
def test_accelerator_invalid_type_devices(_, device_count):
    with pytest.raises(TypeError, match=r"must be an int, a string, a sequence of ints, but you"):
        _ = _Connector(accelerator="gpu", devices=device_count)


@RunIf(min_cuda_gpus=1)
def test_accelerator_gpu():
    connector = _Connector(accelerator="gpu", devices=1)
    assert isinstance(connector.accelerator, CUDAAccelerator)

    connector = _Connector(accelerator="gpu")
    assert isinstance(connector.accelerator, CUDAAccelerator)

    connector = _Connector(accelerator="auto", devices=1)
    assert isinstance(connector.accelerator, CUDAAccelerator)


@pytest.mark.parametrize(("devices", "strategy_class"), [(1, SingleDeviceStrategy), (5, DDPStrategy)])
def test_accelerator_cpu_with_devices(devices, strategy_class):
    connector = _Connector(accelerator="cpu", devices=devices)
    assert connector._parallel_devices == [torch.device("cpu")] * devices
    assert isinstance(connector.strategy, strategy_class)
    assert isinstance(connector.accelerator, CPUAccelerator)


@RunIf(min_cuda_gpus=2)
@pytest.mark.parametrize(
    ("devices", "strategy_class"), [(1, SingleDeviceStrategy), ([1], SingleDeviceStrategy), (2, DDPStrategy)]
)
def test_accelerator_gpu_with_devices(devices, strategy_class):
    connector = _Connector(accelerator="gpu", devices=devices)
    assert len(connector._parallel_devices) == len(devices) if isinstance(devices, list) else devices
    assert isinstance(connector.strategy, strategy_class)
    assert isinstance(connector.accelerator, CUDAAccelerator)


@RunIf(min_cuda_gpus=1)
def test_accelerator_auto_with_devices_gpu():
    connector = _Connector(accelerator="auto", devices=1)
    assert isinstance(connector.accelerator, CUDAAccelerator)
    assert connector._parallel_devices == [torch.device("cuda", 0)]


def test_set_devices_if_none_cpu():
    connector = _Connector(accelerator="cpu", devices=3)
    assert connector._parallel_devices == [torch.device("cpu")] * 3


@RunIf(mps=False)
def test_unsupported_strategy_types_on_cpu_and_fallback():
    with pytest.warns(UserWarning, match="is not supported on CPUs, hence setting `strategy='ddp"):
        connector = _Connector(accelerator="cpu", strategy="dp", devices=2)
    assert isinstance(connector.strategy, DDPStrategy)


def test_invalid_accelerator_choice():
    with pytest.raises(ValueError, match="You selected an invalid accelerator name: `accelerator='cocofruit'`"):
        _Connector(accelerator="cocofruit")


@pytest.mark.parametrize("invalid_strategy", ["cocofruit", object()])
def test_invalid_strategy_choice(invalid_strategy):
    with pytest.raises(ValueError, match="You selected an invalid strategy name:"):
        _Connector(strategy=invalid_strategy)


@pytest.mark.parametrize(
    ("strategy", "strategy_class"),
    [
        ("ddp_spawn", DDPStrategy),
        ("ddp", DDPStrategy),
    ],
)
def test_strategy_choice_cpu_str(strategy, strategy_class):
    connector = _Connector(strategy=strategy, accelerator="cpu", devices=2)
    assert isinstance(connector.strategy, strategy_class)


@RunIf(min_cuda_gpus=2)
@pytest.mark.parametrize(
    ("strategy", "strategy_class"),
    [
        ("ddp_spawn", DDPStrategy),
        ("ddp", DDPStrategy),
        ("dp", DataParallelStrategy),
        pytest.param("deepspeed", DeepSpeedStrategy, marks=RunIf(deepspeed=True)),
    ],
)
def test_strategy_choice_gpu_str(strategy, strategy_class):
    connector = _Connector(strategy=strategy, accelerator="gpu", devices=2)
    assert isinstance(connector.strategy, strategy_class)


def test_device_type_when_strategy_instance_cpu_passed():
    connector = _Connector(strategy=DDPStrategy(), accelerator="cpu", devices=2)
    assert isinstance(connector.strategy, DDPStrategy)
    assert isinstance(connector.accelerator, CPUAccelerator)


@RunIf(min_cuda_gpus=2)
def test_device_type_when_strategy_instance_gpu_passed():
    connector = _Connector(strategy=DDPStrategy(), accelerator="gpu", devices=2)
    assert isinstance(connector.strategy, DDPStrategy)
    assert isinstance(connector.accelerator, CUDAAccelerator)


@pytest.mark.parametrize("precision", [1, 12, "invalid"])
def test_validate_precision_type(precision):
    with pytest.raises(ValueError, match=f"Precision {repr(precision)} is invalid"):
        _Connector(precision=precision)


@pytest.mark.parametrize(
    ("precision", "expected_precision", "should_warn"),
    [
        (16, "16-mixed", True),
        ("16", "16-mixed", True),
        ("16-mixed", "16-mixed", False),
        ("bf16", "bf16-mixed", True),
        ("bf16-mixed", "bf16-mixed", False),
        (32, "32-true", False),
        ("32", "32-true", False),
        ("32-true", "32-true", False),
        (64, "64-true", False),
        ("64", "64-true", False),
        ("64-true", "64-true", False),
    ],
)
# mock cuda as available to not be limited by dtype and accelerator compatibility - this is tested elsewhere
@mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=1)
@mock.patch("lightning.fabric.accelerators.mps.MPSAccelerator.is_available", return_value=False)
def test_precision_conversion(patch1, patch2, precision, expected_precision, should_warn):
    warn_context = pytest.warns if should_warn else no_warning_call
    with warn_context(
        UserWarning,
        match=(
            f"{precision}` is supported for historical reasons but its usage is discouraged. "
            f"Please set your precision to {expected_precision} instead!"
        ),
    ):
        connector = _Connector(precision=precision, accelerator="cuda")
    assert connector._precision_input == expected_precision


def test_multi_device_default_strategy():
    """The default strategy when multiple devices are selected is "ddp" with the subprocess launcher."""
    connector = _Connector(strategy="auto", accelerator="cpu", devices=2)
    assert isinstance(connector.accelerator, CPUAccelerator)
    assert isinstance(connector.strategy, DDPStrategy)
    assert connector.strategy._start_method == "popen"
    assert isinstance(connector.strategy.launcher, _SubprocessScriptLauncher)


def test_strategy_choice_ddp_spawn_cpu():
    connector = _Connector(strategy="ddp_spawn", accelerator="cpu", devices=2)
    assert isinstance(connector.accelerator, CPUAccelerator)
    assert isinstance(connector.strategy, DDPStrategy)
    assert isinstance(connector.strategy.cluster_environment, LightningEnvironment)
    assert connector.strategy._start_method == "spawn"
    assert connector.strategy.launcher._start_method == "spawn"


@RunIf(skip_windows=True)
@mock.patch("lightning.fabric.connector._IS_INTERACTIVE", True)
def test_strategy_choice_ddp_fork_in_interactive():
    """Test that when strategy is unspecified, the connector chooses DDP Fork in interactive environments by
    default."""
    connector = _Connector(accelerator="cpu", devices=2)
    assert isinstance(connector.accelerator, CPUAccelerator)
    assert isinstance(connector.strategy, DDPStrategy)
    assert isinstance(connector.strategy.cluster_environment, LightningEnvironment)
    assert connector.strategy._start_method == "fork"
    assert connector.strategy.launcher._start_method == "fork"


@RunIf(skip_windows=True)
def test_strategy_choice_ddp_fork_cpu():
    connector = _Connector(strategy="ddp_fork", accelerator="cpu", devices=2)
    assert isinstance(connector.accelerator, CPUAccelerator)
    assert isinstance(connector.strategy, DDPStrategy)
    assert isinstance(connector.strategy.cluster_environment, LightningEnvironment)
    assert connector.strategy._start_method == "fork"
    assert connector.strategy.launcher._start_method == "fork"


@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"})
@mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=2)
@mock.patch("lightning.fabric.accelerators.mps.MPSAccelerator.is_available", return_value=False)
def test_strategy_choice_ddp(*_):
    connector = _Connector(strategy="ddp", accelerator="gpu", devices=1)
    assert isinstance(connector.accelerator, CUDAAccelerator)
    assert isinstance(connector.strategy, DDPStrategy)
    assert isinstance(connector.strategy.cluster_environment, LightningEnvironment)


@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"})
@mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=2)
@mock.patch("lightning.fabric.accelerators.mps.MPSAccelerator.is_available", return_value=False)
def test_strategy_choice_ddp_spawn(*_):
    connector = _Connector(strategy="ddp_spawn", accelerator="gpu", devices=1)
    assert isinstance(connector.accelerator, CUDAAccelerator)
    assert isinstance(connector.strategy, DDPStrategy)
    assert isinstance(connector.strategy.cluster_environment, LightningEnvironment)


@mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=2)
@pytest.mark.parametrize(
    ("job_name", "expected_env"), [("some_name", SLURMEnvironment), ("bash", LightningEnvironment)]
)
@pytest.mark.parametrize("strategy", ["auto", "ddp", DDPStrategy])
def test_strategy_choice_ddp_slurm(_, strategy, job_name, expected_env):
    if strategy and not isinstance(strategy, str):
        strategy = strategy()

    with mock.patch.dict(
        os.environ,
        {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "SLURM_NTASKS": "2",
            "SLURM_NTASKS_PER_NODE": "1",
            "SLURM_JOB_NAME": job_name,
            "SLURM_NODEID": "0",
            "SLURM_PROCID": "1",
            "SLURM_LOCALID": "1",
        },
    ):
        connector = _Connector(strategy=strategy, accelerator="cuda", devices=2)
        assert isinstance(connector.accelerator, CUDAAccelerator)
        assert isinstance(connector.strategy, DDPStrategy)
        assert isinstance(connector.strategy.cluster_environment, expected_env)


@mock.patch.dict(
    os.environ,
    {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "WORLD_SIZE": "2",
        "LOCAL_WORLD_SIZE": "2",
        "RANK": "1",
        "LOCAL_RANK": "1",
        "GROUP_RANK": "0",
        "TORCHELASTIC_RUN_ID": "1",
    },
)
@mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=2)
@mock.patch("lightning.fabric.accelerators.mps.MPSAccelerator.is_available", return_value=False)
def test_strategy_choice_ddp_torchelastic(*_):
    connector = _Connector(accelerator="gpu", devices=2)
    assert isinstance(connector.accelerator, CUDAAccelerator)
    assert isinstance(connector.strategy, DDPStrategy)
    assert isinstance(connector.strategy.cluster_environment, TorchElasticEnvironment)
    assert connector.strategy.cluster_environment.local_rank() == 1
    assert connector.strategy.local_rank == 1


@mock.patch.dict(
    os.environ,
    {
        "TORCHELASTIC_RUN_ID": "1",
        "SLURM_NTASKS": "2",
        "WORLD_SIZE": "2",
        "RANK": "1",
        "LOCAL_RANK": "1",
    },
)
@mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=2)
@mock.patch("lightning.fabric.accelerators.mps.MPSAccelerator.is_available", return_value=False)
def test_torchelastic_priority_over_slurm(*_):
    """Test that the TorchElastic cluster environment is chosen over SLURM when both are detected."""
    assert TorchElasticEnvironment.detect()
    assert SLURMEnvironment.detect()
    connector = _Connector(strategy="ddp")
    assert isinstance(connector.strategy.cluster_environment, TorchElasticEnvironment)


@mock.patch.dict(
    os.environ,
    {
        "CUDA_VISIBLE_DEVICES": "0",
        "KUBERNETES_PORT": "tcp://127.0.0.1:443",
        "MASTER_ADDR": "1.2.3.4",
        "MASTER_PORT": "500",
        "WORLD_SIZE": "20",
        "RANK": "1",
    },
)
@mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=2)
@mock.patch("lightning.fabric.accelerators.mps.MPSAccelerator.is_available", return_value=False)
def test_strategy_choice_ddp_kubeflow(*_):
    connector = _Connector(accelerator="gpu", devices=2, plugins=KubeflowEnvironment())
    assert isinstance(connector.accelerator, CUDAAccelerator)
    assert isinstance(connector.strategy, DDPStrategy)
    assert isinstance(connector.strategy.cluster_environment, KubeflowEnvironment)
    assert connector.strategy.cluster_environment.local_rank() == 0
    assert connector.strategy.local_rank == 0


@mock.patch.dict(
    os.environ,
    {
        "KUBERNETES_PORT": "tcp://127.0.0.1:443",
        "MASTER_ADDR": "1.2.3.4",
        "MASTER_PORT": "500",
        "WORLD_SIZE": "20",
        "RANK": "1",
    },
)
def test_strategy_choice_ddp_cpu_kubeflow():
    connector = _Connector(accelerator="cpu", devices=2, plugins=KubeflowEnvironment())
    assert isinstance(connector.accelerator, CPUAccelerator)
    assert isinstance(connector.strategy, DDPStrategy)
    assert isinstance(connector.strategy.cluster_environment, KubeflowEnvironment)
    assert connector.strategy.cluster_environment.local_rank() == 0
    assert connector.strategy.local_rank == 0


@mock.patch.dict(
    os.environ,
    {
        "SLURM_NTASKS": "2",
        "SLURM_NTASKS_PER_NODE": "1",
        "SLURM_JOB_NAME": "SOME_NAME",
        "SLURM_NODEID": "0",
        "LOCAL_RANK": "0",
        "SLURM_PROCID": "0",
        "SLURM_LOCALID": "0",
    },
)
@pytest.mark.parametrize("strategy", ["auto", "ddp", DDPStrategy()])
def test_strategy_choice_ddp_cpu_slurm(strategy):
    connector = _Connector(strategy=strategy, accelerator="cpu", devices=2)
    assert isinstance(connector.accelerator, CPUAccelerator)
    assert isinstance(connector.strategy, DDPStrategy)
    assert isinstance(connector.strategy.cluster_environment, SLURMEnvironment)
    assert connector.strategy.local_rank == 0


@mock.patch.dict(os.environ, {}, clear=True)
@mock.patch("lightning.fabric.accelerators.mps.MPSAccelerator.is_available", return_value=False)
def test_unsupported_tpu_choice(_, tpu_available):
    # if user didn't set strategy, _Connector will choose the SingleDeviceXLAStrategy or XLAStrategy
    with pytest.raises(ValueError, match="XLAAccelerator` can only be used with a `SingleDeviceXLAStrategy`"):
        _Connector(accelerator="tpu", precision="16-true", strategy="ddp")

    # wrong precision plugin type
    with pytest.raises(TypeError, match="can only work with the `XLAPrecision` plugin"):
        XLAStrategy(accelerator=XLAAccelerator(), precision=Precision())

    # wrong strategy type
    strategy = DDPStrategy(accelerator=XLAAccelerator(), precision=XLAPrecision(precision="16-true"))
    with pytest.raises(ValueError, match="XLAAccelerator` can only be used with a `SingleDeviceXLAStrategy`"):
        _Connector(strategy=strategy)


@RunIf(skip_windows=True)
def test_connector_with_tpu_accelerator_instance(tpu_available, monkeypatch):
    monkeypatch.setattr(torch, "device", DeviceMock())

    accelerator = XLAAccelerator()
    connector = _Connector(accelerator=accelerator, devices=1)
    assert connector.accelerator is accelerator
    assert isinstance(connector.strategy, SingleDeviceXLAStrategy)

    connector = _Connector(accelerator=accelerator)
    assert connector.accelerator is accelerator
    assert isinstance(connector.strategy, XLAStrategy)


@RunIf(mps=True)
def test_devices_auto_choice_mps():
    connector = _Connector(accelerator="auto", devices="auto")
    assert isinstance(connector.accelerator, MPSAccelerator)
    assert isinstance(connector.strategy, SingleDeviceStrategy)
    assert connector.strategy.root_device == torch.device("mps", 0)
    assert connector._parallel_devices == [torch.device("mps", 0)]


@pytest.mark.parametrize(
    ("parallel_devices", "accelerator"),
    [([torch.device("cpu")], "cuda"), ([torch.device("cuda", i) for i in range(8)], "tpu")],
)
def test_parallel_devices_in_strategy_conflict_with_accelerator(parallel_devices, accelerator):
    with pytest.raises(ValueError, match=r"parallel_devices set through"):
        _Connector(strategy=DDPStrategy(parallel_devices=parallel_devices), accelerator=accelerator)


@pytest.mark.parametrize(
    ("plugins", "expected"),
    [
        ([LightningEnvironment(), SLURMEnvironment()], "ClusterEnvironment"),
        ([TorchCheckpointIO(), TorchCheckpointIO()], "CheckpointIO"),
        (
            [Precision(), DoublePrecision(), LightningEnvironment(), SLURMEnvironment()],
            "Precision, ClusterEnvironment",
        ),
    ],
)
def test_plugin_only_one_instance_for_one_type(plugins, expected):
    with pytest.raises(ValueError, match=f"Received multiple values for {expected}"):
        _Connector(plugins=plugins)


@pytest.mark.parametrize("accelerator", ["cpu", "cuda", "mps", "tpu"])
@pytest.mark.parametrize("devices", ["0", 0, []])
def test_passing_zero_and_empty_list_to_devices_flag(accelerator, devices):
    with pytest.raises(ValueError, match="value is not a valid input using"):
        _Connector(accelerator=accelerator, devices=devices)


@pytest.mark.parametrize(
    ("expected_accelerator_flag", "expected_accelerator_class"),
    [
        pytest.param("cuda", CUDAAccelerator, marks=RunIf(min_cuda_gpus=1)),
        pytest.param("mps", MPSAccelerator, marks=RunIf(mps=True)),
    ],
)
def test_gpu_accelerator_backend_choice(expected_accelerator_flag, expected_accelerator_class):
    connector = _Connector(accelerator="gpu")
    assert connector._accelerator_flag == expected_accelerator_flag
    assert isinstance(connector.accelerator, expected_accelerator_class)


@mock.patch("lightning.fabric.accelerators.mps.MPSAccelerator.is_available", return_value=False)
@mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=1)
def test_gpu_accelerator_backend_choice_cuda(*_):
    connector = _Connector(accelerator="gpu")
    assert connector._accelerator_flag == "cuda"
    assert isinstance(connector.accelerator, CUDAAccelerator)


@mock.patch("lightning.fabric.accelerators.mps.MPSAccelerator.is_available", return_value=True)
@mock.patch("lightning.fabric.accelerators.mps._get_all_available_mps_gpus", return_value=[0])
@mock.patch("torch.device", DeviceMock)
def test_gpu_accelerator_backend_choice_mps(*_: object) -> object:
    connector = _Connector(accelerator="gpu")
    assert connector._accelerator_flag == "mps"
    assert isinstance(connector.accelerator, MPSAccelerator)


@mock.patch("lightning.fabric.accelerators.mps.MPSAccelerator.is_available", return_value=False)
@mock.patch("lightning.fabric.accelerators.cuda.CUDAAccelerator.is_available", return_value=False)
def test_gpu_accelerator_no_gpu_backend_found_error(*_):
    with pytest.raises(RuntimeError, match="No supported gpu backend found!"):
        _Connector(accelerator="gpu")


@pytest.mark.parametrize("strategy", _DDP_FORK_ALIASES)
@mock.patch(
    "lightning.fabric.connector.torch.multiprocessing.get_all_start_methods",
    return_value=[],
)
@mock.patch("lightning.fabric.accelerators.mps.MPSAccelerator.is_available", return_value=False)
def test_ddp_fork_on_unsupported_platform(_, __, strategy):
    with pytest.raises(ValueError, match="process forking is not supported on this platform"):
        _Connector(strategy=strategy)


@pytest.mark.parametrize(
    ("precision_str", "strategy_str", "expected_precision_cls"),
    [
        ("64-true", "auto", DoublePrecision),
        ("32-true", "auto", Precision),
        ("16-true", "auto", HalfPrecision),
        ("bf16-true", "auto", HalfPrecision),
        ("16-mixed", "auto", MixedPrecision),
        ("bf16-mixed", "auto", MixedPrecision),
        pytest.param("32-true", "fsdp", FSDPPrecision, marks=RunIf(min_cuda_gpus=1)),
        pytest.param("16-true", "fsdp", FSDPPrecision, marks=RunIf(min_cuda_gpus=1)),
        pytest.param("bf16-true", "fsdp", FSDPPrecision, marks=RunIf(min_cuda_gpus=1)),
        pytest.param("16-mixed", "fsdp", FSDPPrecision, marks=RunIf(min_cuda_gpus=1)),
        pytest.param("bf16-mixed", "fsdp", FSDPPrecision, marks=RunIf(min_cuda_gpus=1)),
        pytest.param("32-true", "deepspeed", DeepSpeedPrecision, marks=RunIf(deepspeed=True, mps=False)),
        pytest.param("16-true", "deepspeed", DeepSpeedPrecision, marks=RunIf(deepspeed=True, mps=False)),
        pytest.param("bf16-true", "deepspeed", DeepSpeedPrecision, marks=RunIf(deepspeed=True, mps=False)),
        pytest.param("16-mixed", "deepspeed", DeepSpeedPrecision, marks=RunIf(deepspeed=True, mps=False)),
        pytest.param("bf16-mixed", "deepspeed", DeepSpeedPrecision, marks=RunIf(deepspeed=True, mps=False)),
    ],
)
def test_precision_selection(precision_str, strategy_str, expected_precision_cls):
    connector = _Connector(precision=precision_str, strategy=strategy_str)
    assert isinstance(connector.precision, expected_precision_cls)


def test_precision_selection_16_on_cpu_warns():
    with pytest.warns(
        UserWarning,
        match=r"precision='16-mixed'\)` but AMP with fp16 is not supported on CPU. Using `precision='bf16-mixed'",
    ):
        _Connector(accelerator="cpu", precision="16-mixed")


class MyAMP(MixedPrecision):
    pass


@RunIf(mps=False)
@pytest.mark.parametrize(("strategy", "devices"), [("ddp", 2), ("ddp_spawn", 2)])
@pytest.mark.parametrize(
    ("is_custom_plugin", "plugin_cls"),
    [(False, MixedPrecision), (True, MyAMP)],
)
def test_precision_selection_amp_ddp(strategy, devices, is_custom_plugin, plugin_cls):
    plugin = None
    precision = None
    if is_custom_plugin:
        plugin = plugin_cls("16-mixed", "cpu")
    else:
        precision = "16-mixed"
    connector = _Connector(
        accelerator="cpu",
        precision=precision,
        devices=devices,
        strategy=strategy,
        plugins=plugin,
    )
    assert isinstance(connector.precision, plugin_cls)


@RunIf(min_torch="2.4")
@pytest.mark.parametrize(
    ("precision", "raises"),
    [("32-true", False), ("16-true", False), ("bf16-true", False), ("16-mixed", True), ("bf16-mixed", False)],
)
@mock.patch("lightning.fabric.accelerators.mps.MPSAccelerator.is_available", return_value=False)
def test_precision_selection_model_parallel(_, precision, raises):
    error_context = pytest.raises(ValueError, match=f"does not support .*{precision}") if raises else nullcontext()
    with error_context:
        _Connector(precision=precision, strategy=ModelParallelStrategy(lambda x, _: x))


def test_bitsandbytes_precision_cuda_required(monkeypatch):
    monkeypatch.setattr(lightning.fabric.plugins.precision.bitsandbytes, "_BITSANDBYTES_AVAILABLE", True)
    monkeypatch.setitem(sys.modules, "bitsandbytes", Mock())
    with pytest.raises(RuntimeError, match="Bitsandbytes is only supported on CUDA GPUs"):
        _Connector(accelerator="cpu", plugins=BitsandbytesPrecision(mode="int8"))


@pytest.mark.parametrize(("strategy", "strategy_cls"), [("DDP", DDPStrategy), ("Ddp", DDPStrategy)])
@mock.patch("lightning.fabric.accelerators.mps.MPSAccelerator.is_available", return_value=False)
def test_strategy_str_passed_being_case_insensitive(_, strategy, strategy_cls):
    connector = _Connector(strategy=strategy)
    assert isinstance(connector.strategy, strategy_cls)


@pytest.mark.parametrize(
    ("precision", "expected"),
    [
        (None, Precision),
        ("64-true", DoublePrecision),
        ("32-true", Precision),
        ("16-true", HalfPrecision),
        ("16-mixed", MixedPrecision),
    ],
)
@mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=1)
def test_precision_from_environment(_, precision, expected):
    """Test that the precision input can be set through the environment variable."""
    env_vars = {"LT_CLI_USED": "1"}
    if precision is not None:
        env_vars["LT_PRECISION"] = precision
    with mock.patch.dict(os.environ, env_vars):
        connector = _Connector(accelerator="cuda")  # need to use cuda, because AMP not available on CPU
    assert isinstance(connector.precision, expected)


@pytest.mark.parametrize(
    ("accelerator", "strategy", "expected_accelerator", "expected_strategy"),
    [
        (None, None, CPUAccelerator, SingleDeviceStrategy),
        ("cpu", None, CPUAccelerator, SingleDeviceStrategy),
        ("cpu", "ddp", CPUAccelerator, DDPStrategy),
        pytest.param("mps", None, MPSAccelerator, SingleDeviceStrategy, marks=RunIf(mps=True)),
        pytest.param("cuda", "dp", CUDAAccelerator, DataParallelStrategy, marks=RunIf(min_cuda_gpus=1)),
        pytest.param(
            "cuda", "deepspeed", CUDAAccelerator, DeepSpeedStrategy, marks=RunIf(min_cuda_gpus=1, deepspeed=True)
        ),
    ],
)
def test_accelerator_strategy_from_environment(accelerator, strategy, expected_accelerator, expected_strategy):
    """Test that the accelerator and strategy input can be set through the environment variables."""
    env_vars = {"LT_CLI_USED": "1"}
    if accelerator is not None:
        env_vars["LT_ACCELERATOR"] = accelerator
    if strategy is not None:
        env_vars["LT_STRATEGY"] = strategy

    with mock.patch.dict(os.environ, env_vars):
        connector = _Connector(accelerator="cpu" if accelerator is None else "auto")
        assert isinstance(connector.accelerator, expected_accelerator)
        assert isinstance(connector.strategy, expected_strategy)


@mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=8)
def test_devices_from_environment(*_):
    """Test that the devices and number of nodes can be set through the environment variables."""
    with mock.patch.dict(os.environ, {"LT_DEVICES": "2", "LT_NUM_NODES": "3", "LT_CLI_USED": "1"}):
        connector = _Connector(accelerator="cuda")
        assert isinstance(connector.accelerator, CUDAAccelerator)
        assert isinstance(connector.strategy, DDPStrategy)
        assert len(connector._parallel_devices) == 2
        assert connector._num_nodes_flag == 3


def test_arguments_from_environment_collision():
    """Test that the connector raises an error when the CLI settings conflict with settings in the code."""

    # Do not raise an error about collisions unless the CLI was used
    with mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu"}):
        _Connector(accelerator="cuda")

    with (
        mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu", "LT_CLI_USED": "1"}),
        pytest.raises(ValueError, match="`Fabric\\(accelerator='cuda', ...\\)` but .* `--accelerator=cpu`"),
    ):
        _Connector(accelerator="cuda")

    with (
        mock.patch.dict(os.environ, {"LT_STRATEGY": "ddp", "LT_CLI_USED": "1"}),
        pytest.raises(ValueError, match="`Fabric\\(strategy='ddp_spawn', ...\\)` but .* `--strategy=ddp`"),
    ):
        _Connector(strategy="ddp_spawn")

    with (
        mock.patch.dict(os.environ, {"LT_DEVICES": "2", "LT_CLI_USED": "1"}),
        pytest.raises(ValueError, match="`Fabric\\(devices=3, ...\\)` but .* `--devices=2`"),
    ):
        _Connector(devices=3)

    with (
        mock.patch.dict(os.environ, {"LT_NUM_NODES": "3", "LT_CLI_USED": "1"}),
        pytest.raises(ValueError, match="`Fabric\\(num_nodes=2, ...\\)` but .* `--num_nodes=3`"),
    ):
        _Connector(num_nodes=2)

    with (
        mock.patch.dict(os.environ, {"LT_PRECISION": "16-mixed", "LT_CLI_USED": "1"}),
        pytest.raises(ValueError, match="`Fabric\\(precision='64-true', ...\\)` but .* `--precision=16-mixed`"),
    ):
        _Connector(precision="64-true")


@mock.patch("lightning.fabric.accelerators.mps.MPSAccelerator.is_available", return_value=False)
def test_fsdp_unsupported_on_cpu(_):
    """Test that we raise an error if attempting to run FSDP without GPU."""
    with pytest.raises(ValueError, match="You selected the FSDP strategy but FSDP is only available on GPU"):
        _Connector(accelerator="cpu", strategy="fsdp")

    class FSDPStrategySubclass(FSDPStrategy):
        pass

    class AcceleratorSubclass(CPUAccelerator):
        pass

    # we allow subclasses of FSDPStrategy to be used with other accelerators
    _Connector(accelerator="cpu", strategy=FSDPStrategySubclass())
    _Connector(accelerator=AcceleratorSubclass(), strategy=FSDPStrategySubclass())


def test_connector_defaults_match_fabric_defaults():
    """Test that the default values for the init arguments of Connector match the ones in Fabric."""

    def get_defaults(cls):
        init_signature = inspect.signature(cls)
        return {k: v.default for k, v in init_signature.parameters.items()}

    fabric_defaults = get_defaults(Fabric)
    connector_defaults = get_defaults(_Connector)

    # defaults should match on the intersection of argument names
    for name, connector_default in connector_defaults.items():
        assert connector_default == fabric_defaults[name]


@pytest.mark.parametrize("is_interactive", [False, True])
@RunIf(min_python="3.9")  # mocking issue
def test_connector_auto_selection(monkeypatch, is_interactive):
    no_cuda = mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=0)
    single_cuda = mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=1)
    multi_cuda = mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=4)
    no_mps = mock.patch("lightning.fabric.accelerators.mps.MPSAccelerator.is_available", return_value=False)
    single_mps = mock.patch("lightning.fabric.accelerators.mps.MPSAccelerator.is_available", return_value=True)

    def _mock_interactive():
        monkeypatch.setattr(lightning.fabric.utilities.imports, "_IS_INTERACTIVE", is_interactive)
        monkeypatch.setattr(lightning.fabric.connector, "_IS_INTERACTIVE", is_interactive)
        if _IS_WINDOWS:
            # simulate fork support on windows
            monkeypatch.setattr(torch.multiprocessing, "get_all_start_methods", lambda: ["fork", "spawn"])

    _mock_interactive()

    # CPU
    with no_cuda, no_mps, monkeypatch.context():
        mock_tpu_available(monkeypatch, False)
        connector = _Connector()
    assert isinstance(connector.accelerator, CPUAccelerator)
    assert isinstance(connector.strategy, SingleDeviceStrategy)
    assert connector._devices_flag == 1

    # single CUDA
    with single_cuda, no_mps, monkeypatch.context():
        mock_tpu_available(monkeypatch, False)
        connector = _Connector()
    assert isinstance(connector.accelerator, CUDAAccelerator)
    assert isinstance(connector.strategy, SingleDeviceStrategy)
    assert connector._devices_flag == [0]

    # multi CUDA
    with multi_cuda, no_mps, monkeypatch.context():
        mock_tpu_available(monkeypatch, False)
        connector = _Connector()
    assert isinstance(connector.accelerator, CUDAAccelerator)
    assert isinstance(connector.strategy, (SingleDeviceStrategy if is_interactive else DDPStrategy))
    assert connector._devices_flag == [0] if is_interactive else list(range(4))
    if not is_interactive:
        assert isinstance(connector.strategy.cluster_environment, LightningEnvironment)
        assert connector.strategy._start_method == "fork" if is_interactive else "popen"
        assert connector.strategy.launcher.is_interactive_compatible == is_interactive

    # MPS (there's no distributed)
    with no_cuda, single_mps, monkeypatch.context():
        mock_tpu_available(monkeypatch, False)
        connector = _Connector()
    assert isinstance(connector.accelerator, MPSAccelerator)
    assert isinstance(connector.strategy, SingleDeviceStrategy)
    assert connector._devices_flag == [0]

    # single TPU
    with no_cuda, no_mps, monkeypatch.context():
        mock_tpu_available(monkeypatch, True)
        monkeypatch.setattr(lightning.fabric.accelerators.XLAAccelerator, "auto_device_count", lambda *_: 1)
        monkeypatch.setattr(torch, "device", DeviceMock())
        connector = _Connector()
    assert isinstance(connector.accelerator, XLAAccelerator)
    assert isinstance(connector.strategy, SingleDeviceXLAStrategy)
    assert connector._devices_flag == 1

    monkeypatch.undo()  # for some reason `.context()` is not working properly
    _mock_interactive()

    # Multi TPU
    with no_cuda, no_mps, monkeypatch.context():
        mock_tpu_available(monkeypatch, True)
        connector = _Connector()
    assert isinstance(connector.accelerator, XLAAccelerator)
    assert isinstance(connector.strategy, XLAStrategy)
    assert connector._devices_flag == 8
    assert isinstance(connector.strategy.cluster_environment, XLAEnvironment)
    assert connector.strategy.launcher._start_method == "fork"
    assert connector.strategy.launcher.is_interactive_compatible

    # TPU and CUDA: prefers TPU
    with multi_cuda, no_mps, monkeypatch.context():
        mock_tpu_available(monkeypatch, True)
        connector = _Connector()
    assert isinstance(connector.accelerator, XLAAccelerator)
    assert isinstance(connector.strategy, XLAStrategy)
    assert connector._devices_flag == 8
    assert isinstance(connector.strategy.cluster_environment, XLAEnvironment)
    assert connector.strategy.launcher._start_method == "fork"
    assert connector.strategy.launcher.is_interactive_compatible


@mock.patch("lightning.fabric.accelerators.mps.MPSAccelerator.is_available", return_value=False)
def test_xla_fsdp_automatic_strategy_selection(monkeypatch, tpu_available):
    import lightning.fabric.strategies as strategies

    added_fsdp = False
    # manually register fsdp for when torch.distributed.is_initialized() != True
    if "fsdp" not in strategies.STRATEGY_REGISTRY.available_strategies():
        strategies.STRATEGY_REGISTRY.register("fsdp", FSDPStrategy)
        added_fsdp = True

    connector = _Connector(accelerator="tpu", strategy="fsdp")
    assert isinstance(connector.strategy, XLAFSDPStrategy)

    connector = _Connector(accelerator="tpu", strategy="xla_fsdp")
    assert isinstance(connector.strategy, XLAFSDPStrategy)

    connector = _Connector(accelerator="auto", strategy="fsdp")
    assert isinstance(connector.strategy, XLAFSDPStrategy)

    connector = _Connector(accelerator="auto", strategy="xla_fsdp")
    assert isinstance(connector.strategy, XLAFSDPStrategy)

    if added_fsdp:
        strategies.STRATEGY_REGISTRY.pop("fsdp")
