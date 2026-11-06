import functools
import os
from functools import partial
from pathlib import Path
from unittest import mock
from unittest.mock import Mock

import pytest
import torch
from lightning_utilities.core.imports import RequirementCache

from lightning.fabric.accelerators import CPUAccelerator, CUDAAccelerator, MPSAccelerator
from lightning.fabric.plugins.environments import LightningEnvironment
from lightning.fabric.strategies import DDPStrategy, SingleDeviceStrategy
from lightning.fabric.strategies.launchers.multiprocessing import _MultiProcessingLauncher
from lightning.fabric.utilities.distributed import (
    DistributedSamplerWrapper,
    _destroy_dist_connection,
    _gather_all_tensors,
    _get_default_process_group_backend_for_device,
    _InfiniteBarrier,
    _init_dist_connection,
    _is_dtensor,
    _set_num_threads_if_needed,
    _suggested_max_num_threads,
    _sync_ddp,
    is_shared_filesystem,
)
from tests_fabric.helpers.runif import RunIf


def wrap_launch_function(fn, strategy, *args, **kwargs):
    # the launcher does not manage this automatically. explanation available in:
    # https://github.com/Lightning-AI/pytorch-lightning/pull/14926#discussion_r982976718
    strategy.setup_environment()
    return fn(*args, **kwargs)


def spawn_launch(fn, parallel_devices):
    """Copied from ``tests_pytorch.core.test_results.spawn_launch``"""
    # TODO: the accelerator and cluster_environment should be optional to just launch processes, but this requires lazy
    # initialization to be implemented
    device_to_accelerator = {"cuda": CUDAAccelerator, "mps": MPSAccelerator, "cpu": CPUAccelerator}
    accelerator_cls = device_to_accelerator[parallel_devices[0].type]
    strategy = DDPStrategy(
        accelerator=accelerator_cls(),
        parallel_devices=parallel_devices,
        cluster_environment=LightningEnvironment(),
        start_method="spawn",
    )
    launcher = _MultiProcessingLauncher(strategy=strategy)
    wrapped = partial(wrap_launch_function, fn, strategy)
    return launcher.launch(wrapped, strategy)


def _test_all_gather_uneven_tensors(strategy):
    rank = strategy.local_rank
    device = strategy.root_device
    world_size = strategy.num_processes

    tensor = torch.ones(rank, device=device)
    result = _gather_all_tensors(tensor)
    assert len(result) == world_size
    for idx in range(world_size):
        assert len(result[idx]) == idx
        assert (result[idx] == torch.ones_like(result[idx])).all()


def _test_all_gather_uneven_tensors_multidim(strategy):
    rank = strategy.local_rank
    device = strategy.root_device
    world_size = strategy.num_processes

    tensor = torch.ones(rank + 1, 2 - rank, device=device)
    result = _gather_all_tensors(tensor)
    assert len(result) == world_size
    for idx in range(world_size):
        val = result[idx]
        assert val.shape == (idx + 1, 2 - idx)
        assert (val == torch.ones_like(val)).all()


def _test_all_reduce(strategy):
    rank = strategy.local_rank
    device = strategy.root_device
    world_size = strategy.num_processes

    for dtype in (torch.long, torch.int, torch.float, torch.half):
        # max
        tensor = torch.tensor(rank + 1, device=device, dtype=dtype)
        expected = torch.tensor(2, device=device, dtype=dtype)
        result = _sync_ddp(tensor, reduce_op="max")
        assert torch.equal(result, expected)
        assert result is tensor  # inplace
        # sum
        tensor = torch.tensor(rank + 1, device=device, dtype=dtype)
        expected = torch.tensor(sum(range(1, world_size + 1)), device=device, dtype=dtype)
        result = _sync_ddp(tensor, reduce_op="sum")
        assert torch.equal(result, expected)
        assert result is tensor  # inplace
        # average
        tensor = torch.tensor(rank + 1, device=device, dtype=dtype)
        expected = torch.tensor(sum(range(1, world_size + 1)) / 2, device=device, dtype=dtype)
        result = _sync_ddp(tensor, reduce_op="avg")
        assert torch.equal(result, expected)
        assert result is tensor  # inplace


# flaky with "torch.multiprocessing.spawn.ProcessExitedException: process 0 terminated with signal SIGABRT" (GLOO)
@pytest.mark.flaky(reruns=3)
@RunIf(skip_windows=True)
@pytest.mark.parametrize(
    "process",
    [
        _test_all_gather_uneven_tensors_multidim,
        _test_all_gather_uneven_tensors,
        _test_all_reduce,
    ],
)
@pytest.mark.parametrize(
    "devices",
    [
        pytest.param([torch.device("cuda:0"), torch.device("cuda:1")], marks=RunIf(min_cuda_gpus=2)),
        [torch.device("cpu"), torch.device("cpu")],
    ],
)
def test_collective_operations(devices, process):
    spawn_launch(process, devices)


@pytest.mark.skipif(
    RequirementCache("numpy>=2.0"),
    reason="torch.distributed not compatible with numpy>=2.0",
)
@RunIf(min_torch="2.4", skip_windows=True)
@pytest.mark.flaky(reruns=3)  # flaky with "process 0 terminated with signal SIGABRT" (GLOO)
def test_is_shared_filesystem(tmp_path, monkeypatch):
    # In the non-distributed case, every location is interpreted as 'shared'
    assert is_shared_filesystem(SingleDeviceStrategy(torch.device("cpu")))

    test_fn = functools.partial(_test_is_shared_filesystem, tmp_path=tmp_path, monkeypatch=monkeypatch)
    spawn_launch(test_fn, [torch.device("cpu"), torch.device("cpu")])


def _test_is_shared_filesystem(strategy, tmp_path, monkeypatch):
    # Path doesn't exist
    with pytest.raises(FileNotFoundError, match="Unable to determine if the path belongs to a shared filesystem"):
        is_shared_filesystem(strategy, path="not/exist")

    # Path exists but not the same on all ranks
    file = tmp_path / f"file-rank-{strategy.global_rank}"
    file.touch()
    folder = tmp_path / f"folder-rank-{strategy.global_rank}"
    folder.mkdir()
    assert not is_shared_filesystem(strategy, path=file)
    assert not is_shared_filesystem(strategy, path=folder)

    # Path exists
    folder = tmp_path / "folder"
    file = folder / "file"
    if strategy.global_rank == 0:
        folder.mkdir()
        file.touch()
    strategy.barrier()
    assert folder.exists()
    assert is_shared_filesystem(strategy, path=folder)
    assert is_shared_filesystem(strategy, path=file)
    assert os.listdir(folder) == ["file"]  # rank test files got cleaned up

    # Path defaults to CWD
    monkeypatch.chdir(tmp_path)
    assert Path.cwd() == tmp_path
    assert is_shared_filesystem(strategy)
    monkeypatch.undo()

    # Path is a symlink
    linked = Path(tmp_path / "linked")
    if strategy.global_rank == 0:
        linked.symlink_to(tmp_path / "folder", target_is_directory=True)
    assert is_shared_filesystem(strategy, path=folder)

    # Remote path is considered shared
    assert is_shared_filesystem(strategy, path="s3://my-bucket/data")


@pytest.mark.parametrize("invalid", [-1, 0])
def test_suggested_max_num_threads(invalid):
    with pytest.raises(ValueError, match="should be >= 1"):
        _suggested_max_num_threads(invalid)


@mock.patch.dict(os.environ, {}, clear=True)
@mock.patch("lightning.fabric.utilities.distributed.torch.set_num_threads")
@mock.patch("lightning.fabric.utilities.distributed._num_cpus_available", return_value=4)
@pytest.mark.parametrize(("num_processes", "expected"), [(1, 4), (2, 2), (3, 1), (4, 1), (8, 1)])
def test_set_num_threads_if_needed(_, set_num_threads_mock, num_processes, expected):
    assert "OMP_NUM_THREADS" not in os.environ
    _set_num_threads_if_needed(num_processes)
    set_num_threads_mock.assert_called_with(expected)
    assert os.environ["OMP_NUM_THREADS"] == str(expected)

    # if env variable is already set, no change
    set_num_threads_mock.reset_mock()
    _set_num_threads_if_needed(1)
    set_num_threads_mock.assert_not_called()
    assert os.environ["OMP_NUM_THREADS"] == str(expected)


def test_infinite_barrier():
    # distributed not available
    barrier = _InfiniteBarrier()
    assert barrier.group is None
    with mock.patch("lightning.fabric.utilities.distributed._distributed_is_initialized", return_value=False):
        barrier.__enter__()
        assert barrier.group is None
        barrier()
        barrier.__exit__(None, None, None)
        assert barrier.group is None

    # distributed available
    barrier = _InfiniteBarrier()
    with (
        mock.patch("lightning.fabric.utilities.distributed._distributed_is_initialized", return_value=True),
        mock.patch("lightning.fabric.utilities.distributed.torch.distributed") as dist_mock,
    ):
        barrier.__enter__()
        dist_mock.new_group.assert_called_once()
        assert barrier.barrier == barrier.group.monitored_barrier
        assert barrier.barrier.call_count == 0
        barrier()
        assert barrier.barrier.call_count == 1
        barrier.__exit__(None, None, None)
        assert barrier.barrier.call_count == 2
        dist_mock.destroy_process_group.assert_called_once()


@mock.patch("lightning.fabric.utilities.distributed.atexit")
@mock.patch("lightning.fabric.utilities.distributed.torch.distributed.init_process_group")
def test_init_dist_connection_registers_destruction_handler(_, atexit_mock):
    _init_dist_connection(LightningEnvironment(), "nccl")
    atexit_mock.register.assert_called_once_with(_destroy_dist_connection)
    atexit_mock.reset_mock()
    _init_dist_connection(LightningEnvironment(), "gloo")
    atexit_mock.register.assert_called_once_with(_destroy_dist_connection)


def test_get_default_process_group_backend_for_device(monkeypatch):
    """Test that each device type maps to its correct default process group backend."""
    # patch the map instead of registering a real backend: `rename_privateuse1_backend` is irreversible
    # and would leak a dangling "pcu" device into subsequent tests
    monkeypatch.setitem(torch.distributed.Backend.default_device_backend_map, "pcu", "pccl")

    # test that the default backend is correctly set for each device
    devices = [torch.device("cpu"), torch.device("cuda:0"), Mock(type="pcu")]
    backends = ["gloo", "nccl", "pccl"]
    for device, backend in zip(devices, backends):
        assert _get_default_process_group_backend_for_device(device) == backend


@RunIf(min_torch="2.4")
def test_is_dtensor():
    from torch.distributed._tensor import DTensor

    assert _is_dtensor(Mock(spec=DTensor))
    assert not _is_dtensor(torch.zeros(2, 2))


class _CustomSampler(torch.utils.data.Sampler):
    """A custom sampler for testing DistributedSamplerWrapper."""

    def __init__(self, data_source, non_callable_set_epoch: bool = False):
        self.data_source = data_source
        if non_callable_set_epoch:
            self.set_epoch = "not a method"  # attribute exists but is not callable

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        return iter(range(len(self.data_source)))


class _CustomSamplerWithSetEpoch(_CustomSampler):
    """A custom sampler that tracks set_epoch calls for testing."""

    def __init__(self, data_source):
        super().__init__(data_source)
        self.epoch = 0
        self.set_epoch_call_count = 0

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.set_epoch_call_count += 1


def test_distributed_sampler_wrapper_set_epoch():
    """Test that DistributedSamplerWrapper correctly handles set_epoch for various sampler types.

    Reproduces issue #21454: When a sampler is wrapped by DistributedSamplerWrapper, calling set_epoch on the wrapper
    should forward the call to the underlying sampler if it supports the method.

    """
    data_source = list(range(100))

    # Case 1: Sampler WITH set_epoch method - should forward the call
    sampler_with_set_epoch = _CustomSamplerWithSetEpoch(data_source)
    wrapper = DistributedSamplerWrapper(sampler_with_set_epoch, num_replicas=2, rank=0)

    assert sampler_with_set_epoch.epoch == 0
    assert sampler_with_set_epoch.set_epoch_call_count == 0

    wrapper.set_epoch(5)
    assert wrapper.epoch == 5
    assert sampler_with_set_epoch.epoch == 5, "set_epoch was not forwarded to the underlying sampler"
    assert sampler_with_set_epoch.set_epoch_call_count == 1

    wrapper.set_epoch(10)
    assert wrapper.epoch == 10
    assert sampler_with_set_epoch.epoch == 10
    assert sampler_with_set_epoch.set_epoch_call_count == 2

    # Case 2: Sampler WITHOUT set_epoch method - should not fail
    sampler_without_set_epoch = _CustomSampler(data_source)
    wrapper = DistributedSamplerWrapper(sampler_without_set_epoch, num_replicas=2, rank=0)

    wrapper.set_epoch(5)  # Should not raise
    assert wrapper.epoch == 5

    # Case 3: Sampler with non-callable set_epoch attribute - should not fail or call it
    sampler_non_callable = _CustomSampler(data_source, non_callable_set_epoch=True)
    wrapper = DistributedSamplerWrapper(sampler_non_callable, num_replicas=2, rank=0)

    wrapper.set_epoch(5)  # Should not raise
    assert wrapper.epoch == 5
    assert sampler_non_callable.set_epoch == "not a method"  # Should remain unchanged
