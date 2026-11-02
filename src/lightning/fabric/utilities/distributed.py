import atexit
import contextlib
import logging
import os
import signal
import time
from collections.abc import Iterable, Iterator, Sized
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DistributedSampler, Sampler
from typing_extensions import Self, TypeGuard, override

from lightning.fabric.utilities.cloud_io import _is_local_file_protocol
from lightning.fabric.utilities.data import _num_cpus_available
from lightning.fabric.utilities.rank_zero import rank_zero_info
from lightning.fabric.utilities.types import _PATH, ReduceOp

if torch.distributed.is_available():
    from torch.distributed import group
else:

    class group:  # type: ignore
        WORLD = None


if TYPE_CHECKING:
    from torch.distributed._tensor import DTensor

    from lightning.fabric.plugins import ClusterEnvironment
    from lightning.fabric.strategies import Strategy


log = logging.getLogger(__name__)


def is_shared_filesystem(strategy: "Strategy", path: Optional[_PATH] = None, timeout: int = 3) -> bool:
    """Checks whether the filesystem under the given path is shared across all processes.

    This function should only be used in a context where distributed is initialized.

    Args:
        strategy: The strategy being used, either from Fabric (``fabric.strategy``) or from Trainer
            (``trainer.strategy``).
        path: The path to check. Defaults to the current working directory. The user must have permissions to write
            to this path or the parent folder, and the filesystem must be writable.
        timeout: If any of the processes can't list the file created by rank 0 within this many seconds, the
            filesystem is determined to be not shared.

    """
    # Fast path: Any non-local filesystem is considered shared (e.g., S3)
    if path is not None and not _is_local_file_protocol(path):
        return True

    path = Path(Path.cwd() if path is None else path).resolve()

    # Fast path: Only distributed strategies can detect shared filesystems
    if not hasattr(strategy, "world_size") or strategy.world_size == 1:
        return True

    # Fast path: If the path is not the same on all ranks we know it's not a shared filesystem
    rank_zero_path = strategy.broadcast(path)
    if not strategy.reduce_boolean_decision(rank_zero_path == path, all=True):
        return False

    if not strategy.reduce_boolean_decision(path.exists(), all=True):
        raise FileNotFoundError(
            f"Unable to determine if the path belongs to a shared filesystem. The path does not exist: {path}"
        )

    path = path.parent if path.is_file() else path
    check_file = path / ".lightning_shared_fs_check"
    check_file.unlink(missing_ok=True)

    strategy.barrier()
    if strategy.is_global_zero:
        # Rank 0 creates the file
        check_file.touch()
        found = True
    else:
        # All other ranks will wait until they find the file or timeout
        start = time.perf_counter()
        found = False
        while not found and (time.perf_counter() - start) < timeout:
            found = check_file.exists()
    strategy.barrier()

    all_found = strategy.reduce_boolean_decision(found, all=True)

    with contextlib.suppress(OSError):  # handle race condition on deletion
        check_file.unlink()

    return all_found


def _gather_all_tensors(result: Tensor, group: Optional[Any] = None) -> list[Tensor]:
    """Function to gather all tensors from several DDP processes onto a list that is broadcasted to all processes.

    Works on tensors that have the same number of dimensions, but where each dimension may differ. In this case
    tensors are padded, gathered and then trimmed to secure equal workload for all processes.

    Args:
        result: The value to sync
        group: The process group to gather results from. Defaults to all processes (world)

    Return:
        gathered_result: List with size equal to the process group where
            gathered_result[i] corresponds to result tensor from process i

    """
    if group is None:
        group = torch.distributed.group.WORLD

    # Convert tensors to contiguous format
    result = result.contiguous()

    world_size = torch.distributed.get_world_size(group)
    torch.distributed.barrier(group=group)

    # If the tensor is scalar, things are easy
    if result.ndim == 0:
        return _simple_gather_all_tensors(result, group, world_size)

    # 1. Gather sizes of all tensors in a single device-to-host transfer
    local_size = torch.tensor(result.shape, device=result.device)
    local_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    torch.distributed.all_gather(local_sizes, local_size, group=group)
    shapes: list[list[int]] = torch.stack(local_sizes).cpu().tolist()
    all_sizes_equal = all(s == shapes[0] for s in shapes)

    # 2. If shapes are all the same, then do a simple gather:
    if all_sizes_equal:
        return _simple_gather_all_tensors(result, group, world_size)

    # 3. If not, we need to pad each local tensor to maximum size, gather and then truncate
    max_size = [max(dim_sizes) for dim_sizes in zip(*shapes)]
    pad_dims = []
    for max_dim, cur_dim in zip(reversed(max_size), reversed(result.shape)):
        pad_dims.append(0)
        pad_dims.append(max_dim - cur_dim)
    result_padded = F.pad(result, pad_dims)
    gathered_result = [torch.zeros_like(result_padded) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_result, result_padded, group)
    for idx, item_size in enumerate(shapes):
        slice_param = tuple(slice(dim_size) for dim_size in item_size)
        gathered_result[idx] = gathered_result[idx][slice_param]
    return gathered_result


def _simple_gather_all_tensors(result: Tensor, group: Any, world_size: int) -> list[Tensor]:
    gathered_result = [torch.zeros_like(result) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_result, result, group)
    return gathered_result


def _sync_ddp_if_available(
    result: Tensor, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = None
) -> Tensor:
    """Function to reduce a tensor across worker processes during distributed training.

    Args:
        result: The value to sync and reduce (typically tensor or number)
        group: The process group to gather results from. Defaults to all processes (world)
        reduce_op: The reduction operation. Defaults to sum.
            Can also be a string of 'avg', 'mean' to calculate the mean during reduction.

    Return:
        reduced value

    """
    if _distributed_is_initialized():
        return _sync_ddp(result, group=group, reduce_op=reduce_op)
    return result


def _sync_ddp(result: Tensor, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = None) -> Tensor:
    """Reduces a tensor across several distributed processes.

    This operation is performed in-place, meaning the result will be placed back into the input tensor on all processes.

    Args:
        result: The value to sync and reduce (typically tensor or number)
        group: The process group to gather results from. Defaults to all processes (world)
        reduce_op: The reduction operation. Defaults to sum.
            Can also be a string of 'avg', 'mean' to calculate the mean during reduction.

    Return:
        The reduced value.

    """
    divide_by_world_size = False
    group = torch.distributed.group.WORLD if group is None else group

    op: Optional[ReduceOp]
    if isinstance(reduce_op, str):
        reduce_op = "avg" if reduce_op == "mean" else reduce_op
        if reduce_op.lower() == "avg" and torch.distributed.get_backend(group) == "gloo":
            # The GLOO backend does not support the `ReduceOp.AVG` operation
            op = ReduceOp.SUM  # type: ignore[assignment]
            divide_by_world_size = True
        else:
            op = getattr(ReduceOp, reduce_op.upper())
    else:
        op = reduce_op

    # Sync all processes before reduction
    torch.distributed.barrier(group=group)
    torch.distributed.all_reduce(result, op=op, group=group, async_op=False)
    world_size = torch.distributed.get_world_size(group)

    if not divide_by_world_size:
        return result
    # `torch.distributed.all_reduce` is in-place, so we should do the division in-place to leave the modified tensors
    # with the expected value
    if not torch.is_floating_point(result):
        return result.copy_(result / world_size)
    return result.div_(world_size)


def _all_gather_ddp_if_available(
    tensor: Tensor, group: Optional["torch.distributed.ProcessGroup"] = None, sync_grads: bool = False
) -> Tensor:
    """Function to gather a tensor from several distributed processes.

    Args:
        tensor: Tensor of shape (batch, ...)
        group: The process group to gather results from. Defaults to all processes (world)
        sync_grads: Flag that allows users to synchronize gradients for all_gather op

    Return:
        A tensor of shape (world_size, batch, ...)

    """
    if not _distributed_is_initialized():
        return tensor

    from torch.distributed.nn.functional import all_gather

    tensor = tensor.contiguous()  # https://github.com/pytorch/pytorch/issues/73515
    with nullcontext() if sync_grads else torch.no_grad():
        gathered_tensors = all_gather(tensor, group)
    return torch.stack(gathered_tensors)


def _init_dist_connection(
    cluster_environment: "ClusterEnvironment",
    torch_distributed_backend: str,
    global_rank: Optional[int] = None,
    world_size: Optional[int] = None,
    **kwargs: Any,
) -> None:
    """Utility function to initialize distributed connection by setting env variables and initializing the distributed
    process group.

    Args:
        cluster_environment: ``ClusterEnvironment`` instance
        torch_distributed_backend: Backend to use (includes `nccl` and `gloo`)
        global_rank: Rank of the current process
        world_size: Number of processes in the group
        kwargs: Kwargs for ``init_process_group``

    Raises:
        RuntimeError:
            If ``torch.distributed`` is not available

    """
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is not available. Cannot initialize distributed process group")
    if torch.distributed.is_initialized():
        log.debug("torch.distributed is already initialized. Exiting early")
        return
    global_rank = global_rank if global_rank is not None else cluster_environment.global_rank()
    world_size = world_size if world_size is not None else cluster_environment.world_size()
    os.environ["MASTER_ADDR"] = cluster_environment.main_address
    os.environ["MASTER_PORT"] = str(cluster_environment.main_port)
    log.info(f"Initializing distributed: GLOBAL_RANK: {global_rank}, MEMBER: {global_rank + 1}/{world_size}")
    torch.distributed.init_process_group(torch_distributed_backend, rank=global_rank, world_size=world_size, **kwargs)

    if torch_distributed_backend == "nccl":
        # PyTorch >= 2.4 warns about undestroyed NCCL process group, so we need to do it at program exit
        atexit.register(_destroy_dist_connection)

    # On rank=0 let everyone know training is starting
    rank_zero_info(
        f"{'-' * 100}\n"
        f"distributed_backend={torch_distributed_backend}\n"
        f"All distributed processes registered. Starting with {world_size} processes\n"
        f"{'-' * 100}\n"
    )


def _destroy_dist_connection() -> None:
    # Don't allow Ctrl+C to interrupt this handler
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if _distributed_is_initialized():
        torch.distributed.destroy_process_group()
    signal.signal(signal.SIGINT, signal.SIG_DFL)


def _get_default_process_group_backend_for_device(device: torch.device) -> str:
    """Return corresponding distributed backend for a given device."""
    device_backend_map = torch.distributed.Backend.default_device_backend_map
    if device.type in device_backend_map:
        return device_backend_map[device.type]
    return "gloo"


class _DatasetSamplerWrapper(Dataset):
    """Dataset to create indexes from `Sampler` or `Iterable`"""

    def __init__(self, sampler: Union[Sampler, Iterable]) -> None:
        if not isinstance(sampler, Sized):
            raise TypeError(
                "You seem to have configured a sampler in your DataLoader which"
                " does not provide `__len__` method. The sampler was about to be"
                " replaced by `DistributedSamplerWrapper` since `use_distributed_sampler`"
                " is True and you are using distributed training. Either provide `__len__`"
                " method in your sampler, remove it from DataLoader or set `use_distributed_sampler=False`"
                " if you want to handle distributed sampling yourself."
            )
        if len(sampler) == float("inf"):
            raise TypeError(
                "You seem to have configured a sampler in your DataLoader which"
                " does not provide finite `__len__` method. The sampler was about to be"
                " replaced by `DistributedSamplerWrapper` since `use_distributed_sampler`"
                " is True and you are using distributed training. Either provide `__len__`"
                " method in your sampler which returns a finite number, remove it from DataLoader"
                " or set `use_distributed_sampler=False` if you want to handle distributed sampling yourself."
            )
        self._sampler = sampler
        # defer materializing an iterator until it is necessary
        self._sampler_list: Optional[list[Any]] = None

    @override
    def __getitem__(self, index: int) -> Any:
        if self._sampler_list is None:
            self._sampler_list = list(self._sampler)
        return self._sampler_list[index]

    def __len__(self) -> int:
        return len(self._sampler)

    def reset(self) -> None:
        """Reset the sampler list in order to get new sampling."""
        self._sampler_list = list(self._sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """Wrapper over ``Sampler`` for distributed training.

    Allows you to use any sampler in distributed mode. It will be automatically used by Lightning in distributed mode if
    sampler replacement is enabled.

    Note:
        The purpose of this wrapper is to take care of sharding the sampler indices. It is up to the underlying
        sampler to handle randomness and shuffling. The ``shuffle`` and ``seed`` arguments on this wrapper won't
        have any effect.

    """

    def __init__(self, sampler: Union[Sampler, Iterable], *args: Any, **kwargs: Any) -> None:
        super().__init__(_DatasetSamplerWrapper(sampler), *args, **kwargs)

    @override
    def __iter__(self) -> Iterator:
        self.dataset.reset()
        return (self.dataset[index] for index in super().__iter__())

    @override
    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        # Forward set_epoch to the original sampler if it supports it
        original_sampler = self.dataset._sampler
        if hasattr(original_sampler, "set_epoch") and callable(original_sampler.set_epoch):
            original_sampler.set_epoch(epoch)


def _suggested_max_num_threads(num_processes: int = 1) -> int:
    if num_processes < 1:
        raise ValueError(f"`num_processes` should be >= 1, got {num_processes}.")
    return max(1, _num_cpus_available() // num_processes)


def _set_num_threads_if_needed(num_processes: int = 1) -> None:
    if "OMP_NUM_THREADS" not in os.environ:
        num_threads = _suggested_max_num_threads(num_processes)
        torch.set_num_threads(num_threads)
        os.environ["OMP_NUM_THREADS"] = str(num_threads)


def _distributed_is_initialized() -> bool:
    # `is_initialized` is only defined conditionally
    # https://github.com/pytorch/pytorch/blob/v2.1.0/torch/distributed/__init__.py#L25
    # this might happen to MacOS builds from source (default) or any build from source that sets `USE_DISTRIBUTED=0`
    return torch.distributed.is_available() and torch.distributed.is_initialized()


class _InfiniteBarrier:
    """A barrier with an infinite timeout.

    Creates a new process group with the GLOO backend with a very high timeout that makes the barrier effectively wait
    forever. This is useful in cases where you want to execute a long-running operation on a subset of ranks that should
    not be subject to the regular collective timeout.

    """

    def __init__(self) -> None:
        self.group = None
        self.barrier = lambda: None

    def __call__(self) -> None:
        self.barrier()

    def __enter__(self) -> Self:
        if _distributed_is_initialized():
            # Create a barrier with an 'infinite' timeout (only reliably possible over the GLOO backend)
            self.group = torch.distributed.new_group(backend="gloo", timeout=timedelta(days=10000))
            self.barrier = self.group.monitored_barrier
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.barrier()
        if self.group is not None:
            torch.distributed.destroy_process_group(self.group)


def _is_dtensor(tensor: Tensor) -> TypeGuard["DTensor"]:
    from torch.distributed._tensor import DTensor

    return isinstance(tensor, DTensor)
