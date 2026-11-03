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
import itertools
from collections.abc import Iterable, Iterator, Sized
from typing import Any, Callable, Optional, Union, cast

import torch
from torch import Tensor
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DistributedSampler, Sampler
from typing_extensions import Self, override

from lightning.fabric.utilities.distributed import _DatasetSamplerWrapper
from lightning.pytorch.utilities.rank_zero import rank_zero_debug, rank_zero_info
from lightning.pytorch.utilities.types import _SizedIterable


def _find_tensors(
    obj: Union[Tensor, list, tuple, dict, Any],
) -> Union[list[Tensor], itertools.chain]:  # pragma: no-cover
    """Recursively find all tensors contained in the specified object."""
    if isinstance(obj, Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return itertools.chain(*map(_find_tensors, obj))
    if isinstance(obj, dict):
        return itertools.chain(*map(_find_tensors, obj.values()))
    return []


# In manual_optimization, we need to call reducer prepare_for_backward.
# Note: Keep track of PyTorch DDP and update if there is a change
# https://github.com/pytorch/pytorch/blob/v2.0.0/torch/nn/parallel/distributed.py#L1163-L1178
def prepare_for_backward(model: DistributedDataParallel, output: Any) -> None:
    # `prepare_for_backward` is `DistributedDataParallel` specific.
    if torch.is_grad_enabled() and model.require_backward_grad_sync:
        model.require_forward_param_sync = True
        # We'll return the output object verbatim since it is a freeform
        # object. We need to find any tensors in this object, though,
        # because we need to figure out which parameters were used during
        # this forward pass, to ensure we short circuit reduction for any
        # unused parameters. Only if `find_unused_parameters` is set.
        args = list(_find_tensors(output)) if model.find_unused_parameters and not model.static_graph else []
        reducer = cast(torch._C._distributed_c10d.Reducer, model.reducer)
        reducer._rebuild_buckets()  # avoids "INTERNAL ASSERT FAILED" with `find_unused_parameters=False`
        reducer.prepare_for_backward(args)
    else:
        model.require_forward_param_sync = False


def _register_ddp_comm_hook(
    model: DistributedDataParallel,
    ddp_comm_state: Optional[object] = None,
    ddp_comm_hook: Optional[Callable] = None,
    ddp_comm_wrapper: Optional[Callable] = None,
) -> None:
    """Function to register communication hook for DDP model https://pytorch.org/docs/master/ddp_comm_hooks.html.

    Args:
        model:
            DDP model
        ddp_comm_state:
            state is passed to the hook and can be used to maintain
            and update any state information that users would like to
            maintain as part of the training process. Examples: error
            feedback in gradient compression, peers to communicate with
            next in GossipGrad etc.
        ddp_comm_hook:
            hook(state: object, bucket: dist._GradBucket) -> torch.futures.Future

            This callable function is called once the bucket is ready. The
            hook can perform whatever processing is needed and return
            a Future indicating completion of any async work (ex: allreduce).
            If the hook doesn't perform any communication, it can also
            just return a completed Future. The Future should hold the
            new value of grad bucket's tensors. Once a bucket is ready,
            c10d reducer would call this hook and use the tensors returned
            by the Future and copy grads to individual parameters.
        ddp_comm_wrapper:
            communication hook wrapper to support a communication hook such
            as FP16 compression as wrapper, which could be combined with
            ddp_comm_hook

    Examples::

        from torch.distributed.algorithms.ddp_comm_hooks import (
            default_hooks as default,
            powerSGD_hook as powerSGD,
            post_localSGD_hook as post_localSGD,
        )

        # fp16_compress_hook for compress gradients
        ddp_model = ...
        _register_ddp_comm_hook(
            model=ddp_model,
            ddp_comm_hook=default.fp16_compress_hook,
        )

        # powerSGD_hook
        ddp_model = ...
        _register_ddp_comm_hook(
            model=ddp_model,
            ddp_comm_state=powerSGD.PowerSGDState(
                process_group=None,
                matrix_approximation_rank=1,
                start_powerSGD_iter=5000,
            ),
            ddp_comm_hook=powerSGD.powerSGD_hook,
        )

        # post_localSGD_hook
        subgroup, _ = torch.distributed.new_subgroups()
        ddp_model = ...
        _register_ddp_comm_hook(
            model=ddp_model,
            state=post_localSGD.PostLocalSGDState(
                process_group=None,
                subgroup=subgroup,
                start_localSGD_iter=1_000,
            ),
            ddp_comm_hook=post_localSGD.post_localSGD_hook,
        )

        # fp16_compress_wrapper combined with other communication hook
        ddp_model = ...
        _register_ddp_comm_hook(
            model=ddp_model,
            ddp_comm_state=powerSGD.PowerSGDState(
                process_group=None,
                matrix_approximation_rank=1,
                start_powerSGD_iter=5000,
            ),
            ddp_comm_hook=powerSGD.powerSGD_hook,
            ddp_comm_wrapper=default.fp16_compress_wrapper,
        )

    """
    if ddp_comm_hook is None:
        return
    # inform mypy that ddp_comm_hook is callable
    ddp_comm_hook: Callable = ddp_comm_hook

    if ddp_comm_wrapper is not None:
        rank_zero_info(
            f"DDP comm wrapper is provided, apply {ddp_comm_wrapper.__qualname__}({ddp_comm_hook.__qualname__})."
        )
        ddp_comm_hook = ddp_comm_wrapper(ddp_comm_hook)

    rank_zero_debug(f"Registering DDP comm hook: {ddp_comm_hook.__qualname__}.")
    model.register_comm_hook(state=ddp_comm_state, hook=ddp_comm_hook)


def _sync_module_states(module: torch.nn.Module) -> None:
    """Taken from https://github.com/pytorch/pytorch/blob/v2.0.0/torch/nn/parallel/distributed.py#L675-L682."""
    parameters_to_ignore = set(getattr(module, "_ddp_params_and_buffers_to_ignore", []))
    from torch.distributed.distributed_c10d import _get_default_group
    from torch.distributed.utils import _sync_module_states as torch_sync_module_states

    torch_sync_module_states(
        module,
        _get_default_group(),
        250 * 1024 * 1024,
        src=0,
        params_and_buffers_to_ignore=parameters_to_ignore,
    )


class UnrepeatedDistributedSampler(DistributedSampler):
    """A fork of the PyTorch DistributedSampler that doesn't repeat data, instead allowing the number of batches per
    process to be off-by-one from each other. This makes this sampler usable for predictions (it's deterministic and
    doesn't require shuffling). It is potentially unsafe to use this sampler for training, because during training the
    DistributedDataParallel syncs buffers on each forward pass, so it could freeze if one of the processes runs one
    fewer batch. During prediction, buffers are only synced on the first batch, so this is safe to use as long as each
    process runs at least one batch. We verify this in an assert.

    Taken from https://github.com/jpuigcerver/PyLaia/blob/v1.0.0/laia/data/unpadded_distributed_sampler.py and
    https://github.com/pytorch/pytorch/issues/25162#issuecomment-634146002

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(self.dataset, Sized):
            raise TypeError("The given dataset must implement the `__len__` method.")
        self.num_samples = len(range(self.rank, len(self.dataset), self.num_replicas))
        self.total_size = len(self.dataset)
        # If any process has at least one batch, every other process needs to
        # have at least one batch, or the DistributedDataParallel could lock up.
        assert self.num_samples >= 1 or self.total_size == 0

    @override
    def __iter__(self) -> Iterator[list[int]]:
        if not isinstance(self.dataset, Sized):
            raise TypeError("The given dataset must implement the `__len__` method.")
        indices: Union[range, list[int]]
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            permutation = torch.randperm(len(self.dataset), generator=g)
            assert len(permutation) == self.total_size
            # subsample before converting to a list so that only this rank's indices become Python objects
            indices = permutation[self.rank : self.total_size : self.num_replicas].tolist()
        else:
            # a strided range yields this rank's indices lazily without materializing the full index list
            indices = range(self.rank, self.total_size, self.num_replicas)
        assert len(indices) == self.num_samples

        return iter(indices)


class UnrepeatedDistributedSamplerWrapper(UnrepeatedDistributedSampler):
    """Equivalent class to ``DistributedSamplerWrapper`` but for the ``UnrepeatedDistributedSampler``."""

    def __init__(self, sampler: Union[Sampler, Iterable], *args: Any, **kwargs: Any) -> None:
        super().__init__(_DatasetSamplerWrapper(sampler), *args, **kwargs)

    @override
    def __iter__(self) -> Iterator:
        self.dataset.reset()
        return (self.dataset[index] for index in super().__iter__())


class _IndexBatchSamplerWrapper:
    """This class is used to wrap a :class:`torch.utils.data.BatchSampler` and capture its indices."""

    def __init__(self, batch_sampler: _SizedIterable) -> None:
        # do not call super().__init__() on purpose
        self.seen_batch_indices: list[list[int]] = []

        self.__dict__ = {
            k: v
            for k, v in batch_sampler.__dict__.items()
            if k not in ("__next__", "__iter__", "__len__", "__getstate__")
        }
        self._batch_sampler = batch_sampler
        self._iterator: Optional[Iterator[list[int]]] = None

    def __next__(self) -> list[int]:
        assert self._iterator is not None
        batch = next(self._iterator)
        self.seen_batch_indices.append(batch)
        return batch

    def __iter__(self) -> Self:
        self.seen_batch_indices = []
        self._iterator = iter(self._batch_sampler)
        return self

    def __len__(self) -> int:
        return len(self._batch_sampler)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_iterator"] = None  # cannot pickle 'generator' object
        return state
