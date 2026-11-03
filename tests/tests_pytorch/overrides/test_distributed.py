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
from collections.abc import Iterable

import pytest
import torch
from torch.utils.data import BatchSampler, SequentialSampler

from lightning.fabric.utilities.data import has_len
from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.overrides.distributed import (
    UnrepeatedDistributedSampler,
    UnrepeatedDistributedSamplerWrapper,
    _IndexBatchSamplerWrapper,
)
from tests_pytorch.helpers.runif import RunIf


class MyModel(LightningModule):
    def setup(self, stage: str) -> None:
        self.layer = torch.nn.Linear(1, 1)
        weights = self.layer.weight.item(), self.layer.bias.item()
        self.rank_0_weights = self.trainer.strategy.broadcast(weights)

    def test_step(self, batch, batch_idx):
        current = self.layer.weight.item(), self.layer.bias.item()
        assert self.rank_0_weights == current
        gathered = self.all_gather(current)
        # the weights have been synced
        assert all(torch.all(t == t[0]) for t in gathered), gathered


@RunIf(standalone=True)
def test_params_synced_during_nonfit():
    model = MyModel()
    trainer = Trainer(
        barebones=True,
        devices=2,
        accelerator="cpu",
        strategy="ddp",
    )
    trainer.test(model, [0])


@pytest.mark.parametrize("shuffle", [False, True])
def test_unrepeated_distributed_sampler(shuffle):
    """Test each rank will receive a different number of elements."""
    seed_everything(42)
    world_size = 4
    samplers = []
    dataset = range(103)
    for rank in range(world_size):
        samplers.append(UnrepeatedDistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=shuffle))

    indices = [list(s) for s in samplers]
    assert len(indices[0]) == 26
    assert len(indices[1]) == 26
    assert len(indices[2]) == 26
    assert len(indices[3]) == 25

    assert indices[0][-1] == 18 if shuffle else 100
    assert indices[1][-1] == 30 if shuffle else 101
    assert indices[2][-1] == 29 if shuffle else 102
    assert indices[3][-1] == 35 if shuffle else 99


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize(("length", "world_size"), [(103, 4), (8, 8), (13, 1), (0, 3), (5, 5)])
def test_unrepeated_distributed_sampler_partition(shuffle, length, world_size):
    """Test that the ranks get a disjoint partition of the dataset that matches the reference strided subsampling."""
    dataset = range(length)
    epoch = 3
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(epoch)
        reference = torch.randperm(length, generator=generator).tolist()
    else:
        reference = list(range(length))

    gathered = []
    for rank in range(world_size):
        sampler = UnrepeatedDistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=shuffle)
        sampler.set_epoch(epoch)
        indices = list(sampler)
        assert all(type(i) is int for i in indices)
        assert indices == reference[rank::world_size]
        assert len(indices) == len(sampler)
        assert list(sampler) == indices  # re-iterating gives the same indices
        gathered.extend(indices)
    assert sorted(gathered) == list(range(length))


def test_unrepeated_distributed_sampler_shuffle_depends_on_epoch():
    dataset = range(50)
    sampler = UnrepeatedDistributedSampler(dataset, rank=1, num_replicas=4, shuffle=True)
    sampler.set_epoch(0)
    epoch0 = list(sampler)
    sampler.set_epoch(1)
    epoch1 = list(sampler)
    assert epoch0 != epoch1
    assert sorted(epoch0) != epoch0
    sampler.set_epoch(0)
    assert list(sampler) == epoch0


@pytest.mark.parametrize("world_size", [1, 3])
def test_unrepeated_distributed_sampler_wrapper_matches_sequential(world_size):
    """Test that wrapping a ``SequentialSampler`` is equivalent to sampling the dataset directly."""
    dataset = range(10)
    for rank in range(world_size):
        wrapper = UnrepeatedDistributedSamplerWrapper(
            SequentialSampler(dataset), rank=rank, num_replicas=world_size, shuffle=False
        )
        direct = UnrepeatedDistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=False)
        assert list(wrapper) == list(direct)


def test_index_batch_sampler():
    """Test `IndexBatchSampler` properly extracts indices."""
    dataset = range(15)
    sampler = SequentialSampler(dataset)
    batch_sampler = BatchSampler(sampler, 3, False)
    index_batch_sampler = _IndexBatchSamplerWrapper(batch_sampler)

    assert batch_sampler.batch_size == index_batch_sampler.batch_size
    assert batch_sampler.drop_last == index_batch_sampler.drop_last
    assert batch_sampler.sampler is sampler
    assert index_batch_sampler.sampler is sampler
    assert list(index_batch_sampler) == index_batch_sampler.seen_batch_indices
    assert list(index_batch_sampler) == list(batch_sampler)

    assert isinstance(index_batch_sampler, Iterable)
    assert has_len(index_batch_sampler)

    iterator = iter(index_batch_sampler)
    assert index_batch_sampler.seen_batch_indices == []
    b0 = next(iterator)
    assert b0 == [0, 1, 2]
    assert index_batch_sampler.seen_batch_indices == [b0]
    b1 = next(iterator)
    assert b1 == [3, 4, 5]
    assert index_batch_sampler.seen_batch_indices == [b0, b1]
