import os
from unittest.mock import ANY

import pytest
import torch
from lightning.data.streaming.cache import Cache
from lightning.data.streaming.combined import CombinedStreamingDataset
from lightning.data.streaming.dataloader import StreamingDataLoader
from lightning.data.streaming.dataset import Dir, StreamingDataset
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader


def test_combined_dataset_num_samples_yield():
    dataset = CombinedStreamingDataset([range(10), range(0, -10, -1)], 42, weights=(0.5, 0.5))
    dataset_iter = iter(dataset)

    data = list(dataset_iter)
    assert data == [0, 0, 1, 2, -1, -2, -3, 3, 4, 5, 6, -4, 7, 8, -5, -6, 9, -7, -8]

    dataset = CombinedStreamingDataset([range(10), range(0, -10, -1)], 37, weights=(0.5, 0.5))
    dataset_iter = iter(dataset)

    data = list(dataset_iter)
    assert data == [0, 0, -1, -2, -3, -4, -5, 1, -6, 2, -7, -8, 3, 4, -9, 5]

    dataset = CombinedStreamingDataset([range(10), range(0, -10, -1)], 23, weights=(0.5, 0.5))
    dataset_iter = iter(dataset)

    data = [next(dataset_iter) for _ in range(5)]
    assert data == [0, -1, -2, 0, -3]
    assert dataset._iterator._num_samples_yielded == [1, 4]
    assert next(dataset_iter) == 1
    assert dataset._iterator._num_samples_yielded == [2, 4]


class TestStatefulDataset:
    def __init__(self, size, step):
        self.size = size
        self.step = step
        self.counter = 0

    def __len__(self):
        return self.size

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        if self.counter == self.size:
            raise StopIteration
        value = self.step * self.counter
        self.counter += 1
        return value

    def state_dict(self, *args, **kwargs):
        return {"counter": self.counter}

    def load_state_dict(self, state_dict):
        self.counter = state_dict["counter"]


def test_combined_dataset_state_dict():
    dataset = CombinedStreamingDataset(
        [TestStatefulDataset(10, 1), TestStatefulDataset(10, -1)], 42, weights=(0.5, 0.5)
    )
    assert dataset.state_dict(0, 1) == {}
    dataset_iter = iter(dataset)
    assert dataset.state_dict(0, 1) == {"0": {"counter": 0}, "1": {"counter": 0}}

    dataset2 = CombinedStreamingDataset(
        [TestStatefulDataset(10, 1), TestStatefulDataset(10, -1)], 42, weights=(0.5, 0.5)
    )
    assert dataset2.state_dict(0, 1) == {}

    data = []
    states = []
    for i, value in enumerate(dataset_iter):
        state = dataset.state_dict(i, 1)
        data.append(value)
        states.append(state)

    assert data == [0, 0, 1, 2, -1, -2, -3, 3, 4, 5, 6, -4, 7, 8, -5, -6, 9, -7, -8]
    assert states == [
        {"0": {"counter": 0}, "1": {"counter": 1}},
        {"0": {"counter": 1}, "1": {"counter": 1}},
        {"0": {"counter": 2}, "1": {"counter": 1}},
        {"0": {"counter": 3}, "1": {"counter": 1}},
        {"0": {"counter": 3}, "1": {"counter": 2}},
        {"0": {"counter": 3}, "1": {"counter": 3}},
        {"0": {"counter": 3}, "1": {"counter": 4}},
        {"0": {"counter": 4}, "1": {"counter": 4}},
        {"0": {"counter": 5}, "1": {"counter": 4}},
        {"0": {"counter": 6}, "1": {"counter": 4}},
        {"0": {"counter": 7}, "1": {"counter": 4}},
        {"0": {"counter": 7}, "1": {"counter": 5}},
        {"0": {"counter": 8}, "1": {"counter": 5}},
        {"0": {"counter": 9}, "1": {"counter": 5}},
        {"0": {"counter": 9}, "1": {"counter": 6}},
        {"0": {"counter": 9}, "1": {"counter": 7}},
        {"0": {"counter": 10}, "1": {"counter": 7}},
        {"0": {"counter": 10}, "1": {"counter": 8}},
        {"0": {"counter": 10}, "1": {"counter": 9}},
    ]

    dataset2 = CombinedStreamingDataset(
        [TestStatefulDataset(10, 1), TestStatefulDataset(10, -1)], 42, weights=(0.5, 0.5)
    )
    assert dataset2.state_dict(0, 1) == {}
    dataset2_iter = iter(dataset2)

    data_2 = []
    for state in states:
        dataset.load_state_dict(state)
        data_2.append(next(dataset2_iter))

    assert data == data_2


@pytest.mark.parametrize(
    ("weights", "expected"),
    [
        ([1], [1]),
        ([2], [1]),
        ([2, 0.5], [0.8, 0.2]),
        ([1, 1, 1], [1 / 3, 1 / 3, 1 / 3]),
        ([0.3, 0, 0], [1.0, 0, 0]),
        (None, [0.5, 0.5]),
    ],
)
def test_combined_dataset_normalizes_weights(weights, expected):
    combined_dataset = CombinedStreamingDataset([[1], [2, 3]], weights=weights, seed=1)
    assert combined_dataset._weights == expected


class SimpleDataset(IterableDataset):
    def __init__(self, start, end):
        super().__init__()
        self._start = start
        self._end = end

    def __iter__(self):
        return iter(range(self._start, self._end))

    def state_dict(self, **kwargs):
        return kwargs


def test_combined_dataset():
    dataset1 = SimpleDataset(0, 10)
    dataset2 = SimpleDataset(10, 20)
    dataset = CombinedStreamingDataset(datasets=[dataset1, dataset2], weights=[1.0, 0.0], seed=12345)

    res = list(dataset)
    assert res == list(range(0, 10))

    dataset1 = SimpleDataset(0, 10)
    dataset2 = SimpleDataset(10, 20)
    dataset = CombinedStreamingDataset(datasets=[dataset1, dataset2], weights=[0.0, 1.0], seed=12345)

    res = list(dataset)
    assert res == list(range(10, 20))

    dataset1 = SimpleDataset(0, 10)
    dataset2 = SimpleDataset(10, 20)
    dataset = CombinedStreamingDataset(datasets=[dataset1, dataset2], weights=[0.5, 0.5], seed=12345)

    res = list(dataset)
    assert 9 in res or 19 in res
    if len(res) > 10:
        assert 0 in res
        assert 10 in res

    dataset1 = SimpleDataset(0, 10)
    dataset2 = SimpleDataset(10, 20)
    dataset = CombinedStreamingDataset(datasets=[dataset1, dataset2], weights=[0.5, 0.5], seed=12345)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=1)
    dataloader_iter = iter(dataloader)
    assert torch.equal(next(dataloader_iter), torch.Tensor([0, 1]))


@pytest.mark.parametrize("batch_size", [1, 2])
def test_combined_dataset_with_dataloader_and_one_worker(batch_size):
    dataset1 = SimpleDataset(0, 10)
    dataset2 = SimpleDataset(10, 20)
    dataset = CombinedStreamingDataset(datasets=[dataset1, dataset2], weights=[0.5, 0.5], seed=12345)
    dataloader = StreamingDataLoader(dataset, num_workers=1, batch_size=batch_size, prefetch_factor=1)
    dataloader_iter = iter(dataloader)

    if batch_size == 2:
        assert torch.equal(next(dataloader_iter), torch.Tensor([0, 1]))
        assert torch.equal(next(dataloader_iter), torch.Tensor([10, 2]))
        assert torch.equal(next(dataloader_iter), torch.Tensor([3, 4]))
        assert torch.equal(next(dataloader_iter), torch.Tensor([11, 5]))
        assert torch.equal(next(dataloader_iter), torch.Tensor([6, 7]))
        assert torch.equal(next(dataloader_iter), torch.Tensor([12, 8]))

    else:
        assert torch.equal(next(dataloader_iter), torch.Tensor([0]))
        assert torch.equal(next(dataloader_iter), torch.Tensor([1]))
        assert torch.equal(next(dataloader_iter), torch.Tensor([10]))
        assert torch.equal(next(dataloader_iter), torch.Tensor([2]))
        assert torch.equal(next(dataloader_iter), torch.Tensor([3]))
        assert torch.equal(next(dataloader_iter), torch.Tensor([4]))
        assert torch.equal(next(dataloader_iter), torch.Tensor([11]))
        assert torch.equal(next(dataloader_iter), torch.Tensor([5]))
        assert torch.equal(next(dataloader_iter), torch.Tensor([6]))
        assert torch.equal(next(dataloader_iter), torch.Tensor([7]))
        assert torch.equal(next(dataloader_iter), torch.Tensor([12]))
        assert torch.equal(next(dataloader_iter), torch.Tensor([8]))

    assert dataloader.state_dict() == {
        "0": {"num_samples_yielded": 9, "num_workers": 1, "batch_size": batch_size},
        "1": {"num_samples_yielded": 3, "num_workers": 1, "batch_size": batch_size},
    }


def test_combined_dataset_with_dataloader_2_epochs(tmpdir):
    data_dir_1 = os.path.join(tmpdir, "data_1")
    data_dir_2 = os.path.join(tmpdir, "data_2")
    cache_dir_1 = os.path.join(tmpdir, "cache_dir_1")
    cache_dir_2 = os.path.join(tmpdir, "cache_dir_2")

    os.makedirs(data_dir_1)
    os.makedirs(data_dir_2)
    os.makedirs(cache_dir_1)
    os.makedirs(cache_dir_2)

    cache = Cache(input_dir=str(data_dir_1), chunk_size=2)

    for i in range(10):
        cache[i] = i

    cache.done()
    cache.merge()

    cache = Cache(input_dir=str(data_dir_2), chunk_size=2)

    for i in range(10):
        cache[i] = i + 5

    cache.done()
    cache.merge()

    dataset1 = StreamingDataset(input_dir=Dir(cache_dir_1, data_dir_1), shuffle=True)
    dataset2 = StreamingDataset(input_dir=Dir(cache_dir_2, data_dir_2), shuffle=True)
    dataset = CombinedStreamingDataset(datasets=[dataset1, dataset2], weights=[0.5, 0.5], seed=12345)
    dataloader = StreamingDataLoader(dataset, num_workers=1, batch_size=2)

    assert dataset1.current_epoch == 1
    assert dataset2.current_epoch == 1

    batches_1 = []
    states_1 = []
    for batch in dataloader:
        batches_1.append(batch)
        states_1.append(dataloader.state_dict())

    assert dataset1.current_epoch == 1
    assert dataset2.current_epoch == 1

    batches_2 = []
    states_2 = []
    for batch in dataloader:
        batches_2.append(batch)
        states_2.append(dataloader.state_dict())
    assert dataset1.current_epoch == 2
    assert dataset2.current_epoch == 2

    assert sum(torch.equal(b1, b2) for b1, b2 in zip(batches_1, batches_2)) == 0

    assert states_1 == [
        {
            "0": {
                "0": {
                    "num_samples_yielded": 2,
                    "num_workers": 1,
                    "batch_size": 2,
                    "current_epoch": 1,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                }
            },
            "1": {
                "0": {
                    "num_samples_yielded": 0,
                    "num_workers": 1,
                    "batch_size": 2,
                    "current_epoch": 1,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                }
            },
        },
        {
            "0": {
                "0": {
                    "num_samples_yielded": 3,
                    "num_workers": 1,
                    "batch_size": 2,
                    "current_epoch": 1,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                }
            },
            "1": {
                "0": {
                    "num_samples_yielded": 1,
                    "num_workers": 1,
                    "batch_size": 2,
                    "current_epoch": 1,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                }
            },
        },
        {
            "0": {
                "0": {
                    "num_samples_yielded": 5,
                    "num_workers": 1,
                    "batch_size": 2,
                    "current_epoch": 1,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                }
            },
            "1": {
                "0": {
                    "num_samples_yielded": 1,
                    "num_workers": 1,
                    "batch_size": 2,
                    "current_epoch": 1,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                }
            },
        },
        {
            "0": {
                "0": {
                    "num_samples_yielded": 6,
                    "num_workers": 1,
                    "batch_size": 2,
                    "current_epoch": 1,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                }
            },
            "1": {
                "0": {
                    "num_samples_yielded": 2,
                    "num_workers": 1,
                    "batch_size": 2,
                    "current_epoch": 1,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                }
            },
        },
        {
            "0": {
                "0": {
                    "num_samples_yielded": 8,
                    "num_workers": 1,
                    "batch_size": 2,
                    "current_epoch": 1,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                }
            },
            "1": {
                "0": {
                    "num_samples_yielded": 2,
                    "num_workers": 1,
                    "batch_size": 2,
                    "current_epoch": 1,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                }
            },
        },
        {
            "0": {
                "0": {
                    "num_samples_yielded": 9,
                    "num_workers": 1,
                    "batch_size": 2,
                    "current_epoch": 1,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                }
            },
            "1": {
                "0": {
                    "num_samples_yielded": 3,
                    "num_workers": 1,
                    "batch_size": 2,
                    "current_epoch": 1,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                }
            },
        },
        {
            "0": {
                "0": {
                    "num_samples_yielded": 10,
                    "num_workers": 1,
                    "batch_size": 2,
                    "current_epoch": 1,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                }
            },
            "1": {
                "0": {
                    "num_samples_yielded": 4,
                    "num_workers": 1,
                    "batch_size": 2,
                    "current_epoch": 1,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                }
            },
        },
        {
            "0": {
                "0": {
                    "num_samples_yielded": 11,
                    "num_workers": 1,
                    "batch_size": 2,
                    "current_epoch": 1,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                }
            },
            "1": {
                "0": {
                    "num_samples_yielded": 5,
                    "num_workers": 1,
                    "batch_size": 2,
                    "current_epoch": 1,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                }
            },
        },
    ]

    assert states_2 == [
        {
            "0": {
                "0": {
                    "num_samples_yielded": 2,
                    "num_workers": 1,
                    "batch_size": 2,
                    "current_epoch": 2,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                }
            },
            "1": {
                "0": {
                    "num_samples_yielded": 0,
                    "num_workers": 1,
                    "batch_size": 2,
                    "current_epoch": 2,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                }
            },
        },
        {
            "0": {
                "0": {
                    "num_samples_yielded": 3,
                    "num_workers": 1,
                    "batch_size": 2,
                    "current_epoch": 2,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                }
            },
            "1": {
                "0": {
                    "num_samples_yielded": 1,
                    "num_workers": 1,
                    "batch_size": 2,
                    "current_epoch": 2,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                }
            },
        },
        {
            "0": {
                "0": {
                    "num_samples_yielded": 5,
                    "num_workers": 1,
                    "batch_size": 2,
                    "current_epoch": 2,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                }
            },
            "1": {
                "0": {
                    "num_samples_yielded": 1,
                    "num_workers": 1,
                    "batch_size": 2,
                    "current_epoch": 2,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                }
            },
        },
        {
            "0": {
                "0": {
                    "num_samples_yielded": 6,
                    "num_workers": 1,
                    "batch_size": 2,
                    "current_epoch": 2,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                }
            },
            "1": {
                "0": {
                    "num_samples_yielded": 2,
                    "num_workers": 1,
                    "batch_size": 2,
                    "current_epoch": 2,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                }
            },
        },
        {
            "0": {
                "0": {
                    "num_samples_yielded": 8,
                    "num_workers": 1,
                    "batch_size": 2,
                    "current_epoch": 2,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                }
            },
            "1": {
                "0": {
                    "num_samples_yielded": 2,
                    "num_workers": 1,
                    "batch_size": 2,
                    "current_epoch": 2,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                }
            },
        },
        {
            "0": {
                "0": {
                    "num_samples_yielded": 9,
                    "num_workers": 1,
                    "batch_size": 2,
                    "current_epoch": 2,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                }
            },
            "1": {
                "0": {
                    "num_samples_yielded": 3,
                    "num_workers": 1,
                    "batch_size": 2,
                    "current_epoch": 2,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                }
            },
        },
        {
            "0": {
                "0": {
                    "num_samples_yielded": 10,
                    "num_workers": 1,
                    "batch_size": 2,
                    "current_epoch": 2,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                }
            },
            "1": {
                "0": {
                    "num_samples_yielded": 4,
                    "num_workers": 1,
                    "batch_size": 2,
                    "current_epoch": 2,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                }
            },
        },
        {
            "0": {
                "0": {
                    "num_samples_yielded": 11,
                    "num_workers": 1,
                    "batch_size": 2,
                    "current_epoch": 2,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                }
            },
            "1": {
                "0": {
                    "num_samples_yielded": 5,
                    "num_workers": 1,
                    "batch_size": 2,
                    "current_epoch": 2,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                }
            },
        },
    ]
