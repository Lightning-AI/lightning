import os
import shutil
from unittest import mock

import numpy as np
from lightning.data.streaming.cache import Cache
from lightning.data.streaming.config import ChunkedIndex
from lightning.data.streaming.reader import _get_folder_size, _maybe_flush_cache, _try_to_delete_oldest_chunk
from lightning_cloud.resolver import Dir


def test_reader_chunk_removal(tmpdir, monkeypatch):
    cache_dir = os.path.join(tmpdir, "cache_dir")
    remote_dir = os.path.join(tmpdir, "remote_dir")
    os.makedirs(cache_dir, exist_ok=True)
    cache = Cache(input_dir=Dir(path=cache_dir, url=remote_dir), chunk_size=2, max_cache_size=28020)

    for i in range(25):
        cache[i] = i

    cache.done()
    cache.merge()

    shutil.copytree(cache_dir, remote_dir)
    shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    for i in range(25):
        index = ChunkedIndex(i, cache._get_chunk_index_from_index(i), is_last_index=i == 24)
        assert cache[index] == i

    assert len(os.listdir(cache_dir)) == 14

    cache = Cache(input_dir=Dir(path=cache_dir, url=remote_dir), chunk_size=2, max_cache_size=2800)

    shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    generated = []
    for i in range(25):
        generated.append([i, len(os.listdir(cache_dir))])
        index = ChunkedIndex(i, cache._get_chunk_index_from_index(i), is_last_index=i == 24)
        assert cache[index] == i

    assert generated == [
        [0, 0],
        [1, 2],
        [2, 2],
        [3, 2],
        [4, 2],
        [5, 2],
        [6, 2],
        [7, 2],
        [8, 2],
        [9, 2],
        [10, 2],
        [11, 2],
        [12, 2],
        [13, 2],
        [14, 2],
        [15, 2],
        [16, 2],
        [17, 2],
        [18, 2],
        [19, 2],
        [20, 2],
        [21, 2],
        [22, 2],
        [23, 2],
        [24, 2],
    ]

    assert len(os.listdir(cache_dir)) == 2


def test_get_folder_size(tmpdir):
    array = np.zeros((10, 10))

    np.save(os.path.join(tmpdir, "array_1.npy"), array)
    np.save(os.path.join(tmpdir, "array_2.npy"), array)

    assert _get_folder_size(tmpdir) == 928 * 2


def test_try_to_delete_oldest_chunk(tmpdir):
    with open(os.path.join(tmpdir, "chunk_0.bin"), "w") as f:
        f.write("Hello World")

    with open(os.path.join(tmpdir, "chunk_1.bin"), "w") as f:
        f.write("Hello World")

    assert len(os.listdir(tmpdir)) == 2

    assert _try_to_delete_oldest_chunk(tmpdir)
    assert os.listdir(tmpdir) == ["chunk_1.bin"]

    assert _try_to_delete_oldest_chunk(tmpdir)
    assert os.listdir(tmpdir) == []

    assert not _try_to_delete_oldest_chunk(tmpdir)


def test_maybe_flush_cache(tmpdir):
    with open(os.path.join(tmpdir, "chunk_0.bin"), "w") as f:
        f.write("Hello World")

    with open(os.path.join(tmpdir, "chunk_1.bin"), "w") as f:
        f.write("Hello World")

    assert len(os.listdir(tmpdir)) == 2

    config = mock.MagicMock()
    config._cache_dir = tmpdir

    config.__getitem__.return_value = (os.path.join(tmpdir, "a.txt"), 1, 1)

    _maybe_flush_cache(tmpdir, 0, 0.1, config)

    assert len(os.listdir(tmpdir)) == 0
