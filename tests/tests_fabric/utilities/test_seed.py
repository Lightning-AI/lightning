import os
import random
import warnings
from unittest import mock
from unittest.mock import Mock

import numpy
import pytest
import torch

from lightning.fabric.utilities.seed import (
    _collect_rng_states,
    _set_rng_states,
    pl_worker_init_function,
    reset_seed,
    seed_everything,
)


@mock.patch.dict(os.environ, clear=True)
def test_default_seed():
    """Test that the default seed is 0 when no seed provided and no environment variable set."""
    assert seed_everything() == 0
    assert os.environ["PL_GLOBAL_SEED"] == "0"


@mock.patch.dict(os.environ, {}, clear=True)
def test_seed_stays_same_with_multiple_seed_everything_calls():
    """Ensure that after the initial seed everything, the seed stays the same for the same run."""
    with pytest.warns(UserWarning, match="No seed found"):
        seed_everything()
    initial_seed = os.environ.get("PL_GLOBAL_SEED")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        seed_everything()
    seed = os.environ.get("PL_GLOBAL_SEED")

    assert initial_seed == seed


@mock.patch.dict(os.environ, {"PL_GLOBAL_SEED": "2020"}, clear=True)
def test_correct_seed_with_environment_variable():
    """Ensure that the PL_GLOBAL_SEED environment is read."""
    assert seed_everything() == 2020


@mock.patch.dict(os.environ, {"PL_GLOBAL_SEED": "invalid"}, clear=True)
def test_invalid_seed():
    """Ensure that we still fix the seed even if an invalid seed is given."""
    with pytest.warns(UserWarning, match="Invalid seed found"):
        seed = seed_everything()
    assert seed == 0


@mock.patch.dict(os.environ, {}, clear=True)
@pytest.mark.parametrize("seed", [10e9, -10e9])
def test_out_of_bounds_seed(seed):
    """Ensure that we still fix the seed even if an out-of-bounds seed is given."""
    with pytest.warns(UserWarning, match="is not in bounds"):
        actual = seed_everything(seed)
    assert actual == 0


def test_reset_seed_no_op():
    """Test that the reset_seed function is a no-op when seed_everything() was not used."""
    assert "PL_GLOBAL_SEED" not in os.environ
    seed_before = torch.initial_seed()
    reset_seed()
    assert torch.initial_seed() == seed_before
    assert "PL_GLOBAL_SEED" not in os.environ


@pytest.mark.parametrize("workers", [True, False])
def test_reset_seed_everything(workers):
    """Test that we can reset the seed to the initial value set by seed_everything()"""
    assert "PL_GLOBAL_SEED" not in os.environ
    assert "PL_SEED_WORKERS" not in os.environ

    seed_everything(123, workers)
    before = torch.rand(1)
    assert os.environ["PL_GLOBAL_SEED"] == "123"
    assert os.environ["PL_SEED_WORKERS"] == str(int(workers))

    reset_seed()
    after = torch.rand(1)
    assert os.environ["PL_GLOBAL_SEED"] == "123"
    assert os.environ["PL_SEED_WORKERS"] == str(int(workers))
    assert torch.allclose(before, after)


def test_reset_seed_non_verbose(caplog):
    seed_everything(123)
    assert len(caplog.records) == 1
    caplog.clear()
    reset_seed()  # should call `seed_everything(..., verbose=False)`
    assert len(caplog.records) == 0


def test_backward_compatibility_rng_states_dict():
    """Test that an older rng_states_dict without the "torch.cuda" key does not crash."""
    states = _collect_rng_states()
    assert "torch.cuda" in states
    states.pop("torch.cuda")
    _set_rng_states(states)


@mock.patch("lightning.fabric.utilities.seed.torch.cuda.is_available", Mock(return_value=False))
@mock.patch("lightning.fabric.utilities.seed.torch.cuda.get_rng_state_all")
def test_collect_rng_states_if_cuda_init_fails(get_rng_state_all_mock):
    """Test that the `torch.cuda` rng states are only requested if CUDA is available."""
    get_rng_state_all_mock.side_effect = RuntimeError("The NVIDIA driver on your system is too old")
    states = _collect_rng_states()
    assert states["torch.cuda"] == []


@pytest.mark.parametrize(("num_workers", "num_ranks"), [(64, 64)])
@pytest.mark.parametrize("base_seed", [100, 1024, 2**32 - 1])
def test_pl_worker_init_function(base_seed, num_workers, num_ranks):
    """Test that Lightning's `worker_init_fn` sets unique seeds per worker/rank derived from the base seed."""
    torch_rands = set()
    stdlib_rands = set()
    numpy_rands = set()

    for worker_id in range(num_workers):
        for rank in range(num_ranks):
            seed_everything(base_seed)
            pl_worker_init_function(worker_id, rank)
            torch_rands.add(tuple(torch.randint(0, 1_000_000, (100,)).tolist()))
            stdlib_rands.add(tuple(random.randint(0, 1_000_000) for _ in range(100)))
            numpy_rands.add(tuple(numpy.random.randint(0, 1_000_000, (100,)).tolist()))

    # Assert there are no duplicates (no collisions)
    assert len(torch_rands) == num_ranks * num_workers
    assert len(stdlib_rands) == num_ranks * num_workers
    assert len(numpy_rands) == num_ranks * num_workers
    assert len(torch_rands | stdlib_rands | numpy_rands) == 3 * num_workers * num_ranks
