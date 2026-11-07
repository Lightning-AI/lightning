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
import contextlib
from collections.abc import Iterable, Iterator
from typing import Any, Callable, Literal, Optional, Union

from torch.utils.data.dataloader import _BaseDataLoaderIter, _MultiProcessingDataLoaderIter
from typing_extensions import Self, TypedDict, override

from lightning.fabric.utilities.data import sized_len
from lightning.fabric.utilities.types import _Stateful
from lightning.pytorch.utilities._pytree import _map_and_unflatten, _tree_flatten, tree_unflatten

_ITERATOR_RETURN = tuple[Any, int, int]  # batch, batch_idx, dataloader_idx


class _ModeIterator(Iterator[_ITERATOR_RETURN]):
    def __init__(self, iterables: list[Iterable], limits: Optional[list[Union[int, float]]] = None) -> None:
        if limits is not None and len(limits) != len(iterables):
            raise ValueError(f"Mismatch in number of limits ({len(limits)}) and number of iterables ({len(iterables)})")
        self.iterables = iterables
        self.iterators: list[Iterator] = []
        self._idx = 0  # what would be batch_idx
        self.limits = limits

    @override
    def __next__(self) -> _ITERATOR_RETURN:
        raise NotImplementedError

    @override
    def __iter__(self) -> Self:
        self.iterators = [iter(iterable) for iterable in self.iterables]
        self._idx = 0
        return self

    def __len__(self) -> int:
        raise NotImplementedError

    def reset(self) -> None:
        self.iterators = []
        self._idx = 0

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()

        # workaround an inconvenient `NotImplementedError`:
        # https://github.com/pytorch/pytorch/blob/v2.0.0/torch/utils/data/dataloader.py#L652-L658
        state["iterators"] = [
            None if isinstance(iterator, _BaseDataLoaderIter) else iterator_state
            for iterator, iterator_state in zip(self.iterators, state["iterators"])
        ]

        return state


class _MaxSizeCycle(_ModeIterator):
    def __init__(self, iterables: list[Iterable], limits: Optional[list[Union[int, float]]] = None) -> None:
        super().__init__(iterables, limits)
        self._consumed: list[bool] = []

    @override
    def __next__(self) -> _ITERATOR_RETURN:
        n = len(self.iterators)
        out = [None] * n  # values per iterator
        for i in range(n):
            try:
                out[i] = next(self.iterators[i])
            except StopIteration:
                self._consumed[i] = True
                if all(self._consumed):
                    raise
                # reset the consumed dataloader
                self.iterators[i] = iter(self.iterables[i])
                out[i] = next(self.iterators[i])
        index = self._idx
        self._idx += 1
        return out, index, 0

    @override
    def __iter__(self) -> Self:
        super().__iter__()
        self._consumed = [False] * len(self.iterables)
        return self

    @override
    def __len__(self) -> int:
        lengths = _get_iterables_lengths(self.iterables)
        if self.limits is not None:
            return max(min(length, limit) for length, limit in zip(lengths, self.limits))  # type: ignore[return-value]
        return max(lengths)  # type: ignore[return-value]

    @override
    def reset(self) -> None:
        super().reset()
        self._consumed = []


class _MinSize(_ModeIterator):
    @override
    def __next__(self) -> _ITERATOR_RETURN:
        out = [next(it) for it in self.iterators]
        index = self._idx
        self._idx += 1
        return out, index, 0

    @override
    def __len__(self) -> int:
        lengths = _get_iterables_lengths(self.iterables)
        return min(lengths + self.limits) if self.limits is not None else min(lengths)  # type: ignore[return-value]


class _Sequential(_ModeIterator):
    def __init__(self, iterables: list[Iterable], limits: Optional[list[Union[int, float]]] = None) -> None:
        super().__init__(iterables, limits)
        self._iterator_idx = 0  # what would be dataloader_idx

    @override
    def __next__(self) -> _ITERATOR_RETURN:
        n = len(self.iterables)
        while True:
            if n == 0 or self._iterator_idx >= n:
                raise StopIteration

            # if limits are set, go to the correct iterator
            if self.limits is not None:
                while self.limits[self._iterator_idx] <= self._idx:
                    self._use_next_iterator()
                    if self._iterator_idx >= n:
                        raise StopIteration

            try:
                out = next(self.iterators[0])
                break
            except StopIteration:
                # try the next iterator
                self._use_next_iterator()

        index = self._idx
        self._idx += 1
        return out, index, self._iterator_idx

    @override
    def __iter__(self) -> Self:
        self._iterator_idx = 0
        self._idx = 0
        self._load_current_iterator()
        return self

    @override
    def __len__(self) -> int:
        lengths = _get_iterables_lengths(self.iterables)
        if self.limits is not None:
            return sum(min(length, limit) for length, limit in zip(lengths, self.limits))  # type: ignore[misc]
        return sum(lengths)  # type: ignore[arg-type]

    @override
    def reset(self) -> None:
        super().reset()
        self._iterator_idx = 0

    def _load_current_iterator(self) -> None:
        # Load a single DataLoader, prevents multiple sets of workers from starting unnecessarily
        if self._iterator_idx < len(self.iterables):
            self.iterators = [iter(self.iterables[self._iterator_idx])]
        else:
            # No more iterables to step through, return an empty list
            self.iterators = []

    def _use_next_iterator(self) -> None:
        self._iterator_idx += 1
        self._idx = 0
        self._load_current_iterator()


class _MaxSize(_ModeIterator):
    @override
    def __next__(self) -> _ITERATOR_RETURN:
        n = len(self.iterators)
        out = [None] * n
        all_exhausted = True
        for i in range(n):
            with contextlib.suppress(StopIteration):
                out[i] = next(self.iterators[i])
                all_exhausted = False
        if all_exhausted:
            raise StopIteration
        index = self._idx
        self._idx += 1
        return out, index, 0

    @override
    def __len__(self) -> int:
        lengths = _get_iterables_lengths(self.iterables)
        if self.limits is not None:
            return max(min(length, limit) for length, limit in zip(lengths, self.limits))  # type: ignore[return-value]
        return max(lengths)  # type: ignore[return-value]


class _CombinationMode(TypedDict):
    fn: Callable[[list[int]], int]
    iterator: type[_ModeIterator]


_SUPPORTED_MODES = {
    "min_size": _CombinationMode(fn=min, iterator=_MinSize),
    "max_size_cycle": _CombinationMode(fn=max, iterator=_MaxSizeCycle),
    "max_size": _CombinationMode(fn=max, iterator=_MaxSize),
    "sequential": _CombinationMode(fn=sum, iterator=_Sequential),
}
_LITERAL_SUPPORTED_MODES = Literal["min_size", "max_size_cycle", "max_size", "sequential"]


class CombinedLoader(Iterable):
    """Combines different iterables under specific sampling modes.

    Args:
        iterables: the iterable or collection of iterables to sample from.
        mode: the mode to use. The following modes are supported:

            * ``min_size``: stops after the shortest iterable (the one with the lowest number of items) is done.
            * ``max_size_cycle``: stops after the longest iterable (the one with most items) is done, while cycling
              through the rest of the iterables.
            * ``max_size``: stops after the longest iterable (the one with most items) is done, while returning None
              for the exhausted iterables.
            * ``sequential``: completely consumes each iterable sequentially, and returns a triplet
              ``(data, idx, iterable_idx)``

    Examples:
        >>> from torch.utils.data import DataLoader
        >>> iterables = {'a': DataLoader(range(6), batch_size=4),
        ...              'b': DataLoader(range(15), batch_size=5)}
        >>> combined_loader = CombinedLoader(iterables, 'max_size_cycle')
        >>> _ = iter(combined_loader)
        >>> len(combined_loader)
        3
        >>> for batch, batch_idx, dataloader_idx in combined_loader:
        ...     print(f"{batch}, {batch_idx=}, {dataloader_idx=}")
        {'a': tensor([0, 1, 2, 3]), 'b': tensor([0, 1, 2, 3, 4])}, batch_idx=0, dataloader_idx=0
        {'a': tensor([4, 5]), 'b': tensor([5, 6, 7, 8, 9])}, batch_idx=1, dataloader_idx=0
        {'a': tensor([0, 1, 2, 3]), 'b': tensor([10, 11, 12, 13, 14])}, batch_idx=2, dataloader_idx=0

        >>> combined_loader = CombinedLoader(iterables, 'max_size')
        >>> _ = iter(combined_loader)
        >>> len(combined_loader)
        3
        >>> for batch, batch_idx, dataloader_idx in combined_loader:
        ...     print(f"{batch}, {batch_idx=}, {dataloader_idx=}")
        {'a': tensor([0, 1, 2, 3]), 'b': tensor([0, 1, 2, 3, 4])}, batch_idx=0, dataloader_idx=0
        {'a': tensor([4, 5]), 'b': tensor([5, 6, 7, 8, 9])}, batch_idx=1, dataloader_idx=0
        {'a': None, 'b': tensor([10, 11, 12, 13, 14])}, batch_idx=2, dataloader_idx=0

        >>> combined_loader = CombinedLoader(iterables, 'min_size')
        >>> _ = iter(combined_loader)
        >>> len(combined_loader)
        2
        >>> for batch, batch_idx, dataloader_idx in combined_loader:
        ...     print(f"{batch}, {batch_idx=}, {dataloader_idx=}")
        {'a': tensor([0, 1, 2, 3]), 'b': tensor([0, 1, 2, 3, 4])}, batch_idx=0, dataloader_idx=0
        {'a': tensor([4, 5]), 'b': tensor([5, 6, 7, 8, 9])}, batch_idx=1, dataloader_idx=0

        >>> combined_loader = CombinedLoader(iterables, 'sequential')
        >>> _ = iter(combined_loader)
        >>> len(combined_loader)
        5
        >>> for batch, batch_idx, dataloader_idx in combined_loader:
        ...     print(f"{batch}, {batch_idx=}, {dataloader_idx=}")
        tensor([0, 1, 2, 3]), batch_idx=0, dataloader_idx=0
        tensor([4, 5]), batch_idx=1, dataloader_idx=0
        tensor([0, 1, 2, 3, 4]), batch_idx=0, dataloader_idx=1
        tensor([5, 6, 7, 8, 9]), batch_idx=1, dataloader_idx=1
        tensor([10, 11, 12, 13, 14]), batch_idx=2, dataloader_idx=1

    """

    def __init__(self, iterables: Any, mode: _LITERAL_SUPPORTED_MODES = "min_size") -> None:
        if mode not in _SUPPORTED_MODES:
            raise ValueError(f"Unsupported mode {mode!r}, please select one of: {list(_SUPPORTED_MODES)}.")
        self._iterables = iterables
        self._flattened, self._spec = _tree_flatten(iterables)
        self._mode = mode
        self._iterator: Optional[_ModeIterator] = None
        self._limits: Optional[list[Union[int, float]]] = None

    @property
    def iterables(self) -> Any:
        """Return the original collection of iterables."""
        return self._iterables

    @property
    def sampler(self) -> Any:
        """Return a collections of samplers extracted from iterables."""
        return _map_and_unflatten(lambda x: getattr(x, "sampler", None), self.flattened, self._spec)

    @property
    def batch_sampler(self) -> Any:
        """Return a collections of batch samplers extracted from iterables."""
        return _map_and_unflatten(lambda x: getattr(x, "batch_sampler", None), self.flattened, self._spec)

    @property
    def flattened(self) -> list[Any]:
        """Return the flat list of iterables."""
        return self._flattened

    @flattened.setter
    def flattened(self, flattened: list[Any]) -> None:
        """Setter to conveniently update the list of iterables."""
        if len(flattened) != len(self._flattened):
            raise ValueError(
                f"Mismatch in flattened length ({len(flattened)}) and existing length ({len(self._flattened)})"
            )
        # update the iterable collection
        self._iterables = tree_unflatten(flattened, self._spec)
        self._flattened = flattened

    @property
    def limits(self) -> Optional[list[Union[int, float]]]:
        """Optional limits per iterator."""
        return self._limits

    @limits.setter
    def limits(self, limits: Optional[Union[int, float, list[Union[int, float]]]]) -> None:
        if isinstance(limits, (int, float)):
            limits = [limits] * len(self.flattened)
        elif isinstance(limits, list) and len(limits) != len(self.flattened):
            raise ValueError(
                f"Mismatch in number of limits ({len(limits)}) and number of iterables ({len(self.flattened)})"
            )
        self._limits = limits

    def __next__(self) -> _ITERATOR_RETURN:
        assert self._iterator is not None
        out = next(self._iterator)
        if isinstance(self._iterator, _Sequential):
            return out
        out, batch_idx, dataloader_idx = out
        return tree_unflatten(out, self._spec), batch_idx, dataloader_idx

    @override
    def __iter__(self) -> Self:
        cls = _SUPPORTED_MODES[self._mode]["iterator"]
        iterator = cls(self.flattened, self._limits)
        iter(iterator)
        self._iterator = iterator
        return self

    def __len__(self) -> int:
        """Compute the number of batches."""
        if self._iterator is None:
            raise RuntimeError("Please call `iter(combined_loader)` first.")
        return len(self._iterator)

    def reset(self) -> None:
        """Reset the state and shutdown any workers."""
        if self._iterator is not None:
            self._iterator.reset()
            self._iterator = None
        for iterable in self.flattened:
            _shutdown_workers_and_reset_iterator(iterable)

    def _dataset_length(self) -> int:
        """Compute the total length of the datasets according to the current mode."""
        datasets = [getattr(dl, "dataset", None) for dl in self.flattened]
        lengths = [length for ds in datasets if (length := sized_len(ds)) is not None]
        if not lengths:
            raise NotImplementedError("All datasets are iterable-style datasets.")
        fn = _SUPPORTED_MODES[self._mode]["fn"]
        return fn(lengths)

    def _state_dicts(self) -> list[dict[str, Any]]:
        """Returns the list of state dicts for iterables in `self.flattened` that are stateful."""
        return [loader.state_dict() for loader in self.flattened if isinstance(loader, _Stateful)]

    def _load_state_dicts(self, states: list[dict[str, Any]]) -> None:
        """Loads the state dicts for iterables in `self.flattened` that are stateful."""
        if not states:
            return
        stateful_loaders = [loader for loader in self.flattened if isinstance(loader, _Stateful)]
        if len(stateful_loaders) != len(states):
            raise RuntimeError(
                f"The CombinedLoader has {len(stateful_loaders)} stateful loaders, but found {len(states)} states"
                " in the checkpoint. Please make sure you define the same dataloaders that were used when saving"
                " the checkpoint."
            )
        for loader, state_dict in zip(stateful_loaders, states):
            loader.load_state_dict(state_dict)


def _shutdown_workers_and_reset_iterator(dataloader: object) -> None:
    if hasattr(dataloader, "_iterator"):
        if isinstance(dataloader._iterator, _MultiProcessingDataLoaderIter):
            del dataloader._iterator
        dataloader._iterator = None


def _get_iterables_lengths(iterables: list[Iterable]) -> list[Union[int, float]]:
    return [(float("inf") if (length := sized_len(iterable)) is None else length) for iterable in iterables]
