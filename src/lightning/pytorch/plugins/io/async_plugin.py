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

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Optional

import torch
from lightning_utilities.core.apply_func import apply_to_collection
from typing_extensions import override

from lightning.pytorch.plugins.io.wrapper import _WrappingCheckpointIO

if TYPE_CHECKING:
    from lightning.fabric.plugins import CheckpointIO


class AsyncCheckpointIO(_WrappingCheckpointIO):
    """``AsyncCheckpointIO`` enables saving the checkpoints asynchronously in a thread.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    Args:
        checkpoint_io: A checkpoint IO plugin that is used as the basis for async checkpointing.

    """

    _executor: Optional[ThreadPoolExecutor]
    _error: Optional[BaseException]

    def __init__(self, checkpoint_io: Optional["CheckpointIO"] = None) -> None:
        super().__init__(checkpoint_io)
        self._executor = None
        self._error = None

    # CheckpointIO doesn't have a setup method so we have to do something like.
    def _ensure_setup(self) -> None:
        """Ensures that the executor is setup.

        We can't do setup in __init__ because if train or validate is called more than once, the teardown method deletes
        the executor.

        """
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1)

    @override
    def save_checkpoint(self, *args: Any, **kwargs: Any) -> None:
        """Uses the ``ThreadPoolExecutor`` to save the checkpoints using the base ``checkpoint_io``."""

        self._ensure_setup()

        # rebuild args/kwargs with a cloned checkpoint (supports positional or kw form)
        if "checkpoint" in kwargs:
            kwargs = {**kwargs, "checkpoint": apply_to_collection(kwargs["checkpoint"], torch.Tensor, _clone_tensor)}
        elif len(args) >= 1:
            args = (apply_to_collection(args[0], torch.Tensor, _clone_tensor), *args[1:])

        def _save_checkpoint(*args: Any, **kwargs: Any) -> None:
            try:
                assert self.checkpoint_io is not None
                self.checkpoint_io.save_checkpoint(*args, **kwargs)
            except BaseException as ex:
                self._error = ex

        assert self._executor is not None
        self._executor.submit(_save_checkpoint, *args, **kwargs)

        # if an error was raised between the previous time `save_checkpoint`` was called and now,
        # because `executor.submit` is not blocking
        if self._error:
            raise self._error

    @override
    def teardown(self) -> None:
        """Wait for asynchronous saves, then close the wrapped checkpoint IO."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

        try:
            if self.checkpoint_io is not None:
                self.checkpoint_io.teardown()
        finally:
            # Preserve the background save failure if backend cleanup also fails.
            if self._error:
                raise self._error


# snapshot the checkpoint payload on the caller thread to avoid races with parameter mutation
def _clone_tensor(t: torch.Tensor) -> torch.Tensor:
    """Clones a tensor on the caller thread."""
    # detach to avoid autograd history and clone to take a point-in-time copy
    return t.detach().clone()
