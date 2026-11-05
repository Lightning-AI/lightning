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
"""Abstract base class used to build new loggers."""

import statistics
from abc import ABC
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Optional

from typing_extensions import override

from lightning.fabric.loggers import Logger as FabricLogger
from lightning.fabric.loggers.logger import _DummyExperiment as DummyExperiment  # for backward compatibility
from lightning.fabric.loggers.logger import rank_zero_experiment  # noqa: F401  # for backward compatibility
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint


class Logger(FabricLogger, ABC):
    """Base class for experiment loggers."""

    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        """Called after model checkpoint callback saves a new checkpoint.

        Args:
            checkpoint_callback: the model checkpoint callback instance

        """
        pass

    @property
    def save_dir(self) -> Optional[str]:
        """Return the root directory where experiment logs get saved, or `None` if the logger does not save data
        locally."""
        return None


class DummyLogger(Logger):
    """Dummy logger for internal use.

    It is useful if we want to disable user's logger for a feature, but still ensure that user code can run

    """

    def __init__(self) -> None:
        super().__init__()
        self._experiment = DummyExperiment()

    @property
    def experiment(self) -> DummyExperiment:
        """Return the experiment object associated with this logger."""
        return self._experiment

    @override
    def log_metrics(self, *args: Any, **kwargs: Any) -> None:
        pass

    @override
    def log_hyperparams(self, *args: Any, **kwargs: Any) -> None:
        pass

    @property
    @override
    def name(self) -> str:
        """Return the experiment name."""
        return ""

    @property
    @override
    def version(self) -> str:
        """Return the experiment version."""
        return ""

    def __getitem__(self, idx: int) -> "DummyLogger":
        # enables self.logger[0].experiment.add_image(...)
        return self

    def __getattr__(self, name: str) -> Callable:
        """Allows the DummyLogger to be called with arbitrary methods, to avoid AttributeErrors."""

        def method(*args: Any, **kwargs: Any) -> None:
            return None

        return method


# TODO: this should have been deprecated
def merge_dicts(  # pragma: no cover
    dicts: Sequence[Mapping],
    agg_key_funcs: Optional[Mapping] = None,
    default_func: Callable[[Sequence[float]], float] = statistics.mean,
) -> dict:
    """Merge a sequence with dictionaries into one dictionary by aggregating the same keys with some given function.

    Args:
        dicts:
            Sequence of dictionaries to be merged.
        agg_key_funcs:
            Mapping from key name to function. This function will aggregate a
            list of values, obtained from the same key of all dictionaries.
            If some key has no specified aggregation function, the default one
            will be used. Default is: ``None`` (all keys will be aggregated by the
            default function).
        default_func:
            Default function to aggregate keys, which are not presented in the
            `agg_key_funcs` map.

    Returns:
        Dictionary with merged values.

    Examples:
        >>> import pprint
        >>> d1 = {'a': 1.7, 'b': 2.0, 'c': 1, 'd': {'d1': 1, 'd3': 3}}
        >>> d2 = {'a': 1.1, 'b': 2.2, 'v': 1, 'd': {'d1': 2, 'd2': 3}}
        >>> d3 = {'a': 1.1, 'v': 2.3, 'd': {'d3': 3, 'd4': {'d5': 1}}}
        >>> dflt_func = min
        >>> agg_funcs = {'a': statistics.mean, 'v': max, 'd': {'d1': sum}}
        >>> pprint.pprint(merge_dicts([d1, d2, d3], agg_funcs, dflt_func))
        {'a': 1.3,
         'b': 2.0,
         'c': 1,
         'd': {'d1': 3, 'd2': 3, 'd3': 3, 'd4': {'d5': 1}},
         'v': 2.3}

    """
    agg_key_funcs = agg_key_funcs or {}
    d_out: dict = {}
    for k in dict.fromkeys(k for d in dicts for k in d):
        values_to_agg = [v for d_in in dicts if (v := d_in.get(k)) is not None]
        if not values_to_agg:
            continue
        fn = agg_key_funcs.get(k)
        if isinstance(values_to_agg[0], dict):
            d_out[k] = merge_dicts(values_to_agg, fn, default_func)
        else:
            d_out[k] = (fn or default_func)(values_to_agg)

    return d_out
