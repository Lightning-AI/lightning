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

import inspect
import json
from argparse import Namespace
from collections.abc import Mapping, MutableMapping
from dataclasses import asdict, is_dataclass
from typing import Any, Optional, Union

from torch import Tensor

from lightning.fabric.utilities.imports import _NUMPY_AVAILABLE


def _convert_params(params: Optional[Union[dict[str, Any], Namespace]]) -> dict[str, Any]:
    """Ensure parameters are a dict or convert to dict if necessary.

    Args:
        params: Target to be converted to a dictionary

    Returns:
        params as a dictionary

    """
    # in case converting from namespace
    if isinstance(params, Namespace):
        params = vars(params)

    if params is None:
        params = {}

    return params


def _sanitize_callable_params(params: dict[str, Any]) -> dict[str, Any]:
    """Sanitize callable params dict, e.g. ``{'a': <function_**** at 0x****>} -> {'a': 'function_****'}``.

    Args:
        params: Dictionary containing the hyperparameters

    Returns:
        dictionary with all callables sanitized

    """

    def _sanitize_callable(val: Any) -> Any:
        if inspect.isclass(val):
            # If it's a class, don't try to instantiate it, just return the name
            return val.__name__
        if callable(val):
            # Callables get a chance to return a name
            try:
                _val = val()
                if callable(_val):
                    return val.__name__
                return _val
            # todo: specify the possible exception
            except Exception:
                return getattr(val, "__name__", None)
        return val

    return {key: _sanitize_callable(val) for key, val in params.items()}


def _flatten_dict(params: MutableMapping[Any, Any], delimiter: str = "/", parent_key: str = "") -> dict[str, Any]:
    """Flatten hierarchical dict, e.g. ``{'a': {'b': 'c'}} -> {'a/b': 'c'}``.

    Args:
        params: Dictionary containing the hyperparameters
        delimiter: Delimiter to express the hierarchy. Defaults to ``'/'``.

    Returns:
        Flattened dict.

    Examples:
        >>> _flatten_dict({'a': {'b': 'c'}})
        {'a/b': 'c'}
        >>> _flatten_dict({'a': {'b': 123}})
        {'a/b': 123}
        >>> _flatten_dict({5: {'a': 123}})
        {'5/a': 123}
        >>> _flatten_dict({"dl": [{"a": 1, "c": 3}, {"b": 2, "d": 5}], "l": [1, 2, 3, 4]})
        {'dl/0/a': 1, 'dl/0/c': 3, 'dl/1/b': 2, 'dl/1/d': 5, 'l': [1, 2, 3, 4]}

    """
    result: dict[str, Any] = {}
    for k, v in params.items():
        new_key = parent_key + delimiter + str(k) if parent_key else str(k)
        if is_dataclass(v) and not isinstance(v, type):
            v = asdict(v)
        elif isinstance(v, Namespace):
            v = vars(v)

        if isinstance(v, MutableMapping):
            result = {**result, **_flatten_dict(v, parent_key=new_key, delimiter=delimiter)}
        # Also handle the case where v is a list of dictionaries
        elif isinstance(v, list) and all(isinstance(item, MutableMapping) for item in v):
            for i, item in enumerate(v):
                result = {**result, **_flatten_dict(item, parent_key=f"{new_key}/{i}", delimiter=delimiter)}
        else:
            result[new_key] = v
    return result


def _sanitize_params(params: dict[str, Any]) -> dict[str, Any]:
    """Returns params with non-primitvies converted to strings for logging.

    >>> import torch
    >>> params = {"float": 0.3,
    ...           "int": 1,
    ...           "string": "abc",
    ...           "bool": True,
    ...           "list": [1, 2, 3],
    ...           "namespace": Namespace(foo=3),
    ...           "layer": torch.nn.BatchNorm1d}
    >>> import pprint
    >>> pprint.pprint(_sanitize_params(params))  # doctest: +NORMALIZE_WHITESPACE
    {'bool': True,
        'float': 0.3,
        'int': 1,
        'layer': "<class 'torch.nn.modules.batchnorm.BatchNorm1d'>",
        'list': '[1, 2, 3]',
        'namespace': 'Namespace(foo=3)',
        'string': 'abc'}

    """
    for k in params:
        if _NUMPY_AVAILABLE:
            import numpy as np

            if isinstance(params[k], (np.bool_, np.integer, np.floating)):
                params[k] = params[k].item()
        if type(params[k]) not in [bool, int, float, str, Tensor]:
            params[k] = str(params[k])
    return params


def _convert_json_serializable(params: dict[str, Any]) -> dict[str, Any]:
    """Convert non-serializable objects in params to string."""
    return {k: str(v) if not _is_json_serializable(v) else v for k, v in params.items()}


def _is_json_serializable(value: Any) -> bool:
    """Test whether a variable can be encoded as json."""
    if value is None or isinstance(value, (bool, int, float, str, list, dict)):  # fast path
        return True
    try:
        json.dumps(value)
        return True
    except (TypeError, OverflowError):
        # OverflowError is raised if number is too large to encode
        return False


def _add_prefix(
    metrics: Mapping[str, Union[Tensor, float]], prefix: str, separator: str
) -> Mapping[str, Union[Tensor, float]]:
    """Insert prefix before each key in a dict, separated by the separator.

    Args:
        metrics: Dictionary with metric names as keys and measured quantities as values
        prefix: Prefix to insert before each key
        separator: Separates prefix and original key name

    Returns:
        Dictionary with prefix and separator inserted before each key

    """
    if not prefix:
        return metrics
    return {f"{prefix}{separator}{k}": v for k, v in metrics.items()}
