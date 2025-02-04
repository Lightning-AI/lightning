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
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from lightning.fabric.utilities.logger import (
    _add_prefix,
    _convert_json_serializable,
    _convert_params,
    _flatten_dict,
    _sanitize_callable_params,
    _sanitize_params,
)


def test_convert_params():
    """Test conversion of params to a dict."""
    # Test normal dict, make sure it is unchanged
    params = {"string": "string", "int": 1, "float": 0.1, "bool": True, "none": None}
    expected = params.copy()
    assert _convert_params(params) == expected

    # Test None conversion
    assert _convert_params(None) == {}

    # Test conversion of argparse Namespace
    params = Namespace(string="string", int=1, float=0.1, bool=True, none=None)
    expected = vars(params)
    assert _convert_params(params) == expected


def test_flatten_dict():
    """Validate flatten_dict can handle nested dictionaries and argparse Namespace."""
    # Test basic dict flattening with custom delimiter
    params = {"a": {"b": "c"}}
    params = _flatten_dict(params, "--")

    assert "a" not in params
    assert params["a--b"] == "c"

    # Test complex nested dict flattening
    params = {"a": {5: {"foo": "bar"}}, "b": 6, "c": {7: [1, 2, 3, 4], 8: "foo", 9: {10: "bar"}}}
    params = _flatten_dict(params)

    assert "a" not in params
    assert params["a/5/foo"] == "bar"
    assert params["b"] == 6
    assert params["c/7"] == [1, 2, 3, 4]
    assert params["c/8"] == "foo"
    assert params["c/9/10"] == "bar"

    # Test list of nested dicts flattening
    params = {"dl": [{"a": 1, "c": 3}, {"b": 2, "d": 5}], "l": [1, 2, 3, 4]}
    params = _flatten_dict(params)

    assert params == {"dl/0/a": 1, "dl/0/c": 3, "dl/1/b": 2, "dl/1/d": 5, "l": [1, 2, 3, 4]}

    # Test flattening of argparse Namespace
    params = Namespace(a=1, b=2)
    wrapping_dict = {"params": params}
    params = _flatten_dict(wrapping_dict)

    params_type = type(params)  # way around needed for Ruff's `isinstance` suggestion
    assert params_type is dict
    assert params["params/a"] == 1
    assert params["params/b"] == 2
    assert "a" not in params
    assert "b" not in params

    # Test flattening of dataclass objects
    @dataclass
    class A:
        c: int
        d: int

    @dataclass
    class B:
        a: A
        b: int

    params = {"params": B(a=A(c=1, d=2), b=3), "param": 4}
    params = _flatten_dict(params)
    assert params == {"param": 4, "params/b": 3, "params/a/c": 1, "params/a/d": 2}


def test_sanitize_callable_params():
    """Callback functions are not serializable.

    Therefore, we get them a chance to return something and if the returned type is not accepted, return None.

    """

    def return_something():
        return "something"

    def wrapper_something():
        return return_something

    class ClassNoArgs:
        def __init__(self):
            pass

    class ClassWithCall:
        def __call__(self):
            return "name"

    params = Namespace(
        foo="bar",
        something=return_something,
        wrapper_something_wo_name=(lambda: lambda: "1"),
        wrapper_something=wrapper_something,
        class_no_args=ClassNoArgs,
        class_with_call=ClassWithCall,
    )

    params = _convert_params(params)
    params = _flatten_dict(params)
    params = _sanitize_callable_params(params)
    assert params["foo"] == "bar"
    assert params["something"] == "something"
    assert params["wrapper_something"] == "wrapper_something"
    assert params["wrapper_something_wo_name"] == "<lambda>"
    assert params["class_no_args"] == "ClassNoArgs"
    assert params["class_with_call"] == "ClassWithCall"


def test_sanitize_params():
    """Verify sanitize params converts various types to loggable strings."""
    params = {
        "float": 0.3,
        "int": 1,
        "string": "abc",
        "bool": True,
        "list": [1, 2, 3],
        "np_bool": np.bool_(False),
        "np_int": np.int_(5),
        "np_double": np.double(3.14159),
        "namespace": Namespace(foo=3),
        "layer": torch.nn.BatchNorm1d,
        "tensor": torch.ones(3),
    }
    params = _sanitize_params(params)

    assert params["float"] == 0.3
    assert params["int"] == 1
    assert params["string"] == "abc"
    assert params["bool"] is True
    assert params["list"] == "[1, 2, 3]"
    assert params["np_bool"] is False
    assert params["np_int"] == 5
    assert params["np_double"] == 3.14159
    assert params["namespace"] == "Namespace(foo=3)"
    assert params["layer"] == "<class 'torch.nn.modules.batchnorm.BatchNorm1d'>"
    assert torch.equal(params["tensor"], torch.ones(3))


def test_add_prefix():
    """Verify add_prefix modifies the dict keys correctly."""
    metrics = {"metric1": 1, "metric2": 2}
    metrics = _add_prefix(metrics, "prefix", "-")

    assert "prefix-metric1" in metrics
    assert "prefix-metric2" in metrics
    assert "metric1" not in metrics
    assert "metric2" not in metrics

    metrics = _add_prefix(metrics, "prefix2", "_")

    assert "prefix2_prefix-metric1" in metrics
    assert "prefix2_prefix-metric2" in metrics
    assert "prefix-metric1" not in metrics
    assert "prefix-metric2" not in metrics
    assert metrics["prefix2_prefix-metric1"] == 1
    assert metrics["prefix2_prefix-metric2"] == 2


def test_convert_json_serializable():
    data = {
        # JSON-serializable
        "none": None,
        "int": 1,
        "float": 1.1,
        "bool": True,
        "dict": {"a": 1},
        "list": [2, 3, 4],
        # not JSON-serializable
        "path": Path("path"),
        "tensor": torch.tensor(1),
    }
    expected = {
        "none": None,
        "int": 1,
        "float": 1.1,
        "bool": True,
        "dict": {"a": 1},
        "list": [2, 3, 4],
        "path": "path",
        "tensor": "tensor(1)",
    }
    assert _convert_json_serializable(data) == expected
