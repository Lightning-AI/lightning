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
from typing import Union

import torch
from torch._dynamo import OptimizedModule

import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy, DeepSpeedStrategy, FSDPStrategy, SingleDeviceStrategy, Strategy
from lightning.pytorch.utilities.model_helpers import _check_mixed_imports


def from_compiled(model: OptimizedModule) -> "pl.LightningModule":
    """Returns an instance LightningModule from the output of ``torch.compile``.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    The ``torch.compile`` function returns a ``torch._dynamo.OptimizedModule``, which wraps the LightningModule
    passed in as an argument, but doesn't inherit from it. This means that the output of ``torch.compile`` behaves
    like a LightningModule, but it doesn't inherit from it (i.e. `isinstance` will fail).

    Use this method to obtain a LightningModule that still runs with all the optimizations from ``torch.compile``.

    """
    if not isinstance(model, OptimizedModule):
        raise ValueError(f"`model` is required to be a `OptimizedModule`. Found a `{type(model).__name__}` instead.")

    orig_module = model._orig_mod

    if not isinstance(orig_module, pl.LightningModule):
        _check_mixed_imports(model)
        raise ValueError(
            f"`model` is expected to be a compiled LightningModule. Found a `{type(orig_module).__name__}` instead"
        )

    # A previous conversion replaced these methods already. Keep the original methods, not nested wrappers.
    if orig_module._compiler_ctx is not None:
        to_uncompiled(orig_module)

    orig_module._compiler_ctx = {
        "compiler": "dynamo",
        "dynamo_ctx": model.dynamo_ctx,
        "original_forward": orig_module.forward,
        "original_training_step": orig_module.training_step,
        "original_validation_step": orig_module.validation_step,
        "original_test_step": orig_module.test_step,
        "original_predict_step": orig_module.predict_step,
    }

    orig_module.forward = model.dynamo_ctx(orig_module.forward)  # type: ignore[method-assign]
    orig_module.training_step = model.dynamo_ctx(orig_module.training_step)  # type: ignore[method-assign]
    orig_module.validation_step = model.dynamo_ctx(orig_module.validation_step)  # type: ignore[method-assign]
    orig_module.test_step = model.dynamo_ctx(orig_module.test_step)  # type: ignore[method-assign]
    orig_module.predict_step = model.dynamo_ctx(orig_module.predict_step)  # type: ignore[method-assign]
    return orig_module


def to_uncompiled(model: Union["pl.LightningModule", "torch._dynamo.OptimizedModule"]) -> "pl.LightningModule":
    """Returns an instance of LightningModule without any compilation optimizations from a compiled model.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    This takes either a ``torch._dynamo.OptimizedModule`` returned by ``torch.compile()`` or a ``LightningModule``
    returned by ``from_compiled``.

    Note: this method will in-place modify the ``LightningModule`` that is passed in.

    """
    if isinstance(model, OptimizedModule):
        original = model._orig_mod
        if not isinstance(original, pl.LightningModule):
            raise TypeError(
                f"Unexpected error, the wrapped model should be a LightningModule, found {type(model).__name__}"
            )

    elif isinstance(model, pl.LightningModule):
        if model._compiler_ctx is None:
            raise ValueError(
                "`model` is required to be a compiled LightningModule. Found a non-compiled LightningModule instead."
            )
        original = model

    else:
        raise ValueError("`model` must either be an instance of OptimizedModule or LightningModule")

    ctx = original._compiler_ctx
    if ctx is not None:
        original.forward = ctx["original_forward"]  # type: ignore[method-assign]
        original.training_step = ctx["original_training_step"]  # type: ignore[method-assign]
        original.validation_step = ctx["original_validation_step"]  # type: ignore[method-assign]
        original.test_step = ctx["original_test_step"]  # type: ignore[method-assign]
        original.predict_step = ctx["original_predict_step"]  # type: ignore[method-assign]
        original._compiler_ctx = None

    return original


def _maybe_unwrap_optimized(model: object) -> "pl.LightningModule":
    if isinstance(model, OptimizedModule):
        return from_compiled(model)
    if isinstance(model, pl.LightningModule):
        return model
    _check_mixed_imports(model)
    raise TypeError(
        f"`model` must be a `LightningModule` or `torch._dynamo.OptimizedModule`, got `{type(model).__qualname__}`"
    )


def _verify_strategy_supports_compile(model: "pl.LightningModule", strategy: Strategy) -> None:
    if model._compiler_ctx is not None:
        supported_strategies = (SingleDeviceStrategy, DDPStrategy, FSDPStrategy)
        if not isinstance(strategy, supported_strategies) or isinstance(strategy, DeepSpeedStrategy):
            supported_strategy_names = ", ".join(s.__name__ for s in supported_strategies)
            raise RuntimeError(
                f"Using a compiled model is incompatible with the current strategy: `{type(strategy).__name__}`."
                f" Only {supported_strategy_names} support compilation. Either switch to one of the supported"
                " strategies or avoid passing in compiled model."
            )
