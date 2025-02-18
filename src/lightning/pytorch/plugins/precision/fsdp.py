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
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
from lightning_utilities import apply_to_collection
from torch import Tensor
from torch.nn import Module
from typing_extensions import get_args, override

import lightning.pytorch as pl
from lightning.fabric.plugins.precision.amp import _optimizer_handles_unscaling
from lightning.fabric.plugins.precision.fsdp import _PRECISION_INPUT
from lightning.fabric.plugins.precision.utils import _convert_fp_tensor, _DtypeContextManager
from lightning.fabric.utilities.types import Optimizable
from lightning.pytorch.plugins.precision.precision import Precision
from lightning.pytorch.utilities.exceptions import MisconfigurationException

if TYPE_CHECKING:
    from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision as TorchMixedPrecision
    from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler


class FSDPPrecision(Precision):
    """Precision plugin for training with Fully Sharded Data Parallel (FSDP).

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    Args:
        precision: Full precision (32-true), half precision (16-true, bf16-true) or
            mixed precision (16-mixed, bf16-mixed).
        scaler: An optional :class:`torch.distributed.fsdp.sharded_grad_scaler.ShardedGradScaler` to use.

    Raises:
        ValueError:
            If unsupported ``precision`` is provided.

    """

    def __init__(
        self,
        precision: _PRECISION_INPUT,
        scaler: Optional["ShardedGradScaler"] = None,
        device_type: Optional[str] = None,
    ) -> None:
        supported_precision = get_args(_PRECISION_INPUT)
        if precision not in supported_precision:
            raise ValueError(
                f"`precision={precision!r})` is not supported in FSDP."
                f" `precision` must be one of: {supported_precision}."
            )
        self.device_type = device_type if device_type is not None else "cuda"

        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

        if scaler is not None and self.precision != "16-mixed":
            raise ValueError(f"`precision={precision!r}` does not use a scaler, found {scaler}.")

        self.scaler = ShardedGradScaler() if scaler is None and precision == "16-mixed" else None
        self.precision = precision

        precision_to_type = {
            "bf16-mixed": torch.float32,
            "16-mixed": torch.float32,
            "bf16-true": torch.bfloat16,
            "16-true": torch.float16,
            "32-true": torch.float32,
        }
        self._desired_input_dtype = precision_to_type[self.precision]

    @override
    def convert_module(self, module: Module) -> Module:
        if "true" in self.precision:
            return module.to(dtype=self._desired_input_dtype)
        return module

    @override
    def clip_grad_by_norm(self, *_: Any, **__: Any) -> None:
        # see https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.clip_grad_norm_
        # section `Gradient Clipping`, using `torch.nn.utils.clip_grad_norm_` is incorrect with FSDP.
        # To overcome this we need to call root_sharded_module.clip_grad_norm(clip_val), but we don't have a reference
        # to the root module
        raise MisconfigurationException(
            f"`gradient_clip_algorithm='norm'` is currently not supported for `{self.__class__.__name__}`"
        )

    @property
    def mixed_precision_config(self) -> "TorchMixedPrecision":
        from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision as TorchMixedPrecision

        if self.precision == "16-mixed":
            param_dtype = torch.float32
            reduce_dtype = buffer_dtype = torch.float16
        elif self.precision == "bf16-mixed":
            param_dtype = torch.float32
            reduce_dtype = buffer_dtype = torch.bfloat16
        elif self.precision == "16-true":
            param_dtype = reduce_dtype = buffer_dtype = torch.float16
        elif self.precision == "bf16-true":
            param_dtype = reduce_dtype = buffer_dtype = torch.bfloat16
        elif self.precision == "32-true":
            param_dtype = torch.float32
            reduce_dtype = buffer_dtype = torch.float32
        else:
            raise MisconfigurationException(f"Was unable to infer precision type, received {self.precision!r}.")

        return TorchMixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype,
        )

    @override
    def tensor_init_context(self) -> AbstractContextManager:
        return _DtypeContextManager(self._desired_input_dtype)

    @override
    def module_init_context(self) -> AbstractContextManager:
        return _DtypeContextManager(self.mixed_precision_config.param_dtype or torch.float32)

    @override
    def forward_context(self) -> AbstractContextManager:
        if "mixed" in self.precision:
            return torch.autocast(
                self.device_type, dtype=(torch.bfloat16 if self.precision == "bf16-mixed" else torch.float16)
            )
        return _DtypeContextManager(self._desired_input_dtype)

    @override
    def convert_input(self, data: Any) -> Any:
        return apply_to_collection(data, function=_convert_fp_tensor, dtype=Tensor, dst_type=self._desired_input_dtype)

    @override
    def convert_output(self, data: Any) -> Any:
        return apply_to_collection(data, function=_convert_fp_tensor, dtype=Tensor, dst_type=torch.get_default_dtype())

    @override
    def pre_backward(self, tensor: Tensor, module: "pl.LightningModule") -> Tensor:  # type: ignore[override]
        if self.scaler is not None:
            tensor = self.scaler.scale(tensor)
        return super().pre_backward(tensor, module)

    @override
    def optimizer_step(  # type: ignore[override]
        self,
        optimizer: Optimizable,
        model: "pl.LightningModule",
        closure: Callable[[], Any],
        **kwargs: Any,
    ) -> Any:
        if self.scaler is None:
            # skip scaler logic, as bfloat16 does not require scaler
            return super().optimizer_step(optimizer, model=model, closure=closure, **kwargs)
        closure_result = closure()

        if not _optimizer_handles_unscaling(optimizer):
            # Unscaling needs to be performed here in case we are going to apply gradient clipping.
            # Optimizers that perform unscaling in their `.step()` method are not supported (e.g., fused Adam).
            # Note: `unscale` happens after the closure is executed, but before the `on_before_optimizer_step` hook.
            self.scaler.unscale_(optimizer)  # type: ignore[arg-type]

        self._after_closure(model, optimizer)
        skipped_backward = closure_result is None
        # in manual optimization, the closure does not return a value
        if not model.automatic_optimization or not skipped_backward:
            # note: the scaler will skip the `optimizer.step` if nonfinite gradients are found
            step_output = self.scaler.step(optimizer, **kwargs)  # type: ignore[arg-type]
            self.scaler.update()
            return step_output
        return closure_result

    @override
    def state_dict(self) -> dict[str, Any]:
        if self.scaler is not None:
            return self.scaler.state_dict()
        return {}

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if self.scaler is not None:
            self.scaler.load_state_dict(state_dict)
