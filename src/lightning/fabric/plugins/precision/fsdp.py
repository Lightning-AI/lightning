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
from typing import TYPE_CHECKING, Any, Literal, Optional

import torch
from lightning_utilities import apply_to_collection
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from typing_extensions import get_args, override

from lightning.fabric.plugins.precision.amp import _optimizer_handles_unscaling
from lightning.fabric.plugins.precision.precision import Precision
from lightning.fabric.plugins.precision.utils import _convert_fp_tensor, _DtypeContextManager
from lightning.fabric.utilities.types import Optimizable

if TYPE_CHECKING:
    from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision as TorchMixedPrecision
    from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

_PRECISION_INPUT = Literal["32-true", "16-true", "bf16-true", "16-mixed", "bf16-mixed"]


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
            raise ValueError(f"Was unable to infer precision type, received {self.precision!r}.")

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
        return self.tensor_init_context()

    @override
    def convert_input(self, data: Any) -> Any:
        return apply_to_collection(data, function=_convert_fp_tensor, dtype=Tensor, dst_type=self._desired_input_dtype)

    @override
    def convert_output(self, data: Any) -> Any:
        return apply_to_collection(data, function=_convert_fp_tensor, dtype=Tensor, dst_type=torch.get_default_dtype())

    @override
    def backward(self, tensor: Tensor, model: Optional[Module], *args: Any, **kwargs: Any) -> None:
        if self.scaler is not None:
            tensor = self.scaler.scale(tensor)
        super().backward(tensor, model, *args, **kwargs)

    @override
    def optimizer_step(
        self,
        optimizer: Optimizable,
        **kwargs: Any,
    ) -> Any:
        if self.scaler is None:
            # skip scaler logic, as bfloat16 does not require scaler
            return super().optimizer_step(optimizer, **kwargs)
        # note: the scaler will skip the `optimizer.step` if nonfinite gradients are found
        step_output = self.scaler.step(optimizer, **kwargs)  # type: ignore[arg-type]
        self.scaler.update()
        return step_output

    @override
    def unscale_gradients(self, optimizer: Optimizer) -> None:
        scaler = self.scaler
        if scaler is not None:
            if _optimizer_handles_unscaling(optimizer):
                raise NotImplementedError("Gradient clipping is not implemented for optimizers handling the unscaling.")
            scaler.unscale_(optimizer)

    @override
    def state_dict(self) -> dict[str, Any]:
        if self.scaler is not None:
            return self.scaler.state_dict()
        return {}

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if self.scaler is not None:
            self.scaler.load_state_dict(state_dict)
