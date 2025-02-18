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
from typing import Any, Literal, Optional

import torch
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
from torch.nn import Module
from torch.optim import LBFGS, Optimizer
from typing_extensions import override

from lightning.fabric.plugins.precision.precision import Precision
from lightning.fabric.plugins.precision.utils import _convert_fp_tensor
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_4
from lightning.fabric.utilities.types import Optimizable


class MixedPrecision(Precision):
    """Plugin for Automatic Mixed Precision (AMP) training with ``torch.autocast``.

    Args:
        precision: Whether to use ``torch.float16`` (``'16-mixed'``) or ``torch.bfloat16`` (``'bf16-mixed'``).
        device: The device for ``torch.autocast``.
        scaler: An optional :class:`torch.cuda.amp.GradScaler` to use.

    """

    def __init__(
        self,
        precision: Literal["16-mixed", "bf16-mixed"],
        device: str,
        scaler: Optional["torch.amp.GradScaler"] = None,
    ) -> None:
        if precision not in ("16-mixed", "bf16-mixed"):
            raise ValueError(
                f"Passed `{type(self).__name__}(precision={precision!r})`."
                " Precision must be '16-mixed' or 'bf16-mixed'."
            )

        self.precision = precision
        if scaler is None and self.precision == "16-mixed":
            scaler = torch.amp.GradScaler(device=device) if _TORCH_GREATER_EQUAL_2_4 else torch.cuda.amp.GradScaler()
        if scaler is not None and self.precision == "bf16-mixed":
            raise ValueError(f"`precision='bf16-mixed'` does not use a scaler, found {scaler}.")
        self.device = device
        self.scaler = scaler

        self._desired_input_dtype = torch.bfloat16 if self.precision == "bf16-mixed" else torch.float16

    @override
    def forward_context(self) -> AbstractContextManager:
        return torch.autocast(self.device, dtype=self._desired_input_dtype)

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
        if isinstance(optimizer, LBFGS):
            raise TypeError("AMP and the LBFGS optimizer are not compatible.")
        previous_scale = self.scaler.get_scale()
        # note: the scaler will skip the `optimizer.step` if nonfinite gradients are found
        step_output = self.scaler.step(optimizer, **kwargs)  # type: ignore[arg-type]
        self.scaler.update()
        optimizer._skip_next_scheduler_step = self.scaler.get_scale() < previous_scale
        return step_output

    @override
    def state_dict(self) -> dict[str, Any]:
        if self.scaler is not None:
            return self.scaler.state_dict()
        return {}

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if self.scaler is not None:
            self.scaler.load_state_dict(state_dict)

    @override
    def unscale_gradients(self, optimizer: Optimizer) -> None:
        scaler = self.scaler
        if scaler is not None:
            if _optimizer_handles_unscaling(optimizer):
                raise NotImplementedError("Gradient clipping is not implemented for optimizers handling the unscaling.")
            scaler.unscale_(optimizer)


def _optimizer_handles_unscaling(optimizer: Any) -> bool:
    """Determines whether a PyTorch optimizer handles unscaling gradients in the step method rather than through the
    :class:`torch.cuda.amp.GradScaler`.

    Since, the current implementation of this function checks a PyTorch internal variable on the optimizer, the return
    value will only be reliable for built-in PyTorch optimizers.

    """
    return getattr(optimizer, "_step_supports_amp_scaling", False)
