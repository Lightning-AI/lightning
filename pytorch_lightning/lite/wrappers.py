# Copyright The PyTorch Lightning team.
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
from typing import Any, Callable, Generator, Iterator, Optional, Union

import torch
from torch import nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.utilities.apply_func import apply_to_collection, move_data_to_device


def _do_nothing_closure() -> None:
    return None


class _LiteOptimizer:
    def __init__(self, optimizer: Optimizer, accelerator: Accelerator) -> None:
        """LiteOptimizer is a thin wrapper around the :class:`~torch.optim.Optimizer` that delegates the optimizer
        step calls to the accelerator/strategy plugin.

        The underlying wrapped optimizer object can be accessed via the property :attr:`optimizer`.

        Args:
            optimizer: The optimizer to wrap
            accelerator: Reference to the accelerator for handling the optimizer step
        """
        # `__del__` is skipped in case the optimizer has implemented custom destructor logic which we would
        # not want to call on desturction of the `_LiteOptimizer`
        self.__dict__ = {k: v for k, v in optimizer.__dict__.items() if k not in ("step", "__del__")}
        self.__class__ = type("Lite" + optimizer.__class__.__name__, (self.__class__, optimizer.__class__), {})
        self._optimizer = optimizer
        self._accelerator = accelerator

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    def step(self, closure: Optional[Callable] = None) -> None:
        closure = closure or _do_nothing_closure
        self._accelerator.optimizer_step(
            self.optimizer,
            opt_idx=0,
            lambda_closure=closure,
            model=self._accelerator.model,
        )


class _LiteModule(nn.Module):
    # TODO: Pass in the precision plugin instead of accelerator
    def __init__(self, module: nn.Module, accelerator: Accelerator) -> None:
        """The LiteModule is a thin wrapper around the :class:`torch.nn.Module` and handles precision / autocast
        automatically for the forward pass.

        The underlying wrapped module can be accessed via the property :attr:`module`.

        Args:
            module: The module to wrap
            accelerator: Reference to the accelerator for handling precision context
        """
        super().__init__()
        self._module = module
        self._accelerator = accelerator

    @property
    def module(self) -> nn.Module:
        return self._module

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Casts all inputs to the right precision and handles autocast for operations in the module forward
        method."""
        precision = self._accelerator.precision_plugin.precision
        precision_to_type = {
            "bf16": torch.bfloat16,
            16: torch.float16,
            32: torch.float32,
            64: torch.float64,
        }
        # TODO (@awaelchli): let the precision plugin handle the conversion
        to_type = precision_to_type[precision]
        args, kwargs = apply_to_collection([args, kwargs], function=lambda t: t.to(to_type), dtype=Tensor)

        with self._accelerator.precision_plugin.forward_context():
            output = self.module(*args, **kwargs)

        output = apply_to_collection(output, function=lambda t: t.to(torch.get_default_dtype()), dtype=Tensor)
        return output


class _LiteDataLoader(DataLoader):
    def __init__(self, device: Optional[torch.device] = None, **dl_kwargs: Any) -> None:
        """The LiteDataLoader is an extension of the PyTorch :class:`~torch.utils.data.DataLoader` that adds
        additional features such as moving the data to the device automatically.

        Args:
            device: The device to which the data should be moved. By default the device is `None` and no data
                transfers will be made (identical behavior as :class:`~torch.utils.data.DataLoader`).
            **dl_kwargs: Accepts all arguments that the PyTorch :class:`~torch.utils.data.DataLoader` accepts.
        """
        super().__init__(**dl_kwargs)
        self._device = device

    @property
    def device(self) -> Optional[torch.device]:
        return self._device

    def __iter__(self) -> Union[Iterator[Any], Generator[Any, None, None]]:
        iterator = super().__iter__()
        if self._device is None:
            return iterator

        for item in iterator:
            yield move_data_to_device(item, self._device)
