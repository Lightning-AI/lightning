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
r"""
ModelPruning
^^^^^^^^^^^^
"""

import inspect
import logging
from collections.abc import Sequence
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Optional, Union

import torch.nn.utils.prune as pytorch_prune
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor, nn
from typing_extensions import TypedDict, override

import lightning.pytorch as pl
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.core.module import LightningModule
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.rank_zero import rank_zero_debug, rank_zero_only

log = logging.getLogger(__name__)

_PYTORCH_PRUNING_FUNCTIONS = {
    "ln_structured": pytorch_prune.ln_structured,
    "l1_unstructured": pytorch_prune.l1_unstructured,
    "random_structured": pytorch_prune.random_structured,
    "random_unstructured": pytorch_prune.random_unstructured,
}

_PYTORCH_PRUNING_METHOD = {
    "ln_structured": pytorch_prune.LnStructured,
    "l1_unstructured": pytorch_prune.L1Unstructured,
    "random_structured": pytorch_prune.RandomStructured,
    "random_unstructured": pytorch_prune.RandomUnstructured,
}

_PARAM_TUPLE = tuple[nn.Module, str]
_PARAM_LIST = Sequence[_PARAM_TUPLE]
_MODULE_CONTAINERS = (LightningModule, nn.Sequential, nn.ModuleList, nn.ModuleDict)


class _LayerRef(TypedDict):
    data: nn.Module
    names: list[tuple[int, str]]


class ModelPruning(Callback):
    PARAMETER_NAMES = ("weight", "bias")

    def __init__(
        self,
        pruning_fn: Union[Callable, str],
        parameters_to_prune: _PARAM_LIST = (),
        parameter_names: Optional[list[str]] = None,
        use_global_unstructured: bool = True,
        amount: Union[int, float, Callable[[int], Union[int, float]]] = 0.5,
        apply_pruning: Union[bool, Callable[[int], bool]] = True,
        make_pruning_permanent: bool = True,
        use_lottery_ticket_hypothesis: Union[bool, Callable[[int], bool]] = True,
        resample_parameters: bool = False,
        pruning_dim: Optional[int] = None,
        pruning_norm: Optional[int] = None,
        verbose: int = 0,
        prune_on_train_epoch_end: bool = True,
    ) -> None:
        """Model pruning Callback, using PyTorch's prune utilities. This callback is responsible of pruning networks
        parameters during training.

        To learn more about pruning with PyTorch, please take a look at
        `this tutorial <https://pytorch.org/tutorials/intermediate/pruning_tutorial.html>`_.

        .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

        .. code-block:: python

            parameters_to_prune = [(model.mlp_1, "weight"), (model.mlp_2, "weight")]

            trainer = Trainer(
                callbacks=[
                    ModelPruning(
                        pruning_fn="l1_unstructured",
                        parameters_to_prune=parameters_to_prune,
                        amount=0.01,
                        use_global_unstructured=True,
                    )
                ]
            )

        When ``parameters_to_prune`` is ``None``, ``parameters_to_prune`` will contain all parameters from the model.
        The user can override ``filter_parameters_to_prune`` to filter any ``nn.Module`` to be pruned.

        Args:

            pruning_fn: Function from torch.nn.utils.prune module or your own PyTorch ``BasePruningMethod`` subclass.
                Can also be string e.g. `"l1_unstructured"`. See pytorch docs for more details.

            parameters_to_prune: List of tuples ``(nn.Module, "parameter_name_string")``.

            parameter_names: List of parameter names to be pruned from the nn.Module.
                Can either be ``"weight"`` or ``"bias"``.

            use_global_unstructured: Whether to apply pruning globally on the model.
                If ``parameters_to_prune`` is provided, global unstructured will be restricted on them.

            amount: Quantity of parameters to prune:

                - ``float``. Between 0.0 and 1.0. Represents the fraction of parameters to prune.
                - ``int``. Represents the absolute number of parameters to prune.
                - ``Callable``. For dynamic values. Will be called every epoch. Should return a value.

            apply_pruning: Whether to apply pruning.

                - ``bool``. Always apply it or not.
                - ``Callable[[epoch], bool]``. For dynamic values. Will be called every epoch.

            make_pruning_permanent: Whether to remove all reparameterization pre-hooks and apply masks
                when training ends or the model is saved.

            use_lottery_ticket_hypothesis: See `The lottery ticket hypothesis <https://arxiv.org/abs/1803.03635>`_:

                - ``bool``. Whether to apply it or not.
                - ``Callable[[epoch], bool]``. For dynamic values. Will be called every epoch.

            resample_parameters: Used with ``use_lottery_ticket_hypothesis``. If True, the model parameters will
                be resampled, otherwise, the exact original parameters will be used.

            pruning_dim: If you are using a structured pruning method you need to specify the dimension.

            pruning_norm: If you are using ``ln_structured`` you need to specify the norm.

            verbose: Verbosity level. 0 to disable, 1 to log overall sparsity, 2 to log per-layer sparsity

            prune_on_train_epoch_end: whether to apply pruning at the end of the training epoch.
                If this is ``False``, then the check runs at the end of the validation epoch.

        Raises:
            MisconfigurationException:
                If ``parameter_names`` is neither ``"weight"`` nor ``"bias"``,
                if the provided ``pruning_fn`` is not supported,
                if ``pruning_dim`` is not provided when ``"unstructured"``,
                if ``pruning_norm`` is not provided when ``"ln_structured"``,
                if ``pruning_fn`` is neither ``str`` nor :class:`torch.nn.utils.prune.BasePruningMethod`, or
                if ``amount`` is none of ``int``, ``float`` and ``Callable``.

        """

        self._use_global_unstructured = use_global_unstructured
        self._parameters_to_prune = parameters_to_prune
        self._use_lottery_ticket_hypothesis = use_lottery_ticket_hypothesis
        self._resample_parameters = resample_parameters
        self._prune_on_train_epoch_end = prune_on_train_epoch_end
        self._parameter_names = parameter_names or self.PARAMETER_NAMES
        self._global_kwargs: dict[str, Any] = {}
        self._original_layers: Optional[dict[int, _LayerRef]] = None
        self._pruning_method_name: Optional[str] = None

        for name in self._parameter_names:
            if name not in self.PARAMETER_NAMES:
                raise MisconfigurationException(
                    f"The provided `parameter_names` name: {name} isn't in {self.PARAMETER_NAMES}"
                )

        if isinstance(pruning_fn, str):
            pruning_kwargs = {}
            pruning_fn = pruning_fn.lower()
            if pruning_fn not in _PYTORCH_PRUNING_FUNCTIONS:
                raise MisconfigurationException(
                    f"The provided `pruning_fn` {pruning_fn} isn't available in PyTorch's"
                    f" built-in functions: {list(_PYTORCH_PRUNING_FUNCTIONS.keys())} "
                )
            if pruning_fn.endswith("_structured"):
                if pruning_dim is None:
                    raise MisconfigurationException(
                        "When requesting `structured` pruning, the `pruning_dim` should be provided."
                    )
                if pruning_fn == "ln_structured":
                    if pruning_norm is None:
                        raise MisconfigurationException(
                            "When requesting `ln_structured` pruning, the `pruning_norm` should be provided."
                        )
                    pruning_kwargs["n"] = pruning_norm
                pruning_kwargs["dim"] = pruning_dim
            pruning_fn = self._create_pruning_fn(pruning_fn, **pruning_kwargs)
        elif self._is_pruning_method(pruning_fn):
            if not use_global_unstructured:
                raise MisconfigurationException(
                    "PyTorch `BasePruningMethod` is currently only supported with `use_global_unstructured=True`."
                )
        else:
            raise MisconfigurationException(
                f"`pruning_fn` is expected to be a str in {list(_PYTORCH_PRUNING_FUNCTIONS.keys())}"
                f" or a PyTorch `BasePruningMethod`. Found: {pruning_fn}."
                " HINT: if passing a `BasePruningMethod`, pass the class, not an instance"
            )

        # need to ignore typing here since pytorch base class does not define the PRUNING_TYPE attribute
        if use_global_unstructured and pruning_fn.PRUNING_TYPE != "unstructured":  # type: ignore
            raise MisconfigurationException(
                'Only the "unstructured" PRUNING_TYPE is supported with `use_global_unstructured=True`.'
                f" Found method {pruning_fn} of type {pruning_fn.PRUNING_TYPE}. "  # type: ignore[union-attr]
            )

        self.pruning_fn = pruning_fn
        self._apply_pruning = apply_pruning
        self._make_pruning_permanent = make_pruning_permanent

        if not (isinstance(amount, (int, float)) or callable(amount)):
            raise MisconfigurationException(
                "`amount` should be provided and be either an int, a float or Callable function."
            )

        self.amount = amount

        if verbose not in (0, 1, 2):
            raise MisconfigurationException("`verbose` must be any of (0, 1, 2)")

        self._verbose = verbose

    def filter_parameters_to_prune(self, parameters_to_prune: _PARAM_LIST = ()) -> _PARAM_LIST:
        """This function can be overridden to control which module to prune."""
        return parameters_to_prune

    def _create_pruning_fn(self, pruning_fn: str, **kwargs: Any) -> Union[Callable, pytorch_prune.BasePruningMethod]:
        """This function takes `pruning_fn`, a function name.

        IF use_global_unstructured, pruning_fn will be resolved into its associated ``PyTorch BasePruningMethod`` ELSE,
        pruning_fn will be resolved into its function counterpart from `torch.nn.utils.prune`.

        """
        pruning_meth = (
            _PYTORCH_PRUNING_METHOD[pruning_fn]
            if self._use_global_unstructured
            else _PYTORCH_PRUNING_FUNCTIONS[pruning_fn]
        )
        assert callable(pruning_meth), "Selected pruning method is not callable"
        if self._use_global_unstructured:
            self._global_kwargs = kwargs
        # save the function __name__ now because partial does not include it
        # and there are issues setting the attribute manually in ddp.
        self._pruning_method_name = pruning_meth.__name__
        if self._use_global_unstructured:
            return pruning_meth
        return ModelPruning._wrap_pruning_fn(pruning_meth, **kwargs)

    @staticmethod
    def _wrap_pruning_fn(pruning_fn: Callable, **kwargs: Any) -> Callable:
        return partial(pruning_fn, **kwargs)

    def make_pruning_permanent(self, module: nn.Module) -> None:
        """Removes pruning buffers from any pruned modules.

        Adapted from https://github.com/pytorch/pytorch/blob/v1.7.1/torch/nn/utils/prune.py#L1118-L1122

        """
        for _, module in module.named_modules():
            for k in list(module._forward_pre_hooks):
                hook = module._forward_pre_hooks[k]
                if isinstance(hook, pytorch_prune.BasePruningMethod):
                    hook.remove(module)
                    del module._forward_pre_hooks[k]

    @staticmethod
    def _copy_param(new: nn.Module, old: nn.Module, name: str) -> None:
        # Check if the parameter has been pruned (has _orig suffix)
        dst = getattr(new, name + "_orig") if hasattr(new, name + "_orig") else getattr(new, name)
        src = getattr(old, name)
        if dst is None or src is None or not isinstance(dst, Tensor) or not isinstance(src, Tensor):
            return
        dst.data = src.data.to(dst.device).clone()

    def apply_lottery_ticket_hypothesis(self) -> None:
        r"""Lottery ticket hypothesis algorithm (see page 2 of the paper):

            1. Randomly initialize a neural network :math:`f(x; \theta_0)` (where :math:`\theta_0 \sim \mathcal{D}_\theta`).
            2. Train the network for :math:`j` iterations, arriving at parameters :math:`\theta_j`.
            3. Prune :math:`p\%` of the parameters in :math:`\theta_j`, creating a mask :math:`m`.
            4. Reset the remaining parameters to their values in :math:`\theta_0`, creating the winning ticket :math:`f(x; m \odot \theta_0)`.

        This function implements the step 4.

        The ``resample_parameters`` argument can be used to reset the parameters with a new :math:`\theta_z \sim \mathcal{D}_\theta`

        """  # noqa: E501
        assert self._original_layers is not None
        for d in self._original_layers.values():
            copy = d["data"]
            names = d["names"]
            if self._resample_parameters and hasattr(copy, "reset_parameters") and callable(copy.reset_parameters):
                copy = deepcopy(copy)  # keep the original parameters
                copy.reset_parameters()
            for i, name in names:
                new, _ = self._parameters_to_prune[i]
                self._copy_param(new, copy, name)

    def _apply_local_pruning(self, amount: float) -> None:
        for module, name in self._parameters_to_prune:
            self.pruning_fn(module, name=name, amount=amount)  # type: ignore[call-arg]

    def _resolve_global_kwargs(self, amount: float) -> dict[str, Any]:
        self._global_kwargs["amount"] = amount
        params = set(inspect.signature(self.pruning_fn).parameters)
        params.discard("self")
        return {k: v for k, v in self._global_kwargs.items() if k in params}

    def _apply_global_pruning(self, amount: float) -> None:
        pytorch_prune.global_unstructured(
            self._parameters_to_prune, pruning_method=self.pruning_fn, **self._resolve_global_kwargs(amount)
        )

    @staticmethod
    def _get_pruned_stats(module: nn.Module, name: str) -> tuple[int, int]:
        attr = f"{name}_mask"
        if not hasattr(module, attr):
            return 0, 1
        mask = getattr(module, attr)
        return (mask == 0).sum().item(), mask.numel()

    def apply_pruning(self, amount: Union[int, float]) -> None:
        """Applies pruning to ``parameters_to_prune``."""
        if self._verbose:
            prev_stats = [self._get_pruned_stats(m, n) for m, n in self._parameters_to_prune]

        if self._use_global_unstructured:
            self._apply_global_pruning(amount)
        else:
            self._apply_local_pruning(amount)

        if self._verbose:
            curr_stats = [self._get_pruned_stats(m, n) for m, n in self._parameters_to_prune]
            self._log_sparsity_stats(prev_stats, curr_stats, amount=amount)

    @rank_zero_only
    def _log_sparsity_stats(
        self, prev: list[tuple[int, int]], curr: list[tuple[int, int]], amount: Union[int, float] = 0
    ) -> None:
        total_params = sum(total for _, total in curr)
        prev_total_zeros = sum(zeros for zeros, _ in prev)
        curr_total_zeros = sum(zeros for zeros, _ in curr)
        log.info(
            f"Applied `{self._pruning_method_name}`. Pruned:"
            f" {prev_total_zeros}/{total_params} ({prev_total_zeros / total_params:.2%}) ->"
            f" {curr_total_zeros}/{total_params} ({curr_total_zeros / total_params:.2%})"
        )
        if self._verbose == 2:
            for i, (module, name) in enumerate(self._parameters_to_prune):
                prev_mask_zeros, prev_mask_size = prev[i]
                curr_mask_zeros, curr_mask_size = curr[i]
                log.info(
                    f"Applied `{self._pruning_method_name}` to `{module!r}.{name}` with amount={amount}. Pruned:"
                    f" {prev_mask_zeros} ({prev_mask_zeros / prev_mask_size:.2%}) ->"
                    f" {curr_mask_zeros} ({curr_mask_zeros / curr_mask_size:.2%})"
                )

    @override
    def setup(self, trainer: "pl.Trainer", pl_module: LightningModule, stage: str) -> None:
        parameters_to_prune = self.sanitize_parameters_to_prune(
            pl_module, self._parameters_to_prune, parameter_names=self._parameter_names
        )

        self._parameters_to_prune = self.filter_parameters_to_prune(parameters_to_prune)

        if self._use_lottery_ticket_hypothesis:
            # group modules by id. Each entry has a copy of the initial data
            # and a list of the associated parameter names to prune
            self._original_layers = {}
            for i, (module, name) in enumerate(self._parameters_to_prune):
                id_ = id(module)
                self._original_layers.setdefault(id_, _LayerRef(data=deepcopy(module), names=[]))
                self._original_layers[id_]["names"].append((i, name))

    def _run_pruning(self, current_epoch: int) -> None:
        prune = self._apply_pruning(current_epoch) if callable(self._apply_pruning) else self._apply_pruning
        amount = self.amount(current_epoch) if callable(self.amount) else self.amount
        if not prune or not amount:
            return
        self.apply_pruning(amount)

        if (
            self._use_lottery_ticket_hypothesis(current_epoch)
            if callable(self._use_lottery_ticket_hypothesis)
            else self._use_lottery_ticket_hypothesis
        ):
            self.apply_lottery_ticket_hypothesis()

    @override
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: LightningModule) -> None:
        if self._prune_on_train_epoch_end:
            rank_zero_debug("`ModelPruning.on_train_epoch_end`. Applying pruning")
            self._run_pruning(pl_module.current_epoch)

    @override
    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: LightningModule) -> None:
        if not trainer.sanity_checking and not self._prune_on_train_epoch_end:
            rank_zero_debug("`ModelPruning.on_validation_epoch_end`. Applying pruning")
            self._run_pruning(pl_module.current_epoch)

    @override
    def on_train_end(self, trainer: "pl.Trainer", pl_module: LightningModule) -> None:
        if self._make_pruning_permanent:
            rank_zero_debug("`ModelPruning.on_train_end`. Pruning is made permanent for this checkpoint")
            self.make_pruning_permanent(pl_module)

    def _make_pruning_permanent_on_state_dict(self, pl_module: LightningModule) -> dict[str, Any]:
        state_dict = pl_module.state_dict()

        # find the mask and the original weights.
        map_pruned_params = {k.replace("_mask", "") for k in state_dict if k.endswith("_mask")}
        for tensor_name in map_pruned_params:
            orig = state_dict.pop(tensor_name + "_orig")
            mask = state_dict.pop(tensor_name + "_mask")
            # make weights permanent
            state_dict[tensor_name] = mask.to(dtype=orig.dtype) * orig

        def move_to_cpu(tensor: Tensor) -> Tensor:
            # each tensor and move them on cpu
            return tensor.cpu()

        return apply_to_collection(state_dict, Tensor, move_to_cpu)

    @override
    def on_save_checkpoint(self, trainer: "pl.Trainer", pl_module: LightningModule, checkpoint: dict[str, Any]) -> None:
        if self._make_pruning_permanent:
            rank_zero_debug("`ModelPruning.on_save_checkpoint`. Pruning is made permanent for this checkpoint")
            # manually prune the weights so training can keep going with the same buffers
            checkpoint["state_dict"] = self._make_pruning_permanent_on_state_dict(pl_module)

    @staticmethod
    def sanitize_parameters_to_prune(
        pl_module: LightningModule, parameters_to_prune: _PARAM_LIST = (), parameter_names: Sequence[str] = ()
    ) -> _PARAM_LIST:
        """This function is responsible of sanitizing ``parameters_to_prune`` and ``parameter_names``. If
        ``parameters_to_prune is None``, it will be generated with all parameters of the model.

        Raises:
            MisconfigurationException:
                If ``parameters_to_prune`` doesn't exist in the model, or
                if ``parameters_to_prune`` is neither a list nor a tuple.

        """
        parameters = parameter_names or ModelPruning.PARAMETER_NAMES

        current_modules = [m for m in pl_module.modules() if not isinstance(m, _MODULE_CONTAINERS)]

        if not parameters_to_prune:
            parameters_to_prune = [
                (m, p)
                for p in parameters
                for m in current_modules
                if getattr(m, p, None) is not None and isinstance(getattr(m, p, None), nn.Parameter)
            ]
        elif (
            isinstance(parameters_to_prune, (list, tuple))
            and len(parameters_to_prune) > 0
            and all(len(p) == 2 for p in parameters_to_prune)
            and all(isinstance(a, nn.Module) and isinstance(b, str) for a, b in parameters_to_prune)
        ):
            missing_modules, missing_parameters = [], []
            for module, name in parameters_to_prune:
                if module not in current_modules:
                    missing_modules.append(module)
                    continue
                if not hasattr(module, name):
                    missing_parameters.append(name)

            if missing_modules or missing_parameters:
                raise MisconfigurationException(
                    "Some provided `parameters_to_prune` don't exist in the model."
                    f" Found missing modules: {missing_modules} and missing parameters: {missing_parameters}"
                )
        else:
            raise MisconfigurationException(
                "The provided `parameters_to_prune` should either be list of tuple"
                " with 2 elements: (nn.Module, parameter_name_to_prune) or None"
            )

        return parameters_to_prune

    @staticmethod
    def _is_pruning_method(method: Any) -> bool:
        if not inspect.isclass(method):
            return False
        return issubclass(method, pytorch_prune.BasePruningMethod)
