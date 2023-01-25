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
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Union

import torch.distributed
from torch.nn import Module
from torch.optim.optimizer import Optimizer

import pytorch_lightning as pl
from lightning_fabric.plugins import CheckpointIO, ClusterEnvironment
from lightning_fabric.utilities.distributed import group as _group
from pytorch_lightning.overrides.torch_distributed import broadcast_object_list
from pytorch_lightning.plugins.io.hpu_plugin import HPUCheckpointIO
from pytorch_lightning.plugins.io.wrapper import _WrappingCheckpointIO
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _HPU_AVAILABLE
from pytorch_lightning.utilities.types import STEP_OUTPUT

if _HPU_AVAILABLE:
    import habana_frameworks.torch.core as htcore
    import habana_frameworks.torch.distributed.hccl  # noqa: F401

log = logging.getLogger(__name__)


class HPUParallelStrategy(DDPStrategy):
    """Strategy for distributed training on multiple HPU devices."""

    strategy_name = "hpu_parallel"

    def __init__(
        self,
        accelerator: Optional["pl.accelerators.Accelerator"] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
        ddp_comm_state: Optional[object] = None,
        ddp_comm_hook: Optional[Callable] = None,
        ddp_comm_wrapper: Optional[Callable] = None,
        model_averaging_period: Optional[int] = None,
        process_group_backend: Optional[str] = "hccl",
        **kwargs: Any,
    ) -> None:

        if not _HPU_AVAILABLE:
            raise MisconfigurationException("`HPUParallelStrategy` requires HPU devices to run")

        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
            ddp_comm_state=ddp_comm_state,
            ddp_comm_hook=ddp_comm_hook,
            ddp_comm_wrapper=ddp_comm_wrapper,
            model_averaging_period=model_averaging_period,
            process_group_backend=process_group_backend,
            **kwargs,
        )

    @property
    def checkpoint_io(self) -> CheckpointIO:
        if self._checkpoint_io is None:
            self._checkpoint_io = HPUCheckpointIO()
        elif isinstance(self._checkpoint_io, _WrappingCheckpointIO):
            self._checkpoint_io.checkpoint_io = HPUCheckpointIO()

        return self._checkpoint_io

    @checkpoint_io.setter
    def checkpoint_io(self, io: Optional[CheckpointIO]) -> None:
        self._checkpoint_io = io

    def setup_environment(self) -> None:

        os.environ["ID"] = str(self.local_rank)
        if self._process_group_backend == "hccl":
            # this env is used in overrides to check the backend initiated
            os.environ["HCCL_DISTRIBUTED_BACKEND"] = str(1)
        super().setup_environment()

    def determine_ddp_device_ids(self) -> None:
        return None

    def _pre_configure_ddp(self) -> None:
        # if unset, default `find_unused_parameters` `True`
        # Many models require setting this parameter to True, as there are corner cases
        # when not all parameter backward hooks are fired by the autograd engine even if require_grad is set to True.
        # This flag does come with a performance hit, so it is suggested to disable in cases where it is possible.
        self._ddp_kwargs["find_unused_parameters"] = self._ddp_kwargs.get("find_unused_parameters", True)

        self._static_graph = False
        static_graph = self._ddp_kwargs.get("static_graph")
        if static_graph:
            # when _set_static_graph() is called find_unused_parameters does not have any significance.
            # Resetting the value of find_unused_parameters to False which is the default value to DDP
            self._ddp_kwargs["find_unused_parameters"] = False
            self._static_graph = True
        if static_graph is not None:
            # DDP does not accept static_graph as a parameter, hence removing it from the list
            del self._ddp_kwargs["static_graph"]

    def broadcast(self, obj: object, src: int = 0) -> object:  # type: ignore
        obj = [obj]
        if self.global_rank != src:
            obj = [None]

        broadcast_object_list(obj, src, group=_group.WORLD)
        return obj[0]

    def on_after_backward(self) -> None:
        # Break lazy accumulation of graph after fwd+bwd
        htcore.mark_step()

    def optimizer_step(
        self,
        optimizer: Optimizer,
        opt_idx: int,
        closure: Callable[[], Any],
        model: Optional[Union["pl.LightningModule", Module]] = None,
        **kwargs: Any,
    ) -> Any:
        optimizer_output = super().optimizer_step(optimizer, opt_idx, closure, model, **kwargs)
        # Break lazy accumulation of graph after optimizer
        htcore.mark_step()
        return optimizer_output

    def validation_step_end(self, step_output: STEP_OUTPUT) -> STEP_OUTPUT:
        # Break lazy accumulation of graph after every step
        htcore.mark_step()
        return step_output

    def test_step_end(self, step_output: STEP_OUTPUT) -> STEP_OUTPUT:
        # Break lazy accumulation of graph after every step
        htcore.mark_step()
        return step_output

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description=f"{cls.__class__.__name__}",
        )

    def teardown(self) -> None:
        super().teardown()
        # Was set to local rank
        os.environ.pop("ID", None)
        os.environ.pop("HCCL_DISTRIBUTED_BACKEND", None)
