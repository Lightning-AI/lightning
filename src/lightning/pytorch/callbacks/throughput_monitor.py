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
import time
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import torch
from typing_extensions import override

from lightning.fabric.plugins import Precision as FabricPrecision
from lightning.fabric.utilities.throughput import Throughput, get_available_flops
from lightning.fabric.utilities.throughput import _plugin_to_compute_dtype as fabric_plugin_to_compute_dtype
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.plugins import (
    BitsandbytesPrecision,
    DeepSpeedPrecision,
    DoublePrecision,
    FSDPPrecision,
    HalfPrecision,
    MixedPrecision,
    Precision,
    TransformerEnginePrecision,
    XLAPrecision,
)
from lightning.pytorch.trainer.states import RunningStage, TrainerFn
from lightning.pytorch.utilities.rank_zero import rank_zero_only, rank_zero_warn

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer


class ThroughputMonitor(Callback):
    r"""Computes and logs throughput with the :class:`~lightning.fabric.utilities.throughput.Throughput`

    Example::

        class MyModel(LightningModule):
            def setup(self, stage):
                with torch.device("meta"):
                    model = MyModel()

                    def sample_forward():
                        batch = torch.randn(..., device="meta")
                        return model(batch)

                    self.flops_per_batch = measure_flops(model, sample_forward, loss_fn=torch.Tensor.sum)


        logger = ...
        throughput = ThroughputMonitor(batch_size_fn=lambda batch: batch.size(0))
        trainer = Trainer(max_steps=1000, log_every_n_steps=10, callbacks=throughput, logger=logger)
        model = MyModel()
        trainer.fit(model)

    Notes:
        - It assumes that the batch size is the same during all iterations.
        - It will try to access a ``flops_per_batch`` attribute on your ``LightningModule`` on every iteration.
          We suggest using the :func:`~lightning.fabric.utilities.throughput.measure_flops` function for this.
          You might want to compute it differently each time based on your setup.

    Args:
        batch_size_fn: A function to compute the number of samples given a batch.
        length_fn: A function to compute the number of items in a sample given a batch.
        \**kwargs: See available parameters in
            :class:`~lightning.fabric.utilities.throughput.Throughput`

    """

    def __init__(
        self, batch_size_fn: Callable[[Any], int], length_fn: Optional[Callable[[Any], int]] = None, **kwargs: Any
    ) -> None:
        super().__init__()
        self.kwargs = kwargs
        self.batch_size_fn = batch_size_fn
        self.length_fn = length_fn
        self.available_flops: Optional[int] = None
        self._throughputs: dict[RunningStage, Throughput] = {}
        self._t0s: dict[RunningStage, float] = {}
        self._lengths: dict[RunningStage, int] = {}
        self._samples: dict[RunningStage, int] = {}
        self._batches: dict[RunningStage, int] = {}
        self._module_has_flops: bool | None = None

    @override
    def setup(self, trainer: "Trainer", pl_module: "LightningModule", stage: str) -> None:
        self._module_has_flops = None
        dtype = _plugin_to_compute_dtype(trainer.precision_plugin)
        self.available_flops = get_available_flops(trainer.strategy.root_device, dtype)

        if stage == TrainerFn.FITTING and trainer.enable_validation:
            # `fit` includes validation inside
            throughput = Throughput(available_flops=self.available_flops, world_size=trainer.world_size, **self.kwargs)
            self._throughputs[RunningStage.VALIDATING] = throughput

        throughput = Throughput(available_flops=self.available_flops, world_size=trainer.world_size, **self.kwargs)
        stage = trainer.state.stage
        assert stage is not None
        self._throughputs[stage] = throughput

    def _start(self, trainer: "Trainer") -> None:
        stage = trainer.state.stage
        assert stage is not None

        reset_needed = trainer.state.fn == TrainerFn.FITTING or stage not in self._samples

        if reset_needed:
            self._throughputs[stage].reset()
            self._lengths[stage] = 0
            self._samples[stage] = 0
            self._batches[stage] = 0

        self._t0s[stage] = time.perf_counter()

    @torch.inference_mode()  # in case `length_fn` or `batch_size_fn` computes grads
    def _update(self, trainer: "Trainer", pl_module: "LightningModule", batch: Any, iter_num: int) -> None:
        stage = trainer.state.stage
        assert stage is not None
        throughput = self._throughputs[stage]

        if trainer.strategy.root_device.type == "cuda":
            # required or else perf_counter() won't be correct
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - self._t0s[stage]
        if self.length_fn is not None:
            self._lengths[stage] += self.length_fn(batch)

        if self._module_has_flops is None:
            self._module_has_flops = hasattr(pl_module, "flops_per_batch")
            if not self._module_has_flops:
                rank_zero_warn(
                    "When using the `ThroughputMonitor`, you need to define a `flops_per_batch` attribute or property"
                    f" in {type(pl_module).__name__} to compute the FLOPs."
                )

        flops_per_batch = pl_module.flops_per_batch if self._module_has_flops else None

        self._samples[stage] += self.batch_size_fn(batch)
        self._batches[stage] += 1

        throughput.update(
            time=elapsed,
            batches=self._batches[stage],
            # this assumes that all iterations used the same batch size
            samples=self._samples[stage],
            lengths=None if self.length_fn is None else self._lengths[stage],
            flops=flops_per_batch,  # type: ignore[arg-type]
        )

    def _compute(self, trainer: "Trainer", iter_num: Optional[int] = None) -> None:
        if not trainer._logger_connector.should_update_logs:
            return
        stage = trainer.state.stage
        assert stage is not None
        throughput = self._throughputs[stage]
        metrics = throughput.compute()
        # prefix with the stage to avoid collisions
        metrics = {f"{stage.value}{throughput.separator}{k}": v for k, v in metrics.items()}
        trainer._logger_connector.log_metrics(metrics, step=iter_num)  # type: ignore[arg-type]

    @override
    @rank_zero_only
    def on_train_start(self, trainer: "Trainer", *_: Any) -> None:
        self._start(trainer)

    @override
    @rank_zero_only
    def on_train_batch_end(
        self, trainer: "Trainer", pl_module: "LightningModule", outputs: Any, batch: Any, *_: Any
    ) -> None:
        self._update(trainer, pl_module, batch, trainer.fit_loop.total_batch_idx + 1)
        # log only when gradient accumulation is over. this ensures that we only measure when the effective batch has
        # finished and the `optimizer.step()` time is included
        if not trainer.fit_loop._should_accumulate():
            self._compute(trainer)

    @override
    @rank_zero_only
    def on_validation_start(self, trainer: "Trainer", *_: Any) -> None:
        if trainer.sanity_checking:
            return
        self._start(trainer)

    @override
    @rank_zero_only
    def on_validation_batch_end(
        self, trainer: "Trainer", pl_module: "LightningModule", outputs: Any, batch: Any, *_: Any, **__: Any
    ) -> None:
        if trainer.sanity_checking:
            return
        iter_num = trainer._evaluation_loop.batch_progress.total.ready
        self._update(trainer, pl_module, batch, iter_num)
        self._compute(trainer, iter_num)

    @override
    @rank_zero_only
    def on_validation_end(self, trainer: "Trainer", *_: Any) -> None:
        if trainer.sanity_checking or trainer.state.fn != TrainerFn.FITTING:
            return

        train_times = self._throughputs[RunningStage.TRAINING]._time
        val_times = self._throughputs[RunningStage.VALIDATING]._time

        train_elapsed = train_times[-1] if train_times else 0.0
        val_elapsed = val_times[-1] if val_times else 0.0

        # add the validation time to the training time before continuing to avoid sinking the training throughput
        training_finished = self._t0s[RunningStage.TRAINING] + train_elapsed
        time_between_train_and_val = self._t0s[RunningStage.VALIDATING] - training_finished
        val_time = val_elapsed
        self._t0s[RunningStage.TRAINING] += time_between_train_and_val + val_time

    @override
    @rank_zero_only
    def on_test_start(self, trainer: "Trainer", *_: Any) -> None:
        self._start(trainer)

    @override
    @rank_zero_only
    def on_test_batch_end(
        self, trainer: "Trainer", pl_module: "LightningModule", outputs: Any, batch: Any, *_: Any, **__: Any
    ) -> None:
        iter_num = trainer._evaluation_loop.batch_progress.total.ready
        self._update(trainer, pl_module, batch, iter_num)
        self._compute(trainer, iter_num)

    @override
    @rank_zero_only
    def on_predict_start(self, trainer: "Trainer", *_: Any) -> None:
        self._start(trainer)

    @override
    @rank_zero_only
    def on_predict_batch_end(
        self, trainer: "Trainer", pl_module: "LightningModule", outputs: Any, batch: Any, *_: Any, **__: Any
    ) -> None:
        iter_num = trainer.predict_loop.batch_progress.total.ready
        self._update(trainer, pl_module, batch, iter_num)
        self._compute(trainer, iter_num)


def _plugin_to_compute_dtype(plugin: Union[FabricPrecision, Precision]) -> torch.dtype:
    # TODO: integrate this into the precision plugins
    if not isinstance(plugin, Precision):
        return fabric_plugin_to_compute_dtype(plugin)
    if isinstance(plugin, BitsandbytesPrecision):
        return plugin.dtype
    if isinstance(plugin, HalfPrecision):
        return plugin._desired_input_dtype
    if isinstance(plugin, MixedPrecision):
        return torch.bfloat16 if plugin.precision == "bf16-mixed" else torch.half
    if isinstance(plugin, DoublePrecision):
        return torch.double
    if isinstance(plugin, (XLAPrecision, DeepSpeedPrecision)):
        return plugin._desired_dtype
    if isinstance(plugin, TransformerEnginePrecision):
        return torch.int8
    if isinstance(plugin, FSDPPrecision):
        return plugin.mixed_precision_config.reduce_dtype or torch.float32
    if isinstance(plugin, Precision):
        return torch.float32
    raise NotImplementedError(plugin)
