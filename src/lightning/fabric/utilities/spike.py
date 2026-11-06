import json
import os
import warnings
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import torch

from lightning.fabric.utilities.imports import _TORCHMETRICS_GREATER_EQUAL_1_0_0
from lightning.fabric.utilities.types import _PATH

if TYPE_CHECKING:
    from lightning.fabric.fabric import Fabric


class SpikeDetection:
    """Spike Detection Callback.

    Terminates training with a ``TrainingSpikeException`` when a loss-spike was detected and
    saves the batches to skip when resuming to a file.

    We skip the current and the previous batch since it is unclear whether the previous batch
    altered the weights in a way that it causes the spike or just the current batch is corrupted somehow.

    Args:
        mode: Whether to minimize or maximize the tracked metric
        window: A running mean of metrics with ``window`` size. Serves as reference value for spikes.
        warmup: After how many batches spike-tracking should start
        atol: An absolute tolerance.  Every diff between the running mean and the current value,
            that's not an improvement and above ``atol`` will be considered a spike
        rtol: A relative tolerance. Every diff between the running mean and the current value,
            that's higher than ``rtol * running_mean`` is considered a spike
        exclude_batches_path: Where to save the file that contains the batches to exclude.
            Will default to current directory.
        finite_only: If set to ``False``, consider non-finite values like NaN, inf and -inf a spike as well.

    """

    def __init__(
        self,
        mode: Literal["min", "max"] = "min",
        window: int = 10,
        warmup: int = 1,
        atol: Optional[float] = None,
        rtol: Optional[float] = 2.0,
        exclude_batches_path: Optional[_PATH] = None,
        finite_only: bool = True,
    ):
        if _TORCHMETRICS_GREATER_EQUAL_1_0_0:
            from torchmetrics.aggregation import MeanMetric
            from torchmetrics.wrappers import Running
        else:
            raise RuntimeError("SpikeDetection requires `torchmetrics>=1.0.0` Please upgrade your version.")
        super().__init__()

        self.last_val: Union[torch.Tensor, float] = 0.0
        # spike detection happens individually on each machine
        self.running_mean = Running(MeanMetric(dist_sync_on_step=False, sync_on_compute=False), window=window)
        # workaround for https://github.com/Lightning-AI/torchmetrics/issues/1899
        self.running_mean.dist_sync_on_step = False
        self.running_mean.sync_on_compute = False

        self.mode = mode
        self.warmup = warmup
        self.atol = atol
        self.rtol = rtol
        self.bad_batches: list[int] = []
        self.exclude_batches_path = exclude_batches_path
        self.finite_only = finite_only

    @torch.no_grad()
    def on_train_batch_end(self, fabric: "Fabric", loss: torch.Tensor, batch: Any, batch_idx: int) -> None:
        """Checks if we currently have a loss-spike."""
        if batch_idx == 0:
            self.running_mean.to(fabric.strategy.root_device)

        if self.exclude_batches_path is None:
            self.exclude_batches_path = os.getcwd()

        if not str(self.exclude_batches_path).endswith(".json"):
            self.exclude_batches_path = os.path.join(self.exclude_batches_path, "skip_batches.json")

        is_spike = bool(batch_idx >= self.warmup and self._is_spike(loss))
        fabric.strategy.barrier()

        # While spike-detection happens on a per-rank level, we need to fail all ranks if any rank detected a spike
        is_spike_global = fabric.strategy.reduce_boolean_decision(is_spike, all=False)

        if is_spike_global:
            self._handle_spike(fabric, batch_idx)
        else:
            is_finite_all = self.finite_only or fabric.strategy.reduce_boolean_decision(
                bool(torch.isfinite(loss).all()), all=True
            )
            if is_finite_all:
                self._update_stats(loss)

    def _is_spike(self, loss: torch.Tensor) -> bool:
        if self.finite_only and not torch.isfinite(loss):
            return True

        curr_diff = loss - self.last_val
        if self._is_better(curr_diff):
            return False

        # we might call compute more often than update which is fine as long as the
        # metric has at least one internal value.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            running_val = self.running_mean.compute()

        return self._check_atol(loss, running_val) and self._check_rtol(loss, running_val)

    def _handle_spike(self, fabric: "Fabric", batch_idx: int) -> None:
        # Exclude current and last batch
        # Current batch is excluded since it could be that the data of this batch produces a high loss
        # Last batch is excluded since the previous batch could have "corrupted" the weights
        self.bad_batches.extend([batch_idx - 1, batch_idx])

        if fabric.global_rank == 0:
            assert self.exclude_batches_path is not None
            os.makedirs(os.path.dirname(self.exclude_batches_path), exist_ok=True)

            with open(self.exclude_batches_path, "w") as f:
                json.dump(self.bad_batches, f, indent=4)

        raise TrainingSpikeException(batch_idx=batch_idx)

    def _check_atol(self, val_a: Union[float, torch.Tensor], val_b: Union[float, torch.Tensor]) -> bool:
        return (self.atol is None) or bool(abs(val_a - val_b) >= abs(self.atol))  # type: ignore

    def _check_rtol(self, val_a: Union[float, torch.Tensor], val_b: Union[float, torch.Tensor]) -> bool:
        return (self.rtol is None) or bool(abs(val_a - val_b) >= abs(self.rtol * val_b))  # type: ignore

    def _is_better(self, diff_val: torch.Tensor) -> bool:
        if self.mode == "min":
            return bool((diff_val <= 0.0).all())
        if self.mode == "max":
            return bool((diff_val >= 0).all())

        raise ValueError(f"Invalid mode. Has to be min or max, found {self.mode}")

    def _update_stats(self, val: torch.Tensor) -> None:
        # only update if finite
        self.running_mean.update(val)
        self.last_val = val.detach().clone()

    def state_dict(self) -> dict[str, Any]:
        return {
            "last_val": self.last_val.item() if isinstance(self.last_val, torch.Tensor) else self.last_val,
            "mode": self.mode,
            "warmup": self.warmup,
            "atol": self.atol,
            "rtol": self.rtol,
            "bad_batches": self.bad_batches,
            "bad_batches_path": self.exclude_batches_path,
            "running": self.running_mean.state_dict(),
            "mean": self.running_mean.base_metric.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.last_val = state_dict.pop("last_val")
        self.mode = state_dict.pop("mode")
        self.warmup = state_dict.pop("warmup")
        self.atol = state_dict.pop("atol")
        self.rtol = state_dict.pop("rtol")
        self.bad_batches = state_dict.pop("bad_batches")
        self.exclude_batches_path = state_dict.pop("bad_batches_path")
        self.running.load_state_dict(state_dict.pop("running"))
        self.running_mean.base_metric.load_state_dict(state_dict.pop("mean"))


class TrainingSpikeException(RuntimeError):
    """Exception to be raised with Training Spikes."""

    def __init__(self, batch_idx: int, *args: Any, **kwargs: Any):
        super().__init__(f"Training spike detected in batch {batch_idx}", *args, **kwargs)
