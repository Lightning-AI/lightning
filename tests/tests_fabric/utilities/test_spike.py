import contextlib

import pytest
import torch

from lightning.fabric import Fabric
from lightning.fabric.utilities.imports import _TORCHMETRICS_GREATER_EQUAL_1_0_0
from lightning.fabric.utilities.spike import SpikeDetection, TrainingSpikeException
from tests_fabric.helpers.runif import RunIf


def spike_detection_test(fabric, global_rank_spike, spike_value, should_raise):
    loss_vals = [1 / i for i in range(1, 10)]
    if fabric.global_rank == global_rank_spike:
        if spike_value is None:
            loss_vals[4] = 3
        else:
            loss_vals[4] = spike_value

    for i in range(len(loss_vals)):
        context = pytest.raises(TrainingSpikeException) if i == 4 and should_raise else contextlib.nullcontext()

        with context:
            fabric.call(
                "on_train_batch_end",
                fabric=fabric,
                loss=torch.tensor(loss_vals[i], device=fabric.device),
                batch=None,
                batch_idx=i,
            )


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize(
    ("global_rank_spike", "num_devices", "spike_value", "finite_only"),
    # NOTE FOR ALL FOLLOWING TESTS:
    # adding run on linux only because multiprocessing on other platforms takes forever
    [
        pytest.param(0, 1, None, True),
        pytest.param(0, 1, None, False),
        pytest.param(0, 1, float("inf"), True),
        pytest.param(0, 1, float("inf"), False),
        pytest.param(0, 1, float("-inf"), True),
        pytest.param(0, 1, float("-inf"), False),
        pytest.param(0, 1, float("NaN"), True),
        pytest.param(0, 1, float("NaN"), False),
        pytest.param(0, 2, None, True, marks=RunIf(linux_only=True)),
        pytest.param(0, 2, None, False, marks=RunIf(linux_only=True)),
        pytest.param(1, 2, None, True, marks=RunIf(linux_only=True)),
        pytest.param(1, 2, None, False, marks=RunIf(linux_only=True)),
        pytest.param(0, 2, float("inf"), True, marks=RunIf(linux_only=True)),
        pytest.param(0, 2, float("inf"), False, marks=RunIf(linux_only=True)),
        pytest.param(1, 2, float("inf"), True, marks=RunIf(linux_only=True)),
        pytest.param(1, 2, float("inf"), False, marks=RunIf(linux_only=True)),
        pytest.param(0, 2, float("-inf"), True, marks=RunIf(linux_only=True)),
        pytest.param(0, 2, float("-inf"), False, marks=RunIf(linux_only=True)),
        pytest.param(1, 2, float("-inf"), True, marks=RunIf(linux_only=True)),
        pytest.param(1, 2, float("-inf"), False, marks=RunIf(linux_only=True)),
        pytest.param(0, 2, float("NaN"), True, marks=RunIf(linux_only=True)),
        pytest.param(0, 2, float("NaN"), False, marks=RunIf(linux_only=True)),
        pytest.param(1, 2, float("NaN"), True, marks=RunIf(linux_only=True)),
        pytest.param(1, 2, float("NaN"), False, marks=RunIf(linux_only=True)),
    ],
)
@pytest.mark.skipif(not _TORCHMETRICS_GREATER_EQUAL_1_0_0, reason="requires torchmetrics>=1.0.0")
def test_fabric_spike_detection_integration(tmp_path, global_rank_spike, num_devices, spike_value, finite_only):
    fabric = Fabric(
        accelerator="cpu",
        devices=num_devices,
        callbacks=[SpikeDetection(exclude_batches_path=tmp_path, finite_only=finite_only)],
        strategy="ddp_spawn",
    )

    # spike_value == None -> typical spike detection
    # finite_only -> typical spike detection and raise with NaN +/- inf
    # if inf -> inf >> other values -> typical spike detection
    should_raise = spike_value is None or finite_only or spike_value == float("inf")
    fabric.launch(
        spike_detection_test,
        global_rank_spike=global_rank_spike,
        spike_value=spike_value,
        should_raise=should_raise,
    )


@pytest.mark.skipif(not _TORCHMETRICS_GREATER_EQUAL_1_0_0, reason="requires torchmetrics>=1.0.0")
def test_spike_detection_detaches_and_clones_last_val(monkeypatch):
    detector = SpikeDetection(mode="min", warmup=1, rtol=2.0)

    x = torch.tensor(2.0, requires_grad=True)
    loss = x * 1.0  # has grad_fn
    detector._update_stats(loss)

    assert isinstance(detector.last_val, torch.Tensor)
    assert not detector.last_val.requires_grad
    assert detector.last_val.grad_fn is None

    # In-place mutation of the caller's loss buffer must not corrupt last_val
    loss.detach().fill_(100.0)
    assert float(detector.last_val.item()) == pytest.approx(2.0)

    # Improving loss and non-finite loss (when finite_only=True) should short-circuit
    # before invoking running_mean.compute()
    compute_calls = 0
    orig_compute = detector.running_mean.compute

    def counting_compute():
        nonlocal compute_calls
        compute_calls += 1
        return orig_compute()

    monkeypatch.setattr(detector.running_mean, "compute", counting_compute)

    assert detector._is_spike(torch.tensor(1.5)) is False
    assert detector._is_spike(torch.tensor(float("nan"))) is True
    assert compute_calls == 0

    # Worsening finite loss evaluates running_mean.compute()
    assert detector._is_spike(torch.tensor(10.0)) is True
    assert compute_calls == 1

