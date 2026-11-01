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
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.utilities.compile import from_compiled, to_uncompiled
from tests_pytorch.helpers.runif import RunIf

pytestmark = RunIf(dynamo=True)


class _RoundTripModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return self.weight * x

    def training_step(self, batch, batch_idx):
        return self(batch[0]).square().mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


@pytest.mark.parametrize("conversions", [1, 2])
@pytest.mark.parametrize("via_trainer", [False, True])
def test_roundtrip_restores_methods(tmp_path, conversions, via_trainer):
    m = _RoundTripModel()
    names = ["forward", "training_step", "validation_step", "test_step", "predict_step"]
    originals = {n: getattr(m, n) for n in names}
    compiled = torch.compile(m, backend="eager")
    for _ in range(conversions):
        if via_trainer:
            tr = Trainer(
                default_root_dir=tmp_path,
                max_steps=1,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            tr.fit(compiled, DataLoader(TensorDataset(torch.ones(4, 1)), batch_size=1))
        else:
            from_compiled(compiled)
    to_uncompiled(compiled)
    assert m._compiler_ctx is None
    for n, v in originals.items():
        assert getattr(m, n) == v, n
    torch.testing.assert_close(m(torch.tensor(2.0)), 2 * m.weight)


@pytest.mark.parametrize("conversions", [1, 2])
def test_uncompiled_forward_never_enters_compiler(conversions):
    torch._dynamo.reset()
    compilations = []

    def backend(gm, example_inputs):
        compilations.append(gm)
        return gm.forward

    m = _RoundTripModel()
    compiled = torch.compile(m, backend=backend)
    for _ in range(conversions):
        from_compiled(compiled)
    to_uncompiled(compiled)
    torch._dynamo.reset()
    compilations.clear()
    m(torch.tensor(2.0))
    assert not compilations


class _AllMethods(LightningModule):
    def forward(self, x):
        return x + 1

    def training_step(self, batch, batch_idx):
        return batch + 2

    def validation_step(self, batch, batch_idx):
        return batch + 3

    def test_step(self, batch, batch_idx):
        return batch + 4

    def predict_step(self, batch, batch_idx):
        return batch + 5


@pytest.mark.parametrize(
    ("name", "add"),
    [("forward", 1), ("training_step", 2), ("validation_step", 3), ("test_step", 4), ("predict_step", 5)],
)
@pytest.mark.parametrize("rounds", [1, 2, 3])
def test_all_hooks_uncompiled(name, add, rounds):
    torch._dynamo.reset()
    calls = []

    def backend(gm, inputs):
        calls.append(gm)
        return gm.forward

    m = _AllMethods()
    fn = getattr(m, name)
    compiled = torch.compile(m, backend=backend)
    for _ in range(rounds):
        from_compiled(compiled)
    to_uncompiled(compiled)
    torch._dynamo.reset()
    calls.clear()
    result = getattr(m, name)(torch.tensor(2.0), *(() if name == "forward" else (0,)))
    torch.testing.assert_close(result, torch.tensor(float(2 + add)))
    assert getattr(m, name) == fn
    assert calls == []


@pytest.mark.parametrize("rounds", [2, 3])
def test_latest_compile_context_preserves_uncompiled_originals(rounds):
    torch._dynamo.reset()
    calls = []

    def backend_a(gm, inputs):
        calls.append("a")
        return gm.forward

    def backend_b(gm, inputs):
        calls.append("b")
        return gm.forward

    m = _RoundTripModel()
    originals = {k: getattr(m, k) for k in ["forward", "training_step", "validation_step", "test_step", "predict_step"]}
    compiled_a = torch.compile(m, backend=backend_a)
    compiled_b = torch.compile(m, backend=backend_b)
    for i in range(rounds):
        from_compiled(compiled_a if i % 2 == 0 else compiled_b)
        m(torch.tensor(float(i + 1)))
    to_uncompiled(m)
    calls.clear()
    torch._dynamo.reset()
    for k, v in originals.items():
        assert getattr(m, k) == v, k
    m(torch.tensor(2.0))
    assert calls == []


def test_reconversion_is_single_layer():
    torch._dynamo.reset()
    m = _RoundTripModel()
    compiled = torch.compile(m, backend="eager")
    from_compiled(compiled)
    old = m.forward
    from_compiled(compiled)
    assert m.forward.__wrapped__ == old.__wrapped__
    to_uncompiled(m)


def test_latest_backend_wins_after_reconversion():
    torch._dynamo.reset()
    calls = []

    def backend_a(gm, inputs):
        calls.append("a")
        return gm.forward

    def backend_b(gm, inputs):
        calls.append("b")
        return gm.forward

    m = _RoundTripModel()
    a = torch.compile(m, backend=backend_a)
    b = torch.compile(m, backend=backend_b)
    from_compiled(a)
    m(torch.tensor(2.0))
    assert calls == ["a"]
    torch._dynamo.reset()
    calls.clear()
    from_compiled(b)
    m(torch.tensor(3.0))
    assert calls == ["b"]
    to_uncompiled(m)
    torch._dynamo.reset()
    calls.clear()
    m(torch.tensor(4.0))
    assert calls == []


@pytest.mark.parametrize("stage", ["validate", "test", "predict"])
def test_repeated_evaluation_trainer_restores_hooks(tmp_path, stage):
    m = _AllMethods()
    originals = {n: getattr(m, n) for n in ["forward", "training_step", "validation_step", "test_step", "predict_step"]}
    compiled = torch.compile(m, backend="eager")
    for _ in range(2):
        tr = Trainer(
            default_root_dir=tmp_path,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        getattr(tr, stage)(compiled, DataLoader(torch.arange(3, dtype=torch.float32), batch_size=1))
    to_uncompiled(compiled)
    for n, v in originals.items():
        assert getattr(m, n) == v, n
