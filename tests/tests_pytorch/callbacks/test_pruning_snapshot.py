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
from torch.nn.utils import prune
from torch.utils.data import DataLoader, TensorDataset

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelPruning


class Model(LightningModule):
    def __init__(self, dtype):
        super().__init__()
        self.layer = torch.nn.Linear(4, 2, bias=True, dtype=dtype)
        with torch.no_grad():
            self.layer.weight.copy_(torch.arange(1.0, 9.0, dtype=dtype).view(2, 4))
            self.layer.bias.copy_(torch.tensor([1.0, 2.0], dtype=dtype))

    def training_step(self, batch, batch_idx):
        return self.layer(batch[0].to(self.layer.weight.dtype)).square().mean()

    def validation_step(self, batch, batch_idx):
        return None

    def train_dataloader(self):
        return DataLoader(TensorDataset(torch.ones(3, 4)), batch_size=1)

    def val_dataloader(self):
        return DataLoader(TensorDataset(torch.ones(1, 4)), batch_size=1)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)


@pytest.mark.parametrize("global_pruning", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("at_train_end", [False, True])
@pytest.mark.parametrize("permanent", [False, True])
def test_multiple_resets_keep_snapshot_and_optimizer(tmp_path, global_pruning, dtype, at_train_end, permanent):
    model = Model(dtype)
    initial = {name: p.detach().clone() for (name, p) in model.layer.named_parameters()}
    parameters = dict(model.layer.named_parameters())

    class Inspect(ModelPruning):
        calls = 0

        def apply_lottery_ticket_hypothesis(self):
            masks = {name: getattr(model.layer, name + "_mask").clone() for name in initial}
            super().apply_lottery_ticket_hypothesis()
            self.calls += 1
            for name, expected in initial.items():
                live = getattr(model.layer, name + "_orig")
                assert live is parameters[name]
                torch.testing.assert_close(live, expected)
                torch.testing.assert_close(getattr(model.layer, name + "_mask"), masks[name])
            for saved in self._original_layers.values():
                for name in initial:
                    torch.testing.assert_close(getattr(saved["data"], name), initial[name])

    cb = Inspect(
        "l1_unstructured",
        parameters_to_prune=[(model.layer, "weight"), (model.layer, "bias")],
        amount=0.25,
        use_global_unstructured=global_pruning,
        use_lottery_ticket_hypothesis=True,
        make_pruning_permanent=permanent,
        prune_on_train_epoch_end=at_train_end,
    )
    tr = Trainer(
        default_root_dir=tmp_path,
        max_epochs=3,
        num_sanity_val_steps=0,
        callbacks=[cb],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    tr.fit(model)
    assert cb.calls == 3
    for p in parameters.values():
        assert p in tr.optimizers[0].state
    for name in initial:
        if permanent:
            assert not hasattr(model.layer, name + "_orig")
        else:
            assert getattr(model.layer, name + "_orig") is parameters[name]


@pytest.mark.parametrize("pruned", [False, True])
@pytest.mark.parametrize("name", ["weight", "bias"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_copy_independence_bidirectional(pruned, name, dtype):
    source = torch.nn.Linear(4, 2, dtype=dtype)
    dest = torch.nn.Linear(4, 2, dtype=dtype)
    if pruned:
        prune.l1_unstructured(dest, name, amount=0.25)
    parameter = getattr(dest, name + "_orig" if pruned else name)
    expected = getattr(source, name).detach().clone()
    ModelPruning._copy_param(dest, source, name)
    assert getattr(dest, name + "_orig" if pruned else name) is parameter
    with torch.no_grad():
        parameter.add_(3)
    torch.testing.assert_close(getattr(source, name), expected)
    with torch.no_grad():
        getattr(source, name).sub_(5)
    torch.testing.assert_close(parameter, expected + 3)


@pytest.mark.parametrize("name", ["weight", "bias"])
def test_none_parameters_are_unchanged(name):
    a = torch.nn.Linear(4, 2, bias=False)
    b = torch.nn.Linear(4, 2, bias=False)
    if name == "weight":
        a.weight = None
    ModelPruning._copy_param(b, a, name)


@pytest.mark.parametrize("global_pruning", [False, True])
@pytest.mark.parametrize("resample", [False, True])
@pytest.mark.parametrize("permanent", [False, True])
def test_checkpoint_matches_effective_pruned_weights(tmp_path, global_pruning, resample, permanent):
    m = Model(torch.float32)
    initial = {n: p.detach().clone() for (n, p) in m.layer.named_parameters()}
    cb = ModelPruning(
        "l1_unstructured",
        parameters_to_prune=[(m.layer, "weight"), (m.layer, "bias")],
        amount=0.25,
        use_global_unstructured=global_pruning,
        use_lottery_ticket_hypothesis=True,
        resample_parameters=resample,
        make_pruning_permanent=permanent,
    )
    t = Trainer(
        default_root_dir=tmp_path,
        max_epochs=3,
        num_sanity_val_steps=0,
        callbacks=[cb],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    t.fit(m)
    path = tmp_path / "last.ckpt"
    t.save_checkpoint(path)
    saved = torch.load(path, weights_only=False)["state_dict"]
    for name in initial:
        expected = (
            getattr(m.layer, name) if permanent else getattr(m.layer, name + "_orig") * getattr(m.layer, name + "_mask")
        )
        actual = (
            saved["layer." + name] if permanent else saved["layer." + name + "_orig"] * saved["layer." + name + "_mask"]
        )
        torch.testing.assert_close(actual, expected)
    for snap in cb._original_layers.values():
        for name in initial:
            torch.testing.assert_close(getattr(snap["data"], name), initial[name])


@pytest.mark.parametrize("global_pruning", [False, True])
def test_no_lottery_ticket_control(tmp_path, global_pruning):
    m = Model(torch.float32)
    cb = ModelPruning(
        "l1_unstructured",
        parameters_to_prune=[(m.layer, "weight")],
        amount=0.25,
        use_global_unstructured=global_pruning,
        use_lottery_ticket_hypothesis=False,
    )
    t = Trainer(
        default_root_dir=tmp_path,
        max_epochs=3,
        num_sanity_val_steps=0,
        callbacks=[cb],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    t.fit(m)
    assert cb._original_layers is None


# Snapshot isolation is checked independently of checkpoint serialization.
