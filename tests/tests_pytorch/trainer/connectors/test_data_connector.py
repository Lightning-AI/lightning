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
from collections.abc import Sized
from re import escape
from unittest import mock
from unittest.mock import Mock

import pytest
from lightning_utilities.test.warning import no_warning_call
from torch import Tensor
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler, RandomSampler, Sampler, SequentialSampler

import lightning.fabric
from lightning.fabric.utilities.distributed import DistributedSamplerWrapper
from lightning.fabric.utilities.warnings import PossibleUserWarning
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringDataModule, BoringModel, RandomDataset
from lightning.pytorch.overrides.distributed import UnrepeatedDistributedSampler, UnrepeatedDistributedSamplerWrapper
from lightning.pytorch.trainer.connectors.data_connector import (
    _check_dataloader_iterable,
    _DataHookSelector,
    _DataLoaderSource,
    _get_distributed_sampler,
    _worker_check,
    warning_cache,
)
from lightning.pytorch.trainer.states import RunningStage, TrainerFn
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from lightning.pytorch.utilities.data import _update_dataloader
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from tests_pytorch.helpers.runif import RunIf, _xfail_gloo_windows


@RunIf(skip_windows=True)
@pytest.mark.parametrize("mode", [1, 2])
def test_replace_distributed_sampler(tmp_path, mode):
    class IndexedRandomDataset(RandomDataset):
        def __getitem__(self, index):
            return self.data[index]

    class CustomDataLoader(DataLoader):
        def __init__(self, num_features, dataset, *args, **kwargs):
            # argument `num_features` unused on purpose
            # it gets automatically captured by _replace_dataloader_init_method()
            super().__init__(dataset, *args, **kwargs)

    class CustomBatchSampler(BatchSampler):
        pass

    class TestModel(BoringModel):
        def __init__(self, numbers_test_dataloaders, mode):
            super().__init__()
            self._numbers_test_dataloaders = numbers_test_dataloaders
            self._mode = mode

        def test_step(self, batch, batch_idx, dataloader_idx=0):
            return super().test_step(batch, batch_idx)

        def on_test_start(self) -> None:
            dataloader = self.trainer.test_dataloaders[0]
            assert isinstance(dataloader, CustomDataLoader)
            batch_sampler = dataloader.batch_sampler
            if self._mode == 1:
                assert isinstance(batch_sampler, CustomBatchSampler)
                # the batch_size is set on the batch sampler
                assert dataloader.batch_size is None
            elif self._mode == 2:
                assert type(batch_sampler) is BatchSampler
                assert dataloader.batch_size == self._mode
            assert batch_sampler.batch_size == self._mode
            assert batch_sampler.drop_last
            # the sampler has been replaced
            assert isinstance(batch_sampler.sampler, DistributedSampler)

        def create_dataset(self):
            dataset = IndexedRandomDataset(32, 64)
            if self._mode == 1:
                # with a custom batch sampler
                batch_sampler = CustomBatchSampler(SequentialSampler(dataset), batch_size=1, drop_last=True)
                return CustomDataLoader(32, dataset, batch_sampler=batch_sampler)
            if self._mode == 2:
                # with no batch sampler provided
                return CustomDataLoader(32, dataset, batch_size=2, drop_last=True)
            return None

        def test_dataloader(self):
            return [self.create_dataset()] * self._numbers_test_dataloaders

    model = TestModel(2, mode)

    trainer = Trainer(
        default_root_dir=tmp_path,
        limit_test_batches=2,
        accelerator="cpu",
        devices=1,
        strategy="ddp_find_unused_parameters_false",
    )
    trainer.test(model)


class TestSpawnBoringModel(BoringModel):
    def __init__(self, warning_expected=False):
        super().__init__()
        self.warning_expected = warning_expected

    def on_fit_start(self):
        ctx = pytest.warns if self.warning_expected else no_warning_call
        self.ctx = ctx(UserWarning, match="Consider setting `persistent_workers=True`")
        if self.global_rank == 0:
            self.ctx.__enter__()

    def on_train_end(self):
        if self.global_rank == 0:
            self.ctx.__exit__(None, None, None)


@_xfail_gloo_windows
@pytest.mark.parametrize("num_workers", [0, 1, 2])
def test_dataloader_persistent_workers_performance_warning(num_workers, tmp_path):
    """Test that when the multiprocessing start-method is 'spawn', we recommend setting `persistent_workers=True`."""
    trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        devices=1,
        strategy="ddp_spawn",
        max_steps=1,
        barebones=True,
    )
    model = TestSpawnBoringModel(warning_expected=(num_workers > 0))
    dataloader = DataLoader(RandomDataset(32, 64), num_workers=num_workers)
    trainer.fit(model, dataloader)


@pytest.mark.parametrize(
    ("num_workers", "cpu_count", "expected_warning"),
    [
        (0, 1, False),
        (1, 1, False),
        (2, 1, False),
        (3, 1, False),
        (0, 8, True),
        (1, 8, True),
        (2, 8, False),
        (3, 8, False),
    ],
)
@mock.patch("lightning.fabric.utilities.data.os.cpu_count")
@mock.patch("lightning.pytorch.trainer.connectors.data_connector.mp.get_start_method", return_value="not_spawn")
def test_worker_check(_, cpu_count_mock, num_workers, cpu_count, expected_warning, monkeypatch):
    monkeypatch.delattr(lightning.fabric.utilities.data.os, "sched_getaffinity", raising=False)
    trainer = Mock(spec=Trainer)
    dataloader = Mock(spec=DataLoader, persistent_workers=False)
    trainer.num_devices = 2
    dataloader.num_workers = num_workers
    cpu_count_mock.return_value = cpu_count

    if expected_warning:
        ctx = pytest.warns(UserWarning, match="Consider increasing the value of the `num_workers` argument`")
    else:
        ctx = no_warning_call()

    with ctx:
        _worker_check(trainer, dataloader=dataloader, name="train_dataloader")


@mock.patch("lightning.pytorch.trainer.connectors.data_connector.suggested_max_num_workers", return_value=2)
def test_worker_check_reload_dataloaders_every_n_epochs_limitations(_):
    """Test that we warn about problematic settings when using `dataloaders_every_n_epochs_limitations`."""
    trainer = Mock(reload_dataloaders_every_n_epochs=1)
    dataloader = DataLoader(range(2), num_workers=1, pin_memory=True, persistent_workers=True)
    with pytest.warns(UserWarning, match="The combination of .*reload_dataloaders_every_n_epochs"):
        _worker_check(trainer, dataloader=dataloader, name="train_dataloader")


def test_update_dataloader_raises():
    with pytest.raises(ValueError, match="needs to subclass `torch.utils.data.DataLoader"):
        _update_dataloader(object(), object(), mode="fit")


def test_dataloaders_with_missing_keyword_arguments():
    ds = RandomDataset(10, 20)

    class TestDataLoader(DataLoader):
        def __init__(self, dataset):
            super().__init__(dataset)

    loader = TestDataLoader(ds)
    sampler = SequentialSampler(ds)
    match = escape("missing arguments are ['batch_sampler', 'sampler', 'shuffle']")
    with pytest.raises(MisconfigurationException, match=match):
        _update_dataloader(loader, sampler, mode="fit")
    match = escape("missing arguments are ['batch_sampler', 'batch_size', 'drop_last', 'sampler', 'shuffle']")
    with pytest.raises(MisconfigurationException, match=match):
        _update_dataloader(loader, sampler, mode="predict")

    class TestDataLoader(DataLoader):
        def __init__(self, dataset, *args, **kwargs):
            super().__init__(dataset)

    loader = TestDataLoader(ds)
    sampler = SequentialSampler(ds)
    _update_dataloader(loader, sampler, mode="fit")
    _update_dataloader(loader, sampler, mode="predict")

    class TestDataLoader(DataLoader):
        def __init__(self, *foo, **bar):
            super().__init__(*foo, **bar)

    loader = TestDataLoader(ds)
    sampler = SequentialSampler(ds)
    _update_dataloader(loader, sampler, mode="fit")
    _update_dataloader(loader, sampler, mode="predict")

    class TestDataLoader(DataLoader):
        def __init__(self, num_feat, dataset, *args, shuffle=False):
            self.num_feat = num_feat
            super().__init__(dataset)

    loader = TestDataLoader(1, ds)
    sampler = SequentialSampler(ds)
    match = escape("missing arguments are ['batch_sampler', 'sampler']")
    with pytest.raises(MisconfigurationException, match=match):
        _update_dataloader(loader, sampler, mode="fit")
    match = escape("missing arguments are ['batch_sampler', 'batch_size', 'drop_last', 'sampler']")
    with pytest.raises(MisconfigurationException, match=match):
        _update_dataloader(loader, sampler, mode="predict")

    class TestDataLoader(DataLoader):
        def __init__(self, num_feat, dataset, **kwargs):
            self.feat_num = num_feat
            super().__init__(dataset)

    loader = TestDataLoader(1, ds)
    sampler = SequentialSampler(ds)
    match = escape("missing attributes are ['num_feat']")
    with pytest.raises(MisconfigurationException, match=match):
        _update_dataloader(loader, sampler, mode="fit")
    match = escape("missing attributes are ['num_feat']")
    with pytest.raises(MisconfigurationException, match=match):
        _update_dataloader(loader, sampler, mode="predict")


def test_update_dataloader_with_multiprocessing_context():
    """This test verifies that `use_distributed_sampler` conserves multiprocessing context."""
    train = RandomDataset(32, 64)
    context = "spawn"
    train = DataLoader(train, batch_size=32, num_workers=2, multiprocessing_context=context, shuffle=True)
    new_data_loader = _update_dataloader(train, SequentialSampler(train.dataset))
    assert new_data_loader.multiprocessing_context == train.multiprocessing_context


def test_dataloader_reinit_for_subclass():
    class CustomDataLoader(DataLoader):
        def __init__(
            self,
            dataset,
            batch_size=1,
            shuffle=False,
            sampler=None,
            batch_sampler=None,
            num_workers=0,
            collate_fn=None,
            pin_memory=False,
            drop_last=False,
            timeout=0,
            worker_init_fn=None,
            dummy_kwarg=None,
        ):
            super().__init__(
                dataset,
                batch_size,
                shuffle,
                sampler,
                batch_sampler,
                num_workers,
                collate_fn,
                pin_memory,
                drop_last,
                timeout,
                worker_init_fn,
            )
            self.dummy_kwarg = dummy_kwarg
            self.something_unrelated = 1

    trainer = Trainer(accelerator="cpu", devices=2, strategy="ddp_spawn")
    mode = RunningStage.TRAINING

    class CustomDummyObj:
        sampler = None

    result = trainer._data_connector._prepare_dataloader(CustomDummyObj(), shuffle=True, mode=mode)
    assert isinstance(result, CustomDummyObj), "Wrongly reinstantiated data loader"

    dataset = list(range(10))
    result = trainer._data_connector._prepare_dataloader(CustomDataLoader(dataset), shuffle=True, mode=mode)
    assert isinstance(result, DataLoader)
    assert isinstance(result, CustomDataLoader)
    assert result.dummy_kwarg is None

    # Shuffled DataLoader should also work
    result = trainer._data_connector._prepare_dataloader(
        CustomDataLoader(dataset, shuffle=True), shuffle=True, mode=mode
    )
    assert isinstance(result, DataLoader)
    assert isinstance(result, CustomDataLoader)
    assert result.dummy_kwarg is None

    class CustomSampler(Sampler):
        def __init__(self, data_source: Sized) -> None:
            self.data_source = data_source

        def __len__(self):
            return len(self.data_source)

        def __iter__(self):
            return iter(range(len(self)))

    # Should raise an error if existing sampler is being replaced
    dataloader = CustomDataLoader(dataset, sampler=CustomSampler(dataset))
    result = trainer._data_connector._prepare_dataloader(dataloader, shuffle=False, mode=mode)
    result_dataset = list(result)
    assert len(result_dataset) == 5
    assert result_dataset == [Tensor([x]) for x in [0, 2, 4, 6, 8]]
    assert isinstance(result.sampler, DistributedSamplerWrapper)
    assert isinstance(result.sampler.dataset._sampler, CustomSampler)


class LoaderTestModel(BoringModel):
    def training_step(self, batch, batch_idx):
        assert len(self.trainer.train_dataloader) == 10
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        assert len(self.trainer.val_dataloaders) == 10
        return super().validation_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        assert len(self.trainer.test_dataloaders) == 10
        return super().test_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        assert len(self.trainer.predict_dataloaders) == 10
        return super().predict_step(batch, batch_idx, dataloader_idx=dataloader_idx)


def test_loader_detaching():
    """Checks that the loader has been reset after the entrypoint."""
    loader = DataLoader(RandomDataset(32, 10), batch_size=1)

    model = LoaderTestModel()

    assert len(model.train_dataloader()) == 64
    assert len(model.val_dataloader()) == 64
    assert len(model.predict_dataloader()) == 64
    assert len(model.test_dataloader()) == 64

    trainer = Trainer(fast_dev_run=1)
    trainer.fit(model, loader, loader)

    assert len(model.train_dataloader()) == 64
    assert len(model.val_dataloader()) == 64
    assert len(model.predict_dataloader()) == 64
    assert len(model.test_dataloader()) == 64

    trainer.validate(model, loader)

    assert len(model.train_dataloader()) == 64
    assert len(model.val_dataloader()) == 64
    assert len(model.predict_dataloader()) == 64
    assert len(model.test_dataloader()) == 64

    trainer.predict(model, loader)

    assert len(model.train_dataloader()) == 64
    assert len(model.val_dataloader()) == 64
    assert len(model.predict_dataloader()) == 64
    assert len(model.test_dataloader()) == 64

    trainer.test(model, loader)

    assert len(model.train_dataloader()) == 64
    assert len(model.val_dataloader()) == 64
    assert len(model.predict_dataloader()) == 64
    assert len(model.test_dataloader()) == 64


def test_pre_made_batches():
    """Check that loader works with pre-made batches."""
    loader = DataLoader(RandomDataset(32, 10), batch_size=None)
    trainer = Trainer(fast_dev_run=1)
    trainer.predict(LoaderTestModel(), loader)


def test_error_raised_with_float_limited_eval_batches():
    """Test that an error is raised if there are not enough batches when passed with float value of
    limit_eval_batches."""
    model = BoringModel()
    dl_size = len(model.val_dataloader())
    limit_val_batches = 1 / (dl_size + 2)
    trainer = Trainer(limit_val_batches=limit_val_batches)
    trainer.strategy.connect(model)
    trainer._data_connector.attach_data(model)
    with pytest.raises(
        MisconfigurationException,
        match=rf"{limit_val_batches} \* {dl_size} < 1. Please increase the `limit_val_batches`",
    ):
        trainer.validate_loop.setup_data()


@pytest.mark.parametrize(
    ("val_dl", "warns"),
    [
        (DataLoader(dataset=RandomDataset(32, 64), shuffle=True), True),
        (DataLoader(dataset=RandomDataset(32, 64), sampler=list(range(64))), False),
        (CombinedLoader(DataLoader(dataset=RandomDataset(32, 64), shuffle=True)), True),
        (
            CombinedLoader([
                DataLoader(dataset=RandomDataset(32, 64)),
                DataLoader(dataset=RandomDataset(32, 64), shuffle=True),
            ]),
            True,
        ),
        (
            CombinedLoader({
                "dl1": DataLoader(dataset=RandomDataset(32, 64)),
                "dl2": DataLoader(dataset=RandomDataset(32, 64), shuffle=True),
            }),
            True,
        ),
    ],
)
def test_non_sequential_sampler_warning_is_raised_for_eval_dataloader(val_dl, warns):
    trainer = Trainer()
    model = BoringModel()
    trainer.strategy.connect(model)
    trainer._data_connector.attach_data(model, val_dataloaders=val_dl)
    context = pytest.warns if warns else no_warning_call
    with context(PossibleUserWarning, match="recommended .* turn shuffling off for val/test"):
        trainer.validate_loop.setup_data()


class NoDataLoaderModel(BoringModel):
    def __init__(self):
        super().__init__()
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        self.predict_dataloader = None


@pytest.mark.parametrize(
    ("instance", "available"),
    [
        (None, True),
        (BoringModel().train_dataloader(), True),
        (BoringModel(), True),
        (NoDataLoaderModel(), False),
        (BoringDataModule(), True),
    ],
)
def test_dataloader_source_available(instance, available):
    """Test the availability check for _DataLoaderSource."""
    source = _DataLoaderSource(instance=instance, name="train_dataloader")
    assert source.is_defined() is available


def test_dataloader_source_direct_access():
    """Test requesting a dataloader when the source is already a dataloader."""
    dataloader = BoringModel().train_dataloader()
    source = _DataLoaderSource(instance=dataloader, name="any")
    assert not source.is_module()
    assert source.is_defined()
    assert source.dataloader() is dataloader


def test_dataloader_source_request_from_module():
    """Test requesting a dataloader from a module works."""
    module = BoringModel()
    trainer = Trainer()
    module.trainer = trainer
    module.foo = Mock(return_value=module.train_dataloader())

    source = _DataLoaderSource(module, "foo")
    assert source.is_module()
    module.foo.assert_not_called()
    assert isinstance(source.dataloader(), DataLoader)
    module.foo.assert_called_once()


@pytest.mark.parametrize(
    "hook_name", ["on_before_batch_transfer", "transfer_batch_to_device", "on_after_batch_transfer"]
)
class TestDataHookSelector:
    def overridden_func(self, batch, *args, **kwargs):
        return batch

    def reset_instances(self):
        warning_cache.clear()
        return BoringDataModule(), BoringModel(), Trainer()

    def test_no_datamodule_no_overridden(self, hook_name):
        model, _, trainer = self.reset_instances()
        trainer._data_connector.attach_datamodule(model, datamodule=None)
        with no_warning_call(match=f"have overridden `{hook_name}` in"):
            instance = trainer._data_connector._datahook_selector.get_instance(hook_name)

        assert instance is model

    def test_with_datamodule_no_overridden(self, hook_name):
        model, dm, trainer = self.reset_instances()
        trainer._data_connector.attach_datamodule(model, datamodule=dm)
        with no_warning_call(match=f"have overridden `{hook_name}` in"):
            instance = trainer._data_connector._datahook_selector.get_instance(hook_name)

        assert instance is model

    def test_override_model_hook(self, hook_name):
        model, dm, trainer = self.reset_instances()
        trainer._data_connector.attach_datamodule(model, datamodule=dm)
        with no_warning_call(match=f"have overridden `{hook_name}` in"):
            instance = trainer._data_connector._datahook_selector.get_instance(hook_name)

        assert instance is model

    def test_override_datamodule_hook(self, hook_name):
        model, dm, trainer = self.reset_instances()
        trainer._data_connector.attach_datamodule(model, datamodule=dm)
        setattr(dm, hook_name, self.overridden_func)
        with no_warning_call(match=f"have overridden `{hook_name}` in"):
            instance = trainer._data_connector._datahook_selector.get_instance(hook_name)

        assert instance is dm

    def test_override_both_model_and_datamodule(self, hook_name):
        model, dm, trainer = self.reset_instances()
        trainer._data_connector.attach_datamodule(model, datamodule=dm)
        setattr(model, hook_name, self.overridden_func)
        setattr(dm, hook_name, self.overridden_func)
        with pytest.warns(UserWarning, match=f"have overridden `{hook_name}` in both"):
            instance = trainer._data_connector._datahook_selector.get_instance(hook_name)

        assert instance is dm

    def test_with_datamodule_override_model(self, hook_name):
        model, dm, trainer = self.reset_instances()
        trainer._data_connector.attach_datamodule(model, datamodule=dm)
        setattr(model, hook_name, self.overridden_func)
        with pytest.warns(UserWarning, match=f"have overridden `{hook_name}` in `LightningModule`"):
            instance = trainer._data_connector._datahook_selector.get_instance(hook_name)

        assert instance is model


def test_invalid_hook_passed_in_datahook_selector():
    dh_selector = _DataHookSelector(BoringModel(), None)
    with pytest.raises(ValueError, match="is not a shared hook"):
        dh_selector.get_instance("setup")


@pytest.mark.parametrize(("devices", "warn_context"), [(1, no_warning_call), (2, pytest.warns)])
def test_eval_distributed_sampler_warning(devices, warn_context):
    """Test that a warning is raised when `DistributedSampler` is used with evaluation."""
    model = BoringModel()
    trainer = Trainer(strategy="ddp", devices=devices, accelerator="cpu")
    trainer.strategy.connect(model)
    trainer._data_connector.attach_data(model)

    trainer.state.fn = TrainerFn.VALIDATING
    with warn_context(PossibleUserWarning, match="multi-device settings use `DistributedSampler`"):
        trainer.validate_loop.setup_data()

    trainer.state.fn = TrainerFn.TESTING
    with warn_context(PossibleUserWarning, match="multi-device settings use `DistributedSampler`"):
        trainer.test_loop.setup_data()


@pytest.mark.parametrize("shuffle", [True, False])
def test_eval_shuffle_with_distributed_sampler_replacement(shuffle):
    """Test that shuffle is not changed if set to True."""

    class CustomModel(BoringModel):
        def val_dataloader(self):
            return DataLoader(RandomDataset(32, 64), shuffle=shuffle)

    trainer = Trainer(accelerator="cpu", devices=2, strategy="ddp")
    model = CustomModel()
    trainer.strategy.connect(model)
    trainer._data_connector.attach_data(model)
    trainer.fit_loop.epoch_loop.val_loop.setup_data()
    assert trainer.val_dataloaders.sampler.shuffle == shuffle


def test_error_raised_with_insufficient_float_limit_train_dataloader():
    batch_size = 16
    dl = DataLoader(RandomDataset(32, batch_size * 9), batch_size=batch_size)
    trainer = Trainer(limit_train_batches=0.1)
    model = BoringModel()
    trainer.strategy.connect(model)
    trainer._data_connector.attach_data(model=model, train_dataloaders=dl)
    trainer.state.fn = TrainerFn.FITTING
    trainer.state.stage = RunningStage.TRAINING
    with pytest.raises(
        MisconfigurationException,
        match="Please increase the `limit_train_batches` argument. Try at least",
    ):
        trainer.fit_loop.setup_data()


@pytest.mark.parametrize(
    ("trainer_fn_name", "dataloader_name"),
    [
        ("fit", "train_dataloaders"),
        ("validate", "dataloaders"),
        ("test", "dataloaders"),
        ("predict", "dataloaders"),
    ],
)
def test_attach_data_input_validation_with_none_dataloader(trainer_fn_name, dataloader_name, tmp_path):
    """Test that passing `Trainer.method(x_dataloader=None)` with no module-method implementations available raises an
    error."""
    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=True)
    model = BoringModel()
    datamodule = BoringDataModule()
    trainer_fn = getattr(trainer, trainer_fn_name)

    # Pretend that these methods are not implemented
    model.train_dataloader = None
    model.val_dataloader = None
    model.test_dataloader = None
    model.predict_dataloader = None

    datamodule.train_dataloader = None
    datamodule.val_dataloader = None
    datamodule.test_dataloader = None
    datamodule.predict_dataloader = None

    with pytest.raises(TypeError, match=f"An invalid .*dataloader was passed to `Trainer.{trainer_fn_name}"):
        trainer_fn(model, **{dataloader_name: None}, datamodule=datamodule)

    with pytest.raises(TypeError, match=f"An invalid .*dataloader was passed to `Trainer.{trainer_fn_name}"):
        trainer_fn(model, **{dataloader_name: None}, datamodule=None)


@pytest.mark.parametrize(
    ("trainer_fn_name", "dataloader_name", "stage"),
    [
        ("fit", "train_dataloaders", RunningStage.TRAINING),
        ("validate", "dataloaders", RunningStage.VALIDATING),
        ("test", "dataloaders", RunningStage.TESTING),
        ("predict", "dataloaders", RunningStage.PREDICTING),
    ],
)
@pytest.mark.parametrize("dataloader", [None, object(), [1, object()]])
def test_non_iterables_raise(tmp_path, trainer_fn_name, dataloader_name, stage, dataloader):
    model = BoringModel()

    # Pretend that these methods are not implemented
    model.train_dataloader = None
    model.val_dataloader = None
    model.test_dataloader = None
    model.predict_dataloader = None

    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=1)
    trainer_fn = getattr(trainer, trainer_fn_name)

    with pytest.raises(
        TypeError, match=rf"invalid dataloader was passed to `Trainer.{trainer_fn_name}\({dataloader_name}"
    ):
        trainer_fn(model, **{dataloader_name: dataloader})

    dl_method = stage.dataloader_prefix + "_dataloader"
    setattr(model, dl_method, lambda: dataloader)
    with pytest.raises(TypeError, match=f"invalid dataloader was returned from `BoringModel.{dl_method}"):
        trainer_fn(model)


def test_iterable_check_on_known_iterators():
    """Test that we only call the `iter()` on the dataloader object if it isn't a known type."""
    iterable = Mock()
    iterable.__iter__ = Mock(return_value=iter(range(3)))
    _check_dataloader_iterable(iterable, Mock(), Mock())
    iterable.__iter__.assert_called_once()

    # If it's a datalaoder, we don't call the expensive `__iter__` method
    dataloader = Mock(spec=DataLoader)
    dataloader.__iter__ = Mock()
    _check_dataloader_iterable(dataloader, Mock(), Mock())
    dataloader.__iter__.assert_not_called()


@pytest.mark.parametrize("sampler_cls", [SequentialSampler, RandomSampler])
def test_predict_distributed_sampler_selection(sampler_cls):
    """Test that the prediction sampler avoids materializing indices that are implied by the dataset."""
    dataset = list(range(11))
    dataloader = DataLoader(dataset, sampler=sampler_cls(dataset), batch_size=2)
    kwargs = {"num_replicas": 3, "rank": 1}
    sampler = _get_distributed_sampler(dataloader, False, 0, RunningStage.PREDICTING, **kwargs)
    if sampler_cls is SequentialSampler:
        assert type(sampler) is UnrepeatedDistributedSampler
        assert sampler.dataset is dataset
        assert list(sampler) == [1, 4, 7, 10]
    else:
        assert type(sampler) is UnrepeatedDistributedSamplerWrapper
        assert len(list(sampler)) == 4

    class CustomSequentialSampler(SequentialSampler):
        def __iter__(self):
            return iter(reversed(range(len(self.data_source))))

    dataloader = DataLoader(dataset, sampler=CustomSequentialSampler(dataset))
    sampler = _get_distributed_sampler(dataloader, False, 0, RunningStage.PREDICTING, **kwargs)
    assert type(sampler) is UnrepeatedDistributedSamplerWrapper
    assert list(sampler) == [9, 6, 3, 0]

    # the sampler wraps a different data source than the dataloader dataset
    other = list(range(5))
    dataloader = DataLoader(dataset, sampler=SequentialSampler(other))
    sampler = _get_distributed_sampler(dataloader, False, 0, RunningStage.PREDICTING, **kwargs)
    assert type(sampler) is UnrepeatedDistributedSamplerWrapper
    assert list(sampler) == [1, 4]
