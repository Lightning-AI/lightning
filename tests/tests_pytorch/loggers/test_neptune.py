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
import os
import pickle
from collections import namedtuple
from unittest import mock
from unittest.mock import MagicMock, call

import pytest
import torch

import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.loggers import NeptuneLogger


def _fetchable_paths(value):
    if value == "sys/id":
        return MagicMock(fetch=MagicMock(return_value="TEST-1"))
    if value == "sys/name":
        return MagicMock(fetch=MagicMock(return_value="Run test name"))
    return MagicMock()


def _fit_and_test(logger, model, tmp_path):
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1, limit_train_batches=0.05, logger=logger)
    assert trainer.log_dir == os.path.join(os.getcwd(), ".neptune")
    trainer.fit(model)
    trainer.test(model)
    assert trainer.log_dir == os.path.join(os.getcwd(), ".neptune")


def _get_logger_with_mocks(**kwargs):
    logger = NeptuneLogger(**kwargs)
    run_instance_mock = MagicMock()
    logger._run_instance = run_instance_mock
    logger._run_instance.__getitem__.return_value.fetch.return_value = "exp-name"
    run_attr_mock = MagicMock()
    logger._run_instance.__getitem__.return_value = run_attr_mock

    return logger, run_instance_mock, run_attr_mock


def test_neptune_online(neptune_mock):
    neptune_mock.init_run.return_value.exists.return_value = True
    neptune_mock.init_run.return_value.__getitem__.side_effect = _fetchable_paths

    logger = NeptuneLogger(api_key="test", project="project")
    created_run_mock = logger.run

    assert logger._run_instance == created_run_mock
    created_run_mock.exists.assert_called_once_with("sys/id")
    assert logger.name == "Run test name"
    assert logger.version == "TEST-1"
    assert neptune_mock.init_run.call_count == 1
    assert created_run_mock.__getitem__.call_count == 2
    assert created_run_mock.__setitem__.call_count == 1
    created_run_mock.__getitem__.assert_has_calls([call("sys/id"), call("sys/name")], any_order=True)
    created_run_mock.__setitem__.assert_called_once_with("source_code/integrations/pytorch-lightning", pl.__version__)


def test_neptune_offline(neptune_mock):
    neptune_mock.init_run.return_value.exists.return_value = False

    logger = NeptuneLogger(mode="offline")
    created_run_mock = logger.run
    logger.experiment["foo"] = "bar"

    created_run_mock.exists.assert_called_once_with("sys/id")
    assert logger._run_short_id == "OFFLINE"
    assert logger._run_name == "offline-name"


def test_online_with_custom_run(neptune_mock):
    neptune_mock.init_run.return_value.exists.return_value = True
    neptune_mock.init_run.return_value.__getitem__.side_effect = _fetchable_paths

    created_run = neptune_mock.init_run()
    neptune_mock.init_run.reset_mock()

    logger = NeptuneLogger(run=created_run)
    assert logger._run_instance == created_run
    assert logger._run_instance == created_run
    assert logger.version == "TEST-1"
    assert neptune_mock.init_run.call_count == 0


def test_neptune_pickling(neptune_mock):
    neptune_mock.init_run.return_value.exists.return_value = True
    neptune_mock.init_run.return_value.__getitem__.side_effect = _fetchable_paths

    unpickleable_run = neptune_mock.init_run()
    with pytest.raises(pickle.PicklingError):
        pickle.dumps(unpickleable_run)
    neptune_mock.init_run.reset_mock()

    logger = NeptuneLogger(run=unpickleable_run)
    assert neptune_mock.init_run.call_count == 0

    pickled_logger = pickle.dumps(logger)
    unpickled = pickle.loads(pickled_logger)

    neptune_mock.init_run.assert_called_once_with(name="Run test name", run="TEST-1")
    assert unpickled.experiment is not None


def test_online_with_wrong_kwargs(neptune_mock):
    """Tests combinations of kwargs together with `run` kwarg which makes some of other parameters unavailable in
    init."""
    run = neptune_mock.init_run()

    with pytest.raises(ValueError):
        NeptuneLogger(run="some string")

    with pytest.raises(ValueError):
        NeptuneLogger(run=run, project="redundant project")

    with pytest.raises(ValueError):
        NeptuneLogger(run=run, api_key="redundant api key")

    with pytest.raises(ValueError):
        NeptuneLogger(run=run, name="redundant api name")

    with pytest.raises(ValueError):
        NeptuneLogger(run=run, foo="random **kwarg")

    # this should work
    NeptuneLogger(run=run)
    NeptuneLogger(project="foo")
    NeptuneLogger(foo="bar")


def test_neptune_additional_methods(neptune_mock):
    logger, run_instance_mock, _ = _get_logger_with_mocks(api_key="test", project="project")

    logger.experiment["key1"].log(torch.ones(1))
    run_instance_mock.__getitem__.assert_called_once_with("key1")
    run_instance_mock.__getitem__().log.assert_called_once_with(torch.ones(1))


def test_neptune_leave_open_experiment_after_fit(neptune_mock, tmp_path, monkeypatch):
    """Verify that neptune experiment was NOT closed after training."""
    monkeypatch.chdir(tmp_path)
    logger, run_instance_mock, _ = _get_logger_with_mocks(api_key="test", project="project")
    _fit_and_test(logger=logger, model=BoringModel(), tmp_path=tmp_path)
    assert run_instance_mock.stop.call_count == 0


def test_neptune_log_metrics_on_trained_model(neptune_mock, tmp_path, monkeypatch):
    """Verify that trained models do log data."""
    monkeypatch.chdir(tmp_path)

    class LoggingModel(BoringModel):
        def on_validation_epoch_end(self):
            self.log("some/key", 42)

    logger, run_instance_mock, _ = _get_logger_with_mocks(api_key="test", project="project")
    _fit_and_test(logger=logger, model=LoggingModel(), tmp_path=tmp_path)
    run_instance_mock.__getitem__.assert_any_call("training/some/key")
    run_instance_mock.__getitem__.return_value.append.assert_has_calls([call(42, step=2)])


def test_log_hyperparams(neptune_mock):
    neptune_mock.utils.stringify_unsupported = lambda x: x

    params = {"foo": "bar", "nested_foo": {"bar": 42}}
    test_variants = [
        ({}, "training/hyperparams"),
        ({"prefix": "custom_prefix"}, "custom_prefix/hyperparams"),
        ({"prefix": "custom/nested/prefix"}, "custom/nested/prefix/hyperparams"),
    ]
    for prefix, hyperparams_key in test_variants:
        logger, run_instance_mock, _ = _get_logger_with_mocks(api_key="test", project="project", **prefix)
        logger.log_hyperparams(params)
        assert run_instance_mock.__setitem__.call_count == 1
        assert run_instance_mock.__getitem__.call_count == 0
        run_instance_mock.__setitem__.assert_called_once_with(hyperparams_key, params)


def test_log_metrics(neptune_mock):
    metrics = {
        "foo": 42,
        "bar": 555,
    }
    test_variants = [
        ({}, ("training/foo", "training/bar")),
        ({"prefix": "custom_prefix"}, ("custom_prefix/foo", "custom_prefix/bar")),
        ({"prefix": "custom/nested/prefix"}, ("custom/nested/prefix/foo", "custom/nested/prefix/bar")),
    ]

    for prefix, (metrics_foo_key, metrics_bar_key) in test_variants:
        logger, run_instance_mock, run_attr_mock = _get_logger_with_mocks(api_key="test", project="project", **prefix)
        logger.log_metrics(metrics)
        assert run_instance_mock.__setitem__.call_count == 0
        assert run_instance_mock.__getitem__.call_count == 2
        run_instance_mock.__getitem__.assert_any_call(metrics_foo_key)
        run_instance_mock.__getitem__.assert_any_call(metrics_bar_key)
        run_attr_mock.append.assert_has_calls([call(42, step=None), call(555, step=None)])


def test_log_model_summary(neptune_mock):
    model = BoringModel()
    test_variants = [
        ({}, "training/model/summary"),
        ({"prefix": "custom_prefix"}, "custom_prefix/model/summary"),
        ({"prefix": "custom/nested/prefix"}, "custom/nested/prefix/model/summary"),
    ]

    for prefix, model_summary_key in test_variants:
        logger, run_instance_mock, _ = _get_logger_with_mocks(api_key="test", project="project", **prefix)
        file_from_content_mock = neptune_mock.types.File.from_content()

        logger.log_model_summary(model)

        assert run_instance_mock.__setitem__.call_count == 1
        assert run_instance_mock.__getitem__.call_count == 0
        run_instance_mock.__setitem__.assert_called_once_with(model_summary_key, file_from_content_mock)


@mock.patch("builtins.open", mock.mock_open(read_data="test"))
def test_after_save_checkpoint(neptune_mock):
    test_variants = [
        ({}, "training/model"),
        ({"prefix": "custom_prefix"}, "custom_prefix/model"),
        ({"prefix": "custom/nested/prefix"}, "custom/nested/prefix/model"),
    ]

    for prefix, model_key_prefix in test_variants:
        logger, run_instance_mock, run_attr_mock = _get_logger_with_mocks(api_key="test", project="project", **prefix)
        models_root_dir = os.path.join("path", "to", "models")
        cb_mock = MagicMock(
            dirpath=models_root_dir,
            last_model_path=os.path.join(models_root_dir, "last"),
            best_k_models={
                f"{os.path.join(models_root_dir, 'model1')}": None,
                f"{os.path.join(models_root_dir, 'model2/with/slashes')}": None,
            },
            best_model_path=os.path.join(models_root_dir, "best_model"),
            best_model_score=None,
        )

        mock_file = neptune_mock.types.File
        mock_file.reset_mock()
        mock_file.side_effect = mock.Mock()
        logger.after_save_checkpoint(cb_mock)

        assert run_instance_mock.__setitem__.call_count == 1  # best_model_path
        assert run_instance_mock.__getitem__.call_count == 4  # last_model_path, best_k_models, best_model_path
        assert run_attr_mock.upload.call_count == 4  # last_model_path, best_k_models, best_model_path
        assert mock_file.from_stream.call_count == 0

        run_instance_mock.__getitem__.assert_any_call(f"{model_key_prefix}/checkpoints/model1")
        run_instance_mock.__getitem__.assert_any_call(f"{model_key_prefix}/checkpoints/model2/with/slashes")

        run_attr_mock.upload.assert_has_calls([
            call(os.path.join(models_root_dir, "model1")),
            call(os.path.join(models_root_dir, "model2/with/slashes")),
        ])


def test_save_dir(neptune_mock):
    logger = NeptuneLogger(api_key="test", project="project")
    assert logger.save_dir == os.path.join(os.getcwd(), ".neptune")


def test_get_full_model_name():
    SimpleCheckpoint = namedtuple("SimpleCheckpoint", ["dirpath"])
    test_input_data = [
        ("key", os.path.join("foo", "bar", "key.ext"), SimpleCheckpoint(dirpath=os.path.join("foo", "bar"))),
        (
            "key/in/parts",
            os.path.join("foo", "bar", "key/in/parts.ext"),
            SimpleCheckpoint(dirpath=os.path.join("foo", "bar")),
        ),
        ("key", os.path.join("../foo", "bar", "key.ext"), SimpleCheckpoint(dirpath=os.path.join("../foo", "bar"))),
        ("key", os.path.join("foo", "key.ext"), SimpleCheckpoint(dirpath=os.path.join("./foo", "bar/../"))),
    ]

    for expected_model_name, model_path, checkpoint in test_input_data:
        assert NeptuneLogger._get_full_model_name(model_path, checkpoint) == expected_model_name


def test_get_full_model_names_from_exp_structure():
    input_dict = {
        "foo": {
            "bar": {
                "lvl1_1": {"lvl2": {"lvl3_1": "some non important value", "lvl3_2": "some non important value"}},
                "lvl1_2": "some non important value",
            },
            "other_non_important": {"val100": 100},
        },
        "other_non_important": {"val42": 42},
    }
    expected_keys = {"lvl1_1/lvl2/lvl3_1", "lvl1_1/lvl2/lvl3_2", "lvl1_2"}
    assert NeptuneLogger._get_full_model_names_from_exp_structure(input_dict, "foo/bar") == expected_keys


def test_inactive_run(neptune_mock, tmp_path, monkeypatch):
    from neptune.exceptions import InactiveRunException

    monkeypatch.chdir(tmp_path)
    logger, run_instance_mock, _ = _get_logger_with_mocks(api_key="test", project="project")
    run_instance_mock.__setitem__.side_effect = InactiveRunException

    # this should work without any exceptions
    _fit_and_test(logger=logger, model=BoringModel(), tmp_path=tmp_path)
