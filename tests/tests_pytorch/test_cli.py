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
import glob
import inspect
import json
import operator
import os
import sys
from contextlib import ExitStack, contextmanager, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Callable, Optional, Union
from unittest import mock
from unittest.mock import ANY

import pytest
import torch
import yaml
from lightning_utilities import compare_version
from lightning_utilities.test.warning import no_warning_call
from packaging.version import Version
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.plugins.hparams.plugin_data_pb2 import HParamsPluginData
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from lightning.fabric.plugins.environments import SLURMEnvironment
from lightning.pytorch import Callback, LightningDataModule, LightningModule, Trainer, __version__, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.cli import (
    _JSONARGPARSE_SIGNATURES_AVAILABLE,
    LightningArgumentParser,
    LightningCLI,
    LRSchedulerCallable,
    LRSchedulerTypeTuple,
    OptimizerCallable,
    SaveConfigCallback,
    instantiate_class,
)
from lightning.pytorch.demos.boring_classes import BoringDataModule, BoringModel
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.imports import _TORCHVISION_AVAILABLE
from tests_pytorch.helpers.runif import RunIf

if _JSONARGPARSE_SIGNATURES_AVAILABLE:
    from jsonargparse import Namespace, lazy_instance
else:
    from argparse import Namespace

    def lazy_instance(*args, **kwargs):
        return None


_xfail_python_ge_3_11_9 = pytest.mark.xfail(
    # https://github.com/omni-us/jsonargparse/issues/484
    Version(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}") >= Version("3.11.9"),
    strict=False,
    reason="jsonargparse + Python 3.11.9 compatibility issue",
)


@contextmanager
def mock_subclasses(baseclass, *subclasses):
    """Mocks baseclass so that it only has the given child subclasses."""
    with ExitStack() as stack:
        mgr = mock.patch.object(baseclass, "__subclasses__", return_value=[*subclasses])
        stack.enter_context(mgr)
        for mgr in [mock.patch.object(s, "__subclasses__", return_value=[]) for s in subclasses]:
            stack.enter_context(mgr)
        yield None


@pytest.fixture
def cleandir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return


@pytest.fixture(autouse=True)
def ensure_cleandir():
    yield
    # make sure tests don't leave configuration files
    assert not glob.glob("*.yaml")


@pytest.mark.parametrize("cli_args", [["--callbacks=1", "--logger"], ["--foo", "--bar=1"]])
def test_add_argparse_args_redefined_error(cli_args, monkeypatch):
    """Asserts error raised in case of passing not default cli arguments."""

    class _UnkArgError(Exception):
        pass

    def _raise():
        raise _UnkArgError

    parser = LightningArgumentParser(add_help=False, parse_as_dict=False)
    parser.add_lightning_class_args(Trainer, None)

    monkeypatch.setattr(parser, "exit", lambda *args: _raise(), raising=True)

    with pytest.raises(_UnkArgError):
        parser.parse_args(cli_args)


class Model(LightningModule):
    def __init__(self, model_param: int):
        super().__init__()
        self.model_param = model_param


def _model_builder(model_param: int) -> Model:
    return Model(model_param)


def _trainer_builder(
    limit_train_batches: int, fast_dev_run: bool = False, callbacks: Optional[Union[list[Callback], Callback]] = None
) -> Trainer:
    return Trainer(limit_train_batches=limit_train_batches, fast_dev_run=fast_dev_run, callbacks=callbacks)


@pytest.mark.parametrize(("trainer_class", "model_class"), [(Trainer, Model), (_trainer_builder, _model_builder)])
def test_lightning_cli(trainer_class, model_class, monkeypatch):
    """Test that LightningCLI correctly instantiates model, trainer and calls fit."""
    expected_model = {"model_param": 7}
    expected_trainer = {"limit_train_batches": 100}

    def fit(trainer, model):
        for k, v in expected_model.items():
            assert getattr(model, k) == v
        for k, v in expected_trainer.items():
            assert getattr(trainer, k) == v
        save_callback = [x for x in trainer.callbacks if isinstance(x, SaveConfigCallback)]
        assert len(save_callback) == 1
        save_callback[0].on_train_start(trainer, model)

    def on_train_start(callback, trainer, _):
        config_dump = callback.parser.dump(callback.config, skip_none=False)
        for k, v in expected_model.items():
            assert f"  {k}: {v}" in config_dump
        for k, v in expected_trainer.items():
            assert f"  {k}: {v}" in config_dump
        trainer.ran_asserts = True

    monkeypatch.setattr(Trainer, "fit", fit)
    monkeypatch.setattr(SaveConfigCallback, "on_train_start", on_train_start)

    with mock.patch("sys.argv", ["any.py", "fit", "--model.model_param=7", "--trainer.limit_train_batches=100"]):
        cli = LightningCLI(model_class, trainer_class=trainer_class, save_config_callback=SaveConfigCallback)
        assert hasattr(cli.trainer, "ran_asserts")
        assert cli.trainer.ran_asserts


def test_lightning_cli_args_callbacks(cleandir):
    callbacks = [
        {
            "class_path": "lightning.pytorch.callbacks.LearningRateMonitor",
            "init_args": {"logging_interval": "epoch", "log_momentum": True},
        },
        {"class_path": "lightning.pytorch.callbacks.ModelCheckpoint", "init_args": {"monitor": "NAME"}},
    ]

    class TestModel(BoringModel):
        def on_fit_start(self):
            callback = [c for c in self.trainer.callbacks if isinstance(c, LearningRateMonitor)]
            assert len(callback) == 1
            assert callback[0].logging_interval == "epoch"
            assert callback[0].log_momentum is True
            callback = [c for c in self.trainer.callbacks if isinstance(c, ModelCheckpoint)]
            assert len(callback) == 1
            assert callback[0].monitor == "NAME"
            self.trainer.ran_asserts = True

    with mock.patch("sys.argv", ["any.py", "fit", f"--trainer.callbacks={json.dumps(callbacks)}"]):
        cli = LightningCLI(
            TestModel, trainer_defaults={"fast_dev_run": True, "logger": lazy_instance(CSVLogger, save_dir=".")}
        )

    assert cli.trainer.ran_asserts


def test_lightning_cli_single_arg_callback():
    with mock.patch("sys.argv", ["any.py", "--trainer.callbacks=DeviceStatsMonitor"]):
        cli = LightningCLI(BoringModel, run=False)

    assert cli.config.trainer.callbacks.class_path == "lightning.pytorch.callbacks.DeviceStatsMonitor"
    assert not isinstance(cli.config_init.trainer, list)


@pytest.mark.parametrize("run", [False, True])
def test_lightning_cli_configurable_callbacks(cleandir, run):
    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.add_lightning_class_args(LearningRateMonitor, "learning_rate_monitor")

        def fit(self, **_):
            pass

    cli_args = ["fit"] if run else []
    cli_args += ["--learning_rate_monitor.logging_interval=epoch"]

    with mock.patch("sys.argv", ["any.py"] + cli_args):
        cli = MyLightningCLI(BoringModel, run=run)

    callback = [c for c in cli.trainer.callbacks if isinstance(c, LearningRateMonitor)]
    assert len(callback) == 1
    assert callback[0].logging_interval == "epoch"


def test_lightning_cli_args_cluster_environments(cleandir):
    plugins = [{"class_path": "lightning.fabric.plugins.environments.SLURMEnvironment"}]

    class TestModel(BoringModel):
        def on_fit_start(self):
            # Ensure SLURMEnvironment is set, instead of default LightningEnvironment
            assert isinstance(self.trainer._accelerator_connector.cluster_environment, SLURMEnvironment)
            self.trainer.ran_asserts = True

    with mock.patch("sys.argv", ["any.py", "fit", f"--trainer.plugins={json.dumps(plugins)}"]):
        cli = LightningCLI(TestModel, trainer_defaults={"fast_dev_run": True})

    assert cli.trainer.ran_asserts


class DataDirDataModule(BoringDataModule):
    def __init__(self, data_dir):
        super().__init__()


def test_lightning_cli_args(cleandir):
    cli_args = [
        "fit",
        "--data.data_dir=.",
        "--trainer.max_epochs=1",
        "--trainer.limit_train_batches=1",
        "--trainer.limit_val_batches=0",
        "--trainer.enable_model_summary=False",
        "--trainer.logger=False",
        "--seed_everything=1234",
    ]

    with mock.patch("sys.argv", ["any.py"] + cli_args):
        cli = LightningCLI(BoringModel, DataDirDataModule)

    config_path = "config.yaml"
    assert os.path.isfile(config_path)
    with open(config_path) as f:
        loaded_config = yaml.safe_load(f.read())

    cli_config = cli.config["fit"].as_dict()
    assert cli_config["seed_everything"] == 1234
    assert "model" not in loaded_config
    assert "model" not in cli_config
    assert loaded_config["data"] == cli_config["data"]
    assert loaded_config["trainer"] == cli_config["trainer"]


@pytest.mark.skipif(compare_version("jsonargparse", operator.lt, "4.21.3"), reason="vulnerability with failing imports")
def test_lightning_env_parse(cleandir):
    out = StringIO()
    with mock.patch("sys.argv", ["", "fit", "--help"]), redirect_stdout(out), pytest.raises(SystemExit):
        LightningCLI(BoringModel, DataDirDataModule, parser_kwargs={"default_env": True})
    out = out.getvalue()
    assert "PL_FIT__CONFIG" in out
    assert "PL_FIT__SEED_EVERYTHING" in out
    assert "PL_FIT__TRAINER__LOGGER" in out
    assert "PL_FIT__DATA__DATA_DIR" in out
    assert "PL_FIT__CKPT_PATH" in out

    env_vars = {
        "PL_FIT__DATA__DATA_DIR": ".",
        "PL_FIT__TRAINER__DEFAULT_ROOT_DIR": ".",
        "PL_FIT__TRAINER__MAX_EPOCHS": "1",
        "PL_FIT__TRAINER__LOGGER": "False",
    }
    with mock.patch.dict(os.environ, env_vars), mock.patch("sys.argv", ["", "fit"]):
        cli = LightningCLI(BoringModel, DataDirDataModule, parser_kwargs={"default_env": True})
    assert cli.config.fit.data.data_dir == "."
    assert cli.config.fit.trainer.default_root_dir == "."
    assert cli.config.fit.trainer.max_epochs == 1
    assert cli.config.fit.trainer.logger is False


def test_lightning_cli_save_config_cases(cleandir):
    config_path = "config.yaml"
    cli_args = ["fit", "--trainer.logger=false", "--trainer.fast_dev_run=1"]

    # With fast_dev_run!=False config should not be saved
    with mock.patch("sys.argv", ["any.py"] + cli_args):
        LightningCLI(BoringModel)
    assert not os.path.isfile(config_path)

    # With fast_dev_run==False config should be saved
    cli_args[-1] = "--trainer.max_epochs=1"
    with mock.patch("sys.argv", ["any.py"] + cli_args):
        LightningCLI(BoringModel)
    assert os.path.isfile(config_path)

    # If run again on same directory exception should be raised since config file already exists
    with mock.patch("sys.argv", ["any.py"] + cli_args), pytest.raises(RuntimeError):
        LightningCLI(BoringModel)


def test_lightning_cli_save_config_only_once(cleandir):
    config_path = "config.yaml"
    cli_args = ["--trainer.logger=false", "--trainer.max_epochs=1"]

    with mock.patch("sys.argv", ["any.py"] + cli_args):
        cli = LightningCLI(BoringModel, run=False)

    save_config_callback = next(c for c in cli.trainer.callbacks if isinstance(c, SaveConfigCallback))
    assert not save_config_callback.overwrite
    assert not save_config_callback.already_saved
    cli.trainer.fit(cli.model)
    assert os.path.isfile(config_path)
    assert save_config_callback.already_saved
    cli.trainer.test(cli.model)  # Should not fail because config already saved


def test_lightning_cli_save_config_seed_everything(cleandir):
    config_path = Path("config.yaml")
    cli_args = ["fit", "--seed_everything=true", "--trainer.logger=false", "--trainer.max_epochs=1"]
    with mock.patch("sys.argv", ["any.py"] + cli_args):
        cli = LightningCLI(BoringModel)
    config = yaml.safe_load(config_path.read_text())
    assert isinstance(config["seed_everything"], int)
    assert config["seed_everything"] == cli.config.fit.seed_everything

    cli_args = ["--seed_everything=true", "--trainer.logger=false"]
    with mock.patch("sys.argv", ["any.py"] + cli_args):
        cli = LightningCLI(BoringModel, run=False)
    config = yaml.safe_load(config_path.read_text())
    assert isinstance(config["seed_everything"], int)
    assert config["seed_everything"] == cli.config.seed_everything


def test_save_to_log_dir_false_error():
    with pytest.raises(ValueError):
        SaveConfigCallback(
            LightningArgumentParser(),
            Namespace(),
            save_to_log_dir=False,
        )


@_xfail_python_ge_3_11_9
def test_lightning_cli_logger_save_config(cleandir):
    class LoggerSaveConfigCallback(SaveConfigCallback):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, save_to_log_dir=False, **kwargs)

        def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
            nonlocal config
            config = self.parser.dump(self.config)
            trainer.logger.log_hyperparams({"config": config})

    config = None
    cli_args = [
        "fit",
        "--trainer.max_epochs=1",
        "--trainer.logger=TensorBoardLogger",
        f"--trainer.logger.save_dir={os.getcwd()}",
    ]

    with mock.patch("sys.argv", ["any.py"] + cli_args):
        cli = LightningCLI(
            BoringModel,
            save_config_callback=LoggerSaveConfigCallback,
        )

    assert os.path.isdir(cli.trainer.log_dir)
    assert not os.path.isfile(os.path.join(cli.trainer.log_dir, "config.yaml"))

    events_file = glob.glob(os.path.join(cli.trainer.log_dir, "events.out.tfevents.*"))
    assert len(events_file) == 1
    ea = event_accumulator.EventAccumulator(events_file[0])
    ea.Reload()
    data = ea._plugin_to_tag_to_content["hparams"]["_hparams_/session_start_info"]
    hparam_data = HParamsPluginData.FromString(data).session_start_info.hparams
    assert hparam_data.get("config") is not None
    assert hparam_data["config"].string_value == config


def test_lightning_cli_config_and_subclass_mode(cleandir):
    input_config = {
        "fit": {
            "model": {"class_path": "lightning.pytorch.demos.boring_classes.BoringModel"},
            "data": {
                "class_path": "DataDirDataModule",
                "init_args": {"data_dir": "."},
            },
            "trainer": {"max_epochs": 1, "enable_model_summary": False, "logger": False},
        }
    }
    config_path = "config.yaml"
    with open(config_path, "w") as f:
        f.write(yaml.dump(input_config))

    with (
        mock.patch("sys.argv", ["any.py", "--config", config_path]),
        mock_subclasses(LightningDataModule, DataDirDataModule),
    ):
        cli = LightningCLI(
            BoringModel,
            BoringDataModule,
            subclass_mode_model=True,
            subclass_mode_data=True,
            save_config_kwargs={"overwrite": True},
        )

    config_path = "config.yaml"
    assert os.path.isfile(config_path)
    with open(config_path) as f:
        loaded_config = yaml.safe_load(f.read())

    cli_config = cli.config["fit"].as_dict()
    assert loaded_config["model"] == cli_config["model"]
    assert loaded_config["data"] == cli_config["data"]
    assert loaded_config["trainer"] == cli_config["trainer"]


def any_model_any_data_cli():
    LightningCLI(LightningModule, LightningDataModule, subclass_mode_model=True, subclass_mode_data=True)


@pytest.mark.skipif(compare_version("jsonargparse", operator.lt, "4.21.3"), reason="vulnerability with failing imports")
@pytest.mark.skipif(
    (sys.version_info.major, sys.version_info.minor) == (3, 9)
    and compare_version("jsonargparse", operator.lt, "4.24.0"),
    reason="--trainer.precision is not parsed",
)
def test_lightning_cli_help():
    cli_args = ["any.py", "fit", "--help"]
    out = StringIO()
    with mock.patch("sys.argv", cli_args), redirect_stdout(out), pytest.raises(SystemExit):
        any_model_any_data_cli()
    out = out.getvalue()

    assert "--print_config" in out
    assert "--config" in out
    assert "--seed_everything" in out
    assert "--model.help" in out
    assert "--data.help" in out

    skip_params = {"self"}
    for param in inspect.signature(Trainer.__init__).parameters:
        if param not in skip_params:
            assert f"--trainer.{param}" in out

    cli_args = ["any.py", "fit", "--data.help=DataDirDataModule"]
    out = StringIO()
    with (
        mock.patch("sys.argv", cli_args),
        redirect_stdout(out),
        mock_subclasses(LightningDataModule, DataDirDataModule),
        pytest.raises(SystemExit),
    ):
        any_model_any_data_cli()

    assert ("--data.data_dir" in out.getvalue()) or ("--data.init_args.data_dir" in out.getvalue())


def test_lightning_cli_print_config():
    cli_args = [
        "any.py",
        "predict",
        "--seed_everything=1234",
        "--model=lightning.pytorch.demos.BoringModel",
        "--data=lightning.pytorch.demos.BoringDataModule",
        "--print_config",
    ]
    out = StringIO()
    with mock.patch("sys.argv", cli_args), redirect_stdout(out), pytest.raises(SystemExit):
        any_model_any_data_cli()

    text = out.getvalue()
    # test dump_header
    assert text.startswith(f"# lightning.pytorch=={__version__}")

    outval = yaml.safe_load(text)
    assert outval["seed_everything"] == 1234
    assert outval["model"]["class_path"] == "lightning.pytorch.demos.BoringModel"
    assert outval["data"]["class_path"] == "lightning.pytorch.demos.BoringDataModule"
    assert outval["ckpt_path"] is None


def test_lightning_cli_submodules(cleandir):
    class MainModule(BoringModel):
        def __init__(self, submodule1: LightningModule, submodule2: LightningModule, main_param: int = 1):
            super().__init__()
            self.submodule1 = submodule1
            self.submodule2 = submodule2

    config = """model:
        main_param: 2
        submodule1:
            class_path: lightning.pytorch.demos.boring_classes.BoringModel
        submodule2:
            class_path: lightning.pytorch.demos.boring_classes.BoringModel
    """
    config_path = Path("config.yaml")
    config_path.write_text(config)

    cli_args = [f"--config={config_path}"]
    with mock.patch("sys.argv", ["any.py"] + cli_args):
        cli = LightningCLI(MainModule, run=False)

    assert cli.config["model"]["main_param"] == 2
    assert isinstance(cli.model.submodule1, BoringModel)
    assert isinstance(cli.model.submodule2, BoringModel)


@pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason=str(_TORCHVISION_AVAILABLE))
def test_lightning_cli_torch_modules(cleandir):
    class TestModule(BoringModel):
        def __init__(self, activation: torch.nn.Module = None, transform: Optional[list[torch.nn.Module]] = None):
            super().__init__()
            self.activation = activation
            self.transform = transform

    config = """model:
        activation:
          class_path: torch.nn.LeakyReLU
          init_args:
            negative_slope: 0.2
        transform:
          - class_path: torchvision.transforms.Resize
            init_args:
              size: 64
          - class_path: torchvision.transforms.CenterCrop
            init_args:
              size: 64
    """
    config_path = Path("config.yaml")
    config_path.write_text(config)

    cli_args = [f"--config={config_path}"]
    with mock.patch("sys.argv", ["any.py"] + cli_args):
        cli = LightningCLI(TestModule, run=False)

    assert isinstance(cli.model.activation, torch.nn.LeakyReLU)
    assert cli.model.activation.negative_slope == 0.2
    assert len(cli.model.transform) == 2
    assert all(isinstance(v, torch.nn.Module) for v in cli.model.transform)


class BoringModelRequiredClasses(BoringModel):
    def __init__(self, num_classes: int, batch_size: int = 8):
        super().__init__()
        self.num_classes = num_classes
        self.batch_size = batch_size


class BoringDataModuleBatchSizeAndClasses(BoringDataModule):
    def __init__(self, batch_size: int = 8):
        super().__init__()
        self.batch_size = batch_size
        self.num_classes = 5  # only available after instantiation


def test_lightning_cli_link_arguments():
    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.link_arguments("data.batch_size", "model.batch_size")
            parser.link_arguments("data.num_classes", "model.num_classes", apply_on="instantiate")

    cli_args = ["--data.batch_size=12"]

    with mock.patch("sys.argv", ["any.py"] + cli_args):
        cli = MyLightningCLI(BoringModelRequiredClasses, BoringDataModuleBatchSizeAndClasses, run=False)

    assert cli.model.batch_size == 12
    assert cli.model.num_classes == 5

    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.link_arguments("data.batch_size", "model.init_args.batch_size")
            parser.link_arguments("data.num_classes", "model.init_args.num_classes", apply_on="instantiate")

    cli_args[-1] = "--model=tests_pytorch.test_cli.BoringModelRequiredClasses"

    with mock.patch("sys.argv", ["any.py"] + cli_args):
        cli = MyLightningCLI(
            BoringModelRequiredClasses, BoringDataModuleBatchSizeAndClasses, subclass_mode_model=True, run=False
        )

    assert cli.model.batch_size == 8
    assert cli.model.num_classes == 5


class EarlyExitTestModel(BoringModel):
    def on_fit_start(self):
        raise MisconfigurationException("Error on fit start")


# mps not yet supported by distributed
@RunIf(skip_windows=True, mps=False)
@pytest.mark.parametrize("logger", [False, lazy_instance(TensorBoardLogger, save_dir=".")])
@pytest.mark.parametrize("strategy", ["ddp_spawn", "ddp"])
def test_cli_distributed_save_config_callback(cleandir, logger, strategy):
    from torch.multiprocessing import ProcessRaisedException

    with (
        mock.patch("sys.argv", ["any.py", "fit"]),
        pytest.raises((MisconfigurationException, ProcessRaisedException), match=r"Error on fit start"),
    ):
        LightningCLI(
            EarlyExitTestModel,
            trainer_defaults={
                "logger": logger,
                "max_steps": 1,
                "max_epochs": 1,
                "strategy": strategy,
                "accelerator": "auto",
                "devices": 1,
            },
        )
    if logger:
        config_dir = Path("lightning_logs")
        # no more version dirs should get created
        assert os.listdir(config_dir) == ["version_0"]
        config_path = config_dir / "version_0" / "config.yaml"
    else:
        config_path = "config.yaml"
    assert os.path.isfile(config_path)


def test_cli_config_overwrite(cleandir):
    trainer_defaults = {"max_steps": 1, "max_epochs": 1, "logger": False}

    argv = ["any.py", "fit"]
    with mock.patch("sys.argv", argv):
        LightningCLI(BoringModel, trainer_defaults=trainer_defaults)
    with mock.patch("sys.argv", argv), pytest.raises(RuntimeError, match="Aborting to avoid overwriting"):
        LightningCLI(BoringModel, trainer_defaults=trainer_defaults)
    with mock.patch("sys.argv", argv):
        LightningCLI(BoringModel, save_config_kwargs={"overwrite": True}, trainer_defaults=trainer_defaults)


def test_cli_config_filename(tmp_path):
    with mock.patch("sys.argv", ["any.py", "fit"]):
        LightningCLI(
            BoringModel,
            trainer_defaults={"default_root_dir": str(tmp_path), "logger": False, "max_steps": 1, "max_epochs": 1},
            save_config_kwargs={"config_filename": "name.yaml"},
        )
    assert os.path.isfile(tmp_path / "name.yaml")


@pytest.mark.parametrize("run", [False, True])
def test_lightning_cli_optimizer(run):
    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.add_optimizer_args(torch.optim.Adam)

    match = "BoringModel.configure_optimizers` will be overridden by `MyLightningCLI.configure_optimizers`"
    argv = ["fit", "--trainer.fast_dev_run=1"] if run else []
    with mock.patch("sys.argv", ["any.py"] + argv), pytest.warns(UserWarning, match=match):
        cli = MyLightningCLI(BoringModel, run=run)

    assert cli.model.configure_optimizers is not BoringModel.configure_optimizers

    if not run:
        optimizer = cli.model.configure_optimizers()
        assert isinstance(optimizer, torch.optim.Adam)
    else:
        assert len(cli.trainer.optimizers) == 1
        assert isinstance(cli.trainer.optimizers[0], torch.optim.Adam)
        assert len(cli.trainer.lr_scheduler_configs) == 0


def test_lightning_cli_optimizer_and_lr_scheduler():
    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.add_optimizer_args(torch.optim.Adam)
            parser.add_lr_scheduler_args(torch.optim.lr_scheduler.ExponentialLR)

    cli_args = ["fit", "--trainer.fast_dev_run=1", "--lr_scheduler.gamma=0.8"]

    with mock.patch("sys.argv", ["any.py"] + cli_args):
        cli = MyLightningCLI(BoringModel)

    assert cli.model.configure_optimizers is not BoringModel.configure_optimizers
    assert len(cli.trainer.optimizers) == 1
    assert isinstance(cli.trainer.optimizers[0], torch.optim.Adam)
    assert len(cli.trainer.lr_scheduler_configs) == 1
    assert isinstance(cli.trainer.lr_scheduler_configs[0].scheduler, torch.optim.lr_scheduler.ExponentialLR)
    assert cli.trainer.lr_scheduler_configs[0].scheduler.gamma == 0.8


def test_cli_no_need_configure_optimizers(cleandir):
    class BoringModel(LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(32, 2)

        def training_step(self, *_): ...

        def train_dataloader(self): ...

        # did not define `configure_optimizers`

    from lightning.pytorch.trainer.configuration_validator import __verify_train_val_loop_configuration

    with (
        mock.patch("sys.argv", ["any.py", "fit", "--optimizer=Adam"]),
        mock.patch("lightning.pytorch.Trainer._run_stage") as run,
        mock.patch(
            "lightning.pytorch.trainer.configuration_validator.__verify_train_val_loop_configuration",
            wraps=__verify_train_val_loop_configuration,
        ) as verify,
    ):
        cli = LightningCLI(BoringModel)
    run.assert_called_once()
    verify.assert_called_once_with(cli.trainer, cli.model)


def test_lightning_cli_optimizer_and_lr_scheduler_subclasses(cleandir):
    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.add_optimizer_args((torch.optim.SGD, torch.optim.Adam))
            parser.add_lr_scheduler_args((torch.optim.lr_scheduler.StepLR, torch.optim.lr_scheduler.ExponentialLR))

    optimizer_arg = {"class_path": "torch.optim.Adam", "init_args": {"lr": 0.01}}
    lr_scheduler_arg = {"class_path": "torch.optim.lr_scheduler.StepLR", "init_args": {"step_size": 50}}
    cli_args = [
        "fit",
        "--trainer.max_epochs=1",
        f"--optimizer={json.dumps(optimizer_arg)}",
        f"--lr_scheduler={json.dumps(lr_scheduler_arg)}",
    ]

    with mock.patch("sys.argv", ["any.py"] + cli_args):
        cli = MyLightningCLI(BoringModel)

    assert len(cli.trainer.optimizers) == 1
    assert isinstance(cli.trainer.optimizers[0], torch.optim.Adam)
    assert len(cli.trainer.lr_scheduler_configs) == 1
    assert isinstance(cli.trainer.lr_scheduler_configs[0].scheduler, torch.optim.lr_scheduler.StepLR)
    assert cli.trainer.lr_scheduler_configs[0].scheduler.step_size == 50


@_xfail_python_ge_3_11_9
@RunIf(min_torch="2.2")
@pytest.mark.parametrize("use_generic_base_class", [False, True])
def test_lightning_cli_optimizers_and_lr_scheduler_with_link_to(use_generic_base_class):
    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.add_optimizer_args(
                (torch.optim.Optimizer,) if use_generic_base_class else torch.optim.Adam,
                nested_key="optim1",
                link_to="model.optim1",
            )
            parser.add_optimizer_args((torch.optim.ASGD, torch.optim.SGD), nested_key="optim2", link_to="model.optim2")
            parser.add_lr_scheduler_args(
                LRSchedulerTypeTuple if use_generic_base_class else torch.optim.lr_scheduler.ExponentialLR,
                link_to="model.scheduler",
            )

    class TestModel(BoringModel):
        def __init__(self, optim1: dict, optim2: dict, scheduler: dict):
            super().__init__()
            self.optim1 = instantiate_class(self.parameters(), optim1)
            self.optim2 = instantiate_class(self.parameters(), optim2)
            self.scheduler = instantiate_class(self.optim1, scheduler)

    cli_args = ["fit", "--trainer.fast_dev_run=1"]
    if use_generic_base_class:
        cli_args += [
            "--optim1",
            "Adam",
            "--optim1.weight_decay",
            "0.001",
            "--optim2=SGD",
            "--optim2.lr=0.01",
            "--lr_scheduler=ExponentialLR",
        ]
    else:
        cli_args += ["--optim2=SGD", "--optim2.lr=0.01"]
    cli_args += ["--lr_scheduler.gamma=0.2"]

    with mock.patch("sys.argv", ["any.py"] + cli_args):
        cli = MyLightningCLI(TestModel)

    assert isinstance(cli.model.optim1, torch.optim.Adam)
    assert isinstance(cli.model.optim2, torch.optim.SGD)
    assert cli.model.optim2.param_groups[0]["lr"] == 0.01
    assert isinstance(cli.model.scheduler, torch.optim.lr_scheduler.ExponentialLR)


@_xfail_python_ge_3_11_9
@RunIf(min_torch="2.2")
def test_lightning_cli_optimizers_and_lr_scheduler_with_callable_type():
    class TestModel(BoringModel):
        def __init__(
            self,
            optim1: OptimizerCallable = torch.optim.Adam,
            optim2: OptimizerCallable = torch.optim.Adagrad,
            scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
        ):
            super().__init__()
            self.optim1 = optim1
            self.optim2 = optim2
            self.scheduler = scheduler

        def configure_optimizers(self):
            optim1 = self.optim1(self.parameters())
            optim2 = self.optim2(self.parameters())
            scheduler = self.scheduler(optim2)
            return (
                {"optimizer": optim1},
                {"optimizer": optim2, "lr_scheduler": scheduler},
            )

    out = StringIO()
    with mock.patch("sys.argv", ["any.py", "-h"]), redirect_stdout(out), pytest.raises(SystemExit):
        LightningCLI(TestModel, run=False, auto_configure_optimizers=False)
    out = out.getvalue()
    assert "--optimizer" not in out
    assert "--lr_scheduler" not in out
    assert "--model.optim1" in out
    assert "--model.optim2" in out
    assert "--model.scheduler" in out

    cli_args = [
        "--model.optim1=Adagrad",
        "--model.optim2=SGD",
        "--model.optim2.lr=0.007",
        "--model.scheduler=ExponentialLR",
        "--model.scheduler.gamma=0.3",
    ]
    with mock.patch("sys.argv", ["any.py"] + cli_args):
        cli = LightningCLI(TestModel, run=False, auto_configure_optimizers=False)

    init = cli.model.configure_optimizers()
    assert isinstance(init[0]["optimizer"], torch.optim.Adagrad)
    assert isinstance(init[1]["optimizer"], torch.optim.SGD)
    assert isinstance(init[1]["lr_scheduler"], torch.optim.lr_scheduler.ExponentialLR)
    assert init[1]["optimizer"].param_groups[0]["lr"] == 0.007
    assert init[1]["lr_scheduler"].gamma == 0.3


class TestModelSaveHparams(BoringModel):
    def __init__(
        self,
        optimizer: OptimizerCallable = torch.optim.Adam,
        scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
        activation: torch.nn.Module = lazy_instance(torch.nn.LeakyReLU, negative_slope=0.05),
    ):
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.activation = activation

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        scheduler = self.scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def test_lightning_cli_load_from_checkpoint_dependency_injection(cleandir):
    with mock.patch("sys.argv", ["any.py", "--trainer.max_epochs=1"]):
        cli = LightningCLI(TestModelSaveHparams, run=False, auto_configure_optimizers=False)
    cli.trainer.fit(cli.model)

    hparams_path = Path(cli.trainer.log_dir) / "hparams.yaml"
    assert hparams_path.is_file()
    hparams = yaml.safe_load(hparams_path.read_text())

    expected_keys = ["_instantiator", "activation", "optimizer", "scheduler"]
    expected_instantiator = "lightning.pytorch.cli.instantiate_module"
    expected_activation = "torch.nn.LeakyReLU"
    expected_optimizer = "torch.optim.Adam"
    expected_scheduler = "torch.optim.lr_scheduler.ConstantLR"

    assert sorted(hparams.keys()) == expected_keys
    assert hparams["_instantiator"] == expected_instantiator
    assert hparams["activation"]["class_path"] == expected_activation
    assert hparams["optimizer"] == expected_optimizer or hparams["optimizer"]["class_path"] == expected_optimizer
    assert hparams["scheduler"] == expected_scheduler or hparams["scheduler"]["class_path"] == expected_scheduler

    checkpoint_path = next(Path(cli.trainer.log_dir, "checkpoints").glob("*.ckpt"), None)
    assert checkpoint_path.is_file()
    hparams = torch.load(checkpoint_path, weights_only=True)["hyper_parameters"]
    assert sorted(hparams.keys()) == expected_keys
    assert hparams["_instantiator"] == expected_instantiator
    assert hparams["activation"]["class_path"] == expected_activation
    assert hparams["optimizer"] == expected_optimizer or hparams["optimizer"]["class_path"] == expected_optimizer
    assert hparams["scheduler"] == expected_scheduler or hparams["scheduler"]["class_path"] == expected_scheduler

    model = TestModelSaveHparams.load_from_checkpoint(checkpoint_path)
    assert isinstance(model, TestModelSaveHparams)
    assert isinstance(model.activation, torch.nn.LeakyReLU)
    assert model.activation.negative_slope == 0.05
    optimizer, lr_scheduler = model.configure_optimizers().values()
    assert isinstance(optimizer, torch.optim.Adam)
    assert isinstance(lr_scheduler, torch.optim.lr_scheduler.ConstantLR)


def test_lightning_cli_load_from_checkpoint_dependency_injection_subclass_mode(cleandir):
    with mock.patch("sys.argv", ["any.py", "--trainer.max_epochs=1", "--model=TestModelSaveHparams"]):
        cli = LightningCLI(TestModelSaveHparams, run=False, auto_configure_optimizers=False, subclass_mode_model=True)
    cli.trainer.fit(cli.model)

    expected_keys = ["_class_path", "_instantiator", "activation", "optimizer", "scheduler"]
    expected_instantiator = "lightning.pytorch.cli.instantiate_module"
    expected_class_path = f"{__name__}.TestModelSaveHparams"
    expected_activation = "torch.nn.LeakyReLU"
    expected_optimizer = "torch.optim.Adam"
    expected_scheduler = "torch.optim.lr_scheduler.ConstantLR"

    checkpoint_path = next(Path(cli.trainer.log_dir, "checkpoints").glob("*.ckpt"), None)
    assert checkpoint_path.is_file()
    hparams = torch.load(checkpoint_path, weights_only=True)["hyper_parameters"]

    assert sorted(hparams.keys()) == expected_keys
    assert hparams["_instantiator"] == expected_instantiator
    assert hparams["_class_path"] == expected_class_path
    assert hparams["activation"]["class_path"] == expected_activation
    assert hparams["optimizer"] == expected_optimizer or hparams["optimizer"]["class_path"] == expected_optimizer
    assert hparams["scheduler"] == expected_scheduler or hparams["scheduler"]["class_path"] == expected_scheduler

    model = LightningModule.load_from_checkpoint(checkpoint_path)
    assert isinstance(model, TestModelSaveHparams)
    assert isinstance(model.activation, torch.nn.LeakyReLU)
    assert model.activation.negative_slope == 0.05
    optimizer, lr_scheduler = model.configure_optimizers().values()
    assert isinstance(optimizer, torch.optim.Adam)
    assert isinstance(lr_scheduler, torch.optim.lr_scheduler.ConstantLR)


class TestModelSaveHparamsUntyped(BoringModel):
    def __init__(self, learning_rate, step_size=None, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.kwargs = kwargs


def test_lightning_cli_save_hyperparameters_untyped_module(cleandir):
    config = {
        "model": {
            "class_path": f"{__name__}.TestModelSaveHparamsUntyped",
            "init_args": {"learning_rate": 1e-2},
            "dict_kwargs": {"x": 1},
        }
    }
    with mock.patch("sys.argv", ["any.py", f"--config={json.dumps(config)}", "--trainer.max_epochs=1"]):
        cli = LightningCLI(BoringModel, run=False, auto_configure_optimizers=False, subclass_mode_model=True)
    cli.trainer.fit(cli.model)
    assert isinstance(cli.model, TestModelSaveHparamsUntyped)
    assert cli.model.hparams["learning_rate"] == 1e-2
    assert cli.model.hparams["step_size"] is None
    assert cli.model.hparams["x"] == 1

    checkpoint_path = next(Path(cli.trainer.log_dir, "checkpoints").glob("*.ckpt"), None)
    model = TestModelSaveHparamsUntyped.load_from_checkpoint(checkpoint_path)
    assert model.learning_rate == 1e-2
    assert model.step_size is None
    assert model.kwargs == {"x": 1}


class TestDataSaveHparams(BoringDataModule):
    def __init__(self, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.num_workers = num_workers


def test_lightning_cli_save_hyperparameters_merge(cleandir):
    config = {
        "model": {
            "class_path": f"{__name__}.TestModelSaveHparams",
        },
        "data": {
            "class_path": f"{__name__}.TestDataSaveHparams",
        },
    }
    with mock.patch("sys.argv", ["any.py", "fit", f"--config={json.dumps(config)}", "--trainer.max_epochs=1"]):
        cli = LightningCLI(auto_configure_optimizers=False)
    assert set(cli.model.hparams) == {"optimizer", "scheduler", "activation", "_instantiator", "_class_path"}
    assert set(cli.datamodule.hparams) == {"batch_size", "num_workers", "_instantiator", "_class_path"}


@pytest.mark.parametrize("fn", [fn.value for fn in TrainerFn])
def test_lightning_cli_trainer_fn(fn):
    class TestCLI(LightningCLI):
        def __init__(self, *args, **kwargs):
            self.called = []
            super().__init__(*args, **kwargs)

        def before_fit(self):
            self.called.append("before_fit")

        def fit(self, **_):
            self.called.append("fit")

        def after_fit(self):
            self.called.append("after_fit")

        def before_validate(self):
            self.called.append("before_validate")

        def validate(self, **_):
            self.called.append("validate")

        def after_validate(self):
            self.called.append("after_validate")

        def before_test(self):
            self.called.append("before_test")

        def test(self, **_):
            self.called.append("test")

        def after_test(self):
            self.called.append("after_test")

        def before_predict(self):
            self.called.append("before_predict")

        def predict(self, **_):
            self.called.append("predict")

        def after_predict(self):
            self.called.append("after_predict")

    with mock.patch("sys.argv", ["any.py", fn]):
        cli = TestCLI(BoringModel)
    assert cli.called == [f"before_{fn}", fn, f"after_{fn}"]


def test_lightning_cli_subcommands():
    subcommands = LightningCLI.subcommands()
    trainer = Trainer()
    for subcommand, exclude in subcommands.items():
        fn = getattr(trainer, subcommand)
        parameters = list(inspect.signature(fn).parameters)
        for e in exclude:
            # if this fails, it's because the parameter has been removed from the associated `Trainer` function
            # and the `LightningCLI` subcommand exclusion list needs to be updated
            assert e in parameters


@pytest.mark.skipif(compare_version("jsonargparse", operator.lt, "4.21.3"), reason="vulnerability with failing imports")
def test_lightning_cli_custom_subcommand():
    class TestTrainer(Trainer):
        def foo(self, model: LightningModule, x: int, y: float = 1.0):
            """Sample extra function.

            Args:
                model: A model
                x: The x
                y: The y

            """

    class TestCLI(LightningCLI):
        @staticmethod
        def subcommands():
            subcommands = LightningCLI.subcommands()
            subcommands["foo"] = {"model"}
            return subcommands

    out = StringIO()
    with mock.patch("sys.argv", ["any.py", "-h"]), redirect_stdout(out), pytest.raises(SystemExit):
        TestCLI(BoringModel, trainer_class=TestTrainer)
    out = out.getvalue()
    assert "Sample extra function." in out
    assert "{fit,validate,test,predict,foo}" in out

    out = StringIO()
    with mock.patch("sys.argv", ["any.py", "foo", "-h"]), redirect_stdout(out), pytest.raises(SystemExit):
        TestCLI(BoringModel, trainer_class=TestTrainer)
    out = out.getvalue()
    assert "A model" not in out
    assert "Sample extra function:" in out
    assert "--x X" in out
    assert "The x (required, type: int)" in out
    assert "--y Y" in out
    assert "The y (type: float, default: 1.0)" in out


def test_lightning_cli_run(cleandir):
    with mock.patch("sys.argv", ["any.py"]):
        cli = LightningCLI(BoringModel, run=False)
    assert cli.trainer.global_step == 0
    assert isinstance(cli.trainer, Trainer)
    assert isinstance(cli.model, LightningModule)

    with mock.patch("sys.argv", ["any.py", "fit"]):
        cli = LightningCLI(BoringModel, trainer_defaults={"max_steps": 1, "max_epochs": 1})
    assert cli.trainer.global_step == 1
    assert isinstance(cli.trainer, Trainer)
    assert isinstance(cli.model, LightningModule)


class TestModel(BoringModel):
    def __init__(self, foo, bar=5):
        super().__init__()
        self.foo = foo
        self.bar = bar


@_xfail_python_ge_3_11_9
def test_lightning_cli_model_short_arguments():
    with (
        mock.patch("sys.argv", ["any.py", "fit", "--model=BoringModel"]),
        mock.patch("lightning.pytorch.Trainer._fit_impl") as run,
        mock_subclasses(LightningModule, BoringModel, TestModel),
    ):
        cli = LightningCLI(trainer_defaults={"fast_dev_run": 1})
        assert isinstance(cli.model, BoringModel)
        run.assert_called_once_with(cli.model, ANY, ANY, ANY, ANY)

    with (
        mock.patch("sys.argv", ["any.py", "--model=TestModel", "--model.foo", "123"]),
        mock_subclasses(LightningModule, BoringModel, TestModel),
    ):
        cli = LightningCLI(run=False)
        assert isinstance(cli.model, TestModel)
        assert cli.model.foo == 123
        assert cli.model.bar == 5


class MyDataModule(BoringDataModule):
    def __init__(self, foo, bar=5):
        super().__init__()
        self.foo = foo
        self.bar = bar


@_xfail_python_ge_3_11_9
def test_lightning_cli_datamodule_short_arguments():
    # with set model
    with (
        mock.patch("sys.argv", ["any.py", "fit", "--data=BoringDataModule"]),
        mock.patch("lightning.pytorch.Trainer._fit_impl") as run,
        mock_subclasses(LightningDataModule, BoringDataModule),
    ):
        cli = LightningCLI(BoringModel, trainer_defaults={"fast_dev_run": 1})
        assert isinstance(cli.datamodule, BoringDataModule)
        run.assert_called_once_with(ANY, ANY, ANY, cli.datamodule, ANY)

    with (
        mock.patch("sys.argv", ["any.py", "--data=MyDataModule", "--data.foo", "123"]),
        mock_subclasses(LightningDataModule, MyDataModule),
    ):
        cli = LightningCLI(BoringModel, run=False)
        assert isinstance(cli.datamodule, MyDataModule)
        assert cli.datamodule.foo == 123
        assert cli.datamodule.bar == 5

    # with configurable model
    with (
        mock.patch("sys.argv", ["any.py", "fit", "--model", "BoringModel", "--data=BoringDataModule"]),
        mock.patch("lightning.pytorch.Trainer._fit_impl") as run,
        mock_subclasses(LightningModule, BoringModel),
        mock_subclasses(LightningDataModule, BoringDataModule),
    ):
        cli = LightningCLI(trainer_defaults={"fast_dev_run": 1})
        assert isinstance(cli.model, BoringModel)
        assert isinstance(cli.datamodule, BoringDataModule)
        run.assert_called_once_with(cli.model, ANY, ANY, cli.datamodule, ANY)

    with (
        mock.patch("sys.argv", ["any.py", "--model", "BoringModel", "--data=MyDataModule"]),
        mock_subclasses(LightningModule, BoringModel),
        mock_subclasses(LightningDataModule, MyDataModule),
    ):
        cli = LightningCLI(run=False)
        assert isinstance(cli.model, BoringModel)
        assert isinstance(cli.datamodule, MyDataModule)

    with mock.patch("sys.argv", ["any.py"]):
        cli = LightningCLI(BoringModel, run=False)
        # data was not passed but we are adding it automatically because there are datamodules registered
        assert "data" in cli.parser.groups
        assert not hasattr(cli.parser.groups["data"], "group_class")

    with mock.patch("sys.argv", ["any.py"]):
        cli = LightningCLI(BoringModel, BoringDataModule, run=False)
        # since we are passing the DataModule, that's whats added to the parser
        assert cli.parser.groups["data"].group_class is BoringDataModule


@_xfail_python_ge_3_11_9
@pytest.mark.parametrize("use_class_path_callbacks", [False, True])
def test_callbacks_append(use_class_path_callbacks):
    """This test validates registries are used when simplified command line are being used."""
    cli_args = [
        "--optimizer",
        "Adam",
        "--optimizer.lr",
        "0.0001",
        "--trainer.callbacks+=LearningRateMonitor",
        "--trainer.callbacks.logging_interval=epoch",
        "--trainer.callbacks.log_momentum=True",
        "--model=BoringModel",
        "--trainer.callbacks+",
        "ModelCheckpoint",
        "--trainer.callbacks.monitor=loss",
        "--lr_scheduler",
        "StepLR",
        "--lr_scheduler.step_size=50",
    ]

    extras = []
    if use_class_path_callbacks:
        callbacks = [
            {"class_path": "lightning.pytorch.callbacks.Callback"},
            {"class_path": "lightning.pytorch.callbacks.Callback", "init_args": {}},
        ]
        cli_args += [f"--trainer.callbacks+={json.dumps(callbacks)}"]
        extras = [Callback, Callback]

    with mock.patch("sys.argv", ["any.py"] + cli_args), mock_subclasses(LightningModule, BoringModel):
        cli = LightningCLI(run=False)

    assert isinstance(cli.model, BoringModel)
    optimizers, lr_scheduler = cli.model.configure_optimizers()
    assert isinstance(optimizers[0], torch.optim.Adam)
    assert optimizers[0].param_groups[0]["lr"] == 0.0001
    assert lr_scheduler[0].step_size == 50

    callback_types = [type(c) for c in cli.trainer.callbacks]
    expected = [LearningRateMonitor, SaveConfigCallback, ModelCheckpoint] + extras
    assert all(t in callback_types for t in expected)


@_xfail_python_ge_3_11_9
def test_optimizers_and_lr_schedulers_reload(cleandir):
    base = ["any.py", "--trainer.max_epochs=1"]
    input = base + [
        "--lr_scheduler",
        "OneCycleLR",
        "--lr_scheduler.total_steps=10",
        "--lr_scheduler.max_lr=1",
        "--optimizer",
        "Adam",
        "--optimizer.lr=0.1",
    ]

    # save config
    out = StringIO()
    with mock.patch("sys.argv", input + ["--print_config"]), redirect_stdout(out), pytest.raises(SystemExit):
        LightningCLI(BoringModel, run=False)

    # validate yaml
    yaml_config = out.getvalue()
    dict_config = yaml.safe_load(yaml_config)
    assert dict_config["optimizer"]["class_path"] == "torch.optim.Adam"
    assert dict_config["optimizer"]["init_args"]["lr"] == 0.1
    assert dict_config["lr_scheduler"]["class_path"] == "torch.optim.lr_scheduler.OneCycleLR"

    # reload config
    yaml_config_file = Path("config.yaml")
    yaml_config_file.write_text(yaml_config)
    with mock.patch("sys.argv", base + [f"--config={yaml_config_file}"]):
        LightningCLI(BoringModel, run=False)


@_xfail_python_ge_3_11_9
def test_optimizers_and_lr_schedulers_add_arguments_to_parser_implemented_reload(cleandir):
    class TestLightningCLI(LightningCLI):
        def __init__(self, *args):
            super().__init__(*args, run=False)

        def add_arguments_to_parser(self, parser):
            parser.add_optimizer_args(nested_key="opt1", link_to="model.opt1_config")
            parser.add_optimizer_args(
                (torch.optim.ASGD, torch.optim.SGD), nested_key="opt2", link_to="model.opt2_config"
            )
            parser.add_lr_scheduler_args(link_to="model.sch_config")
            parser.add_argument("--something", type=str, nargs="+")

    class TestModel(BoringModel):
        def __init__(self, opt1_config: dict, opt2_config: dict, sch_config: dict):
            super().__init__()
            self.opt1_config = opt1_config
            self.opt2_config = opt2_config
            self.sch_config = sch_config
            opt1 = instantiate_class(self.parameters(), opt1_config)
            assert isinstance(opt1, torch.optim.Adam)
            opt2 = instantiate_class(self.parameters(), opt2_config)
            assert isinstance(opt2, torch.optim.ASGD)
            sch = instantiate_class(opt1, sch_config)
            assert isinstance(sch, torch.optim.lr_scheduler.OneCycleLR)

    base = ["any.py", "--trainer.max_epochs=1"]
    input = base + [
        "--lr_scheduler",
        "OneCycleLR",
        "--lr_scheduler.total_steps=10",
        "--lr_scheduler.max_lr=1",
        "--opt1",
        "Adam",
        "--opt2=ASGD",
        "--opt2.lr=0.1",
        "--lr_scheduler.anneal_strategy=linear",
        "--something",
        "a",
        "b",
        "c",
    ]

    # save config
    out = StringIO()
    with mock.patch("sys.argv", input + ["--print_config"]), redirect_stdout(out), pytest.raises(SystemExit):
        TestLightningCLI(TestModel)

    # validate yaml
    yaml_config = out.getvalue()
    dict_config = yaml.safe_load(yaml_config)
    assert dict_config["opt1"]["class_path"] == "torch.optim.Adam"
    assert dict_config["opt2"]["class_path"] == "torch.optim.ASGD"
    assert dict_config["opt2"]["init_args"]["lr"] == 0.1
    assert dict_config["lr_scheduler"]["class_path"] == "torch.optim.lr_scheduler.OneCycleLR"
    assert dict_config["lr_scheduler"]["init_args"]["anneal_strategy"] == "linear"
    assert dict_config["something"] == ["a", "b", "c"]

    # reload config
    yaml_config_file = Path("config.yaml")
    yaml_config_file.write_text(yaml_config)
    with mock.patch("sys.argv", base + [f"--config={yaml_config_file}"]):
        cli = TestLightningCLI(TestModel)

    assert cli.model.opt1_config["class_path"] == "torch.optim.Adam"
    assert cli.model.opt2_config["class_path"] == "torch.optim.ASGD"
    assert cli.model.opt2_config["init_args"]["lr"] == 0.1
    assert cli.model.sch_config["class_path"] == "torch.optim.lr_scheduler.OneCycleLR"
    assert cli.model.sch_config["init_args"]["anneal_strategy"] == "linear"


def test_lightning_cli_config_with_subcommand():
    config = {"test": {"trainer": {"limit_test_batches": 1}, "verbose": True, "ckpt_path": "foobar"}}
    with (
        mock.patch("sys.argv", ["any.py", f"--config={config}"]),
        mock.patch("lightning.pytorch.Trainer.test", autospec=True) as test_mock,
    ):
        cli = LightningCLI(BoringModel)

    test_mock.assert_called_once_with(cli.trainer, cli.model, verbose=True, ckpt_path="foobar")
    assert cli.trainer.limit_test_batches == 1


def test_lightning_cli_config_before_subcommand():
    config = {
        "validate": {"trainer": {"limit_val_batches": 1}, "verbose": False, "ckpt_path": "barfoo"},
        "test": {"trainer": {"limit_test_batches": 1}, "verbose": True, "ckpt_path": "foobar"},
    }

    with (
        mock.patch("sys.argv", ["any.py", f"--config={config}", "test"]),
        mock.patch("lightning.pytorch.Trainer.test", autospec=True) as test_mock,
    ):
        cli = LightningCLI(BoringModel)

    test_mock.assert_called_once_with(cli.trainer, model=cli.model, verbose=True, ckpt_path="foobar")
    assert cli.trainer.limit_test_batches == 1

    save_config_callback = cli.trainer.callbacks[0]
    assert save_config_callback.config.trainer.limit_test_batches == 1
    assert save_config_callback.parser.subcommand == "test"

    with (
        mock.patch("sys.argv", ["any.py", f"--config={config}", "validate"]),
        mock.patch("lightning.pytorch.Trainer.validate", autospec=True) as validate_mock,
    ):
        cli = LightningCLI(BoringModel)

    validate_mock.assert_called_once_with(cli.trainer, cli.model, verbose=False, ckpt_path="barfoo")
    assert cli.trainer.limit_val_batches == 1

    save_config_callback = cli.trainer.callbacks[0]
    assert save_config_callback.config.trainer.limit_val_batches == 1
    assert save_config_callback.parser.subcommand == "validate"


def test_lightning_cli_config_before_subcommand_two_configs():
    config1 = {"validate": {"trainer": {"limit_val_batches": 1}, "verbose": False, "ckpt_path": "barfoo"}}
    config2 = {"test": {"trainer": {"limit_test_batches": 1}, "verbose": True, "ckpt_path": "foobar"}}

    with (
        mock.patch("sys.argv", ["any.py", f"--config={config1}", f"--config={config2}", "test"]),
        mock.patch("lightning.pytorch.Trainer.test", autospec=True) as test_mock,
    ):
        cli = LightningCLI(BoringModel)

    test_mock.assert_called_once_with(cli.trainer, model=cli.model, verbose=True, ckpt_path="foobar")
    assert cli.trainer.limit_test_batches == 1

    with (
        mock.patch("sys.argv", ["any.py", f"--config={config1}", f"--config={config2}", "validate"]),
        mock.patch("lightning.pytorch.Trainer.validate", autospec=True) as validate_mock,
    ):
        cli = LightningCLI(BoringModel)

    validate_mock.assert_called_once_with(cli.trainer, cli.model, verbose=False, ckpt_path="barfoo")
    assert cli.trainer.limit_val_batches == 1


def test_lightning_cli_config_after_subcommand():
    config = {"trainer": {"limit_test_batches": 1}, "verbose": True, "ckpt_path": "foobar"}
    with (
        mock.patch("sys.argv", ["any.py", "test", f"--config={config}"]),
        mock.patch("lightning.pytorch.Trainer.test", autospec=True) as test_mock,
    ):
        cli = LightningCLI(BoringModel)

    test_mock.assert_called_once_with(cli.trainer, cli.model, verbose=True, ckpt_path="foobar")
    assert cli.trainer.limit_test_batches == 1


def test_lightning_cli_config_before_and_after_subcommand():
    config1 = {"test": {"trainer": {"limit_test_batches": 1}, "verbose": True, "ckpt_path": "foobar"}}
    config2 = {"trainer": {"fast_dev_run": 1}, "verbose": False, "ckpt_path": "foobar"}
    with (
        mock.patch("sys.argv", ["any.py", f"--config={config1}", "test", f"--config={config2}"]),
        mock.patch("lightning.pytorch.Trainer.test", autospec=True) as test_mock,
    ):
        cli = LightningCLI(BoringModel)

    test_mock.assert_called_once_with(cli.trainer, model=cli.model, verbose=False, ckpt_path="foobar")
    assert cli.trainer.limit_test_batches == 1
    assert cli.trainer.fast_dev_run == 1


def test_lightning_cli_parse_kwargs_with_subcommands(cleandir):
    fit_config = {"trainer": {"limit_train_batches": 2}}
    fit_config_path = Path("fit.yaml")
    fit_config_path.write_text(str(fit_config))

    validate_config = {"trainer": {"limit_val_batches": 3}}
    validate_config_path = Path("validate.yaml")
    validate_config_path.write_text(str(validate_config))

    parser_kwargs = {
        "fit": {"default_config_files": [str(fit_config_path)]},
        "validate": {"default_config_files": [str(validate_config_path)]},
    }

    with (
        mock.patch("sys.argv", ["any.py", "fit"]),
        mock.patch("lightning.pytorch.Trainer.fit", autospec=True) as fit_mock,
    ):
        cli = LightningCLI(BoringModel, parser_kwargs=parser_kwargs)
    fit_mock.assert_called()
    assert cli.trainer.limit_train_batches == 2
    assert cli.trainer.limit_val_batches == 1.0

    with (
        mock.patch("sys.argv", ["any.py", "validate"]),
        mock.patch("lightning.pytorch.Trainer.validate", autospec=True) as validate_mock,
    ):
        cli = LightningCLI(BoringModel, parser_kwargs=parser_kwargs)
    validate_mock.assert_called()
    assert cli.trainer.limit_train_batches == 1.0
    assert cli.trainer.limit_val_batches == 3


def test_lightning_cli_subcommands_common_default_config_files(cleandir):
    class Model(BoringModel):
        def __init__(self, foo: int, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.foo = foo

    config = {"fit": {"model": {"foo": 123}}}
    config_path = Path("default.yaml")
    config_path.write_text(str(config))
    parser_kwargs = {"default_config_files": [str(config_path)]}

    with (
        mock.patch("sys.argv", ["any.py", "fit"]),
        mock.patch("lightning.pytorch.Trainer.fit", autospec=True) as fit_mock,
    ):
        cli = LightningCLI(Model, parser_kwargs=parser_kwargs)
    fit_mock.assert_called()
    assert cli.model.foo == 123


def test_lightning_cli_reinstantiate_trainer():
    with mock.patch("sys.argv", ["any.py"]):
        cli = LightningCLI(BoringModel, run=False)

    assert cli.trainer.max_epochs is None

    class TestCallback(Callback): ...

    # make sure a new trainer can be easily created
    trainer = cli.instantiate_trainer(max_epochs=123, callbacks=[TestCallback()])
    # the new config is used
    assert trainer.max_epochs == 123
    assert {c.__class__ for c in trainer.callbacks} == {c.__class__ for c in cli.trainer.callbacks}.union({
        TestCallback
    })
    # the existing config is not updated
    assert cli.config_init["trainer"]["max_epochs"] is None


def test_cli_configure_optimizers_warning():
    match = "configure_optimizers` will be overridden by `LightningCLI"
    with mock.patch("sys.argv", ["any.py"]), no_warning_call(UserWarning, match=match):
        LightningCLI(BoringModel, run=False)
    with mock.patch("sys.argv", ["any.py", "--optimizer=Adam"]), pytest.warns(UserWarning, match=match):
        LightningCLI(BoringModel, run=False)


def test_cli_help_message():
    # full class path
    cli_args = ["any.py", "--optimizer.help=torch.optim.Adam"]
    classpath_help = StringIO()
    with mock.patch("sys.argv", cli_args), redirect_stdout(classpath_help), pytest.raises(SystemExit):
        LightningCLI(BoringModel, run=False)

    cli_args = ["any.py", "--optimizer.help=Adam"]
    shorthand_help = StringIO()
    with mock.patch("sys.argv", cli_args), redirect_stdout(shorthand_help), pytest.raises(SystemExit):
        LightningCLI(BoringModel, run=False)

    # the help messages should match
    assert shorthand_help.getvalue() == classpath_help.getvalue()
    # make sure it's not empty
    assert "Implements Adam" in shorthand_help.getvalue()


@_xfail_python_ge_3_11_9
def test_cli_reducelronplateau():
    with mock.patch(
        "sys.argv", ["any.py", "--optimizer=Adam", "--lr_scheduler=ReduceLROnPlateau", "--lr_scheduler.monitor=foo"]
    ):
        cli = LightningCLI(BoringModel, run=False)
    config = cli.model.configure_optimizers()
    assert isinstance(config["lr_scheduler"]["scheduler"], ReduceLROnPlateau)
    assert config["lr_scheduler"]["scheduler"].monitor == "foo"


@_xfail_python_ge_3_11_9
def test_cli_configureoptimizers_can_be_overridden():
    class MyCLI(LightningCLI):
        def __init__(self):
            super().__init__(BoringModel, run=False)

        @staticmethod
        def configure_optimizers(self, optimizer, lr_scheduler=None):
            assert isinstance(self, BoringModel)
            assert lr_scheduler is None
            return 123

    with mock.patch("sys.argv", ["any.py", "--optimizer=Adam"]):
        cli = MyCLI()
    assert cli.model.configure_optimizers() == 123

    # with no optimization config, we don't override
    with mock.patch("sys.argv", ["any.py"]):
        cli = MyCLI()
    [optimizer], [scheduler] = cli.model.configure_optimizers()
    assert isinstance(optimizer, SGD)
    assert isinstance(scheduler, StepLR)
    with mock.patch("sys.argv", ["any.py", "--lr_scheduler=StepLR", "--lr_scheduler.step_size=50"]):
        cli = MyCLI()
    [optimizer], [scheduler] = cli.model.configure_optimizers()
    assert isinstance(optimizer, SGD)
    assert isinstance(scheduler, StepLR)


def test_cli_parameter_with_lazy_instance_default():
    class TestModel(BoringModel):
        def __init__(self, activation: torch.nn.Module = lazy_instance(torch.nn.LeakyReLU, negative_slope=0.05)):
            super().__init__()
            self.activation = activation

    model = TestModel()
    assert isinstance(model.activation, torch.nn.LeakyReLU)

    with mock.patch("sys.argv", ["any.py"]):
        cli = LightningCLI(TestModel, run=False)
        assert isinstance(cli.model.activation, torch.nn.LeakyReLU)
        assert cli.model.activation.negative_slope == 0.05
        assert cli.model.activation is not model.activation


@_xfail_python_ge_3_11_9
def test_ddpstrategy_instantiation_and_find_unused_parameters(mps_count_0):
    strategy_default = lazy_instance(DDPStrategy, find_unused_parameters=True)
    with mock.patch("sys.argv", ["any.py", "--trainer.strategy.process_group_backend=group"]):
        cli = LightningCLI(
            BoringModel,
            run=False,
            trainer_defaults={"strategy": strategy_default},
        )

    assert cli.config.trainer.strategy.init_args.find_unused_parameters is True
    assert isinstance(cli.config_init.trainer.strategy, DDPStrategy)
    assert cli.config_init.trainer.strategy.process_group_backend == "group"
    assert strategy_default is not cli.config_init.trainer.strategy


@_xfail_python_ge_3_11_9
def test_cli_logger_shorthand():
    with mock.patch("sys.argv", ["any.py"]):
        cli = LightningCLI(TestModel, run=False, trainer_defaults={"logger": False})
    assert cli.trainer.logger is None

    with mock.patch("sys.argv", ["any.py", "--trainer.logger=TensorBoardLogger", "--trainer.logger.save_dir=foo"]):
        cli = LightningCLI(TestModel, run=False, trainer_defaults={"logger": False})
    assert isinstance(cli.trainer.logger, TensorBoardLogger)

    with mock.patch("sys.argv", ["any.py", "--trainer.logger=False"]):
        cli = LightningCLI(TestModel, run=False)
    assert cli.trainer.logger is None


def _test_logger_init_args(logger_name, init, unresolved=None):
    cli_args = [f"--trainer.logger={logger_name}"]
    cli_args += [f"--trainer.logger.{k}={v}" for k, v in init.items()]
    cli_args += [f"--trainer.logger.dict_kwargs.{k}={v}" for k, v in unresolved.items()]
    cli_args.append("--print_config")

    out = StringIO()
    with mock.patch("sys.argv", ["any.py"] + cli_args), redirect_stdout(out), pytest.raises(SystemExit):
        LightningCLI(TestModel, run=False)

    data = yaml.safe_load(out.getvalue())["trainer"]["logger"]
    assert {k: data["init_args"][k] for k in init} == init
    if unresolved:
        assert data["dict_kwargs"] == unresolved


@_xfail_python_ge_3_11_9
def test_comet_logger_init_args():
    _test_logger_init_args(
        "CometLogger",
        init={
            "experiment_key": "some_key",  # Resolve from CometLogger.__init__
            "workspace": "comet",
        },
        unresolved={
            "save_dir": "comet",  # Resolve from CometLogger.__init__ as kwarg
        },
    )


@pytest.mark.xfail(
    # Only on Windows: TypeError: 'NoneType' object is not subscriptable
    raises=TypeError,
    condition=(sys.platform == "win32"),
    strict=False,
    reason="TypeError on Windows when parsing",
)
@_xfail_python_ge_3_11_9
def test_neptune_logger_init_args():
    _test_logger_init_args(
        "NeptuneLogger",
        init={"name": "neptune"},  # Resolve from NeptuneLogger.__init__
        unresolved={"description": "neptune"},  # Unsupported resolving from neptune.internal.init.run.init_run
    )


@_xfail_python_ge_3_11_9
def test_tensorboard_logger_init_args():
    _test_logger_init_args(
        "TensorBoardLogger",
        init={
            "save_dir": "tb",  # Resolve from TensorBoardLogger.__init__
            "comment": "tb",  # Resolve from FabricTensorBoardLogger.experiment SummaryWriter local import
        },
        unresolved={},
    )


@_xfail_python_ge_3_11_9
def test_wandb_logger_init_args():
    _test_logger_init_args(
        "WandbLogger",
        init={"save_dir": "wandb"},  # Resolve from WandbLogger.__init__
        unresolved={"notes": "wandb"},  # Resolve from wandb.sdk.wandb_init.init
    )


def test_cli_auto_seeding():
    with mock.patch("sys.argv", ["any.py"]):
        cli = LightningCLI(TestModel, run=False, seed_everything_default=False)
    assert cli.seed_everything_default is False
    assert cli.config["seed_everything"] is False

    with mock.patch("sys.argv", ["any.py"]):
        cli = LightningCLI(TestModel, run=False, seed_everything_default=True)
    assert cli.seed_everything_default is True
    assert isinstance(cli.config["seed_everything"], int)

    with mock.patch("sys.argv", ["any.py", "--seed_everything", "3"]):
        cli = LightningCLI(TestModel, run=False, seed_everything_default=False)
    assert cli.seed_everything_default is False
    assert cli.config["seed_everything"] == 3

    with mock.patch("sys.argv", ["any.py", "--seed_everything", "3"]):
        cli = LightningCLI(TestModel, run=False, seed_everything_default=True)
    assert cli.seed_everything_default is True
    assert cli.config["seed_everything"] == 3

    with mock.patch("sys.argv", ["any.py", "--seed_everything", "3"]):
        cli = LightningCLI(TestModel, run=False, seed_everything_default=10)
    assert cli.seed_everything_default == 10
    assert cli.config["seed_everything"] == 3

    with mock.patch("sys.argv", ["any.py", "--seed_everything", "false"]):
        cli = LightningCLI(TestModel, run=False, seed_everything_default=10)
    assert cli.seed_everything_default == 10
    assert cli.config["seed_everything"] is False

    with mock.patch("sys.argv", ["any.py", "--seed_everything", "false"]):
        cli = LightningCLI(TestModel, run=False, seed_everything_default=True)
    assert cli.seed_everything_default is True
    assert cli.config["seed_everything"] is False

    with mock.patch("sys.argv", ["any.py", "--seed_everything", "true"]):
        cli = LightningCLI(TestModel, run=False, seed_everything_default=False)
    assert cli.seed_everything_default is False
    assert isinstance(cli.config["seed_everything"], int)

    seed_everything(123)
    with mock.patch("sys.argv", ["any.py"]):
        cli = LightningCLI(TestModel, run=False)
    assert cli.seed_everything_default is True
    assert cli.config["seed_everything"] == 123  # the original seed is kept


def test_cli_trainer_no_callbacks():
    class MyTrainer(Trainer):
        def __init__(self):
            super().__init__()

    class MyCallback(Callback): ...

    match = "MyTrainer` class does not expose the `callbacks"
    with mock.patch("sys.argv", ["any.py"]), pytest.warns(UserWarning, match=match):
        cli = LightningCLI(
            BoringModel, run=False, trainer_class=MyTrainer, trainer_defaults={"callbacks": MyCallback()}
        )
    assert not any(isinstance(cb, MyCallback) for cb in cli.trainer.callbacks)


def test_unresolvable_import_paths():
    class TestModel(BoringModel):
        def __init__(self, a_func: Callable = torch.nn.Softmax):
            super().__init__()
            self.a_func = a_func

    out = StringIO()
    with mock.patch("sys.argv", ["any.py", "--print_config"]), redirect_stdout(out), pytest.raises(SystemExit):
        LightningCLI(TestModel, run=False)

    assert "a_func: torch.nn.Softmax" in out.getvalue()


@_xfail_python_ge_3_11_9
def test_pytorch_profiler_init_args():
    from lightning.pytorch.profilers import Profiler, PyTorchProfiler

    init = {
        "dirpath": "profiler",  # Resolve from PyTorchProfiler.__init__
        "row_limit": 10,  # Resolve from PyTorchProfiler.__init__
        "group_by_input_shapes": True,  # Resolve from PyTorchProfiler.__init__
    }
    unresolved = {
        "profile_memory": True,  # Not possible to resolve parameters from dynamically chosen Type[_PROFILER]
        "record_shapes": True,  # Resolve from PyTorchProfiler.__init__, gets moved to init_args
    }
    cli_args = ["--trainer.profiler=PyTorchProfiler"]
    cli_args += [f"--trainer.profiler.{k}={v}" for k, v in init.items()]
    cli_args += [f"--trainer.profiler.dict_kwargs.{k}={v}" for k, v in unresolved.items()]

    with mock.patch("sys.argv", ["any.py"] + cli_args), mock_subclasses(Profiler, PyTorchProfiler):
        cli = LightningCLI(TestModel, run=False)

    assert isinstance(cli.config_init.trainer.profiler, PyTorchProfiler)
    init["record_shapes"] = unresolved.pop("record_shapes")  # Test move to init_args
    assert {k: cli.config.trainer.profiler.init_args[k] for k in init} == init
    assert cli.config.trainer.profiler.dict_kwargs == unresolved


@pytest.mark.parametrize(
    "args",
    [
        ["--trainer.logger=False", "--model.foo=456"],
        {"trainer": {"logger": False}, "model": {"foo": 456}},
        Namespace(trainer=Namespace(logger=False), model=Namespace(foo=456)),
    ],
)
def test_lightning_cli_with_args_given(args):
    with mock.patch("sys.argv", [""]):
        cli = LightningCLI(TestModel, run=False, args=args)
    assert isinstance(cli.model, TestModel)
    assert cli.config.trainer.logger is False
    assert cli.model.foo == 456


def test_lightning_cli_args_and_sys_argv_warning():
    with mock.patch("sys.argv", ["", "--model.foo=456"]), pytest.warns(Warning, match="LightningCLI's args parameter "):
        LightningCLI(TestModel, run=False, args=["--model.foo=789"])
