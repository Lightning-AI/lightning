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
import inspect
import os
import sys
from collections.abc import Iterable
from functools import partial, update_wrapper
from types import MethodType
from typing import Any, Callable, Optional, TypeVar, Union

import torch
import yaml
from lightning_utilities.core.imports import RequirementCache
from lightning_utilities.core.rank_zero import _warn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing_extensions import override

import lightning.pytorch as pl
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.pytorch import Callback, LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.core.mixins.hparams_mixin import _given_hyperparameters_context
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning.pytorch.utilities.rank_zero import rank_zero_warn

_JSONARGPARSE_SIGNATURES_AVAILABLE = RequirementCache("jsonargparse[signatures]>=4.27.7")

if _JSONARGPARSE_SIGNATURES_AVAILABLE:
    import docstring_parser
    from jsonargparse import (
        ActionConfigFile,
        ArgumentParser,
        Namespace,
        class_from_function,
        register_unresolvable_import_paths,
        set_config_read_mode,
    )

    register_unresolvable_import_paths(torch)  # Required until fix https://github.com/pytorch/pytorch/issues/74483
    set_config_read_mode(fsspec_enabled=True)
else:
    locals()["ArgumentParser"] = object
    locals()["Namespace"] = object

ModuleType = TypeVar("ModuleType")


class ReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, optimizer: Optimizer, monitor: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(optimizer, *args, **kwargs)
        self.monitor = monitor


# LightningCLI requires the ReduceLROnPlateau defined here, thus it shouldn't accept the one from pytorch:
LRSchedulerTypeTuple = (LRScheduler, ReduceLROnPlateau)
LRSchedulerTypeUnion = Union[LRScheduler, ReduceLROnPlateau]
LRSchedulerType = Union[type[LRScheduler], type[ReduceLROnPlateau]]


# Type aliases intended for convenience of CLI developers
ArgsType = Optional[Union[list[str], dict[str, Any], Namespace]]
OptimizerCallable = Callable[[Iterable], Optimizer]
LRSchedulerCallable = Callable[[Optimizer], Union[LRScheduler, ReduceLROnPlateau]]


class LightningArgumentParser(ArgumentParser):
    """Extension of jsonargparse's ArgumentParser for pytorch-lightning."""

    def __init__(
        self,
        *args: Any,
        description: str = "Lightning Trainer command line tool",
        env_prefix: str = "PL",
        default_env: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize argument parser that supports configuration file input.

        For full details of accepted arguments see `ArgumentParser.__init__
        <https://jsonargparse.readthedocs.io/en/stable/#jsonargparse.ArgumentParser.__init__>`_.

        Args:
            description: Description of the tool shown when running ``--help``.
            env_prefix: Prefix for environment variables. Set ``default_env=True`` to enable env parsing.
            default_env: Whether to parse environment variables.

        """
        if not _JSONARGPARSE_SIGNATURES_AVAILABLE:
            raise ModuleNotFoundError(f"{_JSONARGPARSE_SIGNATURES_AVAILABLE}")
        super().__init__(*args, description=description, env_prefix=env_prefix, default_env=default_env, **kwargs)
        self.callback_keys: list[str] = []
        # separate optimizers and lr schedulers to know which were added
        self._optimizers: dict[str, tuple[Union[type, tuple[type, ...]], str]] = {}
        self._lr_schedulers: dict[str, tuple[Union[type, tuple[type, ...]], str]] = {}

    def add_lightning_class_args(
        self,
        lightning_class: Union[
            Callable[..., Union[Trainer, LightningModule, LightningDataModule, Callback]],
            type[Trainer],
            type[LightningModule],
            type[LightningDataModule],
            type[Callback],
        ],
        nested_key: str,
        subclass_mode: bool = False,
        required: bool = True,
    ) -> list[str]:
        """Adds arguments from a lightning class to a nested key of the parser.

        Args:
            lightning_class: A callable or any subclass of {Trainer, LightningModule, LightningDataModule, Callback}.
            nested_key: Name of the nested namespace to store arguments.
            subclass_mode: Whether allow any subclass of the given class.
            required: Whether the argument group is required.

        Returns:
            A list with the names of the class arguments added.

        """
        if callable(lightning_class) and not isinstance(lightning_class, type):
            lightning_class = class_from_function(lightning_class)

        if isinstance(lightning_class, type) and issubclass(
            lightning_class, (Trainer, LightningModule, LightningDataModule, Callback)
        ):
            if issubclass(lightning_class, Callback):
                self.callback_keys.append(nested_key)
            if subclass_mode:
                return self.add_subclass_arguments(lightning_class, nested_key, fail_untyped=False, required=required)
            return self.add_class_arguments(
                lightning_class,
                nested_key,
                fail_untyped=False,
                instantiate=not issubclass(lightning_class, Trainer),
                sub_configs=True,
            )
        raise MisconfigurationException(
            f"Cannot add arguments from: {lightning_class}. You should provide either a callable or a subclass of: "
            "Trainer, LightningModule, LightningDataModule, or Callback."
        )

    def add_optimizer_args(
        self,
        optimizer_class: Union[type[Optimizer], tuple[type[Optimizer], ...]] = (Optimizer,),
        nested_key: str = "optimizer",
        link_to: str = "AUTOMATIC",
    ) -> None:
        """Adds arguments from an optimizer class to a nested key of the parser.

        Args:
            optimizer_class: Any subclass of :class:`torch.optim.Optimizer`. Use tuple to allow subclasses.
            nested_key: Name of the nested namespace to store arguments.
            link_to: Dot notation of a parser key to set arguments or AUTOMATIC.

        """
        if isinstance(optimizer_class, tuple):
            assert all(issubclass(o, Optimizer) for o in optimizer_class)
        else:
            assert issubclass(optimizer_class, Optimizer)
        kwargs: dict[str, Any] = {"instantiate": False, "fail_untyped": False, "skip": {"params"}}
        if isinstance(optimizer_class, tuple):
            self.add_subclass_arguments(optimizer_class, nested_key, **kwargs)
        else:
            self.add_class_arguments(optimizer_class, nested_key, sub_configs=True, **kwargs)
        self._optimizers[nested_key] = (optimizer_class, link_to)

    def add_lr_scheduler_args(
        self,
        lr_scheduler_class: Union[LRSchedulerType, tuple[LRSchedulerType, ...]] = LRSchedulerTypeTuple,
        nested_key: str = "lr_scheduler",
        link_to: str = "AUTOMATIC",
    ) -> None:
        """Adds arguments from a learning rate scheduler class to a nested key of the parser.

        Args:
            lr_scheduler_class: Any subclass of ``torch.optim.lr_scheduler.{_LRScheduler, ReduceLROnPlateau}``. Use
                tuple to allow subclasses.
            nested_key: Name of the nested namespace to store arguments.
            link_to: Dot notation of a parser key to set arguments or AUTOMATIC.

        """
        if isinstance(lr_scheduler_class, tuple):
            assert all(issubclass(o, LRSchedulerTypeTuple) for o in lr_scheduler_class)
        else:
            assert issubclass(lr_scheduler_class, LRSchedulerTypeTuple)
        kwargs: dict[str, Any] = {"instantiate": False, "fail_untyped": False, "skip": {"optimizer"}}
        if isinstance(lr_scheduler_class, tuple):
            self.add_subclass_arguments(lr_scheduler_class, nested_key, **kwargs)
        else:
            self.add_class_arguments(lr_scheduler_class, nested_key, sub_configs=True, **kwargs)
        self._lr_schedulers[nested_key] = (lr_scheduler_class, link_to)


class SaveConfigCallback(Callback):
    """Saves a LightningCLI config to the log_dir when training starts.

    Args:
        parser: The parser object used to parse the configuration.
        config: The parsed configuration that will be saved.
        config_filename: Filename for the config file.
        overwrite: Whether to overwrite an existing config file.
        multifile: When input is multiple config files, saved config preserves this structure.
        save_to_log_dir: Whether to save the config to the log_dir.

    Raises:
        RuntimeError: If the config file already exists in the directory to avoid overwriting a previous run

    """

    def __init__(
        self,
        parser: LightningArgumentParser,
        config: Namespace,
        config_filename: str = "config.yaml",
        overwrite: bool = False,
        multifile: bool = False,
        save_to_log_dir: bool = True,
    ) -> None:
        self.parser = parser
        self.config = config
        self.config_filename = config_filename
        self.overwrite = overwrite
        self.multifile = multifile
        self.save_to_log_dir = save_to_log_dir
        self.already_saved = False

        if not save_to_log_dir and not is_overridden("save_config", self, SaveConfigCallback):
            raise ValueError(
                "`save_to_log_dir=False` only makes sense when subclassing SaveConfigCallback to implement "
                "`save_config` and it is desired to disable the standard behavior of saving to log_dir."
            )

    @override
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if self.already_saved:
            return

        if self.save_to_log_dir:
            log_dir = trainer.log_dir  # this broadcasts the directory
            assert log_dir is not None
            config_path = os.path.join(log_dir, self.config_filename)
            fs = get_filesystem(log_dir)

            if not self.overwrite:
                # check if the file exists on rank 0
                file_exists = fs.isfile(config_path) if trainer.is_global_zero else False
                # broadcast whether to fail to all ranks
                file_exists = trainer.strategy.broadcast(file_exists)
                if file_exists:
                    raise RuntimeError(
                        f"{self.__class__.__name__} expected {config_path} to NOT exist. Aborting to avoid overwriting"
                        " results of a previous run. You can delete the previous config file,"
                        " set `LightningCLI(save_config_callback=None)` to disable config saving,"
                        ' or set `LightningCLI(save_config_kwargs={"overwrite": True})` to overwrite the config file.'
                    )

            if trainer.is_global_zero:
                # save only on rank zero to avoid race conditions.
                # the `log_dir` needs to be created as we rely on the logger to do it usually
                # but it hasn't logged anything at this point
                fs.makedirs(log_dir, exist_ok=True)
                self.parser.save(
                    self.config, config_path, skip_none=False, overwrite=self.overwrite, multifile=self.multifile
                )

        if trainer.is_global_zero:
            self.save_config(trainer, pl_module, stage)
            self.already_saved = True

        # broadcast so that all ranks are in sync on future calls to .setup()
        self.already_saved = trainer.strategy.broadcast(self.already_saved)

    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Implement to save the config in some other place additional to the standard log_dir.

        Example:
            def save_config(self, trainer, pl_module, stage):
                if isinstance(trainer.logger, Logger):
                    config = self.parser.dump(self.config, skip_none=False)  # Required for proper reproducibility
                    trainer.logger.log_hyperparams({"config": config})

        Note:
            This method is only called on rank zero. This allows to implement a custom save config without having to
            worry about ranks or race conditions. Since it only runs on rank zero, any collective call will make the
            process hang waiting for a broadcast. If you need to make collective calls, implement the setup method
            instead.

        """


class LightningCLI:
    """Implementation of a configurable command line tool for pytorch-lightning."""

    def __init__(
        self,
        model_class: Optional[Union[type[LightningModule], Callable[..., LightningModule]]] = None,
        datamodule_class: Optional[Union[type[LightningDataModule], Callable[..., LightningDataModule]]] = None,
        save_config_callback: Optional[type[SaveConfigCallback]] = SaveConfigCallback,
        save_config_kwargs: Optional[dict[str, Any]] = None,
        trainer_class: Union[type[Trainer], Callable[..., Trainer]] = Trainer,
        trainer_defaults: Optional[dict[str, Any]] = None,
        seed_everything_default: Union[bool, int] = True,
        parser_kwargs: Optional[Union[dict[str, Any], dict[str, dict[str, Any]]]] = None,
        subclass_mode_model: bool = False,
        subclass_mode_data: bool = False,
        args: ArgsType = None,
        run: bool = True,
        auto_configure_optimizers: bool = True,
    ) -> None:
        """Receives as input pytorch-lightning classes (or callables which return pytorch-lightning classes), which are
        called / instantiated using a parsed configuration file and / or command line args.

        Parsing of configuration from environment variables can be enabled by setting ``parser_kwargs={"default_env":
        True}``. A full configuration yaml would be parsed from ``PL_CONFIG`` if set. Individual settings are so parsed
        from variables named for example ``PL_TRAINER__MAX_EPOCHS``.

        For more info, read :ref:`the CLI docs <lightning-cli>`.

        Args:
            model_class: An optional :class:`~lightning.pytorch.core.LightningModule` class to train on or a
                callable which returns a :class:`~lightning.pytorch.core.LightningModule` instance when
                called. If ``None``, you can pass a registered model with ``--model=MyModel``.
            datamodule_class: An optional :class:`~lightning.pytorch.core.datamodule.LightningDataModule` class or a
                callable which returns a :class:`~lightning.pytorch.core.datamodule.LightningDataModule` instance when
                called. If ``None``, you can pass a registered datamodule with ``--data=MyDataModule``.
            save_config_callback: A callback class to save the config.
            save_config_kwargs: Parameters that will be used to instantiate the save_config_callback.
            trainer_class: An optional subclass of the :class:`~lightning.pytorch.trainer.trainer.Trainer` class or a
                callable which returns a :class:`~lightning.pytorch.trainer.trainer.Trainer` instance when called.
            trainer_defaults: Set to override Trainer defaults or add persistent callbacks. The callbacks added through
                this argument will not be configurable from a configuration file and will always be present for
                this particular CLI. Alternatively, configurable callbacks can be added as explained in
                :ref:`the CLI docs <lightning-cli>`.
            seed_everything_default: Number for the :func:`~lightning.fabric.utilities.seed.seed_everything`
                seed value. Set to True to automatically choose a seed value.
                Setting it to False will avoid calling ``seed_everything``.
            parser_kwargs: Additional arguments to instantiate each ``LightningArgumentParser``.
            subclass_mode_model: Whether model can be any `subclass
                <https://jsonargparse.readthedocs.io/en/stable/#class-type-and-sub-classes>`_
                of the given class.
            subclass_mode_data: Whether datamodule can be any `subclass
                <https://jsonargparse.readthedocs.io/en/stable/#class-type-and-sub-classes>`_
                of the given class.
            args: Arguments to parse. If ``None`` the arguments are taken from ``sys.argv``. Command line style
                arguments can be given in a ``list``. Alternatively, structured config options can be given in a
                ``dict`` or ``jsonargparse.Namespace``.
            run: Whether subcommands should be added to run a :class:`~lightning.pytorch.trainer.trainer.Trainer`
                method. If set to ``False``, the trainer and model classes will be instantiated only.

        """
        self.save_config_callback = save_config_callback
        self.save_config_kwargs = save_config_kwargs or {}
        self.trainer_class = trainer_class
        self.trainer_defaults = trainer_defaults or {}
        self.seed_everything_default = seed_everything_default
        self.parser_kwargs = parser_kwargs or {}
        self.auto_configure_optimizers = auto_configure_optimizers

        self.model_class = model_class
        # used to differentiate between the original value and the processed value
        self._model_class = model_class or LightningModule
        self.subclass_mode_model = (model_class is None) or subclass_mode_model

        self.datamodule_class = datamodule_class
        # used to differentiate between the original value and the processed value
        self._datamodule_class = datamodule_class or LightningDataModule
        self.subclass_mode_data = (datamodule_class is None) or subclass_mode_data

        main_kwargs, subparser_kwargs = self._setup_parser_kwargs(self.parser_kwargs)
        self.setup_parser(run, main_kwargs, subparser_kwargs)
        self.parse_arguments(self.parser, args)

        self.subcommand = self.config["subcommand"] if run else None

        self._set_seed()

        self._add_instantiators()
        self.before_instantiate_classes()
        self.instantiate_classes()
        self.after_instantiate_classes()

        if self.subcommand is not None:
            self._run_subcommand(self.subcommand)

    def _setup_parser_kwargs(self, parser_kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        subcommand_names = self.subcommands().keys()
        main_kwargs = {k: v for k, v in parser_kwargs.items() if k not in subcommand_names}
        subparser_kwargs = {k: v for k, v in parser_kwargs.items() if k in subcommand_names}
        return main_kwargs, subparser_kwargs

    def init_parser(self, **kwargs: Any) -> LightningArgumentParser:
        """Method that instantiates the argument parser."""
        kwargs.setdefault("dump_header", [f"lightning.pytorch=={pl.__version__}"])
        parser = LightningArgumentParser(**kwargs)
        parser.add_argument(
            "-c", "--config", action=ActionConfigFile, help="Path to a configuration file in json or yaml format."
        )
        return parser

    def setup_parser(
        self, add_subcommands: bool, main_kwargs: dict[str, Any], subparser_kwargs: dict[str, Any]
    ) -> None:
        """Initialize and setup the parser, subcommands, and arguments."""
        self.parser = self.init_parser(**main_kwargs)
        if add_subcommands:
            self._subcommand_method_arguments: dict[str, list[str]] = {}
            self._add_subcommands(self.parser, **subparser_kwargs)
        else:
            self._add_arguments(self.parser)

    def add_default_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Adds default arguments to the parser."""
        parser.add_argument(
            "--seed_everything",
            type=Union[bool, int],
            default=self.seed_everything_default,
            help=(
                "Set to an int to run seed_everything with this value before classes instantiation."
                "Set to True to use a random seed."
            ),
        )

    def add_core_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Adds arguments from the core classes to the parser."""
        parser.add_lightning_class_args(self.trainer_class, "trainer")
        trainer_defaults = {"trainer." + k: v for k, v in self.trainer_defaults.items() if k != "callbacks"}
        parser.set_defaults(trainer_defaults)

        parser.add_lightning_class_args(self._model_class, "model", subclass_mode=self.subclass_mode_model)

        if self.datamodule_class is not None:
            parser.add_lightning_class_args(self._datamodule_class, "data", subclass_mode=self.subclass_mode_data)
        else:
            # this should not be required because the user might want to use the `LightningModule` dataloaders
            parser.add_lightning_class_args(
                self._datamodule_class, "data", subclass_mode=self.subclass_mode_data, required=False
            )

    def _add_arguments(self, parser: LightningArgumentParser) -> None:
        # default + core + custom arguments
        self.add_default_arguments_to_parser(parser)
        self.add_core_arguments_to_parser(parser)
        self.add_arguments_to_parser(parser)
        # add default optimizer args if necessary
        if self.auto_configure_optimizers:
            if not parser._optimizers:  # already added by the user in `add_arguments_to_parser`
                parser.add_optimizer_args((Optimizer,))
            if not parser._lr_schedulers:  # already added by the user in `add_arguments_to_parser`
                parser.add_lr_scheduler_args(LRSchedulerTypeTuple)
        self.link_optimizers_and_lr_schedulers(parser)

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Implement to add extra arguments to the parser or link arguments.

        Args:
            parser: The parser object to which arguments can be added

        """

    @staticmethod
    def subcommands() -> dict[str, set[str]]:
        """Defines the list of available subcommands and the arguments to skip."""
        return {
            "fit": {"model", "train_dataloaders", "val_dataloaders", "datamodule"},
            "validate": {"model", "dataloaders", "datamodule"},
            "test": {"model", "dataloaders", "datamodule"},
            "predict": {"model", "dataloaders", "datamodule"},
        }

    def _add_subcommands(self, parser: LightningArgumentParser, **kwargs: Any) -> None:
        """Adds subcommands to the input parser."""
        self._subcommand_parsers: dict[str, LightningArgumentParser] = {}
        parser_subcommands = parser.add_subcommands()
        # the user might have passed a builder function
        trainer_class = (
            self.trainer_class if isinstance(self.trainer_class, type) else class_from_function(self.trainer_class)
        )
        # register all subcommands in separate subcommand parsers under the main parser
        for subcommand in self.subcommands():
            fn = getattr(trainer_class, subcommand)
            # extract the first line description in the docstring for the subcommand help message
            description = _get_short_description(fn)
            subparser_kwargs = kwargs.get(subcommand, {})
            subparser_kwargs.setdefault("description", description)
            subcommand_parser = self._prepare_subcommand_parser(trainer_class, subcommand, **subparser_kwargs)
            self._subcommand_parsers[subcommand] = subcommand_parser
            parser_subcommands.add_subcommand(subcommand, subcommand_parser, help=description)

    def _prepare_subcommand_parser(self, klass: type, subcommand: str, **kwargs: Any) -> LightningArgumentParser:
        parser = self.init_parser(**kwargs)
        self._add_arguments(parser)
        # subcommand arguments
        skip: set[Union[str, int]] = set(self.subcommands()[subcommand])
        added = parser.add_method_arguments(klass, subcommand, skip=skip)
        # need to save which arguments were added to pass them to the method later
        self._subcommand_method_arguments[subcommand] = added
        return parser

    @staticmethod
    def link_optimizers_and_lr_schedulers(parser: LightningArgumentParser) -> None:
        """Creates argument links for optimizers and learning rate schedulers that specified a ``link_to``."""
        optimizers_and_lr_schedulers = {**parser._optimizers, **parser._lr_schedulers}
        for key, (class_type, link_to) in optimizers_and_lr_schedulers.items():
            if link_to == "AUTOMATIC":
                continue
            if isinstance(class_type, tuple):
                parser.link_arguments(key, link_to)
            else:
                add_class_path = _add_class_path_generator(class_type)
                parser.link_arguments(key, link_to, compute_fn=add_class_path)

    def parse_arguments(self, parser: LightningArgumentParser, args: ArgsType) -> None:
        """Parses command line arguments and stores it in ``self.config``."""
        if args is not None and len(sys.argv) > 1:
            rank_zero_warn(
                "LightningCLI's args parameter is intended to run from within Python like if it were from the command "
                "line. To prevent mistakes it is not recommended to provide both args and command line arguments, got: "
                f"sys.argv[1:]={sys.argv[1:]}, args={args}."
            )
        if isinstance(args, (dict, Namespace)):
            self.config = parser.parse_object(args)
        else:
            self.config = parser.parse_args(args)

    def _add_instantiators(self) -> None:
        self.config_dump = yaml.safe_load(self.parser.dump(self.config, skip_link_targets=False, skip_none=False))
        if "subcommand" in self.config:
            self.config_dump = self.config_dump[self.config.subcommand]

        self.parser.add_instantiator(
            _InstantiatorFn(cli=self, key="model"),
            _get_module_type(self._model_class),
            subclasses=self.subclass_mode_model,
        )
        self.parser.add_instantiator(
            _InstantiatorFn(cli=self, key="data"),
            _get_module_type(self._datamodule_class),
            subclasses=self.subclass_mode_data,
        )

    def before_instantiate_classes(self) -> None:
        """Implement to run some code before instantiating the classes."""

    def instantiate_classes(self) -> None:
        """Instantiates the classes and sets their attributes."""
        self.config_init = self.parser.instantiate_classes(self.config)
        self.datamodule = self._get(self.config_init, "data")
        self.model = self._get(self.config_init, "model")
        self._add_configure_optimizers_method_to_model(self.subcommand)
        self.trainer = self.instantiate_trainer()

    def after_instantiate_classes(self) -> None:
        """Implement to run some code after instantiating the classes."""

    def instantiate_trainer(self, **kwargs: Any) -> Trainer:
        """Instantiates the trainer.

        Args:
            kwargs: Any custom trainer arguments.

        """
        extra_callbacks = [self._get(self.config_init, c) for c in self._parser(self.subcommand).callback_keys]
        trainer_config = {**self._get(self.config_init, "trainer", default={}), **kwargs}
        return self._instantiate_trainer(trainer_config, extra_callbacks)

    def _instantiate_trainer(self, config: dict[str, Any], callbacks: list[Callback]) -> Trainer:
        key = "callbacks"
        if key in config:
            if config[key] is None:
                config[key] = []
            elif not isinstance(config[key], list):
                config[key] = [config[key]]
            config[key].extend(callbacks)
            if key in self.trainer_defaults:
                value = self.trainer_defaults[key]
                config[key] += value if isinstance(value, list) else [value]
            if self.save_config_callback and not config.get("fast_dev_run", False):
                config_callback = self.save_config_callback(
                    self._parser(self.subcommand),
                    self.config.get(str(self.subcommand), self.config),
                    **self.save_config_kwargs,
                )
                config[key].append(config_callback)
        else:
            rank_zero_warn(
                f"The `{self.trainer_class.__qualname__}` class does not expose the `{key}` argument so they will"
                " not be included."
            )
        return self.trainer_class(**config)

    def _parser(self, subcommand: Optional[str]) -> LightningArgumentParser:
        if subcommand is None:
            return self.parser
        # return the subcommand parser for the subcommand passed
        return self._subcommand_parsers[subcommand]

    @staticmethod
    def configure_optimizers(
        lightning_module: LightningModule, optimizer: Optimizer, lr_scheduler: Optional[LRSchedulerTypeUnion] = None
    ) -> Any:
        """Override to customize the :meth:`~lightning.pytorch.core.LightningModule.configure_optimizers` method.

        Args:
            lightning_module: A reference to the model.
            optimizer: The optimizer.
            lr_scheduler: The learning rate scheduler (if used).

        """
        if lr_scheduler is None:
            return optimizer
        if isinstance(lr_scheduler, ReduceLROnPlateau):
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "monitor": lr_scheduler.monitor},
            }
        return [optimizer], [lr_scheduler]

    def _add_configure_optimizers_method_to_model(self, subcommand: Optional[str]) -> None:
        """Overrides the model's :meth:`~lightning.pytorch.core.LightningModule.configure_optimizers` method if a
        single optimizer and optionally a scheduler argument groups are added to the parser as 'AUTOMATIC'."""
        if not self.auto_configure_optimizers:
            return

        parser = self._parser(subcommand)

        def get_automatic(
            class_type: Union[type, tuple[type, ...]], register: dict[str, tuple[Union[type, tuple[type, ...]], str]]
        ) -> list[str]:
            automatic = []
            for key, (base_class, link_to) in register.items():
                if not isinstance(base_class, tuple):
                    base_class = (base_class,)
                if link_to == "AUTOMATIC" and any(issubclass(c, class_type) for c in base_class):
                    automatic.append(key)
            return automatic

        optimizers = get_automatic(Optimizer, parser._optimizers)
        lr_schedulers = get_automatic(LRSchedulerTypeTuple, parser._lr_schedulers)

        if len(optimizers) == 0:
            return

        if len(optimizers) > 1 or len(lr_schedulers) > 1:
            raise MisconfigurationException(
                f"`{self.__class__.__name__}.add_configure_optimizers_method_to_model` expects at most one optimizer "
                f"and one lr_scheduler to be 'AUTOMATIC', but found {optimizers + lr_schedulers}. In this case the "
                "user is expected to link the argument groups and implement `configure_optimizers`, see "
                "https://lightning.ai/docs/pytorch/stable/common/lightning_cli.html"
                "#optimizers-and-learning-rate-schedulers"
            )

        optimizer_class = parser._optimizers[optimizers[0]][0]
        optimizer_init = self._get(self.config_init, optimizers[0])
        if not isinstance(optimizer_class, tuple):
            optimizer_init = _global_add_class_path(optimizer_class, optimizer_init)
        if not optimizer_init:
            # optimizers were registered automatically but not passed by the user
            return

        lr_scheduler_init = None
        if lr_schedulers:
            lr_scheduler_class = parser._lr_schedulers[lr_schedulers[0]][0]
            lr_scheduler_init = self._get(self.config_init, lr_schedulers[0])
            if not isinstance(lr_scheduler_class, tuple):
                lr_scheduler_init = _global_add_class_path(lr_scheduler_class, lr_scheduler_init)

        if is_overridden("configure_optimizers", self.model):
            _warn(
                f"`{self.model.__class__.__name__}.configure_optimizers` will be overridden by "
                f"`{self.__class__.__name__}.configure_optimizers`."
            )

        self.model.configure_model()
        optimizer = instantiate_class(self.model.parameters(), optimizer_init)
        lr_scheduler = instantiate_class(optimizer, lr_scheduler_init) if lr_scheduler_init else None
        fn = partial(self.configure_optimizers, optimizer=optimizer, lr_scheduler=lr_scheduler)
        update_wrapper(fn, self.configure_optimizers)  # necessary for `is_overridden`
        # override the existing method
        self.model.configure_optimizers = MethodType(fn, self.model)

    def _get(self, config: Namespace, key: str, default: Optional[Any] = None) -> Any:
        """Utility to get a config value which might be inside a subcommand."""
        return config.get(str(self.subcommand), config).get(key, default)

    def _run_subcommand(self, subcommand: str) -> None:
        """Run the chosen subcommand."""
        before_fn = getattr(self, f"before_{subcommand}", None)
        if callable(before_fn):
            before_fn()

        default = getattr(self.trainer, subcommand)
        fn = getattr(self, subcommand, default)
        fn_kwargs = self._prepare_subcommand_kwargs(subcommand)
        fn(**fn_kwargs)

        after_fn = getattr(self, f"after_{subcommand}", None)
        if callable(after_fn):
            after_fn()

    def _prepare_subcommand_kwargs(self, subcommand: str) -> dict[str, Any]:
        """Prepares the keyword arguments to pass to the subcommand to run."""
        fn_kwargs = {
            k: v for k, v in self.config_init[subcommand].items() if k in self._subcommand_method_arguments[subcommand]
        }
        fn_kwargs["model"] = self.model
        if self.datamodule is not None:
            fn_kwargs["datamodule"] = self.datamodule
        return fn_kwargs

    def _set_seed(self) -> None:
        """Sets the seed."""
        config_seed = self._get(self.config, "seed_everything")
        if config_seed is False:
            return
        if config_seed is True:
            # user requested seeding, choose randomly
            config_seed = seed_everything(workers=True)
        else:
            config_seed = seed_everything(config_seed, workers=True)
        if self.subcommand:
            self.config[self.subcommand]["seed_everything"] = config_seed
        else:
            self.config["seed_everything"] = config_seed


def _class_path_from_class(class_type: type) -> str:
    return class_type.__module__ + "." + class_type.__name__


def _global_add_class_path(
    class_type: type, init_args: Optional[Union[Namespace, dict[str, Any]]] = None
) -> dict[str, Any]:
    if isinstance(init_args, Namespace):
        init_args = init_args.as_dict()
    return {"class_path": _class_path_from_class(class_type), "init_args": init_args or {}}


def _add_class_path_generator(class_type: type) -> Callable[[Namespace], dict[str, Any]]:
    def add_class_path(init_args: Namespace) -> dict[str, Any]:
        return _global_add_class_path(class_type, init_args)

    return add_class_path


def instantiate_class(args: Union[Any, tuple[Any, ...]], init: dict[str, Any]) -> Any:
    """Instantiates a class with the given args and init.

    Args:
        args: Positional arguments required for instantiation.
        init: Dict of the form {"class_path":...,"init_args":...}.

    Returns:
        The instantiated class object.

    """
    kwargs = init.get("init_args", {})
    if not isinstance(args, tuple):
        args = (args,)
    class_module, class_name = init["class_path"].rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    args_class = getattr(module, class_name)
    return args_class(*args, **kwargs)


def _get_short_description(component: object) -> Optional[str]:
    if component.__doc__ is None:
        return None
    try:
        docstring = docstring_parser.parse(component.__doc__)
        return docstring.short_description
    except (ValueError, docstring_parser.ParseError) as ex:
        rank_zero_warn(f"Failed parsing docstring for {component}: {ex}")


def _get_module_type(value: Union[Callable, type]) -> type:
    if callable(value) and not isinstance(value, type):
        return inspect.signature(value).return_annotation
    return value


class _InstantiatorFn:
    def __init__(self, cli: LightningCLI, key: str) -> None:
        self.cli = cli
        self.key = key

    def __call__(self, class_type: type[ModuleType], *args: Any, **kwargs: Any) -> ModuleType:
        hparams = self.cli.config_dump.get(self.key, {})
        if "class_path" in hparams:
            # To make hparams backwards compatible, and so that it is the same irrespective of subclass_mode, the
            # parameters are stored directly, and the class_path in a special key `_class_path` to clarify its internal
            # use.
            hparams = {
                "_class_path": hparams["class_path"],
                **hparams.get("init_args", {}),
                **hparams.get("dict_kwargs", {}),
            }
        with _given_hyperparameters_context(
            hparams=hparams,
            instantiator="lightning.pytorch.cli.instantiate_module",
        ):
            return class_type(*args, **kwargs)


def instantiate_module(class_type: type[ModuleType], config: dict[str, Any]) -> ModuleType:
    parser = ArgumentParser(exit_on_error=False)
    if "_class_path" in config:
        parser.add_subclass_arguments(class_type, "module", fail_untyped=False)
        config = {
            "class_path": config["_class_path"],
            "dict_kwargs": {k: v for k, v in config.items() if k != "_class_path"},
        }
    else:
        parser.add_class_arguments(class_type, "module", fail_untyped=False)
    cfg = parser.parse_object({"module": config})
    init = parser.instantiate_classes(cfg)
    return init.module
