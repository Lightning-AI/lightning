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
"""
Neptune Logger
--------------
"""

import contextlib
import logging
import os
from argparse import Namespace
from collections.abc import Generator
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from lightning_utilities.core.imports import RequirementCache
from torch import Tensor
from typing_extensions import override

import lightning.pytorch as pl
from lightning.fabric.utilities.logger import _add_prefix, _convert_params, _sanitize_callable_params
from lightning.pytorch.callbacks import Checkpoint
from lightning.pytorch.loggers.logger import Logger, rank_zero_experiment
from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch.utilities.rank_zero import rank_zero_only

if TYPE_CHECKING:
    from neptune import Run
    from neptune.handler import Handler

log = logging.getLogger(__name__)

# Neptune is available with two names on PyPI : `neptune` and `neptune-client`
# `neptune` was introduced as a name transition of neptune-client and the long-term target is to get
# rid of Neptune-client package completely someday. It was introduced as a part of breaking-changes with a release
# of neptune-client==1.0. neptune-client>=1.0 is just an alias of neptune package and have some breaking-changes
# in compare to neptune-client<1.0.0.
_NEPTUNE_AVAILABLE = RequirementCache("neptune>=1.0")
_INTEGRATION_VERSION_KEY = "source_code/integrations/pytorch-lightning"


# Neptune client throws `InactiveRunException` when trying to log to an inactive run.
# This may happen when the run was stopped through the UI and the logger is still trying to log to it.
def _catch_inactive(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        from neptune.exceptions import InactiveRunException

        with contextlib.suppress(InactiveRunException):
            return func(*args, **kwargs)

    return wrapper


class NeptuneLogger(Logger):
    r"""Log using `Neptune <https://docs.neptune.ai/integrations/lightning/>`_.

    Install it with pip:

    .. code-block:: bash

        pip install neptune

    or conda:

    .. code-block:: bash

        conda install -c conda-forge neptune-client

    **Quickstart**

    Pass a NeptuneLogger instance to the Trainer to log metadata with Neptune:

    .. code-block:: python


        from lightning.pytorch import Trainer
        from lightning.pytorch.loggers import NeptuneLogger
        import neptune

        neptune_logger = NeptuneLogger(
            api_key=neptune.ANONYMOUS_API_TOKEN,  # replace with your own
            project="common/pytorch-lightning-integration",  # format "workspace-name/project-name"
            tags=["training", "resnet"],  # optional
        )
        trainer = Trainer(max_epochs=10, logger=neptune_logger)

    **How to use NeptuneLogger?**

    Use the logger anywhere in your :class:`~lightning.pytorch.core.LightningModule` as follows:

    .. code-block:: python

        from neptune.types import File
        from lightning.pytorch import LightningModule


        class LitModel(LightningModule):
            def training_step(self, batch, batch_idx):
                # log metrics
                acc = ...
                self.append("train/loss", loss)

            def any_lightning_module_function_or_hook(self):
                # log images
                img = ...
                self.logger.experiment["train/misclassified_images"].append(File.as_image(img))

                # generic recipe
                metadata = ...
                self.logger.experiment["your/metadata/structure"] = metadata

    Note that the syntax ``self.logger.experiment["your/metadata/structure"].append(metadata)`` is specific to
    Neptune and extends the logger capabilities. It lets you log various types of metadata, such as
    scores, files, images, interactive visuals, and CSVs.
    Refer to the `Neptune docs <https://docs.neptune.ai/logging/methods>`_
    for details.
    You can also use the regular logger methods ``log_metrics()``, and ``log_hyperparams()`` with NeptuneLogger.

    **Log after fitting or testing is finished**

    You can log objects after the fitting or testing methods are finished:

    .. code-block:: python

        neptune_logger = NeptuneLogger(project="common/pytorch-lightning-integration")

        trainer = pl.Trainer(logger=neptune_logger)
        model = ...
        datamodule = ...
        trainer.fit(model, datamodule=datamodule)
        trainer.test(model, datamodule=datamodule)

        # Log objects after `fit` or `test` methods
        # model summary
        neptune_logger.log_model_summary(model=model, max_depth=-1)

        # generic recipe
        metadata = ...
        neptune_logger.experiment["your/metadata/structure"] = metadata

    **Log model checkpoints**

    If you have :class:`~lightning.pytorch.callbacks.ModelCheckpoint` configured,
    the Neptune logger automatically logs model checkpoints.
    Model weights will be uploaded to the "model/checkpoints" namespace in the Neptune run.
    You can disable this option with:

    .. code-block:: python

        neptune_logger = NeptuneLogger(log_model_checkpoints=False)

    **Pass additional parameters to the Neptune run**

    You can also pass ``neptune_run_kwargs`` to add details to the run, like ``tags`` or ``description``:

    .. testcode::
        :skipif: not _NEPTUNE_AVAILABLE

        from lightning.pytorch import Trainer
        from lightning.pytorch.loggers import NeptuneLogger

        neptune_logger = NeptuneLogger(
            project="common/pytorch-lightning-integration",
            name="lightning-run",
            description="mlp quick run with pytorch-lightning",
            tags=["mlp", "quick-run"],
        )
        trainer = Trainer(max_epochs=3, logger=neptune_logger)

    Check `run documentation <https://docs.neptune.ai/api/neptune/#init_run>`_
    for more info about additional run parameters.

    **Details about Neptune run structure**

    Runs can be viewed as nested dictionary-like structures that you can define in your code.
    Thanks to this you can easily organize your metadata in a way that is most convenient for you.

    The hierarchical structure that you apply to your metadata is reflected in the Neptune web app.

    See also:
        - Read about
          `what objects you can log to Neptune <https://docs.neptune.ai/logging/what_you_can_log/>`_.
        - Check out an `example run <https://app.neptune.ai/o/common/org/pytorch-lightning-integration/e/PTL-1/all>`_
          with multiple types of metadata logged.
        - For more detailed examples, see the
          `user guide <https://docs.neptune.ai/integrations/lightning/>`_.

    Args:
        api_key: Optional.
            Neptune API token, found on https://www.neptune.ai upon registration.
            You should save your token to the `NEPTUNE_API_TOKEN`
            environment variable and leave the api_key argument out of your code.
            Instructions: `Setting your API token <https://docs.neptune.ai/setup/setting_api_token/>`_.
        project: Optional.
            Name of a project in the form "workspace-name/project-name", for example "tom/mask-rcnn".
            If ``None``, the value of `NEPTUNE_PROJECT` environment variable is used.
            You need to create the project on https://www.neptune.ai first.
        name: Optional. Editable name of the run.
            The run name is displayed in the Neptune web app.
        run: Optional. Default is ``None``. A Neptune ``Run`` object.
            If specified, this existing run will be used for logging, instead of a new run being created.
            You can also pass a namespace handler object; for example, ``run["test"]``, in which case all
            metadata is logged under the "test" namespace inside the run.
        log_model_checkpoints: Optional. Default is ``True``. Log model checkpoint to Neptune.
            Works only if ``ModelCheckpoint`` is passed to the ``Trainer``.
        prefix: Optional. Default is ``"training"``. Root namespace for all metadata logging.
        \**neptune_run_kwargs: Additional arguments like ``tags``, ``description``, ``capture_stdout``, etc.
            used when a run is created.

    Raises:
        ModuleNotFoundError:
            If the required Neptune package is not installed.
        ValueError:
            If an argument passed to the logger's constructor is incorrect.

    """

    LOGGER_JOIN_CHAR = "/"
    PARAMETERS_KEY = "hyperparams"
    ARTIFACTS_KEY = "artifacts"

    def __init__(
        self,
        *,  # force users to call `NeptuneLogger` initializer with `kwargs`
        api_key: Optional[str] = None,
        project: Optional[str] = None,
        name: Optional[str] = None,
        run: Optional[Union["Run", "Handler"]] = None,
        log_model_checkpoints: Optional[bool] = True,
        prefix: str = "training",
        **neptune_run_kwargs: Any,
    ):
        if not _NEPTUNE_AVAILABLE:
            raise ModuleNotFoundError(str(_NEPTUNE_AVAILABLE))

        # verify if user passed proper init arguments
        self._verify_input_arguments(api_key, project, name, run, neptune_run_kwargs)
        super().__init__()
        self._log_model_checkpoints = log_model_checkpoints
        self._prefix = prefix
        self._run_name = name
        self._project_name = project
        self._api_key = api_key
        self._run_instance = run
        self._neptune_run_kwargs = neptune_run_kwargs
        self._run_short_id: Optional[str] = None

        if self._run_instance is not None:
            self._retrieve_run_data()

            from neptune.handler import Handler

            # make sure that we've log integration version for outside `Run` instances
            root_obj = self._run_instance
            if isinstance(root_obj, Handler):
                root_obj = root_obj.get_root_object()

            root_obj[_INTEGRATION_VERSION_KEY] = pl.__version__
        self._uploaded_models: Set[str] = set()

    def _retrieve_run_data(self) -> None:
        from neptune.handler import Handler

        assert self._run_instance is not None
        root_obj = self._run_instance
        if isinstance(root_obj, Handler):
            root_obj = root_obj.get_root_object()

        root_obj.wait()

        if root_obj.exists("sys/id"):
            self._run_short_id = root_obj["sys/id"].fetch()
            self._run_name = root_obj["sys/name"].fetch()
        else:
            self._run_short_id = "OFFLINE"
            self._run_name = "offline-name"

    @property
    def _neptune_init_args(self) -> dict:
        args: dict = {}
        # Backward compatibility in case of previous version retrieval
        with contextlib.suppress(AttributeError):
            args = self._neptune_run_kwargs

        if self._project_name is not None:
            args["project"] = self._project_name

        if self._api_key is not None:
            args["api_token"] = self._api_key

        if self._run_short_id is not None:
            args["run"] = self._run_short_id

        # Backward compatibility in case of previous version retrieval
        with contextlib.suppress(AttributeError):
            if self._run_name is not None:
                args["name"] = self._run_name

        return args

    def _construct_path_with_prefix(self, *keys: str) -> str:
        """Return sequence of keys joined by `LOGGER_JOIN_CHAR`, started with `_prefix` if defined."""
        if self._prefix:
            return self.LOGGER_JOIN_CHAR.join([self._prefix, *keys])
        return self.LOGGER_JOIN_CHAR.join(keys)

    @staticmethod
    def _verify_input_arguments(
        api_key: Optional[str],
        project: Optional[str],
        name: Optional[str],
        run: Optional[Union["Run", "Handler"]],
        neptune_run_kwargs: dict,
    ) -> None:
        from neptune import Run
        from neptune.handler import Handler

        # check if user passed the client `Run`/`Handler` object
        if run is not None and not isinstance(run, (Run, Handler)):
            raise ValueError("Run parameter expected to be of type `neptune.Run`, or `neptune.handler.Handler`.")

        # check if user passed redundant neptune.init_run arguments when passed run
        any_neptune_init_arg_passed = any(arg is not None for arg in [api_key, project, name]) or neptune_run_kwargs
        if run is not None and any_neptune_init_arg_passed:
            raise ValueError(
                "When an already initialized run object is provided, you can't provide other `neptune.init_run()`"
                " parameters."
            )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # Run instance can't be pickled
        state["_run_instance"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        import neptune

        self.__dict__ = state
        self._run_instance = neptune.init_run(**self._neptune_init_args)

    @property
    @rank_zero_experiment
    def experiment(self) -> "Run":
        r"""Actual Neptune run object. Allows you to use neptune logging features in your
        :class:`~lightning.pytorch.core.LightningModule`.

        Example::

            class LitModel(LightningModule):
                def training_step(self, batch, batch_idx):
                    # log metrics
                    acc = ...
                    self.logger.experiment["train/acc"].append(acc)

                    # log images
                    img = ...
                    self.logger.experiment["train/misclassified_images"].append(File.as_image(img))

        Note that the syntax ``self.logger.experiment["your/metadata/structure"].append(metadata)``
        is specific to Neptune and extends the logger capabilities.
        It lets you log various types of metadata, such as scores, files,
        images, interactive visuals, and CSVs. Refer to the
        `Neptune docs <https://docs.neptune.ai/logging/methods>`_
        for more detailed explanations.
        You can also use the regular logger methods ``log_metrics()``, and ``log_hyperparams()``
        with NeptuneLogger.

        """
        return self.run

    @property
    @rank_zero_experiment
    def run(self) -> "Run":
        import neptune

        if not self._run_instance:
            self._run_instance = neptune.init_run(**self._neptune_init_args)
            self._retrieve_run_data()
            # make sure that we've log integration version for newly created
            self._run_instance[_INTEGRATION_VERSION_KEY] = pl.__version__

        return self._run_instance

    @override
    @rank_zero_only
    @_catch_inactive
    def log_hyperparams(self, params: Union[dict[str, Any], Namespace]) -> None:
        r"""Log hyperparameters to the run.

        Hyperparameters will be logged under the "<prefix>/hyperparams" namespace.

        Note:

            You can also log parameters by directly using the logger instance:
            ``neptune_logger.experiment["model/hyper-parameters"] = params_dict``.

            In this way you can keep hierarchical structure of the parameters.

        Args:
            params: `dict`.
                Python dictionary structure with parameters.

        Example::

            from lightning.pytorch.loggers import NeptuneLogger
            import neptune

            PARAMS = {
                "batch_size": 64,
                "lr": 0.07,
                "decay_factor": 0.97,
            }

            neptune_logger = NeptuneLogger(
                api_key=neptune.ANONYMOUS_API_TOKEN,
                project="common/pytorch-lightning-integration"
            )

            neptune_logger.log_hyperparams(PARAMS)

        """
        from neptune.utils import stringify_unsupported

        params = _convert_params(params)
        params = _sanitize_callable_params(params)

        parameters_key = self.PARAMETERS_KEY
        parameters_key = self._construct_path_with_prefix(parameters_key)

        self.run[parameters_key] = stringify_unsupported(params)

    @override
    @rank_zero_only
    @_catch_inactive
    def log_metrics(self, metrics: dict[str, Union[Tensor, float]], step: Optional[int] = None) -> None:
        """Log metrics (numeric values) in Neptune runs.

        Args:
            metrics: Dictionary with metric names as keys and measured quantities as values.
            step: Step number at which the metrics should be recorded

        """
        if rank_zero_only.rank != 0:
            raise ValueError("run tried to log from global_rank != 0")

        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)

        for key, val in metrics.items():
            self.run[key].append(val, step=step)

    @override
    @rank_zero_only
    @_catch_inactive
    def finalize(self, status: str) -> None:
        if not self._run_instance:
            # When using multiprocessing, finalize() should be a no-op on the main process, as no experiment has been
            # initialized there
            return
        if status:
            self.run[self._construct_path_with_prefix("status")] = status

        super().finalize(status)

    @property
    @override
    def save_dir(self) -> Optional[str]:
        """Gets the save directory of the experiment which in this case is ``None`` because Neptune does not save
        locally.

        Returns:
            the root directory where experiment logs get saved

        """
        return os.path.join(os.getcwd(), ".neptune")

    @rank_zero_only
    @_catch_inactive
    def log_model_summary(self, model: "pl.LightningModule", max_depth: int = -1) -> None:
        from neptune.types import File

        model_str = str(ModelSummary(model=model, max_depth=max_depth))
        self.run[self._construct_path_with_prefix("model/summary")] = File.from_content(
            content=model_str, extension="txt"
        )

    @override
    @rank_zero_only
    @_catch_inactive
    def after_save_checkpoint(self, checkpoint_callback: Checkpoint) -> None:
        """Automatically log checkpointed model. Called after model checkpoint callback saves a new checkpoint.

        Args:
            checkpoint_callback: the model checkpoint callback instance

        """
        if not self._log_model_checkpoints:
            return

        from neptune.types import File

        file_names = set()
        checkpoints_namespace = self._construct_path_with_prefix("model/checkpoints")

        # save last model
        if hasattr(checkpoint_callback, "last_model_path") and checkpoint_callback.last_model_path:
            model_last_name = self._get_full_model_name(checkpoint_callback.last_model_path, checkpoint_callback)
            file_names.add(model_last_name)
            with open(checkpoint_callback.last_model_path, "rb") as fp:
                self.run[f"{checkpoints_namespace}/{model_last_name}"] = File.from_stream(fp)

        # save best k models
        if hasattr(checkpoint_callback, "best_k_models"):
            for key in checkpoint_callback.best_k_models:
                model_name = self._get_full_model_name(key, checkpoint_callback)
                if model_name not in self._uploaded_models:
                    file_names.add(model_name)
                    self._uploaded_models.add(model_name)
                    self.run[f"{checkpoints_namespace}/{model_name}"].upload(key)

        # log best model path and checkpoint
        if hasattr(checkpoint_callback, "best_model_path") and checkpoint_callback.best_model_path:
            self.run[self._construct_path_with_prefix("model/best_model_path")] = checkpoint_callback.best_model_path

            model_name = self._get_full_model_name(checkpoint_callback.best_model_path, checkpoint_callback)
            file_names.add(model_name)
            with open(checkpoint_callback.best_model_path, "rb") as fp:
                self.run[f"{checkpoints_namespace}/{model_name}"] = File.from_stream(fp)

        # remove old models logged to experiment if they are not part of best k models at this point
        if self.run.exists(checkpoints_namespace):
            exp_structure = self.run.get_structure()
            uploaded_model_names = self._get_full_model_names_from_exp_structure(exp_structure, checkpoints_namespace)

            for file_to_drop in list(uploaded_model_names - file_names):
                del self.run[f"{checkpoints_namespace}/{file_to_drop}"]
                self._uploaded_models.remove(file_to_drop)

        # log best model score
        if hasattr(checkpoint_callback, "best_model_score") and checkpoint_callback.best_model_score:
            self.run[self._construct_path_with_prefix("model/best_model_score")] = (
                checkpoint_callback.best_model_score.cpu().detach().numpy()
            )

    @staticmethod
    def _get_full_model_name(model_path: str, checkpoint_callback: Checkpoint) -> str:
        """Returns model name which is string `model_path` appended to `checkpoint_callback.dirpath`."""
        if hasattr(checkpoint_callback, "dirpath"):
            model_path = os.path.normpath(model_path)
            expected_model_path = os.path.normpath(checkpoint_callback.dirpath)
            if not model_path.startswith(expected_model_path):
                raise ValueError(f"{model_path} was expected to start with {expected_model_path}.")
            # Remove extension from filepath
            filepath, _ = os.path.splitext(model_path[len(expected_model_path) + 1 :])
            return filepath.replace(os.sep, "/")
        return model_path.replace(os.sep, "/")

    @classmethod
    def _get_full_model_names_from_exp_structure(cls, exp_structure: dict[str, Any], namespace: str) -> set[str]:
        """Returns all paths to properties which were already logged in `namespace`"""
        structure_keys: list[str] = namespace.split(cls.LOGGER_JOIN_CHAR)
        for key in structure_keys:
            exp_structure = exp_structure[key]
        uploaded_models_dict = exp_structure
        return set(cls._dict_paths(uploaded_models_dict))

    @classmethod
    def _dict_paths(cls, d: dict[str, Any], path_in_build: Optional[str] = None) -> Generator:
        for k, v in d.items():
            path = f"{path_in_build}/{k}" if path_in_build is not None else k
            if not isinstance(v, dict):
                yield path
            else:
                yield from cls._dict_paths(v, path)

    @property
    @override
    def name(self) -> Optional[str]:
        """Return the experiment name or 'offline-name' when exp is run in offline mode."""
        return self._run_name

    @property
    @override
    def version(self) -> Optional[str]:
        """Return the experiment version.

        It's Neptune Run's short_id

        """
        return self._run_short_id
