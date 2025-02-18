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
Weights and Biases Logger
-------------------------
"""

import os
from argparse import Namespace
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import torch.nn as nn
from lightning_utilities.core.imports import RequirementCache
from torch import Tensor
from typing_extensions import override

from lightning.fabric.utilities.logger import (
    _add_prefix,
    _convert_json_serializable,
    _convert_params,
    _sanitize_callable_params,
)
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers.logger import Logger, rank_zero_experiment
from lightning.pytorch.loggers.utilities import _scan_checkpoints
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.rank_zero import rank_zero_only, rank_zero_warn

if TYPE_CHECKING:
    from wandb import Artifact
    from wandb.sdk.lib import RunDisabled
    from wandb.wandb_run import Run

_WANDB_AVAILABLE = RequirementCache("wandb>=0.12.10")


class WandbLogger(Logger):
    r"""Log using `Weights and Biases <https://docs.wandb.ai/guides/integrations/lightning>`_.

    **Installation and set-up**

    Install with pip:

    .. code-block:: bash

        pip install wandb

    Create a `WandbLogger` instance:

    .. code-block:: python

        from lightning.pytorch.loggers import WandbLogger

        wandb_logger = WandbLogger(project="MNIST")

    Pass the logger instance to the `Trainer`:

    .. code-block:: python

        trainer = Trainer(logger=wandb_logger)

    A new W&B run will be created when training starts if you have not created one manually before with `wandb.init()`.

    **Log metrics**

    Log from :class:`~lightning.pytorch.core.LightningModule`:

    .. code-block:: python

        class LitModule(LightningModule):
            def training_step(self, batch, batch_idx):
                self.log("train/loss", loss)

    Use directly wandb module:

    .. code-block:: python

        wandb.log({"train/loss": loss})

    **Log hyper-parameters**

    Save :class:`~lightning.pytorch.core.LightningModule` parameters:

    .. code-block:: python

        class LitModule(LightningModule):
            def __init__(self, *args, **kwarg):
                self.save_hyperparameters()

    Add other config parameters:

    .. code-block:: python

        # add one parameter
        wandb_logger.experiment.config["key"] = value

        # add multiple parameters
        wandb_logger.experiment.config.update({key1: val1, key2: val2})

        # use directly wandb module
        wandb.config["key"] = value
        wandb.config.update()

    **Log gradients, parameters and model topology**

    Call the `watch` method for automatically tracking gradients:

    .. code-block:: python

        # log gradients and model topology
        wandb_logger.watch(model)

        # log gradients, parameter histogram and model topology
        wandb_logger.watch(model, log="all")

        # change log frequency of gradients and parameters (100 steps by default)
        wandb_logger.watch(model, log_freq=500)

        # do not log graph (in case of errors)
        wandb_logger.watch(model, log_graph=False)

    The `watch` method adds hooks to the model which can be removed at the end of training:

    .. code-block:: python

        wandb_logger.experiment.unwatch(model)

    **Log model checkpoints**

    Log model checkpoints at the end of training:

    .. code-block:: python

        wandb_logger = WandbLogger(log_model=True)

    Log model checkpoints as they get created during training:

    .. code-block:: python

        wandb_logger = WandbLogger(log_model="all")

    Custom checkpointing can be set up through :class:`~lightning.pytorch.callbacks.ModelCheckpoint`:

    .. code-block:: python

        # log model only if `val_accuracy` increases
        wandb_logger = WandbLogger(log_model="all")
        checkpoint_callback = ModelCheckpoint(monitor="val_accuracy", mode="max")
        trainer = Trainer(logger=wandb_logger, callbacks=[checkpoint_callback])

    `latest` and `best` aliases are automatically set to easily retrieve a model checkpoint:

    .. code-block:: python

        # reference can be retrieved in artifacts panel
        # "VERSION" can be a version (ex: "v2") or an alias ("latest or "best")
        checkpoint_reference = "USER/PROJECT/MODEL-RUN_ID:VERSION"

        # download checkpoint locally (if not already cached)
        run = wandb.init(project="MNIST")
        artifact = run.use_artifact(checkpoint_reference, type="model")
        artifact_dir = artifact.download()

        # load checkpoint
        model = LitModule.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")

    **Log media**

    Log text with:

    .. code-block:: python

        # using columns and data
        columns = ["input", "label", "prediction"]
        data = [["cheese", "english", "english"], ["fromage", "french", "spanish"]]
        wandb_logger.log_text(key="samples", columns=columns, data=data)

        # using a pandas DataFrame
        wandb_logger.log_text(key="samples", dataframe=my_dataframe)

    Log images with:

    .. code-block:: python

        # using tensors, numpy arrays or PIL images
        wandb_logger.log_image(key="samples", images=[img1, img2])

        # adding captions
        wandb_logger.log_image(key="samples", images=[img1, img2], caption=["tree", "person"])

        # using file path
        wandb_logger.log_image(key="samples", images=["img_1.jpg", "img_2.jpg"])

    More arguments can be passed for logging segmentation masks and bounding boxes. Refer to
    `Image Overlays documentation <https://docs.wandb.ai/guides/track/log/media#image-overlays>`_.

    **Log Tables**

    `W&B Tables <https://docs.wandb.ai/guides/tables/visualize-tables>`_ can be used to log,
    query and analyze tabular data.

    They support any type of media (text, image, video, audio, molecule, html, etc) and are great for storing,
    understanding and sharing any form of data, from datasets to model predictions.

    .. code-block:: python

        columns = ["caption", "image", "sound"]
        data = [["cheese", wandb.Image(img_1), wandb.Audio(snd_1)], ["wine", wandb.Image(img_2), wandb.Audio(snd_2)]]
        wandb_logger.log_table(key="samples", columns=columns, data=data)


    **Downloading and Using Artifacts**

    To download an artifact without starting a run, call the ``download_artifact``
    function on the class:

    .. code-block:: python

        from lightning.pytorch.loggers import WandbLogger

        artifact_dir = WandbLogger.download_artifact(artifact="path/to/artifact")

    To download an artifact and link it to an ongoing run call the ``download_artifact``
    function on the logger instance:

    .. code-block:: python

        class MyModule(LightningModule):
            def any_lightning_module_function_or_hook(self):
                self.logger.download_artifact(artifact="path/to/artifact")

    To link an artifact from a previous run you can use ``use_artifact`` function:

    .. code-block:: python

        from lightning.pytorch.loggers import WandbLogger

        wandb_logger = WandbLogger(project="my_project", name="my_run")
        wandb_logger.use_artifact(artifact="path/to/artifact")

    See Also:
        - `Demo in Google Colab <http://wandb.me/lightning>`__ with hyperparameter search and model logging
        - `W&B Documentation <https://docs.wandb.ai/guides/integrations/lightning>`__

    Args:
        name: Display name for the run.
        save_dir: Path where data is saved.
        version: Sets the version, mainly used to resume a previous run.
        offline: Run offline (data can be streamed later to wandb servers).
        dir: Same as save_dir.
        id: Same as version.
        anonymous: Enables or explicitly disables anonymous logging.
        project: The name of the project to which this run will belong. If not set, the environment variable
            `WANDB_PROJECT` will be used as a fallback. If both are not set, it defaults to ``'lightning_logs'``.
        log_model: Log checkpoints created by :class:`~lightning.pytorch.callbacks.ModelCheckpoint`
            as W&B artifacts. `latest` and `best` aliases are automatically set.

            * if ``log_model == 'all'``, checkpoints are logged during training.
            * if ``log_model == True``, checkpoints are logged at the end of training, except when
              :paramref:`~lightning.pytorch.callbacks.ModelCheckpoint.save_top_k` ``== -1``
              which also logs every checkpoint during training.
            * if ``log_model == False`` (default), no checkpoint is logged.

        prefix: A string to put at the beginning of metric keys.
        experiment: WandB experiment object. Automatically set when creating a run.
        checkpoint_name: Name of the model checkpoint artifact being logged.
        add_file_policy: If "mutable", copies file to tempdirectory before upload.
        \**kwargs: Arguments passed to :func:`wandb.init` like `entity`, `group`, `tags`, etc.

    Raises:
        ModuleNotFoundError:
            If required WandB package is not installed on the device.
        MisconfigurationException:
            If both ``log_model`` and ``offline`` is set to ``True``.

    """

    LOGGER_JOIN_CHAR = "-"

    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: _PATH = ".",
        version: Optional[str] = None,
        offline: bool = False,
        dir: Optional[_PATH] = None,
        id: Optional[str] = None,
        anonymous: Optional[bool] = None,
        project: Optional[str] = None,
        log_model: Union[Literal["all"], bool] = False,
        experiment: Union["Run", "RunDisabled", None] = None,
        prefix: str = "",
        checkpoint_name: Optional[str] = None,
        add_file_policy: Literal["mutable", "immutable"] = "mutable",
        **kwargs: Any,
    ) -> None:
        if not _WANDB_AVAILABLE:
            raise ModuleNotFoundError(str(_WANDB_AVAILABLE))

        if offline and log_model:
            raise MisconfigurationException(
                f"Providing log_model={log_model} and offline={offline} is an invalid configuration"
                " since model checkpoints cannot be uploaded in offline mode.\n"
                "Hint: Set `offline=False` to log your model."
            )

        super().__init__()
        self._offline = offline
        self._log_model = log_model
        self._prefix = prefix
        self._experiment = experiment
        self._logged_model_time: dict[str, float] = {}
        self._checkpoint_callback: Optional[ModelCheckpoint] = None
        self.add_file_policy = add_file_policy

        # paths are processed as strings
        if save_dir is not None:
            save_dir = os.fspath(save_dir)
        elif dir is not None:
            dir = os.fspath(dir)

        project = project or os.environ.get("WANDB_PROJECT", "lightning_logs")

        # set wandb init arguments
        self._wandb_init: dict[str, Any] = {
            "name": name,
            "project": project,
            "dir": save_dir or dir,
            "id": version or id,
            "resume": "allow",
            "anonymous": ("allow" if anonymous else None),
        }
        self._wandb_init.update(**kwargs)
        # extract parameters
        self._project = self._wandb_init.get("project")
        self._save_dir = self._wandb_init.get("dir")
        self._name = self._wandb_init.get("name")
        self._id = self._wandb_init.get("id")
        self._checkpoint_name = checkpoint_name

    def __getstate__(self) -> dict[str, Any]:
        import wandb

        # Hack: If the 'spawn' launch method is used, the logger will get pickled and this `__getstate__` gets called.
        # We create an experiment here in the main process, and attach to it in the worker process.
        # Using wandb-service, we persist the same experiment even if multiple `Trainer.fit/test/validate` calls
        # are made.
        wandb.require("service")
        _ = self.experiment

        state = self.__dict__.copy()
        # args needed to reload correct experiment
        if self._experiment is not None:
            state["_id"] = getattr(self._experiment, "id", None)
            state["_attach_id"] = getattr(self._experiment, "_attach_id", None)
            state["_name"] = self._experiment.name

        # cannot be pickled
        state["_experiment"] = None
        return state

    @property
    @rank_zero_experiment
    def experiment(self) -> Union["Run", "RunDisabled"]:
        r"""Actual wandb object. To use wandb features in your :class:`~lightning.pytorch.core.LightningModule` do the
        following.

        Example::

        .. code-block:: python

            self.logger.experiment.some_wandb_function()

        """
        import wandb
        from wandb.sdk.lib import RunDisabled
        from wandb.wandb_run import Run

        if self._experiment is None:
            if self._offline:
                os.environ["WANDB_MODE"] = "dryrun"

            attach_id = getattr(self, "_attach_id", None)
            if wandb.run is not None:
                # wandb process already created in this instance
                rank_zero_warn(
                    "There is a wandb run already in progress and newly created instances of `WandbLogger` will reuse"
                    " this run. If this is not desired, call `wandb.finish()` before instantiating `WandbLogger`."
                )
                self._experiment = wandb.run
            elif attach_id is not None and hasattr(wandb, "_attach"):
                # attach to wandb process referenced
                self._experiment = wandb._attach(attach_id)
            else:
                # create new wandb process
                self._experiment = wandb.init(**self._wandb_init)

                # define default x-axis
                if isinstance(self._experiment, (Run, RunDisabled)) and getattr(
                    self._experiment, "define_metric", None
                ):
                    self._experiment.define_metric("trainer/global_step")
                    self._experiment.define_metric("*", step_metric="trainer/global_step", step_sync=True)

        return self._experiment

    def watch(
        self, model: nn.Module, log: Optional[str] = "gradients", log_freq: int = 100, log_graph: bool = True
    ) -> None:
        self.experiment.watch(model, log=log, log_freq=log_freq, log_graph=log_graph)

    @override
    @rank_zero_only
    def log_hyperparams(self, params: Union[dict[str, Any], Namespace]) -> None:
        params = _convert_params(params)
        params = _sanitize_callable_params(params)
        params = _convert_json_serializable(params)
        self.experiment.config.update(params, allow_val_change=True)

    @override
    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"

        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)
        if step is not None:
            self.experiment.log(dict(metrics, **{"trainer/global_step": step}))
        else:
            self.experiment.log(metrics)

    @rank_zero_only
    def log_table(
        self,
        key: str,
        columns: Optional[list[str]] = None,
        data: Optional[list[list[Any]]] = None,
        dataframe: Any = None,
        step: Optional[int] = None,
    ) -> None:
        """Log a Table containing any object type (text, image, audio, video, molecule, html, etc).

        Can be defined either with `columns` and `data` or with `dataframe`.

        """
        import wandb

        metrics = {key: wandb.Table(columns=columns, data=data, dataframe=dataframe)}
        self.log_metrics(metrics, step)

    @rank_zero_only
    def log_text(
        self,
        key: str,
        columns: Optional[list[str]] = None,
        data: Optional[list[list[str]]] = None,
        dataframe: Any = None,
        step: Optional[int] = None,
    ) -> None:
        """Log text as a Table.

        Can be defined either with `columns` and `data` or with `dataframe`.

        """

        self.log_table(key, columns, data, dataframe, step)

    @rank_zero_only
    def log_image(self, key: str, images: list[Any], step: Optional[int] = None, **kwargs: Any) -> None:
        """Log images (tensors, numpy arrays, PIL Images or file paths).

        Optional kwargs are lists passed to each image (ex: caption, masks, boxes).

        """
        if not isinstance(images, list):
            raise TypeError(f'Expected a list as "images", found {type(images)}')
        n = len(images)
        for k, v in kwargs.items():
            if len(v) != n:
                raise ValueError(f"Expected {n} items but only found {len(v)} for {k}")
        kwarg_list = [{k: kwargs[k][i] for k in kwargs} for i in range(n)]

        import wandb

        metrics = {key: [wandb.Image(img, **kwarg) for img, kwarg in zip(images, kwarg_list)]}
        self.log_metrics(metrics, step)  # type: ignore[arg-type]

    @rank_zero_only
    def log_audio(self, key: str, audios: list[Any], step: Optional[int] = None, **kwargs: Any) -> None:
        r"""Log audios (numpy arrays, or file paths).

        Args:
            key: The key to be used for logging the audio files
            audios: The list of audio file paths, or numpy arrays to be logged
            step: The step number to be used for logging the audio files
            \**kwargs: Optional kwargs are lists passed to each ``Wandb.Audio`` instance (ex: caption, sample_rate).

        Optional kwargs are lists passed to each audio (ex: caption, sample_rate).

        """
        if not isinstance(audios, list):
            raise TypeError(f'Expected a list as "audios", found {type(audios)}')
        n = len(audios)
        for k, v in kwargs.items():
            if len(v) != n:
                raise ValueError(f"Expected {n} items but only found {len(v)} for {k}")
        kwarg_list = [{k: kwargs[k][i] for k in kwargs} for i in range(n)]

        import wandb

        metrics = {key: [wandb.Audio(audio, **kwarg) for audio, kwarg in zip(audios, kwarg_list)]}
        self.log_metrics(metrics, step)  # type: ignore[arg-type]

    @rank_zero_only
    def log_video(self, key: str, videos: list[Any], step: Optional[int] = None, **kwargs: Any) -> None:
        """Log videos (numpy arrays, or file paths).

        Args:
            key: The key to be used for logging the video files
            videos: The list of video file paths, or numpy arrays to be logged
            step: The step number to be used for logging the video files
            **kwargs: Optional kwargs are lists passed to each Wandb.Video instance (ex: caption, fps, format).

        Optional kwargs are lists passed to each video (ex: caption, fps, format).

        """
        if not isinstance(videos, list):
            raise TypeError(f'Expected a list as "videos", found {type(videos)}')
        n = len(videos)
        for k, v in kwargs.items():
            if len(v) != n:
                raise ValueError(f"Expected {n} items but only found {len(v)} for {k}")
        kwarg_list = [{k: kwargs[k][i] for k in kwargs} for i in range(n)]

        import wandb

        metrics = {key: [wandb.Video(video, **kwarg) for video, kwarg in zip(videos, kwarg_list)]}
        self.log_metrics(metrics, step)  # type: ignore[arg-type]

    @property
    @override
    def save_dir(self) -> Optional[str]:
        """Gets the save directory.

        Returns:
            The path to the save directory.

        """
        return self._save_dir

    @property
    @override
    def name(self) -> Optional[str]:
        """The project name of this experiment.

        Returns:
            The name of the project the current experiment belongs to. This name is not the same as `wandb.Run`'s
            name. To access wandb's internal experiment name, use ``logger.experiment.name`` instead.

        """
        return self._project

    @property
    @override
    def version(self) -> Optional[str]:
        """Gets the id of the experiment.

        Returns:
            The id of the experiment if the experiment exists else the id given to the constructor.

        """
        # don't create an experiment if we don't have one
        return self._experiment.id if self._experiment else self._id

    @override
    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        # log checkpoints as artifacts
        if self._log_model == "all" or self._log_model is True and checkpoint_callback.save_top_k == -1:
            self._scan_and_log_checkpoints(checkpoint_callback)
        elif self._log_model is True:
            self._checkpoint_callback = checkpoint_callback

    @staticmethod
    @rank_zero_only
    def download_artifact(
        artifact: str,
        save_dir: Optional[_PATH] = None,
        artifact_type: Optional[str] = None,
        use_artifact: Optional[bool] = True,
    ) -> str:
        """Downloads an artifact from the wandb server.

        Args:
            artifact: The path of the artifact to download.
            save_dir: The directory to save the artifact to.
            artifact_type: The type of artifact to download.
            use_artifact: Whether to add an edge between the artifact graph.

        Returns:
            The path to the downloaded artifact.

        """
        import wandb

        if wandb.run is not None and use_artifact:
            artifact = wandb.run.use_artifact(artifact)
        else:
            api = wandb.Api()
            artifact = api.artifact(artifact, type=artifact_type)

        save_dir = None if save_dir is None else os.fspath(save_dir)
        return artifact.download(root=save_dir)

    def use_artifact(self, artifact: str, artifact_type: Optional[str] = None) -> "Artifact":
        """Logs to the wandb dashboard that the mentioned artifact is used by the run.

        Args:
            artifact: The path of the artifact.
            artifact_type: The type of artifact being used.

        Returns:
            wandb Artifact object for the artifact.

        """
        return self.experiment.use_artifact(artifact, type=artifact_type)

    @override
    @rank_zero_only
    def finalize(self, status: str) -> None:
        if status != "success":
            # Currently, checkpoints only get logged on success
            return
        # log checkpoints as artifacts
        if self._checkpoint_callback and self._experiment is not None:
            self._scan_and_log_checkpoints(self._checkpoint_callback)

    def _scan_and_log_checkpoints(self, checkpoint_callback: ModelCheckpoint) -> None:
        import wandb

        # get checkpoints to be saved with associated score
        checkpoints = _scan_checkpoints(checkpoint_callback, self._logged_model_time)

        # log iteratively all new checkpoints
        for t, p, s, tag in checkpoints:
            metadata = {
                "score": s.item() if isinstance(s, Tensor) else s,
                "original_filename": Path(p).name,
                checkpoint_callback.__class__.__name__: {
                    k: getattr(checkpoint_callback, k)
                    for k in [
                        "monitor",
                        "mode",
                        "save_last",
                        "save_top_k",
                        "save_weights_only",
                        "_every_n_train_steps",
                    ]
                    # ensure it does not break if `ModelCheckpoint` args change
                    if hasattr(checkpoint_callback, k)
                },
            }
            if not self._checkpoint_name:
                self._checkpoint_name = f"model-{self.experiment.id}"
            artifact = wandb.Artifact(name=self._checkpoint_name, type="model", metadata=metadata)
            artifact.add_file(p, name="model.ckpt", policy=self.add_file_policy)
            aliases = ["latest", "best"] if p == checkpoint_callback.best_model_path else ["latest"]
            self.experiment.log_artifact(artifact, aliases=aliases)
            # remember logged models - timestamp needed in case filename didn't change (lastkckpt or custom name)
            self._logged_model_time[p] = t
