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
MLflow Logger
-------------
"""

import logging
import os
import re
import tempfile
from argparse import Namespace
from collections.abc import Mapping
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Union

import yaml
from lightning_utilities.core.imports import RequirementCache
from torch import Tensor
from typing_extensions import override

from lightning.fabric.utilities.logger import _add_prefix, _convert_params, _flatten_dict
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers.logger import Logger, rank_zero_experiment
from lightning.pytorch.loggers.utilities import _scan_checkpoints
from lightning.pytorch.utilities.rank_zero import rank_zero_only, rank_zero_warn

if TYPE_CHECKING:
    from mlflow.tracking import MlflowClient

log = logging.getLogger(__name__)
LOCAL_FILE_URI_PREFIX = "file:"
_MLFLOW_AVAILABLE = RequirementCache("mlflow>=1.0.0", "mlflow")
_MLFLOW_SYNCHRONOUS_AVAILABLE = RequirementCache("mlflow>=2.8.0", "mlflow")


class MLFlowLogger(Logger):
    """Log using `MLflow <https://mlflow.org>`_.

    Install it with pip:

    .. code-block:: bash

        pip install mlflow  # or mlflow-skinny

    .. code-block:: python

        from lightning.pytorch import Trainer
        from lightning.pytorch.loggers import MLFlowLogger

        mlf_logger = MLFlowLogger(experiment_name="lightning_logs", tracking_uri="file:./ml-runs")
        trainer = Trainer(logger=mlf_logger)

    Use the logger anywhere in your :class:`~lightning.pytorch.core.LightningModule` as follows:

    .. code-block:: python

        from lightning.pytorch import LightningModule


        class LitModel(LightningModule):
            def training_step(self, batch, batch_idx):
                # example
                self.logger.experiment.whatever_ml_flow_supports(...)

            def any_lightning_module_function_or_hook(self):
                self.logger.experiment.whatever_ml_flow_supports(...)

    Args:
        experiment_name: The name of the experiment.
        run_name: Name of the new run. The `run_name` is internally stored as a ``mlflow.runName`` tag.
            If the ``mlflow.runName`` tag has already been set in `tags`, the value is overridden by the `run_name`.
        tracking_uri: Address of local or remote tracking server.
            If not provided, defaults to `MLFLOW_TRACKING_URI` environment variable if set, otherwise it falls
            back to `file:<save_dir>`.
        tags: A dictionary tags for the experiment.
        save_dir: A path to a local directory where the MLflow runs get saved.
            Defaults to `./mlruns` if `tracking_uri` is not provided.
            Has no effect if `tracking_uri` is provided.
        log_model: Log checkpoints created by :class:`~lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint`
            as MLFlow artifacts.

            * if ``log_model == 'all'``, checkpoints are logged during training.
            * if ``log_model == True``, checkpoints are logged at the end of training, except when
              :paramref:`~lightning.pytorch.callbacks.Checkpoint.save_top_k` ``== -1``
              which also logs every checkpoint during training.
            * if ``log_model == False`` (default), no checkpoint is logged.

        prefix: A string to put at the beginning of metric keys.
        artifact_location: The location to store run artifacts. If not provided, the server picks an appropriate
            default.
        run_id: The run identifier of the experiment. If not provided, a new run is started.
        synchronous: Hints mlflow whether to block the execution for every logging call until complete where
            applicable. Requires mlflow >= 2.8.0

    Raises:
        ModuleNotFoundError:
            If required MLFlow package is not installed on the device.

    """

    LOGGER_JOIN_CHAR = "-"

    def __init__(
        self,
        experiment_name: str = "lightning_logs",
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = os.getenv("MLFLOW_TRACKING_URI"),
        tags: Optional[dict[str, Any]] = None,
        save_dir: Optional[str] = "./mlruns",
        log_model: Literal[True, False, "all"] = False,
        prefix: str = "",
        artifact_location: Optional[str] = None,
        run_id: Optional[str] = None,
        synchronous: Optional[bool] = None,
    ):
        if not _MLFLOW_AVAILABLE:
            raise ModuleNotFoundError(str(_MLFLOW_AVAILABLE))
        if synchronous is not None and not _MLFLOW_SYNCHRONOUS_AVAILABLE:
            raise ModuleNotFoundError("`synchronous` requires mlflow>=2.8.0")
        super().__init__()
        if not tracking_uri:
            tracking_uri = f"{LOCAL_FILE_URI_PREFIX}{save_dir}"

        self._experiment_name = experiment_name
        self._experiment_id: Optional[str] = None
        self._tracking_uri = tracking_uri
        self._run_name = run_name
        self._run_id = run_id
        self.tags = tags
        self._log_model = log_model
        self._logged_model_time: dict[str, float] = {}
        self._checkpoint_callback: Optional[ModelCheckpoint] = None
        self._prefix = prefix
        self._artifact_location = artifact_location
        self._log_batch_kwargs = {} if synchronous is None else {"synchronous": synchronous}
        self._initialized = False

        from mlflow.tracking import MlflowClient

        self._mlflow_client = MlflowClient(tracking_uri)

    @property
    @rank_zero_experiment
    def experiment(self) -> "MlflowClient":
        r"""Actual MLflow object. To use MLflow features in your :class:`~lightning.pytorch.core.LightningModule` do the
        following.

        Example::

            self.logger.experiment.some_mlflow_function()

        """
        import mlflow

        if self._initialized:
            return self._mlflow_client

        mlflow.set_tracking_uri(self._tracking_uri)

        if self._run_id is not None:
            run = self._mlflow_client.get_run(self._run_id)
            self._experiment_id = run.info.experiment_id
            self._initialized = True
            return self._mlflow_client

        if self._experiment_id is None:
            expt = self._mlflow_client.get_experiment_by_name(self._experiment_name)
            if expt is not None and expt.lifecycle_stage != "deleted":
                self._experiment_id = expt.experiment_id
            else:
                log.warning(f"Experiment with name {self._experiment_name} not found. Creating it.")
                self._experiment_id = self._mlflow_client.create_experiment(
                    name=self._experiment_name, artifact_location=self._artifact_location
                )

        if self._run_id is None:
            if self._run_name is not None:
                self.tags = self.tags or {}

                from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME

                if MLFLOW_RUN_NAME in self.tags:
                    log.warning(
                        f"The tag {MLFLOW_RUN_NAME} is found in tags. The value will be overridden by {self._run_name}."
                    )
                self.tags[MLFLOW_RUN_NAME] = self._run_name

            resolve_tags = _get_resolve_tags()
            run = self._mlflow_client.create_run(experiment_id=self._experiment_id, tags=resolve_tags(self.tags))
            self._run_id = run.info.run_id
        self._initialized = True
        return self._mlflow_client

    @property
    def run_id(self) -> Optional[str]:
        """Create the experiment if it does not exist to get the run id.

        Returns:
            The run id.

        """
        _ = self.experiment
        return self._run_id

    @property
    def experiment_id(self) -> Optional[str]:
        """Create the experiment if it does not exist to get the experiment id.

        Returns:
            The experiment id.

        """
        _ = self.experiment
        return self._experiment_id

    @override
    @rank_zero_only
    def log_hyperparams(self, params: Union[dict[str, Any], Namespace]) -> None:
        params = _convert_params(params)
        params = _flatten_dict(params)

        from mlflow.entities import Param

        # Truncate parameter values to 250 characters.
        # TODO: MLflow 1.28 allows up to 500 characters: https://github.com/mlflow/mlflow/releases/tag/v1.28.0
        params_list = [Param(key=k, value=str(v)[:250]) for k, v in params.items()]

        # Log in chunks of 100 parameters (the maximum allowed by MLflow).
        for idx in range(0, len(params_list), 100):
            self.experiment.log_batch(run_id=self.run_id, params=params_list[idx : idx + 100], **self._log_batch_kwargs)

    @override
    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"

        from mlflow.entities import Metric

        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)
        metrics_list: list[Metric] = []

        timestamp_ms = int(time() * 1000)
        for k, v in metrics.items():
            if isinstance(v, str):
                log.warning(f"Discarding metric with string value {k}={v}.")
                continue

            new_k = re.sub("[^a-zA-Z0-9_/. -]+", "", k)
            if k != new_k:
                rank_zero_warn(
                    "MLFlow only allows '_', '/', '.' and ' ' special characters in metric name."
                    f" Replacing {k} with {new_k}.",
                    category=RuntimeWarning,
                )
                k = new_k
            metrics_list.append(Metric(key=k, value=v, timestamp=timestamp_ms, step=step or 0))

        self.experiment.log_batch(run_id=self.run_id, metrics=metrics_list, **self._log_batch_kwargs)

    @override
    @rank_zero_only
    def finalize(self, status: str = "success") -> None:
        if not self._initialized:
            return
        if status == "success":
            status = "FINISHED"
        elif status == "failed":
            status = "FAILED"
        elif status == "finished":
            status = "FINISHED"

        # log checkpoints as artifacts
        if self._checkpoint_callback:
            self._scan_and_log_checkpoints(self._checkpoint_callback)

        if self.experiment.get_run(self.run_id):
            self.experiment.set_terminated(self.run_id, status)

    @property
    @override
    def save_dir(self) -> Optional[str]:
        """The root file directory in which MLflow experiments are saved.

        Return:
            Local path to the root experiment directory if the tracking uri is local.
            Otherwise returns `None`.

        """
        if self._tracking_uri.startswith(LOCAL_FILE_URI_PREFIX):
            return self._tracking_uri[len(LOCAL_FILE_URI_PREFIX) :]
        return None

    @property
    @override
    def name(self) -> Optional[str]:
        """Get the experiment id.

        Returns:
            The experiment id.

        """
        return self.experiment_id

    @property
    @override
    def version(self) -> Optional[str]:
        """Get the run id.

        Returns:
            The run id.

        """
        return self.run_id

    @override
    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        # log checkpoints as artifacts
        if self._log_model == "all" or self._log_model is True and checkpoint_callback.save_top_k == -1:
            self._scan_and_log_checkpoints(checkpoint_callback)
        elif self._log_model is True:
            self._checkpoint_callback = checkpoint_callback

    def _scan_and_log_checkpoints(self, checkpoint_callback: ModelCheckpoint) -> None:
        # get checkpoints to be saved with associated score
        checkpoints = _scan_checkpoints(checkpoint_callback, self._logged_model_time)

        # log iteratively all new checkpoints
        for t, p, s, tag in checkpoints:
            metadata = {
                # Ensure .item() is called to store Tensor contents
                "score": s.item() if isinstance(s, Tensor) else s,
                "original_filename": Path(p).name,
                "Checkpoint": {
                    k: getattr(checkpoint_callback, k)
                    for k in [
                        "monitor",
                        "mode",
                        "save_last",
                        "save_top_k",
                        "save_weights_only",
                        "_every_n_train_steps",
                        "_every_n_val_epochs",
                    ]
                    # ensure it does not break if `Checkpoint` args change
                    if hasattr(checkpoint_callback, k)
                },
            }
            aliases = ["latest", "best"] if p == checkpoint_callback.best_model_path else ["latest"]

            # Artifact path on mlflow
            artifact_path = Path(p).stem

            # Log the checkpoint
            self.experiment.log_artifact(self._run_id, p, artifact_path)

            # Create a temporary directory to log on mlflow
            with tempfile.TemporaryDirectory(prefix="test", suffix="test", dir=os.getcwd()) as tmp_dir:
                # Log the metadata
                with open(f"{tmp_dir}/metadata.yaml", "w") as tmp_file_metadata:
                    yaml.dump(metadata, tmp_file_metadata, default_flow_style=False)

                # Log the aliases
                with open(f"{tmp_dir}/aliases.txt", "w") as tmp_file_aliases:
                    tmp_file_aliases.write(str(aliases))

                # Log the metadata and aliases
                self.experiment.log_artifacts(self._run_id, tmp_dir, artifact_path)

            # remember logged models - timestamp needed in case filename didn't change (lastkckpt or custom name)
            self._logged_model_time[p] = t


def _get_resolve_tags() -> Callable:
    from mlflow.tracking import context

    # before v1.1.0
    if hasattr(context, "resolve_tags"):
        from mlflow.tracking.context import resolve_tags
    # since v1.1.0
    elif hasattr(context, "registry"):
        from mlflow.tracking.context.registry import resolve_tags
    else:
        resolve_tags = lambda tags: tags

    return resolve_tags
