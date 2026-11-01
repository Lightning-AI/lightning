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
from argparse import Namespace
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Optional, Union

from lightning_utilities.core.imports import RequirementCache
from torch import Tensor
from torch.nn import Module
from typing_extensions import override

from lightning.fabric.loggers.logger import Logger, rank_zero_experiment
from lightning.fabric.utilities.cloud_io import _is_dir, get_filesystem
from lightning.fabric.utilities.logger import _add_prefix, _convert_params, _flatten_dict
from lightning.fabric.utilities.logger import _sanitize_params as _utils_sanitize_params
from lightning.fabric.utilities.rank_zero import rank_zero_only, rank_zero_warn
from lightning.fabric.utilities.types import _PATH
from lightning.fabric.wrappers import _unwrap_objects

_TENSORBOARD_AVAILABLE = RequirementCache("tensorboard")
_TENSORBOARDX_AVAILABLE = RequirementCache("tensorboardX")
if TYPE_CHECKING:
    # assumes at least one will be installed when type checking
    if _TENSORBOARD_AVAILABLE:
        from torch.utils.tensorboard import SummaryWriter
    else:
        from tensorboardX import SummaryWriter  # type: ignore[no-redef]


class TensorBoardLogger(Logger):
    r"""Log to local file system in `TensorBoard <https://www.tensorflow.org/tensorboard>`_ format.

    Implemented using :class:`~tensorboardX.SummaryWriter`. Logs are saved to
    ``os.path.join(root_dir, name, version)``. This is the recommended logger in Lightning Fabric.

    Args:
        root_dir: The root directory in which all your experiments with different names and versions will be stored.
        name: Experiment name. Defaults to ``'lightning_logs'``. If it is the empty string then no per-experiment
            subdirectory is used.
        version: Experiment version. If version is not specified the logger inspects the save
            directory for existing versions, then automatically assigns the next available version.
            If it is a string then it is used as the run-specific subdirectory name,
            otherwise ``'version_${version}'`` is used.
        default_hp_metric: Enables a placeholder metric with key `hp_metric` when `log_hyperparams` is
            called without a metric (otherwise calls to ``log_hyperparams`` without a metric are ignored).
        prefix: A string to put at the beginning of all metric keys.
        sub_dir: Sub-directory to group TensorBoard logs. If a ``sub_dir`` argument is passed
            then logs are saved in ``/root_dir/name/version/sub_dir/``. Defaults to ``None`` in which case
            logs are saved in ``/root_dir/name/version/``.
        \**kwargs: Additional arguments used by :class:`tensorboardX.SummaryWriter` can be passed as keyword
            arguments in this logger. To automatically flush to disk, `max_queue` sets the size
            of the queue for pending logs before flushing. `flush_secs` determines how many seconds
            elapses before flushing.


    Example::

        from lightning.fabric.loggers import TensorBoardLogger

        logger = TensorBoardLogger("path/to/logs/root", name="my_model")
        logger.log_hyperparams({"epochs": 5, "optimizer": "Adam"})
        logger.log_metrics({"acc": 0.75})
        logger.finalize("success")

    """

    LOGGER_JOIN_CHAR = "-"

    def __init__(
        self,
        root_dir: _PATH,
        name: Optional[str] = "lightning_logs",
        version: Optional[Union[int, str]] = None,
        default_hp_metric: bool = True,
        prefix: str = "",
        sub_dir: Optional[_PATH] = None,
        **kwargs: Any,
    ):
        if not _TENSORBOARD_AVAILABLE and not _TENSORBOARDX_AVAILABLE:
            raise ModuleNotFoundError(
                "Neither `tensorboard` nor `tensorboardX` is available. Try `pip install`ing either.\n"
                f"{str(_TENSORBOARDX_AVAILABLE)}\n{str(_TENSORBOARD_AVAILABLE)}"
            )
        super().__init__()
        root_dir = os.fspath(root_dir)
        self._root_dir = root_dir
        self._name = name or ""
        self._version = version
        self._sub_dir = None if sub_dir is None else os.fspath(sub_dir)

        self._default_hp_metric = default_hp_metric
        self._prefix = prefix
        self._fs = get_filesystem(root_dir)

        self._experiment: Optional[SummaryWriter] = None
        self._kwargs = kwargs

    @property
    @override
    def name(self) -> str:
        """Get the name of the experiment.

        Returns:
            The name of the experiment.

        """
        return self._name

    @property
    @override
    def version(self) -> Union[int, str]:
        """Get the experiment version.

        Returns:
            The experiment version if specified else the next version.

        """
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    @property
    @override
    def root_dir(self) -> str:
        """Gets the save directory where the TensorBoard experiments are saved.

        Returns:
            The local path to the save directory where the TensorBoard experiments are saved.

        """
        return self._root_dir

    @property
    @override
    def log_dir(self) -> str:
        """The directory for this run's tensorboard checkpoint.

        By default, it is named ``'version_${self.version}'`` but it can be overridden by passing a string value for the
        constructor's version parameter instead of ``None`` or an int.

        """
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        log_dir = os.path.join(self.root_dir, self.name, version)
        if isinstance(self.sub_dir, str):
            log_dir = os.path.join(log_dir, self.sub_dir)
        log_dir = os.path.expandvars(log_dir)
        log_dir = os.path.expanduser(log_dir)
        return log_dir

    @property
    def sub_dir(self) -> Optional[str]:
        """Gets the sub directory where the TensorBoard experiments are saved.

        Returns:
            The local path to the sub directory where the TensorBoard experiments are saved.

        """
        return self._sub_dir

    @property
    @rank_zero_experiment
    def experiment(self) -> "SummaryWriter":
        """Actual tensorboard object. To use TensorBoard features anywhere in your code, do the following.

        Example::

            logger.experiment.some_tensorboard_function()

        """
        if self._experiment is not None:
            return self._experiment

        assert rank_zero_only.rank == 0, "tried to init log dirs in non global_rank=0"
        if self.root_dir:
            self._fs.makedirs(self.root_dir, exist_ok=True)

        if _TENSORBOARD_AVAILABLE:
            from torch.utils.tensorboard import SummaryWriter
        else:
            from tensorboardX import SummaryWriter  # type: ignore[no-redef]

        self._experiment = SummaryWriter(log_dir=self.log_dir, **self._kwargs)
        return self._experiment

    @override
    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"

        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)

        for k, v in metrics.items():
            if isinstance(v, Tensor):
                v = v.item()
            elif hasattr(v, "ndim") and hasattr(v, "item"):
                # Handle numpy arrays and other array-like objects
                if v.ndim == 0:
                    # For 0-dimensional arrays, handle numpy >= 2.4.0 TypeError
                    try:
                        v = v.item()
                    except TypeError:
                        # numpy >= 2.4.0 raises TypeError for 0-dim arrays
                        # Fall back to scalar conversion
                        v = float(v)
                elif v.ndim == 1 and v.size == 1:
                    # Handle 1-dimensional arrays with single element
                    try:
                        v = v.item()
                    except TypeError:
                        # numpy >= 2.4.0 fallback
                        v = float(v.flat[0])
                else:
                    # Multi-dimensional arrays cannot be logged as scalars
                    raise ValueError(
                        f"Cannot log multi-dimensional array with shape {v.shape} as scalar. "
                        f"Try using a scalar value instead."
                    )

            if isinstance(v, dict):
                self.experiment.add_scalars(k, v, step)
            else:
                try:
                    self.experiment.add_scalar(k, v, step)
                # TODO(fabric): specify the possible exception
                except Exception as ex:
                    raise ValueError(
                        f"\n you tried to log {v} which is currently not supported. Try a dict or a scalar/tensor."
                    ) from ex

    @override
    @rank_zero_only
    def log_hyperparams(
        self,
        params: Union[dict[str, Any], Namespace],
        metrics: Optional[dict[str, Any]] = None,
        step: Optional[int] = None,
    ) -> None:
        """Record hyperparameters. TensorBoard logs with and without saved hyperparameters are incompatible, the
        hyperparameters are then not displayed in the TensorBoard. Please delete or move the previously saved logs to
        display the new ones with hyperparameters.

        Args:
            params: A dictionary-like container with the hyperparameters
            metrics: Dictionary with metric names as keys and measured quantities as values
            step: Optional global step number for the logged metrics

        """
        params = _convert_params(params)

        # format params into the suitable for tensorboard
        params = _flatten_dict(params)
        params = self._sanitize_params(params)

        if metrics is None:
            if self._default_hp_metric:
                metrics = {"hp_metric": -1}
        elif not isinstance(metrics, dict):
            metrics = {"hp_metric": metrics}

        if metrics:
            self.log_metrics(metrics, step)

            if _TENSORBOARD_AVAILABLE:
                from torch.utils.tensorboard.summary import hparams
            else:
                from tensorboardX.summary import hparams  # type: ignore[no-redef]

            exp, ssi, sei = hparams(params, metrics)
            writer = self.experiment._get_file_writer()
            writer.add_summary(exp, step)
            writer.add_summary(ssi, step)
            writer.add_summary(sei, step)

    @override
    @rank_zero_only
    def log_graph(self, model: Module, input_array: Optional[Tensor] = None) -> None:
        model_example_input = getattr(model, "example_input_array", None)
        input_array = model_example_input if input_array is None else input_array
        model = _unwrap_objects(model)

        if input_array is None:
            rank_zero_warn(
                "Could not log computational graph to TensorBoard: The `model.example_input_array` attribute"
                " is not set or `input_array` was not given."
            )
        elif not isinstance(input_array, (Tensor, tuple)):
            rank_zero_warn(
                "Could not log computational graph to TensorBoard: The `input_array` or `model.example_input_array`"
                f" has type {type(input_array)} which can't be traced by TensorBoard. Make the input array a tuple"
                f" representing the positional arguments to the model's `forward()` implementation."
            )
        elif callable(getattr(model, "_on_before_batch_transfer", None)) and callable(
            getattr(model, "_apply_batch_transfer_handler", None)
        ):
            # this is probably is a LightningModule
            input_array = model._on_before_batch_transfer(input_array)
            input_array = model._apply_batch_transfer_handler(input_array)
            self.experiment.add_graph(model, input_array)
        else:
            self.experiment.add_graph(model, input_array)

    @override
    @rank_zero_only
    def save(self) -> None:
        self.experiment.flush()

    @override
    @rank_zero_only
    def finalize(self, status: str) -> None:
        if self._experiment is not None:
            self.experiment.flush()
            self.experiment.close()

    def _get_next_version(self) -> int:
        save_dir = os.path.join(self.root_dir, self.name)

        try:
            listdir_info = self._fs.listdir(save_dir)
        except OSError:
            return 0

        existing_versions = []
        for listing in listdir_info:
            d = listing["name"]
            bn = os.path.basename(d)
            if _is_dir(self._fs, d) and bn.startswith("version_"):
                dir_ver = bn.split("_")[1].replace("/", "")
                if dir_ver.isdigit():
                    existing_versions.append(int(dir_ver))
        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1

    @staticmethod
    def _sanitize_params(params: dict[str, Any]) -> dict[str, Any]:
        params = _utils_sanitize_params(params)
        # logging of arrays with dimension > 1 is not supported, sanitize as string
        return {k: str(v) if hasattr(v, "ndim") and v.ndim > 1 else v for k, v in params.items()}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_experiment"] = None
        return state
