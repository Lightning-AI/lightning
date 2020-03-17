"""
Log using `allegro.ai TRAINS <https://github.com/allegroai/trains>'_

.. code-block:: python

    from pytorch_lightning.loggers import TrainsLogger
    trains_logger = TrainsLogger(
        project_name="pytorch lightning",
        task_name="default",
    )
    trainer = Trainer(logger=trains_logger)


Use the logger anywhere in you LightningModule as follows:

.. code-block:: python

    def train_step(...):
        # example
        self.logger.experiment.whatever_trains_supports(...)

    def any_lightning_module_function_or_hook(...):
        self.logger.experiment.whatever_trains_supports(...)

"""

from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

try:
    import trains
    from trains.task import Task
except ImportError:  # pragma: no-cover
    raise ImportError('You want to use `TRAINS` logger which is not installed yet,'  # pragma: no-cover
                      ' install it with `pip install trains`.')

from pytorch_lightning import _logger as log
from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_only


class TrainsLogger(LightningLoggerBase):
    """Logs using TRAINS

    Args:
        project_name: The name of the experiment's project. Defaults to None.
        task_name: The name of the experiment. Defaults to None.
        task_type: The name of the experiment. Defaults to 'training'.
        reuse_last_task_id: Start with the previously used task id. Defaults to True.
        output_uri: Default location for output models. Defaults to None.
        auto_connect_arg_parser: Automatically grab the ArgParser
            and connect it with the task. Defaults to True.
        auto_connect_frameworks: If True, automatically patch to trains backend. Defaults to True.
        auto_resource_monitoring: If true, machine vitals will be
            sent along side the task scalars. Defaults to True.

    Examples:
        >>> logger = TrainsLogger("lightning_log", "my-test", output_uri=".")  # doctest: +ELLIPSIS
        TRAINS Task: ...
        TRAINS results page: https://demoapp.trains.allegro.ai/.../log
        >>> logger.log_metrics({"val_loss": 1.23}, step=0)
        >>> logger.log_text("sample test")
        sample test
        >>> import numpy as np
        >>> logger.log_artifact("confusion matrix", np.ones((2, 3)))
        >>> logger.log_image("passed", "Image 1", np.random.randint(0, 255, (200, 150, 3), dtype=np.uint8))
    """

    def __init__(
            self,
            project_name: Optional[str] = None,
            task_name: Optional[str] = None,
            task_type: str = 'training',
            reuse_last_task_id: bool = True,
            output_uri: Optional[str] = None,
            auto_connect_arg_parser: bool = True,
            auto_connect_frameworks: bool = True,
            auto_resource_monitoring: bool = True
    ) -> None:
        super().__init__()
        self._trains = Task.init(
            project_name=project_name,
            task_name=task_name,
            task_type=task_type,
            reuse_last_task_id=reuse_last_task_id,
            output_uri=output_uri,
            auto_connect_arg_parser=auto_connect_arg_parser,
            auto_connect_frameworks=auto_connect_frameworks,
            auto_resource_monitoring=auto_resource_monitoring
        )

    @property
    def experiment(self) -> Task:
        r"""Actual TRAINS object. To use TRAINS features do the following.

        Example:
            .. code-block:: python

                self.logger.experiment.some_trains_function()

        """
        return self._trains

    @property
    def id(self) -> Union[str, None]:
        """
        ID is a uuid (string) representing this specific experiment in the entire system.
        """
        if not self._trains:
            return None
        return self._trains.id

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        """Log hyperparameters (numeric values) in TRAINS experiments

        Args:
            params:
                The hyperparameters that passed through the model.
        """
        if not self._trains:
            return None
        if not params:
            return
        if isinstance(params, dict):
            self._trains.connect(params)
        else:
            self._trains.connect(vars(params))

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics (numeric values) in TRAINS experiments.
            This method will be called by Trainer.

        Args:
            metrics:
                The dictionary of the metrics.
                If the key contains "/", it will be split by the delimiter,
                then the elements will be logged as "title" and "series" respectively.
            step: Step number at which the metrics should be recorded. Defaults to None.
        """
        if not self._trains:
            return None

        if not step:
            step = self._trains.get_last_iteration()

        for k, v in metrics.items():
            if isinstance(v, str):
                log.warning("Discarding metric with string value {}={}".format(k, v))
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            parts = k.split('/')
            if len(parts) <= 1:
                series = title = k
            else:
                title = parts[0]
                series = '/'.join(parts[1:])
            self._trains.get_logger().report_scalar(
                title=title, series=series, value=v, iteration=step)

    @rank_zero_only
    def log_metric(self, title: str, series: str, value: float, step: Optional[int] = None) -> None:
        """Log metrics (numeric values) in TRAINS experiments.
            This method will be called by the users.

        Args:
            title: The title of the graph to log, e.g. loss, accuracy.
            series: The series name in the graph, e.g. classification, localization.
            value: The value to log.
            step: Step number at which the metrics should be recorded. Defaults to None.
        """
        if not self._trains:
            return None

        if not step:
            step = self._trains.get_last_iteration()

        if isinstance(value, torch.Tensor):
            value = value.item()
        self._trains.get_logger().report_scalar(
            title=title, series=series, value=value, iteration=step)

    @rank_zero_only
    def log_text(self, text: str) -> None:
        """Log console text data in TRAINS experiment

        Args:
            text: The value of the log (data-point).
        """
        if not self._trains:
            return None

        self._trains.get_logger().report_text(text)

    @rank_zero_only
    def log_image(
            self, title: str, series: str,
            image: Union[str, np.ndarray, 'PIL.Image', torch.Tensor],
            step: Optional[int] = None) -> None:
        """Log Debug image in TRAINS experiment

        Args:
            title: The title of the debug image, i.e. "failed", "passed".
            series: The series name of the debug image, i.e. "Image 0", "Image 1".
            image:
                Debug image to log. Can be one of the following types:
                    Torch, Numpy, PIL image, path to image file (str)
                If Numpy or Torch, the image is assume to be the following:
                    shape: CHW
                    color space: RGB
                    value range: [0., 1.] (float) or [0, 255] (uint8)
            step:
                Step number at which the metrics should be recorded. Defaults to None.
        """
        if not self._trains:
            return None

        if not step:
            step = self._trains.get_last_iteration()

        if isinstance(image, str):
            self._trains.get_logger().report_image(
                title=title, series=series, local_path=image, iteration=step)
        else:
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            if isinstance(image, np.ndarray):
                image = image.transpose(1, 2, 0)
            self._trains.get_logger().report_image(
                title=title, series=series, image=image, iteration=step)

    @rank_zero_only
    def log_artifact(
            self, name: str,
            artifact: Union[str, Path, Dict[str, Any], 'pandas.DataFrame', 'numpy.ndarray', 'PIL.Image.Image'],
            metadata: Optional[Dict[str, Any]] = None, delete_after_upload: bool = False) -> None:
        """Save an artifact (file/object) in TRAINS experiment storage.

        Arguments:
            name: Artifact name. Notice! it will override previous artifact
                if name already exists
            artifact: Artifact object to upload. Currently supports:
                - string / pathlib2.Path are treated as path to artifact file to upload
                    If wildcard or a folder is passed, zip file containing the
                    local files will be created and uploaded
                - dict will be stored as .json file and uploaded
                - pandas.DataFrame will be stored as .csv.gz (compressed CSV file) and uploaded
                - numpy.ndarray will be stored as .npz and uploaded
                - PIL.Image will be stored to .png file and uploaded
            metadata:
                Simple key/value dictionary to store on the artifact. Defaults to None.
            delete_after_upload:
                If True local artifact will be deleted (only applies if artifact_object is a
                local file). Defaults to False.
        """
        if not self._trains:
            return None

        self._trains.upload_artifact(
            name=name, artifact_object=artifact, metadata=metadata,
            delete_after_upload=delete_after_upload
        )

    def save(self) -> None:
        pass

    @rank_zero_only
    def finalize(self, status: str) -> None:
        if not self._trains:
            return None
        self._trains.close()
        self._trains = None

    @property
    def name(self) -> Union[str, None]:
        """
        Name is a human readable non-unique name (str) of the experiment.
        """
        if not self._trains:
            return None
        return self._trains.name

    @property
    def version(self) -> Union[str, None]:
        if not self._trains:
            return None
        return self._trains.id

    def __getstate__(self) -> Union[str, None]:
        if not self._trains:
            return None
        return self._trains.id

    def __setstate__(self, state: str) -> None:
        self._rank = 0
        self._trains = None
        if state:
            self._trains = Task.get_task(task_id=state)
