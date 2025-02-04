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
from lightning.pytorch.callbacks.batch_size_finder import BatchSizeFinder
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.callbacks.checkpoint import Checkpoint
from lightning.pytorch.callbacks.device_stats_monitor import DeviceStatsMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.email_callback import EmailCallback
from lightning.pytorch.callbacks.finetuning import BackboneFinetuning, BaseFinetuning
from lightning.pytorch.callbacks.gradient_accumulation_scheduler import GradientAccumulationScheduler
from lightning.pytorch.callbacks.lambda_function import LambdaCallback
from lightning.pytorch.callbacks.lr_finder import LearningRateFinder
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.callbacks.model_summary import ModelSummary
from lightning.pytorch.callbacks.on_exception_checkpoint import OnExceptionCheckpoint
from lightning.pytorch.callbacks.prediction_writer import BasePredictionWriter
from lightning.pytorch.callbacks.progress import ProgressBar, RichProgressBar, TQDMProgressBar
from lightning.pytorch.callbacks.pruning import ModelPruning
from lightning.pytorch.callbacks.rich_model_summary import RichModelSummary
from lightning.pytorch.callbacks.spike import SpikeDetection
from lightning.pytorch.callbacks.stochastic_weight_avg import StochasticWeightAveraging
from lightning.pytorch.callbacks.throughput_monitor import ThroughputMonitor
from lightning.pytorch.callbacks.timer import Timer

__all__ = [
    "BackboneFinetuning",
    "BaseFinetuning",
    "BasePredictionWriter",
    "BatchSizeFinder",
    "Callback",
    "Checkpoint",
    "DeviceStatsMonitor",
    "EarlyStopping",
    "EmailCallback",
    "GradientAccumulationScheduler",
    "LambdaCallback",
    "LearningRateFinder",
    "LearningRateMonitor",
    "ModelCheckpoint",
    "ModelPruning",
    "ModelSummary",
    "OnExceptionCheckpoint",
    "ProgressBar",
    "RichModelSummary",
    "RichProgressBar",
    "StochasticWeightAveraging",
    "SpikeDetection",
    "ThroughputMonitor",
    "Timer",
    "TQDMProgressBar",
]
