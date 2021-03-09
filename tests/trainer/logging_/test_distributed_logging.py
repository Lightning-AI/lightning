# Copyright The PyTorch Lightning team.
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

import platform
from unittest import mock

import pytest
import torch

from pytorch_lightning import Trainer
from tests.helpers import BoringModel


class TestModel(BoringModel):

    def on_pretrain_routine_end(self) -> None:
        with mock.patch('pytorch_lightning.loggers.base.LightningLoggerBase.agg_and_log_metrics') as m:
            self.trainer.logger_connector.log_metrics({'a': 2}, {})
            logged_times = m.call_count
            expected = int(self.trainer.is_global_zero)
            msg = f'actual logger called from non-global zero, logged_times: {logged_times}, expected: {expected}'
            assert logged_times == expected, msg


@pytest.mark.skipif(platform.system() == "Windows", reason="Distributed training is not supported on Windows")
def test_global_zero_only_logging_ddp_cpu(tmpdir):
    """
    Makes sure logging only happens from root zero
    """
    model = TestModel()
    model.training_epoch_end = None
    trainer = Trainer(
        accelerator='ddp_cpu',
        num_processes=2,
        default_root_dir=tmpdir,
        limit_train_batches=1,
        limit_val_batches=1,
        max_epochs=1,
        weights_summary=None,
    )
    trainer.fit(model)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_global_zero_only_logging_ddp_spawn(tmpdir):
    """
    Makes sure logging only happens from root zero
    """
    model = TestModel()
    model.training_epoch_end = None
    trainer = Trainer(
        accelerator='ddp_spawn',
        gpus=2,
        default_root_dir=tmpdir,
        limit_train_batches=1,
        limit_val_batches=1,
        max_epochs=1,
        weights_summary=None,
    )
    trainer.fit(model)


def test_first_logger_call_in_subprocess(tmpdir):
    """
    Test that the Trainer does not call the logger too early. Only when the worker processes are initialized
    do we have access to the rank and know which one is the main process.
    """

    class LoggerCallsObserver(Callback):

        def on_fit_start(self, trainer, pl_module):
            # this hook is executed directly before Trainer.pre_dispatch
            # logger should not write any logs until this point
            assert not trainer.logger.method_calls
            assert not os.listdir(trainer.logger.save_dir)

        def on_train_start(self, trainer, pl_module):
            assert trainer.logger.method_call
            trainer.logger.log_hyperparams.assert_called_once()
            trainer.logger.log_graph.assert_called_once()

    logger = Mock()
    logger.version = "0"
    logger.name = "name"
    logger.save_dir = tmpdir

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=1,
        limit_val_batches=1,
        max_epochs=1,
        logger=logger,
        callbacks=[LoggerCallsObserver()]
    )
    trainer.fit(model)
