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
from datetime import timedelta, datetime
from unittest.mock import Mock, patch

import pytest

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers import BoringModel


@pytest.mark.parametrize("duration,expected", [
    ("00:00:22", timedelta(seconds=22)),
    ("12:34:56", timedelta(hours=12, minutes=34, seconds=56)),
    (timedelta(weeks=52, milliseconds=1), timedelta(weeks=52, milliseconds=1)),
])
def test_timer_parse_duration(duration, expected):
    timer = Timer(duration=duration)
    assert timer.time_remaining == expected


def test_timer_interval_choice():
    Timer(duration=timedelta(), interval="step")
    Timer(duration=timedelta(), interval="epoch")
    with pytest.raises(MisconfigurationException, match="Unsupported parameter value"):
        Timer(duration=timedelta(), interval="invalid")


@patch("pytorch_lightning.callbacks.timer.datetime")
def test_timer_time_remaining(datetime_mock):
    """ Test that the timer tracks the elapsed and remaining time correctly. """
    start_time = datetime.now()
    duration = timedelta(seconds=10)
    datetime_mock.now.return_value = start_time
    timer = Timer(duration=duration)
    assert timer.time_remaining == duration
    assert timer.time_elapsed == timedelta(0)

    # timer not started yet
    datetime_mock.now.return_value = start_time + timedelta(minutes=1)
    assert timer.start_time is None
    assert timer.time_remaining == timedelta(seconds=10)
    assert timer.time_elapsed == timedelta(seconds=0)

    # start timer
    datetime_mock.now.return_value = start_time
    timer.on_train_start(trainer=Mock(), pl_module=Mock())
    assert timer.start_time == start_time

    # pretend time has elapsed
    elapsed = timedelta(seconds=3)
    datetime_mock.now.return_value = start_time + elapsed
    assert timer.start_time == start_time
    assert timer.time_remaining == timedelta(seconds=7)
    assert timer.time_elapsed == timedelta(seconds=3)


def test_timer_stops_training(tmpdir):
    """ Test that the timer stops training before reaching max_epochs """
    model = BoringModel()
    duration = timedelta(milliseconds=100)
    timer = Timer(duration=duration)

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1000,
        callbacks=[timer],
    )
    trainer.fit(model)
    assert trainer.global_step > 1
    assert trainer.current_epoch < 999


@pytest.mark.parametrize("interval", ["step", "epoch"])
def test_timer_zero_duration_stop(tmpdir, interval):
    """ Test that the timer stops training immediately after the first check occurs. """
    model = BoringModel()
    duration = timedelta(0)
    timer = Timer(duration=duration, interval=interval)
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[timer],
    )
    trainer.fit(model)
    if interval == "step":
        # timer triggers stop on step end
        assert trainer.global_step == 1
        assert trainer.current_epoch == 0
    else:
        # timer triggers stop on epoch end
        assert trainer.global_step == len(trainer.train_dataloader)
        assert trainer.current_epoch == 0


@pytest.mark.parametrize("min_steps,min_epochs", [
    (None, 2),
    (3, None),
    (3, 2),
])
def test_timer_duration_min_steps_override(tmpdir, min_steps, min_epochs):
    model = BoringModel()
    duration = timedelta(0)
    timer = Timer(duration=duration)
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[timer],
        min_steps=min_steps,
        min_epochs=min_epochs,
    )
    trainer.fit(model)
    if min_epochs:
        assert trainer.current_epoch >= min_epochs - 1
    if min_steps:
        assert trainer.global_step >= min_steps - 1
    assert timer.time_elapsed > duration


def test_timer_resume_training(tmpdir):
    # TODO
    model = BoringModel()
    timer = Timer(duration=timedelta())
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_steps=1,
        callbacks=[timer]
    )