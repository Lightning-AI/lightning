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
# limitations under the License
import logging
import os
import uuid
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.utilities.memory import garbage_collection_cuda, is_oom_error
from pytorch_lightning.utilities.parsing import lightning_getattr, lightning_setattr
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn

log = logging.getLogger(__name__)


def scale_batch_size(
    trainer: "pl.Trainer",
    model: "pl.LightningModule",
    mode: str = "power",
    steps_per_trial: int = 3,
    init_val: int = 2,
    max_trials: int = 25,
    batch_arg_name: str = "batch_size",
):
    if trainer.fast_dev_run:
        rank_zero_warn("Skipping batch size scaler since `fast_dev_run` is enabled.")
        return

    # Save initial model, that is loaded after batch size is found
    ckpt_path = os.path.join(trainer.default_root_dir, f".scale_batch_size_{uuid.uuid4()}.ckpt")
    trainer.save_checkpoint(ckpt_path)

    # Arguments we adjust during the batch size finder, save for restoring
    params = __scale_batch_dump_params(trainer)

    # Set to values that are required by the algorithm
    __scale_batch_reset_params(trainer, steps_per_trial)

    if trainer.progress_bar_callback:
        trainer.progress_bar_callback.disable()

    new_size, _ = _adjust_batch_size(trainer, batch_arg_name, value=init_val)

    if mode == "power":
        new_size = _run_power_scaling(trainer, model, new_size, batch_arg_name, max_trials, params)
    elif mode == "binsearch":
        new_size = _run_binary_scaling(trainer, model, new_size, batch_arg_name, max_trials, params)

    _collect_garbage(trainer)

    log.info(f"Finished batch size finder, will continue with full run using batch size {new_size}")

    __scale_batch_restore_params(trainer, params)

    if trainer.progress_bar_callback:
        trainer.progress_bar_callback.enable()

    trainer._checkpoint_connector.restore(ckpt_path)
    trainer.strategy.remove_checkpoint(ckpt_path)

    return new_size


def __scale_batch_dump_params(trainer: "pl.Trainer") -> Dict[str, Any]:
    dumped_params = {
        "loggers": trainer.loggers,
        "callbacks": trainer.callbacks,
    }
    if trainer.state.fn == "fit":
        loop = trainer.fit_loop
        dumped_params["max_steps"] = trainer.max_steps
        dumped_params["limit_val_batches"] = trainer.limit_val_batches
    else:
        stage = trainer.state.stage
        loop = getattr(trainer, f"{stage}_loop")
        assert stage is not None
        dumped_params["limit_eval_batches"] = getattr(trainer, f"limit_{stage.dataloader_prefix}_batches")

        if hasattr(loop, "verbose"):
            dumped_params["loop_verbose"] = loop.verbose

    dumped_params["loop_state_dict"] = deepcopy(loop.state_dict())
    return dumped_params


def __scale_batch_reset_params(trainer: "pl.Trainer", steps_per_trial: int) -> None:
    from pytorch_lightning.loggers.logger import DummyLogger

    trainer.logger = DummyLogger() if trainer.logger is not None else None
    trainer.callbacks = []

    if trainer.state.fn == "fit":
        trainer.limit_val_batches = steps_per_trial
        trainer.fit_loop.max_steps = steps_per_trial
    else:
        stage = trainer.state.stage
        loop = getattr(trainer, f"{stage}_loop")
        assert stage is not None
        setattr(trainer, f"limit_{stage.dataloader_prefix}_batches", steps_per_trial)

        if hasattr(loop, "verbose"):
            loop.verbose = False


def __scale_batch_restore_params(trainer: "pl.Trainer", params: Dict[str, Any]) -> None:
    # TODO: There are more states that needs to be reset (#4512 and #4870)
    trainer.loggers = params["loggers"]
    trainer.callbacks = params["callbacks"]

    if trainer.state.fn == "fit":
        loop = trainer.fit_loop
        loop.max_steps = params["max_steps"]
        trainer.limit_val_batches = params["limit_val_batches"]
    else:
        stage = trainer.state.stage
        loop = getattr(trainer, f"{stage}_loop")
        assert stage is not None
        setattr(trainer, f"limit_{stage.dataloader_prefix}_batches", params["limit_eval_batches"])

    loop.load_state_dict(deepcopy(params["loop_state_dict"]))
    loop.restarting = False
    if "loop_verbose" in params:
        loop.verbose = params["loop_verbose"]


def _run_power_scaling(
    trainer: "pl.Trainer", pl_module: "pl.LightningModule", new_size: int, batch_arg_name: str, max_trials: int, params
) -> int:
    """Batch scaling mode where the size is doubled at each iteration until an OOM error is encountered."""
    for _ in range(max_trials):
        _collect_garbage(trainer)

        try:
            _try_loop_run(trainer, params)
            new_size, changed = _adjust_batch_size(trainer, batch_arg_name, factor=2.0, desc="succeeded")

            if changed:
                # Force the dataloaders to reset as the batch size has changed
                _reset_dataloaders(trainer, pl_module)
            else:
                break
        except RuntimeError as exception:
            if is_oom_error(exception):
                _collect_garbage(trainer)

                new_size, _ = _adjust_batch_size(trainer)
                break
            else:
                raise  # some other error not memory related

    return new_size


def _run_binary_scaling(
    trainer: "pl.Trainer", pl_module: "pl.LightningModule", new_size: int, batch_arg_name: str, max_trials: int, params
) -> int:
    """Batch scaling mode where the size is initially is doubled at each iteration until an OOM error is
    encountered.

    Hereafter, the batch size is further refined using a binary search
    """
    low = 1
    high = None
    count = 0
    while True:
        _collect_garbage(trainer)

        try:
            # run loop
            _try_loop_run(trainer, params)
            count += 1
            if count > max_trials:
                break
            # Double in size
            low = new_size
            if high:
                if high - low <= 1:
                    break
                midval = (high + low) // 2
                new_size, changed = _adjust_batch_size(trainer, batch_arg_name, value=midval, desc="succeeded")
            else:
                new_size, changed = _adjust_batch_size(trainer, batch_arg_name, factor=2.0, desc="succeeded")

            if changed:
                # Force the dataloaders to reset as the batch size has changed
                _reset_dataloaders(trainer, pl_module)
            else:
                break

        except RuntimeError as exception:
            # Only these errors should trigger an adjustment
            if is_oom_error(exception):
                # If we fail in power mode, half the size and return
                _collect_garbage(trainer)

                high = new_size
                midval = (high + low) // 2
                new_size, changed = _adjust_batch_size(trainer, value=midval, desc="failed")

                if changed:
                    # Force the dataloaders to reset as the batch size has changed
                    _reset_dataloaders(trainer, pl_module)

                if high - low <= 1:
                    break
            else:
                raise  # some other error not memory related

    return new_size


def _adjust_batch_size(
    trainer: "pl.Trainer",
    batch_arg_name: str = "batch_size",
    factor: float = 1.0,
    value: Optional[int] = None,
    desc: Optional[str] = None,
) -> Tuple[int, bool]:
    """Helper function for adjusting the batch size.

    Args:
        trainer: instance of pytorch_lightning.Trainer
        factor: value which the old batch size is multiplied by to get the
            new batch size
        value: if a value is given, will override the batch size with this value.
            Note that the value of `factor` will not have an effect in this case
        desc: either ``"succeeded"`` or ``"failed"``. Used purely for logging

    Returns:
        The new batch size for the next trial and a bool that signals whether the
        new value is different than the previous batch size.
    """
    model = trainer.lightning_module
    batch_size = lightning_getattr(model, batch_arg_name)
    assert batch_size is not None
    new_size = value if value is not None else int(batch_size * factor)
    if desc:
        rank_zero_info(f"Batch size {batch_size} {desc}, trying batch size {new_size}")

    # TODO improve this for multi eval dataloaders
    if trainer.state.fn == "fit":
        if trainer.train_dataloader is None:
            trainer.reset_train_dataloader()

        assert trainer.train_dataloader is not None
        if not _is_valid_batch_size(new_size, trainer.train_dataloader, trainer):
            new_size = min(new_size, len(trainer.train_dataloader.dataset))
    else:
        stage = trainer.state.stage
        assert stage is not None
        dataloaders = getattr(trainer, f"{stage.dataloader_prefix}_dataloaders")
        if dataloaders is None:
            getattr(trainer, f"reset_{stage.dataloader_prefix}_dataloader")()

        dataloaders = getattr(trainer, f"{stage.dataloader_prefix}_dataloaders")
        assert dataloaders is not None
        if not _is_valid_batch_size(new_size, dataloaders[0], trainer):
            new_size = min(new_size, len(dataloaders[0].dataset))

    changed = new_size != batch_size
    lightning_setattr(model, batch_arg_name, new_size)
    return new_size, changed


def _is_valid_batch_size(batch_size: int, dataloader: DataLoader, trainer: "pl.Trainer") -> bool:
    from pytorch_lightning.utilities.data import has_len_all_ranks

    module = trainer.lightning_module or trainer.datamodule
    return not has_len_all_ranks(dataloader, trainer.strategy, module) or batch_size <= len(dataloader)


def _collect_garbage(trainer: "pl.Trainer") -> None:
    from pytorch_lightning.accelerators.gpu import GPUAccelerator

    if isinstance(trainer.accelerator, GPUAccelerator):
        garbage_collection_cuda()


def _reset_dataloaders(trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
    if trainer.state.fn == "fit":
        trainer.reset_train_dataloader(pl_module)
        trainer.reset_val_dataloader(pl_module)
    else:
        stage = trainer.state.stage
        assert stage is not None
        getattr(trainer, f"reset_{stage.dataloader_prefix}_dataloader")(pl_module)


def _try_loop_run(trainer: "pl.Trainer", params) -> None:
    if trainer.state.fn == "fit":
        loop = trainer.fit_loop
    else:
        loop = getattr(trainer, f"{trainer.state.stage}_loop")

    loop.load_state_dict(deepcopy(params["loop_state_dict"]))
    loop.restarting = False
    loop.run()
