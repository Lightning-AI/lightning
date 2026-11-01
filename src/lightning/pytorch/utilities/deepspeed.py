# Copyright 2020 The PyTorch Lightning team and Microsoft Corporation. All rights reserved.
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
"""Utilities that can be used with Deepspeed."""

from __future__ import annotations

import os
from typing import Any

import torch

from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.strategies.deepspeed import _DEEPSPEED_AVAILABLE

CPU_DEVICE = torch.device("cpu")


def ds_checkpoint_dir(checkpoint_dir: _PATH, tag: str | None = None) -> str:
    if tag is None:
        latest_path = os.path.join(checkpoint_dir, "latest")
        if os.path.isfile(latest_path):
            with open(latest_path) as fd:
                tag = fd.read().strip()
        else:
            raise ValueError(f"Unable to find 'latest' file at {latest_path}")

    directory = os.path.join(checkpoint_dir, tag)

    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory '{directory}' doesn't exist")
    return directory


# Modified script from https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/utils/zero_to_fp32.py
def convert_zero_checkpoint_to_fp32_state_dict(
    checkpoint_dir: _PATH, output_file: _PATH, tag: str | None = None
) -> dict[str, Any]:
    """Convert ZeRO 2 or 3 checkpoint into a single fp32 consolidated ``state_dict`` file that can be loaded with
    ``torch.load(file)`` + ``load_state_dict()`` and used for training without DeepSpeed. It gets copied into the top
    level checkpoint dir, so the user can easily do the conversion at any point in the future. Once extracted, the
    weights don't require DeepSpeed and can be used in any application. Additionally the script has been modified to
    ensure we keep the lightning state inside the state dict for being able to run
    ``LightningModule.load_from_checkpoint('...')```.

    Args:
        checkpoint_dir: path to the desired checkpoint folder.
            (one that contains the tag-folder, like ``global_step14``)
        output_file: path to the pytorch fp32 state_dict output file (e.g. path/pytorch_model.bin)
        tag: checkpoint tag used as a unique identifier for checkpoint. If not provided will attempt
            to load tag in the file named ``latest`` in the checkpoint folder, e.g., ``global_step14``

    Examples::

        # Lightning deepspeed has saved a directory instead of a file
        convert_zero_checkpoint_to_fp32_state_dict(
            "lightning_logs/version_0/checkpoints/epoch=0-step=0.ckpt/",
            "lightning_model.pt"
        )

    """
    if not _DEEPSPEED_AVAILABLE:
        raise ModuleNotFoundError(str(_DEEPSPEED_AVAILABLE))

    from deepspeed.utils.zero_to_fp32 import (
        get_fp32_state_dict_from_zero_checkpoint,
        get_model_state_file,
        get_optim_files,
    )

    state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir, tag)

    # additional logic to ensure we keep the lightning state dict as well from rank 0.
    deepspeed_states = [
        "module",
        "optimizer",
        "lr_scheduler",
        "csr_tensor_module_names",
        "skipped_steps",
        "global_steps",
        "dp_world_size",
        "mp_world_size",
    ]
    checkpoint_dir = ds_checkpoint_dir(checkpoint_dir, tag)
    optim_files = get_optim_files(checkpoint_dir)
    optim_state = torch.load(optim_files[0], map_location=CPU_DEVICE, weights_only=False)
    zero_stage = optim_state["optimizer_state_dict"]["zero_stage"]
    model_file = get_model_state_file(checkpoint_dir, zero_stage)
    client_state = torch.load(model_file, map_location=CPU_DEVICE, weights_only=False)
    client_state = {key: value for key, value in client_state.items() if key not in deepspeed_states}
    # State dict keys will include reference to wrapper _LightningModuleWrapperBase in old checkpoints created in
    # Lightning version < 2.1. Delete the `_forward_module` prefix before saving.
    state_dict = {_remove_prefix(k, "_forward_module."): state_dict[k] for k in state_dict}
    client_state["state_dict"] = state_dict

    print(f"Saving fp32 state dict to {output_file}")
    torch.save(client_state, output_file)

    return client_state


def _remove_prefix(key: str, prefix: str) -> str:
    return key[len(prefix) :] if key.startswith(prefix) else key
