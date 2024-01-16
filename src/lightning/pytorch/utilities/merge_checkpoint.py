import re
from typing import Any, Dict

import torch

from lightning.fabric.utilities.load import _load_distributed_checkpoint
from lightning.fabric.utilities.merge_checkpoint import _parse_cli_args, _process_cli_args


def _format_checkpoint(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """The FSDP strategy saves the checkpoint in a special format.

    This function converts it to the standard format the Lightning Trainer expects.

    """
    # Rename the model key
    checkpoint["state_dict"] = checkpoint.pop("model")

    optimizer_keys = [key for key in checkpoint if re.match("optimizer_[0-9]+", key)]
    if not optimizer_keys:
        return checkpoint

    # Optimizers are saved in special keys named `optimizer_0`, `optimizer_1`, etc.
    # These need to be merged back into a Python list
    checkpoint["optimizer_states"] = [checkpoint.pop(f"optimizer_{opt_idx}") for opt_idx in range(len(optimizer_keys))]
    return checkpoint


if __name__ == "__main__":
    args = _parse_cli_args()
    config = _process_cli_args(args)
    checkpoint = _load_distributed_checkpoint(config.checkpoint_folder)
    checkpoint = _format_checkpoint(checkpoint)
    torch.save(checkpoint, config.output_file)
