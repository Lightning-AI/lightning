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

import gc
from typing import Any, Dict, Union

import torch

RECURSIVE_DICT_WITH_TENSORS = Union[Dict[str, torch.Tensor], Dict[Any, Any]]


def recursive_detach(
    in_dict: RECURSIVE_DICT_WITH_TENSORS,
    to_cpu: bool = False,
) -> Dict[str, Union[Any, Dict[str, torch.Tensor], torch.Tensor]]:
    """Detach all tensors in `in_dict`.

    May operate recursively if some of the values in `in_dict` are dictionaries
    which contain instances of `torch.Tensor`. Other types in `in_dict` are
    not affected by this utility function.

    Args:
        in_dict: Dictionary with tensors to detach
        to_cpu: Whether to move tensor to cpu

    Return:
        out_dict: Dictionary with detached tensors
    """
    out_dict = {}
    for k, v in in_dict.items():
        if isinstance(v, dict):
            v = recursive_detach(v, to_cpu=to_cpu)
        elif callable(getattr(v, 'detach', None)):
            v = v.detach()
            if to_cpu:
                v = v.cpu()
        out_dict[k] = v
    return out_dict


def is_oom_error(exception: Exception) -> bool:
    return is_cuda_out_of_memory(exception) \
        or is_cudnn_snafu(exception) \
        or is_out_of_cpu_memory(exception)


# based on https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
def is_cuda_out_of_memory(exception: Exception) -> bool:
    return isinstance(exception, RuntimeError) \
        and len(exception.args) == 1 \
        and "CUDA" in exception.args[0] \
        and "out of memory" in exception.args[0]


# based on https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
def is_cudnn_snafu(exception: Exception) -> bool:
    # For/because of https://github.com/pytorch/pytorch/issues/4107
    return isinstance(exception, RuntimeError) \
        and len(exception.args) == 1 \
        and "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED." in exception.args[0]


# based on https://github.com/BlackHC/toma/blob/master/toma/cpu_memory.py
def is_out_of_cpu_memory(exception: Exception) -> bool:
    return isinstance(exception, RuntimeError) \
        and len(exception.args) == 1 \
        and "DefaultCPUAllocator: can't allocate memory" in exception.args[0]


# based on https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
def garbage_collection_cuda() -> None:
    """Garbage collection Torch (CUDA) memory."""
    gc.collect()
    if torch.cuda.is_available():
        try:
            # This is the last thing that should cause an OOM error, but seemingly it can.
            torch.cuda.empty_cache()
        except RuntimeError as exception:
            if not is_oom_error(exception):
                # Only handle OOM errors
                raise
