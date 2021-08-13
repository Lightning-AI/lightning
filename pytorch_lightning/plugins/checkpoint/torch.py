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
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.plugins.checkpoint.checkpoint import CheckpointIOPlugin
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.cloud_io import atomic_save
from pytorch_lightning.utilities.cloud_io import load as pl_load


class TorchCheckpointIOPlugin(CheckpointIOPlugin):
    def save_checkpoint(
        self, checkpoint: Dict[str, Any], path: Union[str, Path], storage_options: Optional[Any] = None
    ) -> None:
        try:
            # write the checkpoint dictionary on the file
            atomic_save(checkpoint, path)
        except AttributeError as err:
            key = pl.LightningModule.CHECKPOINT_HYPER_PARAMS_KEY
            checkpoint.pop(key, None)
            rank_zero_warn(f"Warning, `{key}` dropped from checkpoint. An attribute is not picklable: {err}")
            atomic_save(checkpoint, path)

    def load_checkpoint(
        self, path: Union[str, Path], storage_options: Optional[Callable] = lambda storage, loc: storage
    ) -> Dict[str, Any]:
        return pl_load(path, map_location=storage_options)
