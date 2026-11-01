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
"""LightningDataModule for loading DataLoaders with ease."""

import inspect
import os
from collections.abc import Iterable, Sized
from typing import IO, Any, Optional, Union, cast

from lightning_utilities import apply_to_collection
from torch.utils.data import DataLoader, Dataset, IterableDataset
from typing_extensions import Self

import lightning.pytorch as pl
from lightning.fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH
from lightning.pytorch.core.hooks import DataHooks
from lightning.pytorch.core.mixins import HyperparametersMixin
from lightning.pytorch.core.saving import _load_from_checkpoint
from lightning.pytorch.utilities.model_helpers import _restricted_classmethod
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS


class LightningDataModule(DataHooks, HyperparametersMixin):
    """A DataModule standardizes the training, val, test splits, data preparation and transforms. The main advantage is
    consistent data splits, data preparation and transforms across models.

    Example::

        import lightning as L
        import torch.utils.data as data
        from lightning.pytorch.demos.boring_classes import RandomDataset

        class MyDataModule(L.LightningDataModule):
            def prepare_data(self):
                # download, IO, etc. Useful with shared filesystems
                # only called on 1 GPU/TPU in distributed
                ...

            def setup(self, stage):
                # make assignments here (val/train/test split)
                # called on every process in DDP
                dataset = RandomDataset(1, 100)
                self.train, self.val, self.test = data.random_split(
                    dataset, [80, 10, 10], generator=torch.Generator().manual_seed(42)
                )

            def train_dataloader(self):
                return data.DataLoader(self.train)

            def val_dataloader(self):
                return data.DataLoader(self.val)

            def test_dataloader(self):
                return data.DataLoader(self.test)

            def on_exception(self, exception):
                # clean up state after the trainer faced an exception
                ...

            def teardown(self):
                # clean up state after the trainer stops, delete files...
                # called on every process in DDP
                ...

    """

    name: Optional[str] = None
    CHECKPOINT_HYPER_PARAMS_KEY = "datamodule_hyper_parameters"
    CHECKPOINT_HYPER_PARAMS_NAME = "datamodule_hparams_name"
    CHECKPOINT_HYPER_PARAMS_TYPE = "datamodule_hparams_type"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        # Pointer to the trainer object
        self.trainer: Optional[pl.Trainer] = None

    @classmethod
    def from_datasets(
        cls,
        train_dataset: Optional[Union[Dataset, Iterable[Dataset]]] = None,
        val_dataset: Optional[Union[Dataset, Iterable[Dataset]]] = None,
        test_dataset: Optional[Union[Dataset, Iterable[Dataset]]] = None,
        predict_dataset: Optional[Union[Dataset, Iterable[Dataset]]] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        **datamodule_kwargs: Any,
    ) -> "LightningDataModule":
        r"""Create an instance from torch.utils.data.Dataset.

        Args:
            train_dataset: Optional dataset or iterable of datasets to be used for train_dataloader()
            val_dataset: Optional dataset or iterable of datasets to be used for val_dataloader()
            test_dataset: Optional dataset or iterable of datasets to be used for test_dataloader()
            predict_dataset: Optional dataset or iterable of datasets to be used for predict_dataloader()
            batch_size: Batch size to use for each dataloader. Default is 1. This parameter gets forwarded to the
                ``__init__`` if the datamodule has such a name defined in its signature.
            num_workers: Number of subprocesses to use for data loading. 0 means that the
                data will be loaded in the main process. Number of CPUs available. This parameter gets forwarded to the
                ``__init__`` if the datamodule has such a name defined in its signature.
            **datamodule_kwargs: Additional parameters that get passed down to the datamodule's ``__init__``.

        """

        def dataloader(ds: Dataset, shuffle: bool = False) -> DataLoader:
            shuffle &= not isinstance(ds, IterableDataset)
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

        def train_dataloader() -> TRAIN_DATALOADERS:
            return apply_to_collection(train_dataset, Dataset, dataloader, shuffle=True)

        def val_dataloader() -> EVAL_DATALOADERS:
            return apply_to_collection(val_dataset, Dataset, dataloader)

        def test_dataloader() -> EVAL_DATALOADERS:
            return apply_to_collection(test_dataset, Dataset, dataloader)

        def predict_dataloader() -> EVAL_DATALOADERS:
            return apply_to_collection(predict_dataset, Dataset, dataloader)

        candidate_kwargs = {"batch_size": batch_size, "num_workers": num_workers}
        accepted_params = inspect.signature(cls.__init__).parameters
        accepts_kwargs = any(param.kind == param.VAR_KEYWORD for param in accepted_params.values())
        if accepts_kwargs:
            special_kwargs = candidate_kwargs
        else:
            accepted_param_names = set(accepted_params)
            accepted_param_names.discard("self")
            special_kwargs = {k: v for k, v in candidate_kwargs.items() if k in accepted_param_names}

        datamodule = cls(**datamodule_kwargs, **special_kwargs)
        if train_dataset is not None:
            datamodule.train_dataloader = train_dataloader  # type: ignore[method-assign]
        if val_dataset is not None:
            datamodule.val_dataloader = val_dataloader  # type: ignore[method-assign]
        if test_dataset is not None:
            datamodule.test_dataloader = test_dataloader  # type: ignore[method-assign]
        if predict_dataset is not None:
            datamodule.predict_dataloader = predict_dataloader  # type: ignore[method-assign]
        return datamodule

    def state_dict(self) -> dict[str, Any]:
        """Called when saving a checkpoint, implement to generate and save datamodule state.

        Returns:
            A dictionary containing datamodule state.

        """
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Called when loading a checkpoint, implement to reload datamodule state given datamodule state_dict.

        Args:
            state_dict: the datamodule state returned by ``state_dict``.

        """
        pass

    def on_exception(self, exception: BaseException) -> None:
        """Called when the trainer execution is interrupted by an exception."""
        pass

    @_restricted_classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[_PATH, IO],
        map_location: _MAP_LOCATION_TYPE = None,
        hparams_file: Optional[_PATH] = None,
        weights_only: Optional[bool] = None,
        **kwargs: Any,
    ) -> Self:
        r"""Primary way of loading a datamodule from a checkpoint. When Lightning saves a checkpoint it stores the
        arguments passed to ``__init__``  in the checkpoint under ``"datamodule_hyper_parameters"``.

        Any arguments specified through \*\*kwargs will override args stored in ``"datamodule_hyper_parameters"``.

        Args:
            checkpoint_path: Path to checkpoint. This can also be a URL, or file-like object
            map_location:
                If your checkpoint saved a GPU model and you now load on CPUs
                or a different number of GPUs, use this to map to the new setup.
                The behaviour is the same as in :func:`torch.load`.
            hparams_file: Optional path to a ``.yaml`` or ``.csv`` file with hierarchical structure
                as in this example::

                    dataloader:
                        batch_size: 32

                You most likely won't need this since Lightning will always save the hyperparameters
                to the checkpoint.
                However, if your checkpoint weights don't have the hyperparameters saved,
                use this method to pass in a ``.yaml`` file with the hparams you'd like to use.
                These will be converted into a :class:`~dict` and passed into your
                :class:`LightningDataModule` for use.

                If your datamodule's ``hparams`` argument is :class:`~argparse.Namespace`
                and ``.yaml`` file has hierarchical structure, you need to refactor your datamodule to treat
                ``hparams`` as :class:`~dict`.
            weights_only: If ``True``, restricts loading to ``state_dicts`` of plain ``torch.Tensor`` and other
                primitive types. If loading a checkpoint from a trusted source that contains an ``nn.Module``, use
                ``weights_only=False``. If loading checkpoint from an untrusted source, we recommend using
                ``weights_only=True``. For more information, please refer to the
                `PyTorch Developer Notes on Serialization Semantics <https://docs.pytorch.org/docs/main/notes/serialization.html#id3>`_.
            \**kwargs: Any extra keyword args needed to init the datamodule. Can also be used to override saved
                hyperparameter values.

        Return:
            :class:`LightningDataModule` instance with loaded weights and hyperparameters (if available).

        Note:
            ``load_from_checkpoint`` is a **class** method. You must use your :class:`LightningDataModule`
            **class** to call it instead of the :class:`LightningDataModule` instance, or a
            ``TypeError`` will be raised.

        Example::

            # load weights without mapping ...
            datamodule = MyLightningDataModule.load_from_checkpoint('path/to/checkpoint.ckpt')

            # or load weights and hyperparameters from separate files.
            datamodule = MyLightningDataModule.load_from_checkpoint(
                'path/to/checkpoint.ckpt',
                hparams_file='/path/to/hparams_file.yaml'
            )

            # override some of the params with new values
            datamodule = MyLightningDataModule.load_from_checkpoint(
                PATH,
                batch_size=32,
                num_workers=10,
            )

        """
        loaded = _load_from_checkpoint(
            cls,
            checkpoint_path,
            map_location=map_location,
            hparams_file=hparams_file,
            strict=None,
            weights_only=weights_only,
            **kwargs,
        )
        return cast(Self, loaded)

    def __len__(self) -> int:
        """Return the number of samples in the training dataset.

        Override this method in your subclass to return the total number of
        training samples. By default, this method attempts to determine the
        length from the training dataloader's dataset.

        Returns:
            The number of samples in the training dataset.

        Raises:
            TypeError: If ``train_dataloader`` is not configured or the dataset
                does not implement ``__len__``.

        Example::

            class MyDataModule(LightningDataModule):
                def __len__(self) -> int:
                    return len(self.train_dataset)

        """
        try:
            loader = self.train_dataloader()
            dataset = loader.dataset
            return len(dataset)  # type: ignore[arg-type]
        except Exception as ex:
            raise TypeError(
                f"'{type(self).__name__}' object has no len(). "
                "Override `__len__` in your LightningDataModule subclass to return the number of training samples."
            ) from ex

    def __str__(self) -> str:
        """Return a string representation of the datasets that are set up.

        Returns:
            A string representation of the datasets that are setup.

        """

        class dataset_info:
            def __init__(self, available: bool, length: str) -> None:
                self.available = available
                self.length = length

        def retrieve_dataset_info(loader: DataLoader) -> dataset_info:
            """Helper function to compute dataset information."""
            dataset = loader.dataset
            size: str = str(len(dataset)) if isinstance(dataset, Sized) else "NA"

            return dataset_info(True, size)

        def loader_info(
            loader: Union[DataLoader, Iterable[DataLoader]],
        ) -> Union[dataset_info, Iterable[dataset_info]]:
            """Helper function to compute dataset information."""
            return apply_to_collection(loader, DataLoader, retrieve_dataset_info)

        def extract_loader_info(methods: list[tuple[str, str]]) -> dict:
            """Helper function to extract information for each dataloader method."""
            info: dict[str, Union[dataset_info, Iterable[dataset_info]]] = {}
            for loader_name, func_name in methods:
                loader_method = getattr(self, func_name, None)

                try:
                    loader = loader_method()  # type: ignore
                    info[loader_name] = loader_info(loader)
                except Exception:
                    info[loader_name] = dataset_info(False, "")

            return info

        def format_loader_info(info: dict[str, Union[dataset_info, Iterable[dataset_info]]]) -> str:
            """Helper function to format loader information."""
            output = []
            for loader_name, loader_info in info.items():
                # Single dataset
                if isinstance(loader_info, dataset_info):
                    loader_info_formatted = "None" if not loader_info.available else f"size={loader_info.length}"
                # Iterable of datasets
                else:
                    loader_info_formatted = " ; ".join(
                        "None" if not loader_info_i.available else f"{i}. size={loader_info_i.length}"
                        for i, loader_info_i in enumerate(loader_info, start=1)
                    )

                output.append(f"{{{loader_name}: {loader_info_formatted}}}")

            return os.linesep.join(output)

        # Available dataloader methods
        datamodule_loader_methods: list[tuple[str, str]] = [
            ("Train dataloader", "train_dataloader"),
            ("Validation dataloader", "val_dataloader"),
            ("Test dataloader", "test_dataloader"),
            ("Predict dataloader", "predict_dataloader"),
        ]

        # Retrieve information for each dataloader method
        dataloader_info = extract_loader_info(datamodule_loader_methods)
        # Format the information
        dataloader_str = format_loader_info(dataloader_info)
        return dataloader_str
