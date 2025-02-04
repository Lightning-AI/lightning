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

import ast
import contextlib
import csv
import inspect
import logging
import os
from argparse import Namespace
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Callable, Optional, Union
from warnings import warn

import torch
import yaml
from lightning_utilities.core.apply_func import apply_to_collection

import lightning.pytorch as pl
from lightning.fabric.utilities.cloud_io import _is_dir, get_filesystem
from lightning.fabric.utilities.cloud_io import _load as pl_load
from lightning.fabric.utilities.data import AttributeDict
from lightning.fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH
from lightning.pytorch.accelerators import CUDAAccelerator, MPSAccelerator, XLAAccelerator
from lightning.pytorch.utilities.imports import _OMEGACONF_AVAILABLE
from lightning.pytorch.utilities.migration import pl_legacy_patch
from lightning.pytorch.utilities.migration.utils import _pl_migrate_checkpoint
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning.pytorch.utilities.parsing import parse_class_init_keys
from lightning.pytorch.utilities.rank_zero import rank_zero_warn

if TYPE_CHECKING:
    from torch.storage import UntypedStorage

log = logging.getLogger(__name__)
# the older shall be on the top
CHECKPOINT_PAST_HPARAMS_KEYS = ("hparams", "module_arguments")  # used in 0.7.6


def _load_from_checkpoint(
    cls: Union[type["pl.LightningModule"], type["pl.LightningDataModule"]],
    checkpoint_path: Union[_PATH, IO],
    map_location: _MAP_LOCATION_TYPE = None,
    hparams_file: Optional[_PATH] = None,
    strict: Optional[bool] = None,
    **kwargs: Any,
) -> Union["pl.LightningModule", "pl.LightningDataModule"]:
    map_location = map_location or _default_map_location
    with pl_legacy_patch():
        checkpoint = pl_load(checkpoint_path, map_location=map_location)

    # convert legacy checkpoints to the new format
    checkpoint = _pl_migrate_checkpoint(
        checkpoint, checkpoint_path=(checkpoint_path if isinstance(checkpoint_path, (str, Path)) else None)
    )

    if hparams_file is not None:
        extension = str(hparams_file).split(".")[-1]
        if extension.lower() == "csv":
            hparams = load_hparams_from_tags_csv(hparams_file)
        elif extension.lower() in ("yml", "yaml"):
            hparams = load_hparams_from_yaml(hparams_file)
        else:
            raise ValueError(".csv, .yml or .yaml is required for `hparams_file`")

        # overwrite hparams by the given file
        checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY] = hparams

    # TODO: make this a migration:
    # for past checkpoint need to add the new key
    checkpoint.setdefault(cls.CHECKPOINT_HYPER_PARAMS_KEY, {})
    # override the hparams with values that were passed in
    checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY].update(kwargs)

    if issubclass(cls, pl.LightningDataModule):
        return _load_state(cls, checkpoint, **kwargs)
    if issubclass(cls, pl.LightningModule):
        model = _load_state(cls, checkpoint, strict=strict, **kwargs)
        state_dict = checkpoint["state_dict"]
        if not state_dict:
            rank_zero_warn(f"The state dict in {checkpoint_path!r} contains no parameters.")
            return model

        device = next((t for t in state_dict.values() if isinstance(t, torch.Tensor)), torch.tensor(0)).device
        assert isinstance(model, pl.LightningModule)
        if device.type == "cpu" and model.device.type == "cpu":
            return model
        return model.to(device)

    raise NotImplementedError(f"Unsupported {cls}")


def _default_map_location(storage: "UntypedStorage", location: str) -> Optional["UntypedStorage"]:
    if (
        location.startswith("mps")
        and not MPSAccelerator.is_available()
        or location.startswith("cuda")
        and not CUDAAccelerator.is_available()
        or location.startswith("xla")
        and not XLAAccelerator.is_available()
    ):
        return storage.cpu()
    return None  # default behavior by `torch.load()`


def _load_state(
    cls: Union[type["pl.LightningModule"], type["pl.LightningDataModule"]],
    checkpoint: dict[str, Any],
    strict: Optional[bool] = None,
    **cls_kwargs_new: Any,
) -> Union["pl.LightningModule", "pl.LightningDataModule"]:
    cls_spec = inspect.getfullargspec(cls.__init__)
    cls_init_args_name = inspect.signature(cls.__init__).parameters.keys()

    self_var, args_var, kwargs_var = parse_class_init_keys(cls)
    drop_names = [n for n in (self_var, args_var, kwargs_var) if n]
    cls_init_args_name = list(filter(lambda n: n not in drop_names, cls_init_args_name))

    cls_kwargs_loaded = {}
    # pass in the values we saved automatically
    if cls.CHECKPOINT_HYPER_PARAMS_KEY in checkpoint:
        if issubclass(cls, pl.LightningModule):
            # TODO: make this a migration:
            # 1. (backward compatibility) Try to restore model hparams from checkpoint using old/past keys
            for _old_hparam_key in CHECKPOINT_PAST_HPARAMS_KEYS:
                cls_kwargs_loaded.update(checkpoint.get(_old_hparam_key, {}))

        # 2. Try to restore model hparams from checkpoint using the new key
        cls_kwargs_loaded.update(checkpoint.get(cls.CHECKPOINT_HYPER_PARAMS_KEY, {}))

        # 3. Ensure that `cls_kwargs_old` has the right type, back compatibility between dict and Namespace
        cls_kwargs_loaded = _convert_loaded_hparams(cls_kwargs_loaded, checkpoint.get(cls.CHECKPOINT_HYPER_PARAMS_TYPE))

        # 4. Update cls_kwargs_new with cls_kwargs_old, such that new has higher priority
        args_name = checkpoint.get(cls.CHECKPOINT_HYPER_PARAMS_NAME)
        if args_name and args_name in cls_init_args_name:
            cls_kwargs_loaded = {args_name: cls_kwargs_loaded}

    _cls_kwargs = {}
    _cls_kwargs.update(cls_kwargs_loaded)
    _cls_kwargs.update(cls_kwargs_new)

    instantiator = None
    instantiator_path = _cls_kwargs.pop("_instantiator", None)
    if instantiator_path is not None:
        # import custom instantiator
        module_path, name = instantiator_path.rsplit(".", 1)
        instantiator = getattr(__import__(module_path, fromlist=[name]), name)

    if not cls_spec.varkw:
        # filter kwargs according to class init unless it allows any argument via kwargs
        _cls_kwargs = {k: v for k, v in _cls_kwargs.items() if k in cls_init_args_name}

    obj = instantiator(cls, _cls_kwargs) if instantiator else cls(**_cls_kwargs)

    if isinstance(obj, pl.LightningDataModule):
        if obj.__class__.__qualname__ in checkpoint:
            obj.load_state_dict(checkpoint[obj.__class__.__qualname__])
        return obj

    if isinstance(obj, pl.LightningModule):
        if obj._strict_loading is not None and strict is not None and strict != obj.strict_loading:
            raise ValueError(
                f"You set `.load_from_checkpoint(..., strict={strict!r})` which is in conflict with"
                f" `{cls.__name__}.strict_loading={obj.strict_loading!r}. Please set the same value for both of them."
            )
        strict = obj.strict_loading if strict is None else strict

        if is_overridden("configure_model", obj):
            obj.configure_model()

        # give model a chance to load something
        obj.on_load_checkpoint(checkpoint)

    # load the state_dict on the model automatically
    keys = obj.load_state_dict(checkpoint["state_dict"], strict=strict)

    if not strict:
        if keys.missing_keys:
            rank_zero_warn(
                f"Found keys that are in the model state dict but not in the checkpoint: {keys.missing_keys}"
            )
        if keys.unexpected_keys:
            rank_zero_warn(
                f"Found keys that are not in the model state dict but in the checkpoint: {keys.unexpected_keys}"
            )

    return obj


def _convert_loaded_hparams(
    model_args: dict[str, Any], hparams_type: Optional[Union[Callable, str]] = None
) -> dict[str, Any]:
    """Convert hparams according given type in callable or string (past) format."""
    # if not hparams type define
    if not hparams_type:
        return model_args
    # if past checkpoint loaded, convert str to callable
    if isinstance(hparams_type, str):
        hparams_type = AttributeDict
    # convert hparams
    return hparams_type(model_args)


def update_hparams(hparams: dict, updates: dict) -> None:
    """Overrides hparams with new values.

    >>> hparams = {'c': 4}
    >>> update_hparams(hparams, {'a': {'b': 2}, 'c': 1})
    >>> hparams['a']['b'], hparams['c']
    (2, 1)
    >>> update_hparams(hparams, {'a': {'b': 4}, 'c': 7})
    >>> hparams['a']['b'], hparams['c']
    (4, 7)

    Args:
        hparams: the original params and also target object
        updates: new params to be used as update

    """
    for k, v in updates.items():
        # if missing, add the key
        if k not in hparams:
            hparams[k] = v
            continue

        # recurse if dictionary
        if isinstance(v, dict):
            update_hparams(hparams[k], updates[k])
        else:
            # update the value
            hparams.update({k: v})


def load_hparams_from_tags_csv(tags_csv: _PATH) -> dict[str, Any]:
    """Load hparams from a file.

    >>> hparams = Namespace(batch_size=32, learning_rate=0.001, data_root='./any/path/here')
    >>> path_csv = os.path.join('.', 'testing-hparams.csv')
    >>> save_hparams_to_tags_csv(path_csv, hparams)
    >>> hparams_new = load_hparams_from_tags_csv(path_csv)
    >>> vars(hparams) == hparams_new
    True
    >>> os.remove(path_csv)

    """
    fs = get_filesystem(tags_csv)
    if not fs.exists(tags_csv):
        rank_zero_warn(f"Missing Tags: {tags_csv}.", category=RuntimeWarning)
        return {}

    with fs.open(tags_csv, "r", newline="") as fp:
        csv_reader = csv.reader(fp, delimiter=",")
        return {row[0]: convert(row[1]) for row in list(csv_reader)[1:]}


def save_hparams_to_tags_csv(tags_csv: _PATH, hparams: Union[dict, Namespace]) -> None:
    fs = get_filesystem(tags_csv)
    if not _is_dir(fs, os.path.dirname(tags_csv)):
        raise RuntimeError(f"Missing folder: {os.path.dirname(tags_csv)}.")

    if isinstance(hparams, Namespace):
        hparams = vars(hparams)

    with fs.open(tags_csv, "w", newline="") as fp:
        fieldnames = ["key", "value"]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writerow({"key": "key", "value": "value"})
        for k, v in hparams.items():
            writer.writerow({"key": k, "value": v})


def load_hparams_from_yaml(config_yaml: _PATH, use_omegaconf: bool = True) -> dict[str, Any]:
    """Load hparams from a file.

        Args:
            config_yaml: Path to config yaml file
            use_omegaconf: If omegaconf is available and ``use_omegaconf=True``,
                the hparams will be converted to ``DictConfig`` if possible.

    >>> hparams = Namespace(batch_size=32, learning_rate=0.001, data_root='./any/path/here')
    >>> path_yaml = './testing-hparams.yaml'
    >>> save_hparams_to_yaml(path_yaml, hparams)
    >>> hparams_new = load_hparams_from_yaml(path_yaml)
    >>> vars(hparams) == hparams_new
    True
    >>> os.remove(path_yaml)

    """
    fs = get_filesystem(config_yaml)
    if not fs.exists(config_yaml):
        rank_zero_warn(f"Missing Tags: {config_yaml}.", category=RuntimeWarning)
        return {}

    with fs.open(config_yaml, "r") as fp:
        hparams = yaml.full_load(fp)

    if _OMEGACONF_AVAILABLE and use_omegaconf:
        from omegaconf import OmegaConf
        from omegaconf.errors import UnsupportedValueType, ValidationError

        with contextlib.suppress(UnsupportedValueType, ValidationError):
            return OmegaConf.create(hparams)
    return hparams


def save_hparams_to_yaml(config_yaml: _PATH, hparams: Union[dict, Namespace], use_omegaconf: bool = True) -> None:
    """
    Args:
        config_yaml: path to new YAML file
        hparams: parameters to be saved
        use_omegaconf: If omegaconf is available and ``use_omegaconf=True``,
            the hparams will be converted to ``DictConfig`` if possible.

    """
    fs = get_filesystem(config_yaml)
    if not _is_dir(fs, os.path.dirname(config_yaml)):
        raise RuntimeError(f"Missing folder: {os.path.dirname(config_yaml)}.")

    # convert Namespace or AD to dict
    if isinstance(hparams, Namespace):
        hparams = vars(hparams)
    elif isinstance(hparams, AttributeDict):
        hparams = dict(hparams)

    # saving with OmegaConf objects
    if _OMEGACONF_AVAILABLE and use_omegaconf:
        from omegaconf import OmegaConf
        from omegaconf.dictconfig import DictConfig
        from omegaconf.errors import UnsupportedValueType, ValidationError

        # deepcopy: hparams from user shouldn't be resolved
        hparams = deepcopy(hparams)
        hparams = apply_to_collection(hparams, DictConfig, OmegaConf.to_container, resolve=True)
        with fs.open(config_yaml, "w", encoding="utf-8") as fp:
            try:
                OmegaConf.save(hparams, fp)
                return
            except (UnsupportedValueType, ValidationError):
                pass

    if not isinstance(hparams, dict):
        raise TypeError("hparams must be dictionary")

    hparams_allowed = {}
    # drop parameters which contain some strange datatypes as fsspec
    for k, v in hparams.items():
        try:
            v = v.name if isinstance(v, Enum) else v
            yaml.dump(v)
        except (TypeError, ValueError):
            warn(f"Skipping '{k}' parameter because it is not possible to safely dump to YAML.")
            hparams[k] = type(v).__name__
        else:
            hparams_allowed[k] = v

    # saving the standard way
    with fs.open(config_yaml, "w", newline="") as fp:
        yaml.dump(hparams_allowed, fp)


def convert(val: str) -> Union[int, float, bool, str]:
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError) as err:
        log.debug(err)
        return val
