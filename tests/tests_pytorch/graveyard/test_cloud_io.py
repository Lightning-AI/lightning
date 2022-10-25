import glob
import os
import sys
import warnings

import pytest
import torch

import pytorch_lightning  # noqa: F401
from tests_pytorch import _PATH_LEGACY, _PROJECT_ROOT

LEGACY_CHECKPOINTS_PATH = os.path.join(_PATH_LEGACY, "checkpoints")
CHECKPOINT_EXTENSION = ".ckpt"
# load list of all back compatible versions
with open(os.path.join(_PROJECT_ROOT, "legacy", "back-compatible-versions.txt")) as fp:
    LEGACY_BACK_COMPATIBLE_PL_VERSIONS = [ln.strip() for ln in fp.readlines()]


@pytest.mark.parametrize("pl_version", LEGACY_BACK_COMPATIBLE_PL_VERSIONS)
@pytest.mark.skipif(
    not "pytorch_" + "lightning" in sys.modules, reason="This test is only relevant for the standalone package"
)
def test_imports_standalone(pl_version: str):
    assert any(
        key.startswith("pytorch_" + "lightning") for key in sys.modules.keys()
    ), "Imported PL, so it has to be in sys.modules"
    path_legacy = os.path.join(LEGACY_CHECKPOINTS_PATH, pl_version)
    path_ckpts = sorted(glob.glob(os.path.join(path_legacy, f"*{CHECKPOINT_EXTENSION}")))
    assert path_ckpts, f'No checkpoints found in folder "{path_legacy}"'
    path_ckpt = path_ckpts[-1]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        torch.load(path_ckpt)

    assert any(
        key.startswith("pytorch_" + "lightning") for key in sys.modules.keys()
    ), "Imported PL, so it has to be in sys.modules"
    assert not any(
        key.startswith("lightning.pytorch") for key in sys.modules.keys()
    ), "Did not import the unified package, so it should not be in sys.modules"


@pytest.mark.parametrize("pl_version", LEGACY_BACK_COMPATIBLE_PL_VERSIONS)
@pytest.mark.skipif(
    "pytorch_" + "lightning" in sys.modules, reason="This test is only relevant for the unified package"
)
def test_imports_unified(pl_version: str):
    assert any(
        key.startswith("lightning.pytorch") for key in sys.modules.keys()
    ), "Imported unified package, so it has to be in sys.modules"
    assert not any(
        key.startswith("pytorch_" + "lightning") for key in sys.modules.keys()
    ), "Should not import standalone package, all imports should be redirected to the unified package"

    path_legacy = os.path.join(LEGACY_CHECKPOINTS_PATH, pl_version)
    path_ckpts = sorted(glob.glob(os.path.join(path_legacy, f"*{CHECKPOINT_EXTENSION}")))
    assert path_ckpts, f'No checkpoints found in folder "{path_legacy}"'
    path_ckpt = path_ckpts[-1]

    with pytest.warns(match="Redirecting imports of"):
        torch.load(path_ckpt)

    assert any(
        key.startswith("lightning.pytorch") for key in sys.modules.keys()
    ), "Imported unified package, so it has to be in sys.modules"
    assert not any(
        key.startswith("pytorch_" + "lightning") for key in sys.modules.keys()
    ), "Should not import standalone package, all imports should be redirected to the unified package"
