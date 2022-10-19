import pytest

from lightning_app import CloudCompute
from lightning_app.storage import Mount


def test_cloud_compute_unsupported_features():
    with pytest.raises(ValueError, match="Clusters are't supported yet"):
        CloudCompute("gpu", clusters="as")
    with pytest.raises(ValueError, match="Setting a wait timeout isn't supported yet"):
        CloudCompute("gpu", wait_timeout=1)


def test_cloud_compute_names():
    assert CloudCompute().name == "default"
    assert CloudCompute("cpu-small").name == "cpu-small"
    assert CloudCompute("coconut").name == "coconut"  # the backend is responsible for validation of names


def test_cloud_compute_shared_memory():
    cloud_compute = CloudCompute("gpu", shm_size=1100)
    assert cloud_compute.shm_size == 1100


def test_cloud_compute_with_mounts():
    mount_1 = Mount(source="s3://foo/", root_dir="./foo")
    mount_2 = Mount(source="s3://foo/bar/", root_dir="./bar")

    cloud_compute = CloudCompute("gpu", mount=mount_1)
    assert cloud_compute.mount == mount_1

    cloud_compute = CloudCompute("gpu", mount=[mount_1, mount_2])
    assert cloud_compute.mount == [mount_1, mount_2]

    cc_dict = cloud_compute.to_dict()
    assert "mount" in cc_dict
    assert cc_dict["mount"] == [
        {"root_dir": "./foo", "source": "s3://foo/"},
        {"root_dir": "./bar", "source": "s3://foo/bar/"},
    ]

    assert CloudCompute.from_dict(cc_dict) == cloud_compute


def test_cloud_compute_with_non_unique_mount_root_dirs():
    mount_1 = Mount(source="s3://foo/", root_dir="./foo")
    mount_2 = Mount(source="s3://foo/bar/", root_dir="./foo")

    with pytest.raises(ValueError, match="Every Mount attached to a work must have a unique 'root_dir' argument."):
        CloudCompute("gpu", mount=[mount_1, mount_2])
