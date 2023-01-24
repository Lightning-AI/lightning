import os
from unittest import mock

from lightning_app.core.constants import get_cluster_driver, get_lightning_cloud_url


@mock.patch.dict(os.environ, {"LIGHTNING_CLOUD_URL": "https://beta.lightning.ai"})
def test_defaults():
    assert get_lightning_cloud_url() == "https://beta.lightning.ai"


def test_cluster_drive(monkeypatch):
    assert get_cluster_driver() == "k8s"

    monkeypatch.setenv("LIGHTNING_ENABLE_PREEMPTIBLE_WORKS", "1")
    assert get_cluster_driver() == "direct"

    monkeypatch.setenv("LIGHTNING_CLUSTER_DRIVER", "k8s")
    assert get_cluster_driver() == "k8s"
