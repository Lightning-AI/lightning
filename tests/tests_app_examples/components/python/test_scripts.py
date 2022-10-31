import os

import pytest
from click.testing import CliRunner
from tests_app import _PROJECT_ROOT

from lightning_app.cli.lightning_cli import run_app
from lightning_app.testing.helpers import _run_script, _RunIf


@_RunIf(pl=True)
@pytest.mark.parametrize(
    "file",
    [
        pytest.param("component_tracer.py"),
        pytest.param("component_popen.py"),
    ],
)
def test_scripts(file):
    _run_script(str(os.path.join(_PROJECT_ROOT, f"examples/app_components/python/{file}")))


@pytest.mark.skip(reason="causing some issues with CI, not sure if the test is actually needed")
@_RunIf(pl=True)
def test_components_app_example():

    runner = CliRunner()
    result = runner.invoke(
        run_app,
        [
            os.path.join(_PROJECT_ROOT, "examples/app_components/python/app.py"),
            "--blocking",
            "False",
            "--open-ui",
            "False",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert "tracer script succeed" in result.stdout
