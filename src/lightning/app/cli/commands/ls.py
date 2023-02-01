import click
from typing import Optional
from lightning.app.utilities.cloud import _get_project
import os
from lightning.app.utilities.app_helpers import Logger
from lightning.app.utilities.cloud import _get_project
from lightning.app.utilities.network import LightningClient
from lightning.app.cli.commands.connection import _LIGHTNING_CONNECTION_FOLDER
from rich.color import ANSI_COLOR_NAMES
import rich

logger = Logger(__name__)


@click.option("--project_id", required=False)
@click.option("--app_id", required=False)
def ls(path: Optional[str] = None, project_id: Optional[str] = None, app_id: Optional[str] = None) -> None:

    cd_file = os.path.join(_LIGHTNING_CONNECTION_FOLDER, "cd.txt")
    root = '/'
    paths = []

    from rich.columns import Columns
    from rich.panel import Panel
    from rich.live import Live
    from rich.text import Text
    from rich.spinner import Spinner, SPINNERS

    spinner =  Spinner("point", text=Text("pending...", style="white"))

    console = rich.console.Console()

    with Live(spinner, console=console, transient=True) as live:

        if not os.path.exists(_LIGHTNING_CONNECTION_FOLDER):
            os.makedirs(_LIGHTNING_CONNECTION_FOLDER)

        if not os.path.exists(cd_file):
            with open(cd_file, "w") as f:
                f.write(root + "\n")
        else:
            with open(cd_file, "r") as f:
                lines = f.readlines()
                root = lines[0].replace("\n", "")

        client = LightningClient()
        if not project_id:
            project_id = _get_project(client, verbose=False).project_id

        lit_apps = client.lightningapp_instance_service_list_lightningapp_instances(project_id=project_id).lightningapps

        if app_id:
            lit_apps = [lit_app for lit_app in lit_apps if lit_app.id == app_id or lit_app.name == app_id]
        else:
            lit_apps = [lit_app for lit_app in lit_apps]

        if not paths:
            for lit_app in lit_apps:
                if root == '/' and app_id is None:
                    paths.append(_add_colors(lit_app.name, color="blue"))
                else:
                    if not root[1:].startswith(lit_app.name):
                        continue
                    num_split = len([split for split in root.split('/') if split != ''])
                    # TODO: Replace with project level endpoints  
                    response = client.lightningapp_instance_service_list_lightningapp_instance_artifacts(project_id, lit_app.id)
                    for artifact in response.artifacts:
                        path = os.path.join(lit_app.name, artifact.filename)
                        splits = path.split("/")

                        # display files otherwise folders
                        if len(splits) == num_split + 1:
                            color = "white"
                        else:
                            color= "blue"
                        
                        paths.append(_add_colors(splits[num_split], color=color))
        
        os.remove(cd_file)

        with open(cd_file, "w") as f:
            f.write(root + "\n")

    rich.print(*sorted(set(paths)))


def _add_colors(filename: str, color: Optional[str] = None) -> str:
    colors = list(ANSI_COLOR_NAMES)
    if color is None:
        color = "magenta"

        if ".yaml" in filename:
            color = colors[1]

        elif ".ckpt" in filename:
            color = colors[2]

        elif "events.out.tfevents" in filename:
            color = colors[3]

        elif ".py" in filename:
            color = colors[4]

        elif ".png" in filename:
            color = colors[5]

    return f"[{color}]{filename}[/{color}]"