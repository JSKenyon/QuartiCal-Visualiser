import panel as pn
from qcvis.datamanager import DataManager
from qcvis.interface import GainInspector
from qcvis.parameterised import ParamInspector

from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated


def main():
    typer.run(app)


def app(
    gain_path: Annotated[
        Path,
        typer.Argument(
            help="Path to QuartiCal gain.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            writable=True,
            readable=True,
            resolve_path=True
        )
    ],
    port: Annotated[
        int,
        typer.Option(
            help="Port on which to serve the visualiser."
        )
    ] = 5006
):
    
    # Mangle the path into the format required by daskms.
    gain_path = Path(f"{gain_path.parent}::{gain_path.stem}")

    datamanager = DataManager(gain_path, fields=["gains", "gain_flags"])
    interface = GainInspector(datamanager)
    # datamanager = DataManager(gain_path, fields=["params", "param_flags"])
    # interface = ParamInspector(datamanager)
    

    customised_params = pn.Param(
        interface.param,
        show_name=False,
        widgets={
            # 'update': {'visible': False},
            'flag': {
                "type": pn.widgets.Button,
                "description": (
                    "Flag gain solutions corresponding to selected regions."
                )
            },
            # 'correlation': pn.widgets.RadioButtonGroup
        }
    )

    layout = pn.template.MaterialTemplate(
        # site="Panel",
        title="QuartiCal-Visualiser",
        sidebar=customised_params,
        main=[interface.update_plot],
    ).servable()

    pn.serve(
        layout,
        port=port
    )