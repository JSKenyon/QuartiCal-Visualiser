import param
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

    inspectors = {}

    gain_dm = DataManager(gain_path, fields=["gains", "gain_flags"])
    gain_inspector = GainInspector(gain_dm)
    inspectors["Gains"] = gain_inspector

    try:
        param_dm = DataManager(gain_path, fields=["params", "param_flags"])
        param_inspector = ParamInspector(param_dm)
        inspectors["Parameters"] = param_inspector
    except KeyError:
        pass

    def get_widgets(value):
        return inspectors[value].widgets

    def get_plot(value):
        return inspectors[value].update_plot


    plot_type = pn.widgets.RadioButtonGroup(
        name="Inspector Type",
        options=list(inspectors.keys()),
        value=list(inspectors.keys())[0],
        sizing_mode="stretch_width"
    )

    bound_get_widgets = pn.bind(get_widgets, plot_type)
    bound_get_plot = pn.bind(get_plot, plot_type)

    layout = pn.template.MaterialTemplate(
        # site="Panel",
        title="QuartiCal-Visualiser",
        sidebar=[plot_type, bound_get_widgets],
        main=[bound_get_plot],
    ).servable()

    pn.serve(
        layout,
        port=port,
        show=False
    )