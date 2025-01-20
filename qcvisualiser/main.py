import panel as pn

from pathlib import Path

import typer
from typing_extensions import Annotated

from qcvisualiser.gain_inspector import GainInspector
from qcvisualiser.param_inspector import ParamInspector


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

    gain_inspector = GainInspector(gain_path, "gains", "gain_flags")
    inspectors["Gains"] = gain_inspector

    try:
        param_inspector = ParamInspector(gain_path, "params", "param_flags")
        inspectors["Parameters"] = param_inspector
    except KeyError:
        pass

    def get_widgets(value):
        widgets =  inspectors[value].widgets

        # Disable flagging on gains when gain is parameterized.
        if "Parameters" in inspectors and value == "Gains":
            widgets = pn.Column(widgets[0], widgets[1])

        return widgets

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