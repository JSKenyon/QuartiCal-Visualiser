import pandas as pd

import hvplot.pandas  # NOQA - required to register hvpot behaviour.

import numpy as np
from math import prod

import holoviews as hv
from holoviews import opts, streams

import param
import panel as pn

from qcvis.datamanager import DataManager

pd.options.mode.copy_on_write = True
pn.config.throttled = True  # Throttle all sliders.

hv.extension('bokeh', width="stretch_both")


class Inspector(param.Parameterized):

    axis_map = {}  # Specific inspectors hsould provide valid mappings.

    rasterized = param.Boolean(
        label="Rasterize",
        default=True
    )
    # Set the bounds during the init step.
    rasterize_when = param.Integer(
        label="Rasterize Limit",
        bounds=(1, None),
        step=10000,
        default=50000,
    )
    pixel_ratio = param.Number(
        label="Pixel ratio",
        bounds=(0.1, 2),
        step=0.05,
        default=0.25
    )
    flag_mode = param.Selector(
        label='FLAGGING MODE',
        objects=["SELECTED ANTENNA", "ALL ANTENNAS"],
        default="SELECTED ANTENNA"
    )
    flag_axis = param.Selector(
        label='FLAGGING AXIS',
        objects=["SELECTION", "SELECTION (X-AXIS)", "SELECTION (Y-AXIS)"],
        default="SELECTION"
    )
    flag = param.Action(
        lambda x: x.param.trigger('flag'),
        label='APPLY FLAGS'
    )
    reset = param.Action(
        lambda x: x.param.trigger('reset'),
        label='RESET FLAGS'
    )
    save = param.Action(
        lambda x: x.param.trigger('save'),
        label='SAVE FLAGS'
    )

    _selection_parameters = [
        "x_axis",
        "y_axis",
    ]

    _display_parameters = [
        "rasterized",
        "rasterize_when",
        "pixel_ratio",
    ]

    _flag_parameters = [
        "flag",
        "flag_mode",
        "flag_axis",
        "reset",
        "save"
    ]

    def __init__(self, data_path, data_field, flag_field, **params):

        self.dm = DataManager(data_path, fields=[data_field, flag_field])
        self.data_field = data_field
        self.flag_field = flag_field

        dims = list(self.dm.dataset[self.data_field].dims)

        for dim in dims:
            self.param.add_parameter(
                dim,
                param.Selector(
                    label=dim.capitalize(),
                    objects=self.dm.get_coord_values(dim).tolist()                    
                )
            )

        for i, ax in enumerate(["x_axis", "y_axis"]):
            self.param.add_parameter(
                ax,
                param.Selector(
                    label=ax.replace("_", " ").capitalize(),
                    objects=list(self.axis_map.keys()),
                    default=list(self.axis_map.keys())[i]                    
                )
            )

        super().__init__(**params)

        # Configure initial selection.
        self.update_selection()

        # # Ensure that amplitude is added to data on init. TODO: The plottable
        # # axes are term dependent i.e. this shouldn't be here.
        # self.dm.set_otf_columns(amplitude="gains")

        self.param.watch(self.update_flags, ['flag'], queued=True)
        self.param.watch(self.write_flags, ['save'], queued=True)
        self.param.watch(self.reset_flags, ['reset'], queued=True)

        # Automatically update data selection when these fields change.
        self.param.watch(
            self.update_selection,
            dims,
            queued=True
        )

        # Automatically update on-the-fly columns when these fields change.
        self.param.watch(
            self.update_otf_columns,
            ['x_axis', 'y_axis'],
            queued=True
        )

        # Empty Rectangles for overlay
        self.rectangles = hv.Rectangles([])
        # Attach a BoxEdit stream to the Rectangles
        self.box_edit = streams.BoxEdit(source=self.rectangles)

        # Get initial selection so we can reason about it.
        selection = self.dm.get_selection()
        # Start in the appropriate state based on size of selection.
        self.rasterized = prod(selection.sizes.values()) > self.rasterize_when

    def update_flags(self, event):

        if not self.box_edit.data:  # Nothing has been flagged.
            return

        corners = self.box_edit.data
        axes = ["antenna"] if self.flag_mode == "ALL ANTENNAS" else []

        for x_min, y_min, x_max, y_max in zip(*corners.values()):

            criteria = {}

            if self.flag_axis in ["SELECTION", "SELECTION (X-AXIS)"]:
                criteria[self.axis_map[self.x_axis]] = (x_min, x_max)
            if self.flag_axis in ["SELECTION", "SELECTION (Y-AXIS)"]:
                criteria[self.axis_map[self.y_axis]] = (y_min, y_max)

            self.dm.flag_selection(self.flag_field, criteria, axes=axes)

    def reset_flags(self, event=None):
        self.dm.reset()

    def write_flags(self, event=None):
        self.dm.write_flags(self.flag_field)

    def update_selection(self, event=None):
        return NotImplementedError(f"update_selection not yet implemented.")

    def update_otf_columns(self, event=None):
        self.dm.set_otf_columns(
            **{
                self.axis_map[ax]: self.data_field for ax in self.current_axes
                if self.axis_map[ax] in self.dm.otf_column_map
            }
        )

    @property
    def current_axes(self):
        return [self.x_axis, self.y_axis]

    def update_plot(self):

        pn.state.log(f'Plot update triggered.')

        plot_data = self.dm.get_plot_data(
            self.axis_map[self.x_axis],
            self.axis_map[self.y_axis],
            self.data_field,
            self.flag_field
        )

        plot = self.rectangles * plot_data.hvplot.scatter(
            x=self.axis_map[self.x_axis],
            y=self.axis_map[self.y_axis],
            rasterize=self.rasterized,
            # dynspread=True,
            resample_when=self.rasterize_when if self.rasterized else None,
            hover=False,
            responsive=True,
            # logz=True,
            # x_sampling=self.minimum_sampling.get(self.x_axis, None),
            # y_sampling=self.minimum_sampling.get(self.y_axis, None),
            pixel_ratio=self.pixel_ratio,
            xlabel=self.x_axis,
            ylabel=self.y_axis
        )

        pn.state.log(f'Plot update completed.')

        return plot
    
    @property
    def widgets(self):

        widget_opts = {}

        for k in self.param.objects().keys():
            widget_opts[k] = {"sizing_mode": "stretch_width"}

        display_widgets = pn.Param(
            self.param,
            parameters=self._display_parameters,
            name="DISPLAY",
            widgets=widget_opts
        )

        selection_widgets = pn.Param(
            self.param,
            parameters=self._selection_parameters,
            name="SELECTION",
            widgets=widget_opts
        )

        widget_opts["flag_mode"].update(
            {
                "type": pn.widgets.RadioButtonGroup,
                "orientation": "vertical",
                "name": "FLAGGING MODE"
            }
        )

        widget_opts["flag_axis"].update(
            {
                "type": pn.widgets.RadioButtonGroup,
                "orientation": "vertical",
                "name": "FLAGGING AXIS"
            }
        )

        flagging_widgets = pn.Param(
            self.param,
            parameters=self._flag_parameters,
            name="FLAGGING",
            widgets=widget_opts
        )

        return pn.Column(
            pn.WidgetBox(display_widgets),
            pn.WidgetBox(selection_widgets),
            pn.WidgetBox(flagging_widgets)
        )