import pandas as pd

import hvplot.pandas  # NOQA - required to register hvpot behaviour.

import holoviews as hv
from holoviews import opts, streams

import param
import panel as pn

pd.options.mode.copy_on_write = True
pn.config.throttled = True  # Throttle all sliders.

hv.extension('bokeh', width="stretch_width")

axis_map = {
    "Time": "param_time",
    "Frequency": "param_freq",
    "Parameter": "params",
}

class ParamInspector(param.Parameterized):

    antenna = param.Selector(
        label="Antenna",
    )
    direction = param.Selector(
        label="Direction",
    )
    gain_param = param.Selector(
        label="Parameter",
    )
    x_axis = param.Selector(
        label="X Axis",
        objects=list(axis_map.keys()),
        default="Time"
    )
    y_axis = param.Selector(
        label="Y Axis",
        objects=list(axis_map.keys()),
        default="Parameter"
    )
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
    flag_antennas = param.Action(
        lambda x: x.param.trigger('flag_antennas'),
        label='FLAG (ALL ANTENNAS)'
    )
    flag = param.Action(
        lambda x: x.param.trigger('flag'),
        label='FLAG'
    )
    save = param.Action(
        lambda x: x.param.trigger('save'),
        label='SAVE'
    )

    def __init__(self, datamanager, **params):

        self.dm = datamanager
        self.data = self.dm.dataframe

        self.param.antenna.objects = self.dm.get_coord_values("antenna")
        self.param.antenna.default = self.param.antenna.objects[0]

        self.param.direction.objects = self.dm.get_coord_values("direction")
        self.param.direction.default = self.param.direction.objects[0]

        self.param.gain_param.objects = self.dm.get_coord_values("param_name")
        self.param.gain_param.default = self.param.gain_param.objects[0]

        super().__init__(**params)

        # Configure initial selection.
        self.dm.set_selection(
            antenna=self.antenna,
            direction=self.direction,
            param_name=self.gain_param
        )

        # # Ensure that amplitude is added to data on init.
        # self.dm.set_otf_columns("real")

        self.param.watch(
            self.update_flags,
            ['flag', 'flag_antennas'],
            queued=True
        )

        self.param.watch(self.write_flags, ['save'], queued=True)

        # Automatically update data selection when these fields change.
        self.param.watch(
            self.update_selection,
            ['antenna', 'direction', 'gain_param'],
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

        # self.param.rasterize_when.bounds = (1, len(self.current_selection))

    def update_flags(self, event):

        if not self.box_edit.data:  # Nothing has been flagged.
            return

        corners = self.box_edit.data

        for x_min, y_min, x_max, y_max in zip(*corners.values()):

            query = (
                f"{x_min} <= {axis_map[self.x_axis]} <= {x_max} &"
                f"{y_min} <= {axis_map[self.y_axis]} <= {y_max}"
            )

            flag_axes = ["param_name"]

            if event.name == "flag_antennas":
                flag_axes.append("antenna")

            self.dm.flag_selection("param_flags", query, flag_axes)

        self.dm.get_selection.cache_clear()  # Invalidate cache.

    def write_flags(self, event):
        self.dm.write_flags("param_flags")

    def update_selection(self, event):
        self.dm.set_selection(
            antenna=self.antenna,
            direction=self.direction,
            param_name=self.gain_param
        )

    def update_otf_columns(self, event):
        self.dm.set_otf_columns(
            **{
                axis_map[ax]: "params" for ax in self.current_axes
                if axis_map[ax] in self.dm.otf_column_map
            }
        )

    @property
    def current_axes(self):
        return [self.x_axis, self.y_axis]

    @property
    def current_selection(self):
        return self.dm.get_selection()

    def update_plot(self):

        pn.state.log(f'Plot update triggered.')

        sel = self.current_selection

        sel = sel[sel["param_flags"] != 1]

        plot = self.rectangles * sel.hvplot.scatter(
            x=axis_map[self.x_axis],
            y=axis_map[self.y_axis],
            rasterize=self.rasterized,
            # dynspread=True,
            resample_when=self.rasterize_when if self.rasterized else None,
            hover=False,
            responsive=True,
            height=800,
            # logz=True,
            # x_sampling=self.minimum_sampling.get(self.x_axis, None),
            # y_sampling=self.minimum_sampling.get(self.y_axis, None),
            pixel_ratio=self.pixel_ratio,
            xlabel=self.x_axis,
            ylabel=self.y_axis
        )

        pn.state.log(f'Plot update completed.')

        return plot