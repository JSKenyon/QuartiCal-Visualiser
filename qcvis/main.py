''' An interactivate categorized chart based on a movie dataset.
This example shows the ability of Bokeh to create a dashboard with different
sorting options based on a given dataset.

'''
from pathlib import Path

import pandas as pd
import numpy as np
import xarray

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Div, Select, Slider, Button
from bokeh.plotting import figure

import hvplot.pandas

import holoviews as hv
from holoviews import opts, streams
from holoviews.operation.datashader import datashade, rasterize, shade, dynspread, spread
from holoviews.operation.downsample import downsample1d
from holoviews.operation.resample import ResampleOperation2D
from holoviews.operation import decimate

import param
import panel as pn
from panel.io import hold
import panel.widgets as pnw

from timedec import timedec

from daskms.experimental.zarr import xds_from_zarr

pd.options.mode.copy_on_write = True
pn.config.throttled = True  # Throttle all sliders.

hv.extension('bokeh', width="stretch_width")

# TODO: Make programmatic + include concatenation when we have mutiple xdss.
xds = xds_from_zarr("::G")#[:1]

xds = [x[["gains", "gain_flags"]] for x in xds]

directory_contents = Path.cwd().glob("*")

xds = timedec(xarray.combine_by_coords)(xds, combine_attrs="drop_conflicts")

# xds = xds.compute()

ds = xds[["gains", "gain_flags"]].to_dataframe()
ds["rowid"] = np.arange(len(ds))  # This is very important.

axis_map = {
    "Time": "gain_time",
    "Frequency": "gain_freq",
    "Amplitude": "amplitude",
    "Phase": "phase",
    "Real": "real",
    "Imaginary": "imaginary"
}

class GainInspector(param.Parameterized):

    # create a button that when pushed triggers 'button'
    antenna = param.Selector(
        label="Antenna",
        objects=ds.index.unique(level="antenna").tolist(),
        default=ds.index.unique(level="antenna").tolist()[0]
    )
    direction = param.Selector(
        label="Direction",
        objects=ds.index.unique(level="direction").tolist(),
        default=ds.index.unique(level="direction").tolist()[0]
    )
    correlation = param.Selector(
        label="Correlation",
        objects=ds.index.unique(level="correlation").tolist(),
        default=ds.index.unique(level="correlation").tolist()[0]
    )
    x_axis = param.Selector(
        label="X Axis",
        objects=list(axis_map.keys()),
        default="Time"
    )
    y_axis = param.Selector(
        label="Y Axis",
        objects=list(axis_map.keys()),
        default="Amplitude"
    )
    rasterized = param.Boolean(
        label="Rasterize",
        default=True
    )
    # Set the bounds during the init step.
    rasterize_when = param.Integer(
        label="Rasterize Limit",
        bounds=(10000, None),
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

    def __init__(self, **params):
        super().__init__(**params)

        self.data = ds  # Make this an argument.
 
        self.param.watch(self.flag_selection, ['flag'], queued=True)
        self.param.watch(self.flag_selection, ['flag_antennas'], queued=True)

        self.selection_cache = {}

        # Empty Rectangles for overlay
        self.rectangles = hv.Rectangles([])

        # Attach a BoxEdit stream to the Rectangles
        self.box_edit = streams.BoxEdit(source=self.rectangles)

        self.param.rasterize_when.bounds = (10000, len(self.current_selection))

    @property
    def selection_key(self):
        return (
            "x_axis", self.x_axis,
            "y_axis", self.y_axis,
            "antenna", self.antenna,
            "direction", self.direction,
            "correlation", self.correlation 
        )

    @property
    def current_selection(self):

        selection_key = self.selection_key

        pn.state.log(f'Querying cache.')

        if not selection_key in self.selection_cache:

            pn.state.log(f'No cache entry found - fetching data.')

            self.selection_cache = {}  # Empty the cache.

            selection = self.data.loc[
                (
                    slice(None),
                    slice(None),
                    self.antenna,
                    self.direction,
                    self.correlation
                )
            ]

            self.add_derived_columns(selection)

            # NOTE: This is a bit of a hack to work around hvplot not playing
            # well with multiindexes.
            self.selection_cache[selection_key] = selection.reset_index()

        return self.selection_cache[selection_key]

    @property
    def current_axes(self):
        return (self.x_axis, self.y_axis)

    def broadcast_rowids(self, rowids, all_antennas=False):

        corrs = self.param.correlation.objects
        n_corr = len(corrs)
        n_dir = len(self.param.direction.objects)

        stride = n_dir * n_corr  # TODO: Currently flags all directions.

        output_size = len(rowids) * n_corr

        if all_antennas:
            antennas = self.param.antenna.objects
            n_ant = len(antennas)
            ant_id = antennas.index(self.antenna)
            output_size *= n_ant
        else:
            n_ant, ant_id = 1, 0 

        output_rowids = np.zeros(output_size, dtype=np.int64)

        start = -ant_id * stride
        stop = stride * (n_ant - ant_id)

        for i, rowid in enumerate(rowids):
            output_i = i * stride * n_ant
            for i_off, r_off  in enumerate(range(start, stop)):
                output_rowids[output_i + i_off] = rowid + r_off

        return output_rowids

    @timedec
    def flag_selection(self, event):
        if not self.box_edit.data:
            return

        corners = self.box_edit.data

        for x_min, y_min, x_max, y_max in zip(*corners.values()):

            query = (
                f"{x_min} <= {axis_map[self.x_axis]} <= {x_max} &"
                f"{y_min} <= {axis_map[self.y_axis]} <= {y_max}"
            )

            flag_rowids = self.current_selection.query(query).rowid.values

            flag_rowids = self.broadcast_rowids(
                flag_rowids,
                all_antennas=True if event.name == "flag_antennas" else False
            )

            flag_col_loc = self.data.columns.get_loc('gain_flags')

            self.data.iloc[flag_rowids, flag_col_loc] = 1

        self.selection_cache = {}  # Clear the cache.


    @timedec
    def update_plot(self):

        pn.state.log(f'Plot update triggered.')

        sel = self.current_selection

        sel = sel[sel["gain_flags"] != 1]

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

    def add_derived_columns(self, df):
        required_columns = {axis_map[k] for k in [self.x_axis, self.y_axis]}

        available_columns = set(df.columns.to_list() + df.index.names)

        missing_columns = required_columns - available_columns

        func_map = {
            "amplitude": np.abs,
            "phase": lambda arr: np.rad2deg(np.angle(arr)),
            "real": np.real,
            "imaginary": np.imag
        }

        for column in missing_columns:
            df[column] = func_map[column](df["gains"])

action_example = GainInspector()

customised_params= pn.Param(
    action_example.param,
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

pn.template.MaterialTemplate(
    # site="Panel",
    title="QuartiCal-Visualiser",
    sidebar=customised_params,
    main=[action_example.update_plot],
).servable()
