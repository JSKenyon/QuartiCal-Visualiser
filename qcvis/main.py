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

hv.extension('bokeh', width=100)

# Default values suitable for this notebook
decimate.max_samples=1000
dynspread.max_px=20
dynspread.threshold=0.5
ResampleOperation2D.width=500
ResampleOperation2D.height=500

# TODO: Make programmatic + include concatenation when we have mutiple xdss.
xds = xds_from_zarr("::G")#[:1]

xds = [x[["gains", "gain_flags"]] for x in xds]

directory_contents = Path.cwd().glob("*")

xds = timedec(xarray.combine_by_coords)(xds, combine_attrs="drop_conflicts")

# xds = xds.compute()

ds = xds[["gains", "gain_flags"]].to_dataframe()

axis_map = {
    "Time": "gain_time",
    "Frequency": "gain_freq",
    "Amplitude": "amplitude",
    "Phase": "phase",
    "Real": "real",
    "Imaginary": "imaginary"
}

class ActionExample(param.Parameterized):

    # create a button that when pushed triggers 'button'
    flag = param.Action(lambda x: x.param.trigger('flag'), label='FLAG')
    redraw = param.Action(lambda x: x.param.trigger('redraw'), label='REDRAW')
    antenna = param.Selector(label="Antenna", objects=xds.antenna.values.tolist(), default=xds.antenna.values[0])
    direction = param.Selector(label="Direction", objects=xds.direction.values.tolist(), default=xds.direction.values[0])
    correlation = param.Selector(label="Correlation", objects=xds.correlation.values.tolist(), default=xds.correlation.values[0])
    x_axis = param.Selector(label="X Axis", objects=list(axis_map.keys()), default="Time")
    y_axis = param.Selector(label="Y Axis", objects=list(axis_map.keys()), default="Amplitude")
    datashaded = param.Boolean(label="Datashade", default=True)
    # Set the bounds during the init step.
    datashade_when = param.Integer(label="Datashade limit", bounds=(1, 250000), step=10000, default=10000)

    def __init__(self, **params):
        super().__init__(**params)

        self.data = ds  # Make this an argument.
 
        self.param.watch(self.flag_selection, ['flag'], queued=True)

        self.selection_cache = {}

        # Empty Rectangles for overlay
        self.rectangles = hv.Rectangles([])

        # Attach a BoxEdit stream to the Rectangles
        self.box_edit = streams.BoxEdit(source=self.rectangles)

    @property
    def selection_key(self):
        return (
            "antenna", self.antenna,
            "direction", self.direction,
            "correlation", self.correlation 
        )

    @property
    def current_selection(self):

        selection_key = self.selection_key

        if not selection_key in self.selection_cache:
            print("Invalidating cache!")
            self.selection_cache = {}  # Empty the cache.

            self.selection_cache[selection_key] = self.data.loc[
                (
                    slice(None),
                    slice(None),
                    self.antenna,
                    self.direction,
                    self.correlation
                )
            ]

        return self.selection_cache[selection_key]

    @property
    def current_axes(self):
        return (self.x_axis, self.y_axis)

    def flag_selection(self, event):
        if not self.box_edit.data:
            return

        idxs = self.data.index.get_locs((slice(None), slice(None), self.antenna, 0, self.correlation))
        sel = self.data.iloc[idxs].iloc[self.selected_points.index]
        self.data.loc[sel.index, "gain_flags"] = 1

    @timedec
    def update_plot(self):
        print("TRIGGERED UPDATE")

        sel = self.current_selection

        self.add_derived_columns(sel)

        sel["color"] = np.where(sel["gain_flags"], "red", "blue")

        sel = sel.reset_index()  # Workaround - resample when doesn't play nicely with mult-indices.

        plot = self.rectangles * sel.hvplot(
            x=axis_map[self.x_axis],
            y=axis_map[self.y_axis],
            kind="scatter",
            rasterize=self.datashaded,
            # dynspread=True,
            resample_when=self.datashade_when if self.datashaded else None,
            hover=False,
            responsive=True,
            height=800,
            logz=True,
            x_sampling=60  # This needs to be determined from the data.
        )

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

action_example = ActionExample()

customised_params= pn.Param(action_example.param, widgets={
        # 'update': {'visible': False},
        'flag': pn.widgets.Button
    }
)

myrow = pn.Row(customised_params, action_example.update_plot).servable()
