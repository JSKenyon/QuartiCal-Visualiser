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

# import time
# t0 = time.time()
# gains = xds[["gains", "gain_flags"]].to_dataframe().reset_index()
# print(f"{time.time() - t0}")

# t0 = time.time()
# gains["amplitude"] = np.abs(gains["gains"])
# gains["phase"] = np.angle(gains["gains"])

# gains["color"] = np.where(gains.gain_flags == True, "red", "blue")
# gains["alpha"] = np.where(gains.gain_flags == True, 0.25, 0.9)
# gains.fillna(0, inplace=True)  # just replace missing values with zero
# print(f"{time.time() - t0}")
axis_map = {
    "Time": "gain_time",
    "Frequency": "gain_freq",
    "Amplitude": "amplitude",
    "Phase": "phase",
    "Real": "real",
    "Imaginary": "imaginary"
}


index = ds.index

foo = index.unique(level="gain_time")
bar = index.unique(level="gain_freq")
baz = index.unique(level="antenna")
correlation_values = index.unique(level="correlation")

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
    datashade_when = param.Integer(label="Datashade limit", bounds=(0, None), default=10000)

    selected_points = streams.Selection1D()
    visible_points = streams.RangeXY()
    # plot_resetter = streams.PlotReset()

    def __init__(self, **params):
        super().__init__(**params)

        self.data = ds  # Make this an argument.
 
        self.param.watch(self.flag_selection, ['flag'], queued=True)
        # self.param.watch(self.reset_zoom, ['x_axis', 'y_axis'], queued=True)
        # self.plot_resetter.add_subscriber(self.reset, precedence=1)

        self.visible_points.param.watch(self.on_zoom, ["x_range", "y_range"], queued=True)

        self.x_min, self.x_max = None, None
        self.y_min, self.y_max = None, None

        self.selection_cache = {}

        # Empty Rectangles for overlay
        self.rectangles = hv.Rectangles([])

        # Attach a BoxEdit stream to the Rectangles
        self.box_edit = streams.BoxEdit(source=self.rectangles)

    def on_zoom(self, *events):

        print("ZOOMING")
        with hold():
            for event in events:
                if event.name == "x_range":
                    # self.x_lims = event.new
                    self.x_min, self.x_max = np.array(event.new, dtype=np.float64)
                elif event.name == "y_range":
                    # self.y_lims = event.new
                    self.y_min, self.y_max = np.array(event.new, dtype=np.float64)

    # def reset_zoom(self, event):
    #     if event.name == "x_axis":
    #         self.x_min, self.x_max = None, None
    #     elif event.name == "y_axis":
    #         self.y_min, self.y_max = None, None
    #     else:
    #         raise ValueError(f"Reset zoom event not understood: {event}")

    # def reset(self, resetting):
    #     self.x_lims = self.param.x_lims.bounds
    #     # self.y_min, self.y_max = None, None

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

    # def get_limits(self, df, column_or_level):
    #     try:
    #         values = df[column_or_level]
    #     except KeyError:
    #         values = df.index.get_level_values(column_or_level)

    #     return values.min(), values.max()


    @timedec
    def update_plot(self):
        print("TRIGGERED UPDATE")

        sel = self.current_selection

        self.add_derived_columns(sel)

        # if self.x_min is None: 
        #     self.x_min, self.x_max = self.param.x_lims.bounds
        #     self.x_lims = self.param.x_lims.bounds
        # if self.y_min is None: 
        #     self.y_min, self.y_max = self.param.y_lims.bounds
        #     self.y_lims = self.param.y_lims.bounds

        # print(self.x_min, self.x_max)

        # if not None in [self.x_min, self.x_max]:  # Limits are set.
        #     sel = sel.query(
        #         f"{self.x_min} <= {axis_map[self.x_axis]} <= {self.x_max}"
        #     )
            
        # if not None in [self.y_min, self.y_max]:  # Limits are set.
        #     sel = sel.query(
        #         f"{self.y_min} <= {axis_map[self.y_axis]} <= {self.y_max}"
        #     )

        sel["color"] = np.where(sel["gain_flags"], "red", "blue")

        sel = sel.reset_index()  # Workaround - resample when doesn't play nicely with mult-indices.

        plot = self.rectangles * sel.hvplot(
            x=axis_map[self.x_axis],
            y=axis_map[self.y_axis],
            kind="scatter",
            rasterize=True,
            # dynspread=True,
            resample_when=10000,
            tools=['box_select'],
            hover=False,
            responsive=True,
            height=800,
            logz=True,
            x_sampling=60  # This needs to be determined from the data.
        )
        # # with param.parameterized.discard_events(self.visible_points):
        # scatter = hv.Scatter(sel, [axis_map[self.x_axis]], [axis_map[self.y_axis], "color"]).opts(**plot_opts, hooks=[self._set_current_zoom])

        # if self.datashaded:
        #     del plot_opts["color"]
        #     plot = datashade(scatter).opts(**plot_opts, hooks=[self._set_current_zoom])
        # else:
        #     plot = scatter.opts(hooks=[self._set_current_zoom])

        # self.selected_points.source = plot
        # self.visible_points.source = plot
        # self.plot_resetter.source = plot

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

    # def _set_current_zoom(self, plot, element):
    #     # Access the Bokeh plot object and set zoom range
    #     if self.x_min is not None:
    #         self.visible_points.update(x_range=(self.x_min, self.x_max), y_range=(self.y_min, self.y_max))
    #         plot.state.x_range.start = self.x_min
    #         plot.state.x_range.end = self.x_max
    #         plot.state.y_range.start = self.y_min
    #         plot.state.y_range.end = self.y_max

    #     else:
    #         "IT WAS NONE!"



action_example = ActionExample()

customised_params= pn.Param(action_example.param, widgets={
        # 'update': {'visible': False},
        'flag': pn.widgets.Button
    }
)


myrow = pn.Row(customised_params, action_example.update_plot).servable()

# class WidgetParams(param.Parameterized):

#     # create a button that when pushed triggers 'button'
#     flag = param.Action(lambda x: x.param.trigger('flag'), label='FLAG')
#     antenna = param.Selector(label="Antenna", objects=xds.antenna.values.tolist(), default=xds.antenna.values[0])
#     direction = param.Selector(label="Direction", objects=xds.direction.values.tolist(), default=xds.direction.values[0])
#     correlation = param.Selector(label="Correlation", objects=xds.correlation.values.tolist(), default=xds.correlation.values[0])
#     x_axis = param.Selector(label="X Axis", objects=list(axis_map.keys()), default="Time")
#     y_axis = param.Selector(label="Y Axis", objects=list(axis_map.keys()), default="Amplitude")

# tunables = WidgetParams()

# def dmap_update(flag, antenna, direction, correlation, x_axis, y_axis):

#     print("TRIGGERED", antenna, correlation)

#     plot_opts = dict(
#         color='color',
#         height=800,
#         responsive=True,
#         tools=['box_select'],
#         active_tools=['box_select']
#     )
#     sel = ds.loc[(slice(None), slice(None), antenna, 0, correlation)]

#     if "Amplitude" in [x_axis, y_axis]:
#         sel["amplitude"] = np.abs(sel["gains"])
#     if "Phase" in [x_axis, y_axis]:
#         sel["phase"] = np.rad2deg(np.angle(sel["gains"]))
#     if "Real" in [x_axis, y_axis]:
#         sel["real"] = np.real(sel["gains"])
#     if "Imaginary" in [x_axis, y_axis]:
#         sel["imaginary"] = np.imag(sel["gains"])

#     sel["color"] = np.where(sel["gain_flags"], "red", "blue")

#     scatter = hv.Scatter(sel, [axis_map[x_axis]], [axis_map[y_axis], "color"]).opts(**plot_opts)

#     selected_points.source = scatter

#     return scatter

# mystreams = dict(
#     flag=tunables.param.flag,
#     antenna=antenna.param.value, #tunables.param.antenna,
#     direction=tunables.param.direction,
#     correlation=tunables.param.correlation,
#     x_axis=tunables.param.x_axis,
#     y_axis=tunables.param.y_axis
# )

# dmap = hv.DynamicMap(dmap_update, streams=mystreams)

# pn.Row(tunables, pn.panel(dmap)).servable()

