''' An interactivate categorized chart based on a movie dataset.
This example shows the ability of Bokeh to create a dashboard with different
sorting options based on a given dataset.

'''
from pathlib import Path

import pandas as pd
import numpy as np
import xarray
import dask.array as da

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

from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr

from cachetools import cached, LRUCache
from cachetools.keys import hashkey

pd.options.mode.copy_on_write = True
pn.config.throttled = True  # Throttle all sliders.

hv.extension('bokeh', width="stretch_width")

axis_map = {
    "Time": "gain_time",
    "Frequency": "gain_freq",
    "Amplitude": "amplitude",
    "Phase": "phase",
    "Real": "real",
    "Imaginary": "imaginary"
}

class DataManager(object):

    otf_column_map = {
        "amplitude": np.abs,
        "phase": lambda arr: np.rad2deg(np.angle(arr)),
        "real": np.real,
        "imaginary": np.imag
    }

    def __init__(self, path, fields=["gains", "gain_flags"]):

        self.path = path
        # The datasets are lazily evaluated - inexpensive to hold onto them.
        self.datasets = xds_from_zarr(self.path)
        self.consolidated_dataset = xarray.combine_by_coords(
            self.datasets,
            combine_attrs="drop_conflicts"
        )
        # Eagerly evaluated on conversion to pandas dataframe.
        self.dataframe = self.consolidated_dataset[fields].to_dataframe()
        # Add a rowid column to the dataframe to simplify later operations.
        self.dataframe["rowid"] = np.arange(len(self.dataframe))

        # Coordinates and the sizes.
        self.antennas = self.consolidated_dataset.antenna.values
        self.n_ant = len(self.antennas)
        self.directions = self.consolidated_dataset.direction.values
        self.n_dir = len(self.directions)
        self.correlations = self.consolidated_dataset.correlation.values
        self.n_corr = len(self.correlations)

    @cached(
        cache=LRUCache(maxsize=16),
        key=lambda self, otf_columns=[], **coords: hashkey(
            tuple(otf_columns),
            tuple(list(coords.items()))
        )
    )
    def get_selection(self, otf_columns=[], **coords):

        if not isinstance(otf_columns, list):
            raise ValueError("otf_columns must be a list.")

        dataframe_columns = self.dataframe.columns.tolist()
        index_levels = self.dataframe.index.names

        locator = tuple([coords.get(i, slice(None)) for i in index_levels])

        selection = self.dataframe.loc[locator]

        required_columns = set(otf_columns) 
        available_columns = set(dataframe_columns + index_levels)
        missing_columns = required_columns - available_columns

        for column in missing_columns:
            otf_func = self.otf_column_map[column]
            selection[column] = otf_func(selection.gains)

        return selection.reset_index()

    def write_flags(self):
        # TODO: This is likely flawed for multiple spectral windows.
        flags = self.dataframe.gain_flags.values[::self.n_corr].copy()

        array_shape = self.consolidated_dataset.gain_flags.shape
        array_flags = flags.reshape(array_shape)

        offset = 0

        output_xdsl = []

        for ds in self.datasets:

            n_time = ds.sizes["gain_time"]

            updated_xds = ds.assign(
                {
                    "gain_flags": (
                        ds.gain_flags.dims,
                        da.from_array(array_flags[offset: offset + n_time])
                    )
                }
            )

            offset += n_time

            output_xdsl.append(updated_xds)

        writes = xds_to_zarr(
            output_xdsl,
            self.path,
            columns="gain_flags",
            rechunk=True
        )

        da.compute(writes)


class GainInspector(param.Parameterized):

    antenna = param.Selector(
        label="Antenna",
    )
    direction = param.Selector(
        label="Direction",
    )
    correlation = param.Selector(
        label="Correlation",
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
    save = param.Action(
        lambda x: x.param.trigger('save'),
        label='SAVE'
    )

    def __init__(self, path, **params):
        
        self.dm = DataManager(path)
        self.data = self.dm.dataframe

        self.param.antenna.objects = self.dm.antennas
        self.param.antenna.default = self.dm.antennas[0]
        
        self.param.direction.objects = self.dm.directions
        self.param.direction.default = self.dm.directions[0]

        self.param.correlation.objects = self.dm.correlations
        self.param.correlation.default = self.dm.correlations[0]

        super().__init__(**params)

        self.param.watch(self.flag_selection, ['flag'], queued=True)
        self.param.watch(self.flag_selection, ['flag_antennas'], queued=True)
        self.param.watch(self.write_flags, ['save'], queued=True)

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

    def write_flags(self, event):
        self.dm.write_flags()

    @property
    def current_selection(self):

        pn.state.log(f'Attempting to fetch data - checking cache.')

        selection = self.dm.get_selection(
            otf_columns=[axis_map[self.x_axis], axis_map[self.y_axis]],
            antenna=self.antenna,
            direction=self.direction,
            correlation=self.correlation
        )

        return selection

    def flags_from_rowids(self, rowids, all_antennas=False):

        n_corr = self.dm.n_corr

        flags = np.zeros_like(self.data.gain_flags)

        flags[rowids] = 1  # New flags.

        array_shape = self.dm.consolidated_dataset.gain_flags.shape + (n_corr,)
        array_flags = flags.reshape(array_shape)

        or_axes = (-1, -3) if all_antennas else -1

        array_flags = array_flags.any(axis=or_axes, keepdims=True)
        array_flags = np.broadcast_to(array_flags, array_shape)

        return array_flags.ravel().astype(np.int8)

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

            flag_update = self.flags_from_rowids(
                flag_rowids,
                all_antennas=True if event.name == "flag_antennas" else False
            )

            self.data.gain_flags |= flag_update

        self.dm.get_selection.cache_clear()


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

action_example = GainInspector("::G")

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
