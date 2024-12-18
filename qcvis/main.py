''' An interactivate categorized chart based on a movie dataset.
This example shows the ability of Bokeh to create a dashboard with different
sorting options based on a given dataset.

'''
from pathlib import Path

import numpy as np
import xarray

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Div, Select, Slider, Button
from bokeh.plotting import figure

import holoviews as hv
from holoviews import opts, streams
from holoviews.operation.datashader import datashade, rasterize, shade, dynspread, spread
from holoviews.operation.downsample import downsample1d
from holoviews.operation.resample import ResampleOperation2D
from holoviews.operation import decimate

import panel as pn
import panel.widgets as pnw

from timedec import timedec

from daskms.experimental.zarr import xds_from_zarr

hv.extension('bokeh', width=100)

# Default values suitable for this notebook
decimate.max_samples=1000
dynspread.max_px=20
dynspread.threshold=0.5
ResampleOperation2D.width=500
ResampleOperation2D.height=500

# TODO: Make programmatic + include concatenation when we have mutiple xdss.
xds = xds_from_zarr("::G") #[:1]

directory_contents = Path.cwd().glob("*")

xds = timedec(xarray.combine_by_coords)(xds, combine_attrs="drop_conflicts")

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

# desc = Div(
#     text=(Path(__file__).parent / "description.html").read_text("utf8"),
#     sizing_mode="stretch_width"
# )

# Create Input controls
# directories = Select(
#     title="Gain",
#     options=[str(p) for p in directory_contents],
#     value="::G"
# )
# frequency_lower = pnw.DiscreteSlider(
#     name="Frequency (Min)",
#     value=gains.gain_freq.min(),
#     start=gains.gain_freq.min(),
#     end=gains.gain_freq.max(),
#     step=(gains.gain_freq.max() - gains.gain_freq.min()) / len(set(gains.gain_freq))
# )
# frequency_upper = Slider(
#     title="Frequency (Max)",
#     value=gains.gain_freq.max(),
#     start=gains.gain_freq.min(),
#     end=gains.gain_freq.max(),
#     step=(gains.gain_freq.max() - gains.gain_freq.min()) / len(set(gains.gain_freq))
# )
# time_lower = Slider(
#     title="Time (Min)",
#     value=gains.gain_time.min(),
#     start=gains.gain_time.min(),
#     end=gains.gain_time.max(),
#     step=(gains.gain_time.max() - gains.gain_time.min()) / len(set(gains.gain_time))
# )
# time_upper = Slider(
#     title="Time (Max)",
#     value=gains.gain_time.max(),
#     start=gains.gain_time.min(),
#     end=gains.gain_time.max(),
#     step=(gains.gain_time.max() - gains.gain_time.min()) / len(set(gains.gain_time))
# )

antenna = pnw.Select(name="Antenna", options=xds.antenna.values.tolist(), value=xds.antenna.values[0])
correlation = pnw.Select(name="Correlation", options=xds.correlation.values.tolist(), value=xds.correlation.values[0])
x_axis = pnw.Select(name="X Axis", options=list(axis_map.keys()), value="Time")
y_axis = pnw.Select(name="Y Axis", options=list(axis_map.keys()), value="Amplitude")
flag = pnw.Button(name='Flag', button_type='danger')

# size = pnw.Select(name='Size', value='None', options=['None'] + quantileable)
# color = pnw.Select(name='Color', value='None', options=['None'] + quantileable)

selected_points = streams.Selection1D()
active_subset = None

@timedec
@pn.depends(
    x_axis.param.value,
    y_axis.param.value,
    antenna.param.value,
    correlation.param.value,
    watch=True
)
def create_figure(x, y, antenna, correlation):
    active_subset = xds.sel({"antenna": antenna, "correlation": correlation})

    sel = active_subset.to_dataframe()

    plot_opts = dict(
        color='color',
        height=800,
        responsive=True,
        tools=['box_select'],
        active_tools=['box_select']
    )
    if "Amplitude" in [x, y]:
        sel["amplitude"] = np.abs(sel["gains"])
    if "Phase" in [x, y]:
        sel["phase"] = np.rad2deg(np.angle(sel["gains"]))
    if "Real" in [x, y]:
        sel["real"] = np.real(sel["gains"])
    if "Imaginary" in [x, y]:
        sel["imaginary"] = np.imag(sel["gains"])
    sel["color"] = np.where(sel.gain_flags == True, "red", "blue")
    points = timedec(hv.Points)(
        sel,
        [axis_map[x], axis_map[y]],
        vdims="color",
        label="%s vs %s" % (x.title(), y.title())
    ).opts(**plot_opts)

    selected_points.source = points  # Add points as stream source.

    return points # downsample1d(points)

flag.on_click(lambda event: print(selected_points.index))

widgets = pn.WidgetBox(antenna, correlation, x_axis, y_axis, flag) #, width=400)

pn.Row(widgets, create_figure).servable('Cross-selector')