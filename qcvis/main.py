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

import param
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
xds = xds_from_zarr("::G")[:1]

xds = [x[["gains", "gain_flags"]] for x in xds]

directory_contents = Path.cwd().glob("*")

xds = timedec(xarray.combine_by_coords)(xds, combine_attrs="drop_conflicts")

xds = xds.compute()

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
cache = {}

@timedec
@pn.depends(
    x_axis.param.value,
    y_axis.param.value,
    antenna.param.value,
    correlation.param.value,
    flag.param.value
)
def create_figure(x, y, antenna, correlation, foo):

    active_subset = xds.sel({"antenna": antenna, "correlation": correlation})

    cache["active"] = sel = active_subset.to_dataframe()

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

def debug(event):
    if not selected_points.index:
        return

    idxs = ds.index.get_locs((slice(None), slice(None), antenna.value, 0, correlation.value))
    sel = ds.iloc[idxs].iloc[selected_points.index]
    ds.loc[sel.index, "gain_flags"] = 1

    x_axis.param.trigger('value')

    # import ipdb; ipdb.set_trace()

    # idxs = ds.index.get_locs((slice(None), slice(None), antenna.value, 0, correlation.value))
    # sel = ds.iloc[idxs]  # retains indexed out levels.
    # ds.loc[sel.index]["gain_flags"] = 1  # Update dataset.
 
    # sel = ds.loc[(slice(None), slice(None), antenna.value, 0, correlation.value)]

    # sel.iloc[selected_points.index]["gain_flags"] = 1

    # import ipdb; ipdb.set_trace()

    # index = flag_selection.index

    # times, chans, dirs = [index.get_level_values(i) for i in range(3)]

    # ant = np.unique(flag_selection["antenna"]).item()

    # xds.gain_flags.loc[
    #     {
    #         "gain_time": times,
    #         "gain_freq": chans,
    #         "direction": dirs,
    #         "antenna": ant
    #     }
    # ] = 1

flag.on_click(debug)



index = ds.index

foo = index.unique(level="gain_time")
bar = index.unique(level="gain_freq")
baz = index.unique(level="antenna")
correlation_values = index.unique(level="correlation")



@pn.depends(
    antenna,
    correlation,
    x_axis,
    y_axis,
    # flag.param.value
)
def update_plot(*args):

    plot_opts = dict(
        color='color',
        height=800,
        responsive=True,
        tools=['box_select'],
        active_tools=['box_select']
    )
    sel = ds.loc[(slice(None), slice(None), antenna.value, 0, correlation.value)]

    if "Amplitude" in [x_axis.value, y_axis.value]:
        sel["amplitude"] = np.abs(sel["gains"])
    if "Phase" in [x_axis.value, y_axis.value]:
        sel["phase"] = np.rad2deg(np.angle(sel["gains"]))
    if "Real" in [x_axis.value, y_axis.value]:
        sel["real"] = np.real(sel["gains"])
    if "Imaginary" in [x_axis.value, y_axis.value]:
        sel["imaginary"] = np.imag(sel["gains"])

    sel["color"] = np.where(sel["gain_flags"], "red", "blue")

    scatter = hv.Scatter(sel, [axis_map[x_axis.value]], [axis_map[y_axis.value], "color"]).opts(**plot_opts)

    selected_points.source = scatter

    return scatter

# dmap = hv.DynamicMap(pn.bind(update_plot, antenna))

# dmap = hv.DynamicMap(
#     sine,
#     kdims=['antenna', 'direction', 'correlation']).redim.range(direction=(0,1)).redim.values(antenna=baz, correlation=correlation_values)

# dmap_panel = pn.panel(dmap, sizing_mode="stretch_width")

# dmap_pane = pn.pane.HoloViews(dmap, widgets={
#     'x-axis': x_axis
# })

# import ipdb; ipdb.set_trace()   

widgets = pn.WidgetBox(antenna, correlation, x_axis, y_axis, flag)

pn.Row(widgets, update_plot).servable('Cross-selector')

# import ipdb; ipdb.set_trace()

# class ActionExample(param.Parameterized):
       
#     # create a button that when pushed triggers 'button'
#     button = param.Action(lambda x: x.param.trigger('button'), label='Start training model!')
      
#     model_trained = None
    
#     # method keeps on watching whether button is triggered
#     @param.depends('button', watch=True)
#     def train_model(self):
#         self.model_df = pd.DataFrame(np.random.normal(size=[50, 2]), columns=['col1', 'col2'])
#         self.model_trained = True

#     # method is watching whether model_trained is updated
#     @param.depends('model_trained', watch=True)
#     def update_graph(self):
#         if self.model_trained:
#             return hv.Points(self.model_df)
#         else:
#             return "Model not trained yet"

# action_example = ActionExample()

# pn.Row(action_example.param, action_example.update_graph)



# widget = pn.widgets.IntSlider(value=50, start=1, end=100, name="Number of points")

# @pn.depends(widget.param.value)
# def get_plot(n):
#     return hv.Scatter(range(n), kdims="x", vdims="y").opts(
#         height=300, responsive=True, xlim=(0, 100), ylim=(0, 100)
#     )


# # plot = hv.DynamicMap(pn.bind(get_plot, widget))

# pn.Column(widget, get_plot).servable()
