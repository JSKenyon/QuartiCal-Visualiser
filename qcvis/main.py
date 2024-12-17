''' An interactivate categorized chart based on a movie dataset.
This example shows the ability of Bokeh to create a dashboard with different
sorting options based on a given dataset.

'''
from pathlib import Path

import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Div, Select, Slider, Button
from bokeh.plotting import figure

from daskms.experimental.zarr import xds_from_zarr

# TODO: Make programmatic + include concatenation when we have mutiple xdss.
xds = xds_from_zarr("::B")[0]

directory_contents = Path.cwd().glob("*")

# import ipdb; ipdb.set_trace()

# import xarray
# xds = xarray.combine_by_coords(xds, combine_attrs="drop_conflicts")

gains = xds[["gains", "gain_flags"]].to_dataframe().reset_index()

gains["amplitude"] = np.abs(gains["gains"])
gains["phase"] = np.angle(gains["gains"])

gains["color"] = np.where(gains.gain_flags == True, "red", "blue")
gains["alpha"] = np.where(gains.gain_flags == True, 0.25, 0.9)
gains.fillna(0, inplace=True)  # just replace missing values with zero

axis_map = {
    "Time": "gain_time",
    "Frequency": "gain_freq",
    "Amplitude": "amplitude",
    "Phase": "phase"
}

desc = Div(text=(Path(__file__).parent / "description.html").read_text("utf8"), sizing_mode="stretch_width")

# Create Input controls
directories = Select(
    title="Gain",
    options=[str(p) for p in directory_contents],
    value="::G"
)
frequency_lower = Slider(
    title="Frequency (Min)",
    value=gains.gain_freq.min(),
    start=gains.gain_freq.min(),
    end=gains.gain_freq.max(),
    step=(gains.gain_freq.max() - gains.gain_freq.min()) / len(set(gains.gain_freq))
)
frequency_upper = Slider(
    title="Frequency (Max)",
    value=gains.gain_freq.max(),
    start=gains.gain_freq.min(),
    end=gains.gain_freq.max(),
    step=(gains.gain_freq.max() - gains.gain_freq.min()) / len(set(gains.gain_freq))
)
time_lower = Slider(
    title="Time (Min)",
    value=gains.gain_time.min(),
    start=gains.gain_time.min(),
    end=gains.gain_time.max(),
    step=(gains.gain_time.max() - gains.gain_time.min()) / len(set(gains.gain_time))
)
time_upper = Slider(
    title="Time (Max)",
    value=gains.gain_time.max(),
    start=gains.gain_time.min(),
    end=gains.gain_time.max(),
    step=(gains.gain_time.max() - gains.gain_time.min()) / len(set(gains.gain_time))
)
antenna = Select(title="Antenna", options=np.unique(gains.antenna).tolist(), value=gains.antenna[0])
correlation = Select(title="Correlation", options=np.unique(gains.correlation).tolist(), value=gains.correlation[0])
x_axis = Select(title="X Axis", options=sorted(axis_map.keys()), value="Time")
y_axis = Select(title="Y Axis", options=sorted(axis_map.keys()), value="Amplitude")

# Create Column Data Source that will be used by the plot
source = ColumnDataSource(data=dict(x=[], y=[], color=[], amplitude=[], phase=[], gain_flags=[], alpha=[]))

TOOLTIPS=[
    ("Amplitude", "@amplitude"),
    ("Phase", "@phase"),
]

p = figure(
    height=600,
    title="",
    tools=["box_select", "lasso_select", "reset", "save"],
    # toolbar_location=None,
    tooltips=TOOLTIPS,
    sizing_mode="stretch_width",
    output_backend="webgl"
)
p.scatter(x="x", y="y", source=source, size=7, color="color", line_color=None, fill_alpha="alpha")


def select_movies():
    selected = gains[
        (
            (gains.antenna == antenna.value) &
            (gains.correlation == correlation.value) &
            (
                (gains.gain_freq >= frequency_lower.value) &
                (gains.gain_freq <= frequency_upper.value)
            ) &
            (
                (gains.gain_time >= time_lower.value) &
                (gains.gain_time <= time_upper.value)
            )
        )
    ]
    return selected


def update():
    df = select_movies()
    x_name = axis_map[x_axis.value]
    y_name = axis_map[y_axis.value]

    p.xaxis.axis_label = x_axis.value
    p.yaxis.axis_label = y_axis.value
    p.title.text = f"{len(df)} gains selected"
    source.data = dict(
        x=df[x_name],
        y=df[y_name],
        gain_flags=df["gain_flags"],
        color=df["color"],
        alpha=df["alpha"],
    )

flag_button = Button(label="FLAG", button_type="danger")

def flagdata():
    df = select_movies()

    # We are selecting from a selection - we need to propagate these indices
    # back to the original dataframe.
    indices = df.iloc[source.selected.indices].index

    gains.loc[indices, "gain_flags"] = 1
    gains.loc[indices, "color"] = "red"
    update()

    source.selected.indices = []  # Drop the selection when button is clicked.

    # TODO: Add functionality to take interactive flags and write them back to
    # disk. This is tricky because of the conversion to dataframe.

flag_button.on_click(flagdata)

controls = [
    directories,
    antenna,
    correlation,
    x_axis,
    y_axis,
    frequency_lower,
    frequency_upper,
    time_lower,
    time_upper
]

for control in controls:
    control.on_change('value', lambda attr, old, new: update())

controls.append(flag_button)

inputs = column(*controls, width=320, height=800)

layout = column(desc, row(inputs, p, sizing_mode="inherit"), sizing_mode="stretch_width", height=800)

update()  # initial load of the data

curdoc().add_root(layout)
curdoc().title = "Gains"
