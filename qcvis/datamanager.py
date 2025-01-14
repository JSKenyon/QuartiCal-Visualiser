import numpy as np
import xarray
import dask.array as da
import pandas as pd

from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
from cachetools import cached, LRUCache
from cachetools.keys import hashkey


class DataManager(object):

    otf_column_map = {
        "amplitude": np.abs,
        "phase": lambda arr: np.rad2deg(np.angle(arr)),
        "real": np.real,
        "imaginary": np.imag
    }

    def __init__(self, path, fields=["gains", "gain_flags"]):

        self.path = path
        self.fields = fields
        # The datasets are lazily evaluated - inexpensive to hold onto them.
        self.datasets = [xds[self.fields] for xds in xds_from_zarr(self.path)]
        self.dataset = xarray.combine_by_coords(
            self.datasets,
            combine_attrs="drop_conflicts"
        ).compute()

        # Initialise data selection - defaults to all data.
        self.selector = tuple([slice(None) for _ in self.dataset.dims])

        # Initialise columns which should be added on the fly.
        self.otf_columns = {}

    def get_coord_values(self, dim_name):
        if not isinstance(dim_name, str):
            raise ValueError("dim_name expects a string.")
        return self.dataset[dim_name].values

    def get_dim_size(self, dim_name):
        if not isinstance(dim_name, str):
            raise ValueError("dim_name expects a string.")
        return self.dataset.sizes[dim_name]

    def set_otf_columns(self, **columns):
        self.otf_columns = columns

    def set_selection(self, **selections):
        self.selector = tuple(
            [selections.get(i, slice(None)) for i in self.dataset.dims]
        )

    # @cached(
    #     cache=LRUCache(maxsize=16),
    #     key=lambda self: hashkey(
    #         tuple(list(self.dataset.data_vars.keys())),
    #         tuple([None if isinstance(l, slice) else l for l in self.selector])
    #     )
    # )
    def get_selection(self, deselect=[]):

        selector = [
            s if d not in deselect else slice(None)
            for s, d in zip(self.selector, self.dataset.dims)
        ]

        selection = self.dataset.sel(
            {d: v for d, v in zip(self.dataset.dims, selector)}
        )

        # Add supported otf columns e.g. amplitude.
        for column, target in self.otf_columns.items():
            otf_func = self.otf_column_map[column]
            selection = selection.assign(
                {
                    column: (
                        selection[target].dims,
                        otf_func(selection[target].values)
                    )
                }
            )

        return selection
    
    def get_plot_data(
        self,
        x_axis,
        y_axis,
        data_field="gains",
        flag_field="gain_flags"
    ):

        sel = self.get_selection()
        sel = sel.where(sel[flag_field] != 1)

        x_data_array = sel[x_axis]
        y_data_array = sel[y_axis]

        x_slicer = tuple(
            [
                slice(None) if d in x_data_array.dims else np.newaxis
                for d in sel[data_field].dims
            ]
        )
        y_slicer = tuple(
            [
                slice(None) if d in y_data_array.dims else np.newaxis
                for d in sel[data_field].dims
            ]
        )

        x = np.broadcast_to(
            x_data_array.values[x_slicer],
            sel[data_field].shape
        ).ravel()
        y = np.broadcast_to(
            y_data_array.values[y_slicer],
            sel[data_field].shape
        ).ravel()

        return pd.DataFrame({x_axis: x, y_axis: y})

    def flag_selection(self, target, criteria, axes=[]):

        sel = self.get_selection(deselect=axes)

        dim_criteria = [k for k in criteria if k in sel.dims]
        val_criteria = [k for k in criteria if k not in sel.dims]

        for dim in dim_criteria:
            sel = sel.sel({dim: slice(*criteria[dim])})

        bool_arr = np.ones_like(sel[target].values, dtype=bool)
        for val in val_criteria:
            bool_arr[np.where(criteria[val][0] > sel[val].values)] = False
            bool_arr[np.where(criteria[val][1] < sel[val].values)] = False

        sel[target].values[bool_arr] = 1

    def write_flags(self, target):

        output_xdsl = []

        for ds in self.datasets:

            flags = self.dataset[target].sel(ds[target].coords)

            updated_xds = ds.assign(
                {
                    target: (
                        flags.dims,
                        da.from_array(flags.values)
                    )
                }
            )

            output_xdsl.append(updated_xds)

        writes = xds_to_zarr(
            output_xdsl,
            self.path,
            columns=target,
            rechunk=True
        )

        da.compute(writes)

    def reset(self):
        self.dataset = xarray.combine_by_coords(
            self.datasets,
            combine_attrs="drop_conflicts"
        ).compute()