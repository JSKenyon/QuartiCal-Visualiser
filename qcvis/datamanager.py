import numpy as np
import xarray
import dask.array as da

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
        # The datasets are lazily evaluated - inexpensive to hold onto them.
        self.datasets = [xds[fields] for xds in xds_from_zarr(self.path)]
        self.consolidated_dataset = xarray.combine_by_coords(
            self.datasets,
            combine_attrs="drop_conflicts"
        )
        # Eagerly evaluated on conversion to pandas dataframe.
        self.dataframe = self.consolidated_dataset[fields].to_dataframe()
        # Add a rowid column to the dataframe to simplify later operations.
        self.dataframe["rowid"] = np.arange(len(self.dataframe))

        # Initialise data selection - defaults to all data.
        index_levels = self.dataframe.index.names
        self.locator = tuple([slice(None) for _ in index_levels])

        # Initialise columns which should be added on the fly.
        self.otf_columns = {}

    def get_coord_values(self, dim_name):
        if not isinstance(dim_name, str):
            raise ValueError("dim_name expects a string.")
        return self.consolidated_dataset[dim_name].values

    def get_dim_size(self, dim_name):
        if not isinstance(dim_name, str):
            raise ValueError("dim_name expects a string.")
        return self.consolidated_dataset.sizes[dim_name]

    def set_otf_columns(self, **columns):
        self.otf_columns = columns

    def set_selection(self, **selections):
        index_levels = self.dataframe.index.names
        self.locator = tuple(
            [selections.get(i, slice(None)) for i in index_levels]
        )

    @cached(
        cache=LRUCache(maxsize=16),
        key=lambda self: hashkey(
            tuple(self.otf_columns),
            tuple([None if isinstance(l, slice) else l for l in self.locator])
        )
    )
    def get_selection(self):

        selection = self.dataframe.loc[self.locator]

        # Add supported otf columns e.g. amplitude. TODO: These should have
        # a condifurable target rather than defaulting to the gains.
        for column, target in self.otf_columns.items():
            otf_func = self.otf_column_map[column]
            selection[column] = otf_func(selection[target])

        return selection.reset_index()  # Reset to satisfy hvplot - irritating!

    def flag_selection(self, target, query, dims):

        rowids = self.get_selection().query(query).rowid.values

        new_flags = np.zeros_like(self.dataframe[target])

        new_flags[rowids] = 1  # New flags.

        array_shape = self.consolidated_dataset[target].shape
        df_dims = self.dataframe.index.names

        # TODO: This assumes the last dimension was broadcast. This should
        # be replaced with a more thorough approach that doesn't assume which
        # axes have been broadcast.
        if len(df_dims) != len(array_shape):
            array_shape = array_shape + (-1,)

        array_flags = new_flags.reshape(array_shape)
        array_shape = array_flags.shape  # Get the current array shape.
        or_axes = tuple([df_dims.index(dim) for dim in dims])

        array_flags = array_flags.any(axis=or_axes, keepdims=True)
        array_flags = np.broadcast_to(array_flags, array_shape)

        self.dataframe[target] |= array_flags.ravel().astype(np.int8)


    def write_flags(self, target):
        # TODO: This presumes that the only concatenation axis is the first.
        # In general, this may not be true for multi-SPW data and this code
        # will need to be improved. The correct approach is to implement the
        # inverse of combine_by_coords.
        flags = self.dataframe[target].values.copy()

        df_dims = self.dataframe.index.names
        df_sizes = [
            len(self.dataframe.index.unique(level=i))
            for i in range(len(df_dims))
        ]

        flags = flags.reshape(df_sizes)

        ds_dims = self.consolidated_dataset[target].dims

        missing_dims = set(df_dims) - set(ds_dims)

        or_axes = tuple([df_dims.index(dim) for dim in missing_dims])

        if or_axes:
            flags = flags.any(axis=or_axes).astype(np.int8)

        offset = 0

        output_xdsl = []

        for ds in self.datasets:

            ax_size = ds.sizes[ds_dims[0]]

            updated_xds = ds.assign(
                {
                    target: (
                        ds[target].dims,
                        da.from_array(flags[offset: offset + ax_size])
                    )
                }
            )

            offset += ax_size

            output_xdsl.append(updated_xds)

        writes = xds_to_zarr(
            output_xdsl,
            self.path,
            columns=target,
            rechunk=True
        )

        da.compute(writes)