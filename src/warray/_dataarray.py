from __future__ import annotations

import types
from typing import TYPE_CHECKING, Hashable, Mapping, Sequence, cast

import numpy as np

from ._common import AbstractArray
from ._util import (
    ErrorOptionsWithWarn,
    as_compatible_data,
    either_dict_or_kwargs,
    expanded_indexer,
    is_fancy_indexer,
)
from ._variable import Variable, as_variable

if TYPE_CHECKING:
    from typing import Any, Iterator, NoReturn, Self


class DataArray(AbstractArray):
    __module__ = "warray"

    def __init__(
        self,
        data: Any,
        coords: Sequence[Sequence[Any] | DataArray] | Mapping[Any, Any] | None = None,
        dims: Hashable | Sequence[Hashable] | None = None,
        name: Hashable | None = None,
        attrs: Mapping | None = None,
    ) -> None:
        _data = as_compatible_data(data)
        _coords, _dims = _infer_coords_and_dims(_data.shape, coords, dims)
        if name is None:
            name = getattr(data, "name", None)

        self._variable = Variable(_dims, _data)
        self._coords = _coords
        self._dims = _dims
        self._name = name
        self._attrs = attrs

    @property
    def name(self) -> Hashable | None:
        return self._name

    @property
    def variable(self) -> Variable:
        """Low level interface to the Variable object for this DataArray."""
        return self._variable

    @property
    def dtype(self) -> np.dtype:
        return self.variable.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self.variable.shape

    @property
    def size(self) -> int:
        return self.variable.size

    @property
    def nbytes(self) -> int:
        return self.variable.nbytes

    @property
    def ndim(self) -> int:
        return self.variable.ndim

    def __len__(self) -> int:
        return len(self.variable)

    @property
    def data(self) -> Any:
        return self.variable.data

    @data.setter
    def data(self, value: Any) -> None:
        self.variable.data = value

    @property
    def values(self) -> np.ndarray:
        """The array's data as a numpy.ndarray."""
        return self.variable.values

    @property
    def coords(self) -> Coordinates:
        return self._coords

    @property
    def dims(self) -> tuple[Hashable, ...]:
        return self._dims

    @dims.setter
    def dims(self, value: Any) -> NoReturn:
        raise AttributeError(
            "you cannot assign dims on a DataArray. Use "
            ".rename() or .swap_dims() instead."
        )

    @property
    def sizes(self) -> Mapping[Hashable, int]:
        return types.MappingProxyType(dict(zip(self.dims, self.shape)))

    def _item_key_to_dict(self, key: Any) -> Mapping[Hashable, Any]:
        if isinstance(key, Mapping):
            return key
        key = expanded_indexer(key, self.ndim)
        return dict(zip(self.dims, key))

    # def _getitem_coord(self, key: Any) -> Self:
    #     try:
    #         var = self._coords[key]
    #     except KeyError:
    #         raise NotImplementedError("xarray-style indexing not yet implemente")
    #         # dim_sizes = dict(zip(self.dims, self.shape))
    #         # _, key, var = _get_virtual_variable(self._coords, key, dim_sizes)

    #     return self._replace_maybe_drop_dims(var, name=key)

    # def _replace_maybe_drop_dims(
    #     self,
    #     variable: Variable,
    #     name: Hashable | None = None,
    # ) -> Self:
    #     if variable.dims == self.dims and variable.shape == self.shape:
    #         coords = self._coords.copy()
    #         indexes = self._indexes
    #     elif variable.dims == self.dims:
    #         # Shape has changed (e.g. from reduce(..., keepdims=True)
    #         new_sizes = dict(zip(self.dims, variable.shape))
    #         coords = {
    #             k: v
    #             for k, v in self._coords.items()
    #             if v.shape == tuple(new_sizes[d] for d in v.dims)
    #         }
    #         indexes = filter_indexes_from_coords(self._indexes, set(coords))
    #     else:
    #         allowed_dims = set(variable.dims)
    #         coords = {
    #             k: v for k, v in self._coords.items() if set(v.dims) <= allowed_dims
    #         }
    #         indexes = filter_indexes_from_coords(self._indexes, set(coords))
    #     return self._replace(variable, coords, name, indexes=indexes)

    def __getitem__(self, key: Any) -> Self:
        if isinstance(key, str):
            raise NotImplementedError("xarray-style indexing not yet implemented")
            # return self._getitem_coord(key)
        else:
            # xarray-style array indexing
            return self.isel(indexers=self._item_key_to_dict(key))

    def isel(
        self,
        indexers: Mapping[Any, Any] | None = None,
        drop: bool = False,
        missing_dims: ErrorOptionsWithWarn = "raise",
        **indexers_kwargs: Any,
    ) -> Self:
        """Return a new DataArray whose given by selecting indexes along specified dims.

        Parameters
        ----------
        indexers : dict, optional
            A dict with keys matching dimensions and values given
            by integers, slice objects or arrays.
            indexer can be a integer, slice, array-like or DataArray.
            If DataArrays are passed as indexers, xarray-style indexing will be
            carried out. See :ref:`indexing` for the details.
            One of indexers or indexers_kwargs must be provided.
        drop : bool, default: False
            If ``drop=True``, drop coordinates variables indexed by integers
            instead of making them scalar.
        missing_dims : {"raise", "warn", "ignore"}, default: "raise"
            What to do if dimensions that should be selected from are not present in the
            DataArray:
            - "raise": raise an exception
            - "warn": raise a warning, and ignore the missing dimensions
            - "ignore": ignore the missing dimensions
        **indexers_kwargs : {dim: indexer, ...}, optional
            The keyword arguments form of ``indexers``.

        Returns
        -------
        indexed : xarray.DataArray

        See Also
        --------
        Dataset.isel
        DataArray.sel

        :doc:`xarray-tutorial:intermediate/indexing/indexing`
            Tutorial material on indexing with Xarray objects

        :doc:`xarray-tutorial:fundamentals/02.1_indexing_Basic`
            Tutorial material on basics of indexing

        Examples
        --------
        >>> da = xr.DataArray(np.arange(25).reshape(5, 5), dims=("x", "y"))
        >>> da
        <xarray.DataArray (x: 5, y: 5)>
        array([[ 0,  1,  2,  3,  4],
               [ 5,  6,  7,  8,  9],
               [10, 11, 12, 13, 14],
               [15, 16, 17, 18, 19],
               [20, 21, 22, 23, 24]])
        Dimensions without coordinates: x, y

        >>> tgt_x = xr.DataArray(np.arange(0, 5), dims="points")
        >>> tgt_y = xr.DataArray(np.arange(0, 5), dims="points")
        >>> da = da.isel(x=tgt_x, y=tgt_y)
        >>> da
        <xarray.DataArray (points: 5)>
        array([ 0,  6, 12, 18, 24])
        Dimensions without coordinates: points
        """
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "isel")

        if any(is_fancy_indexer(idx) for idx in indexers.values()):
            raise NotImplementedError("fancy indexing is not yet implemented")
            # ds = self._to_temp_dataset()._isel_fancy(
            #     indexers, drop=drop, missing_dims=missing_dims
            # )
            # return self._from_temp_dataset(ds)

        # Much faster algorithm for when all indexers are ints, slices, one-dimensional
        # lists, or zero or one-dimensional np.ndarray's

        variable = self._variable.isel(indexers, missing_dims=missing_dims)
        # indexes, index_variables = isel_indexes(self.xindexes, indexers)

        # HACK: here's one place we're avoiding complexity and losing features
        # xarray uses indices, we're just using coords and dims
        new_dims = tuple(
            dim
            for dim in self._dims
            if dim not in indexers or not isinstance(indexers[dim], (int, np.integer))
        )

        coords = {}
        for coord_name, coord_value in self._coords.items():
            # if coord_name in index_variables:
            # coord_value = index_variables[coord_name]
            # else:
            coord_indexers = {
                k: v for k, v in indexers.items() if k in coord_value.dims
            }
            if coord_indexers:
                coord_value = coord_value.isel(coord_indexers)
                if drop and coord_value.ndim == 0:
                    continue
            coords[coord_name] = coord_value

        # return self._replace(variable=variable, coords=coords, indexes=indexes)
        return self._replace(variable=variable, dims=new_dims, coords=coords)

    def _replace(
        self,
        variable: Variable | None = None,
        coords: Any = None,
        name: Hashable | None = None,
        dims: Hashable | Sequence[Hashable] | None = None,
        indexes: Any = None,
    ) -> Self:
        if variable is None:
            variable = self.variable
        if coords is None:
            coords = self._coords
        # if indexes is None:
        # indexes = self._indexes
        if dims is None:
            dims = self._dims
        if name is None:
            name = self.name
        # return type(self)(variable, coords, name=name, indexes=indexes, fastpath=True)
        return type(self)(variable, coords, dims=dims, name=name)

    def __repr__(self) -> str:
        sizes = ", ".join(f"{k}: {v}" for k, v in self.sizes.items())
        lines = [
            f"<warray.DataArray ({sizes})> Size: {self.nbytes}B",
            repr(self.values),
            "Coordinates:",
        ]
        for dim, coord in self.coords.items():
            star = "*" if dim in self.dims else " "
            dim_names = f"({', '.join(str(x) for x in coord.dims)})"
            dtype = coord.dtype
            line_repr = repr(coord.data)[:50]
            lines.append(f"  {star} {dim:<8} {dim_names} {dtype} {line_repr}")
        return "\n".join(lines)


class Coordinates(Mapping[Hashable, DataArray]):
    def __init__(self, data: Mapping) -> None:
        self._data: Mapping[Hashable, DataArray] = data

    def __getitem__(self, key: Hashable) -> DataArray:
        return self._data[key]

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return repr(self._data)


def _infer_coords_and_dims(
    shape: tuple[int, ...],
    coords: Sequence[Sequence | DataArray] | Mapping[Hashable, Any] | None,
    dims: Hashable | Sequence[Hashable] | None,
) -> tuple[Coordinates, tuple[str, ...]]:
    if (
        coords is not None
        and not isinstance(coords, Mapping)
        and len(coords) != len(shape)
    ):
        raise ValueError(
            f"coords is not dict-like, but it has {len(coords)} items, "
            f"which does not match the {len(shape)} dimensions of the "
            "data"
        )

    if dims is not None and (not isinstance(dims, Sequence) or isinstance(dims, str)):
        dims = (dims,)
    dims = cast("Sequence[Hashable] | None", dims)

    if dims is None:
        dims = [f"dim_{n}" for n in range(len(shape))]
        if coords is not None and len(coords) == len(shape):
            # try to infer dimensions from coords
            if isinstance(coords, Mapping):
                dims = list(coords)
        dims = tuple(dims)
    elif len(dims) != len(shape):
        raise ValueError(
            "different number of dimensions on data "
            f"and dims: {len(shape)} vs {len(dims)}"
        )
    else:
        for d in dims:
            if not isinstance(d, str):
                raise TypeError(f"dimension {d} is not a string")

    if isinstance(coords, Coordinates):
        new_coords = coords
    else:
        tmp_coords: dict[Hashable, Variable] = {}
        if isinstance(coords, Mapping):
            for k, v in coords.items():
                tmp_coords[k] = as_variable(v, name=str(k))
        elif coords is not None:
            for dim, coord in zip(dims, coords):
                var = as_variable(coord, name=str(dim))
                var.dims = (dim,)
                tmp_coords[dim] = var
        # FIXME
        new_coords = Coordinates(tmp_coords)

    _check_coords_dims(shape, new_coords, dims)

    return new_coords, tuple(str(x) for x in dims)


def _check_coords_dims(
    shape: tuple[int, ...], coords: Coordinates, dims: Sequence[Hashable]
) -> None:
    sizes = dict(zip(dims, shape))
    for k, v in coords.items():
        if any(d not in dims for d in v.dims):
            raise ValueError(
                f"coordinate {k} has dimensions {v.dims}, but these "
                "are not a subset of the DataArray "
                f"dimensions {dims}"
            )

        for d, s in v.sizes.items():
            if s != sizes[d]:
                raise ValueError(
                    f"conflicting sizes for dimension {d!r}: "
                    f"length {sizes[d]} on the data but length {s} on "
                    f"coordinate {k!r}"
                )
