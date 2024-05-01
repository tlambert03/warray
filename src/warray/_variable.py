from __future__ import annotations

import copy
import math
import types
from typing import TYPE_CHECKING, Hashable, Iterable, Mapping

import numpy as np

from ._common import AbstractArray
from ._indexing import (
    BASIC_INDEXING_TYPES,
    BasicIndexer,
    as_indexable,
    integer_types,
)
from ._util import (
    DuckArray,
    ErrorOptionsWithWarn,
    T_DuckArray,
    as_compatible_data,
    drop_dims_from_indexers,
    either_dict_or_kwargs,
    expanded_indexer,
)

if TYPE_CHECKING:
    from typing import Any, Iterable, Self, Sequence, TypeAlias

    _Dim: TypeAlias = Hashable
    _Dims: TypeAlias = "tuple[_Dim, ...]"
    _DimsLike:TypeAlias = "str | Iterable[_Dim]"

_default = object()


class Variable(AbstractArray):
    def __init__(
        self,
        dims: Sequence[str],
        data: T_DuckArray,
        attrs: dict[Any, Any] | None = None,
    ) -> None:
        self._data = data
        self._dims = self._parse_dimensions(dims)
        self._attrs = dict(attrs) if attrs else None

    def _parse_dimensions(self, dims: _DimsLike) -> _Dims:
        dims = (dims,) if isinstance(dims, str) else tuple(dims)
        if len(dims) != self.ndim:
            raise ValueError(
                f"dimensions {dims} must have the same length as the "
                f"number of data dimensions, ndim={self.ndim}"
            )
        return dims

    def _check_shape(self, new_data: DuckArray) -> None:
        if new_data.shape != self.shape:
            raise ValueError(
                f"replacement data must match the {self.__class__.__name__}'s shape. "
                f"replacement data has shape {new_data.shape}; "
                f"{self.__class__.__name__} has shape {self.shape}"
            )

    @property
    def data(self) -> Any:
        return self._data

    @data.setter
    def data(self, data: Any) -> None:
        _data = as_compatible_data(data)
        self._check_shape(_data)
        self._data = _data

    @property
    def values(self) -> np.ndarray:
        """The variable's data as a numpy.ndarray."""
        return np.asarray(self._data)

    @property
    def dims(self) -> _Dims:
        """Tuple of dimension names with which this NamedArray is associated."""
        return self._dims

    @dims.setter
    def dims(self, value: _DimsLike) -> None:
        self._dims = self._parse_dimensions(value)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return math.prod(self.shape)

    def __len__(self) -> int:
        try:
            return self.shape[0]
        except Exception as exc:
            raise TypeError("len() of unsized object") from exc

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    @property
    def nbytes(self) -> int:
        if hasattr(self._data, "nbytes"):
            return self._data.nbytes  # type: ignore[no-any-return]
        else:
            return self.size * self.dtype.itemsize

    @property
    def sizes(self) -> Mapping[Hashable, int]:
        return types.MappingProxyType(dict(zip(self.dims, self.shape)))

    def __repr__(self) -> str:
        return f"<Variable {self.dims} {self._data!r}>"

    def isel(
        self,
        indexers: Mapping[Any, Any] | None = None,
        missing_dims: ErrorOptionsWithWarn = "raise",
        **indexers_kwargs: Any,
    ) -> Self:
        """Return a new array indexed along the specified dimension(s).

        Parameters
        ----------
        indexers : {dim: indexer, ...}
            Keyword arguments with names matching dimensions and values given
            by integers, slice objects or arrays.
        missing_dims : {"raise", "warn", "ignore"}, default: "raise"
            What to do if dimensions that should be selected from are not present in the
            DataArray:
            - "raise": raise an exception
            - "warn": raise a warning, and ignore the missing dimensions
            - "ignore": ignore the missing dimensions
        **indexers_kwargs : indexer
            Keyword arguments form of ``indexers``.

        Returns
        -------
        obj : Array object
            A new Array with the selected data and dimensions. In general,
            the new variable's data will be a view of this variable's data,
            unless numpy fancy indexing was triggered by using an array
            indexer, in which case the data will be a copy.
        """
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "isel")

        indexers = drop_dims_from_indexers(indexers, self.dims, missing_dims)

        key = tuple(indexers.get(dim, slice(None)) for dim in self.dims)
        return self[key]

    def __getitem__(self, key: Any) -> Self:
        """Return new Variable consistent with getting `key` from the underlying data.

        NB. __getitem__ and __setitem__ implement xarray-style indexing,
        where if keys are unlabeled arrays, we index the array orthogonally
        with them. If keys are labeled array (such as Variables), they are
        broadcasted with our usual scheme and then the array is indexed with
        the broadcasted key, like numpy's fancy indexing.

        If you really want to do indexing like `x[x > 0]`, manipulate the numpy
        array `x.values` directly.
        """
        dims, indexer, new_order = self._broadcast_indexes(key)
        data = as_indexable(self._data)[indexer]
        if new_order:
            data = np.moveaxis(data, range(len(new_order)), new_order)
        return self._finalize_indexing_result(dims, data)

    def _finalize_indexing_result(self, dims: _Dims, data: Any) -> Self:
        """Used by IndexVariable to return IndexVariable objects when possible."""
        return self._replace(dims=dims, data=data)

    def _replace(
        self,
        dims: _Dims | object = _default,
        data: Any = _default,
        attrs: Any = _default,
        # encoding: Any = _default,
    ) -> Self:
        if dims is _default:
            dims = copy.copy(self._dims)
        if data is _default:
            data = copy.copy(self.data)
        if attrs is _default:
            attrs = copy.copy(self._attrs)

        # if encoding is _default:
        #     encoding = copy.copy(self._encoding)
        # return type(self)(dims, data, attrs, encoding, fastpath=True)
        return type(self)(dims, data, attrs)  # type: ignore

    def _item_key_to_tuple(self, key: Any) -> Any:
        if isinstance(key, Mapping):
            return tuple(key.get(dim, slice(None)) for dim in self.dims)
        else:
            return key

    def _broadcast_indexes(self, key: Any) -> tuple[_Dims, Any, Sequence[int] | None]:
        """Prepare an indexing key for an indexing operation.

        Parameters
        ----------
        key : int, slice, array-like, dict or tuple of integer, slice and array-like
            Any valid input for indexing.

        Returns
        -------
        dims : tuple
            Dimension of the resultant variable.
        indexers : IndexingTuple subclass
            Tuple of integer, array-like, or slices to use when indexing
            self._data. The type of this argument indicates the type of
            indexing to perform, either basic, outer or vectorized.
        new_order : Optional[Sequence[int]]
            Optional reordering to do on the result of indexing. If not None,
            the first len(new_order) indexing should be moved to these
            positions.
        """
        key = self._item_key_to_tuple(key)  # key is a tuple
        # key is a tuple of full size
        exp_key = expanded_indexer(key, self.ndim)
        # Convert a scalar Variable to a 0d-array
        exp_key = tuple(
            k.data if isinstance(k, Variable) and k.ndim == 0 else k for k in exp_key
        )
        # Convert a 0d numpy arrays to an integer
        # dask 0d arrays are passed through
        exp_key = tuple(
            k.item() if isinstance(k, np.ndarray) and k.ndim == 0 else k
            for k in exp_key
        )

        if all(isinstance(k, BASIC_INDEXING_TYPES) for k in exp_key):
            return self._broadcast_indexes_basic(exp_key)

        raise NotImplementedError(f"Cannot yet index with: {type(key)}")

        # self._validate_indexers(key)
        # # Detect it can be mapped as an outer indexer
        # # If all key is unlabeled, or
        # # key can be mapped as an OuterIndexer.
        # if all(not isinstance(k, Variable) for k in key):
        #     return self._broadcast_indexes_outer(key)

        # # If all key is 1-dimensional and there are no duplicate labels,
        # # key can be mapped as an OuterIndexer.
        # dims = []
        # for k, d in zip(key, self.dims):
        #     if isinstance(k, Variable):
        #         if len(k.dims) > 1:
        #             return self._broadcast_indexes_vectorized(key)
        #         dims.append(k.dims[0])
        #     elif not isinstance(k, integer_types):
        #         dims.append(d)
        # if len(set(dims)) == len(dims):
        #     return self._broadcast_indexes_outer(key)

        # return self._broadcast_indexes_vectorized(key)

    def _broadcast_indexes_basic(self, key: tuple) -> tuple[_Dims, BasicIndexer, None]:
        dims = tuple(
            dim for k, dim in zip(key, self.dims) if not isinstance(k, integer_types)
        )
        return dims, BasicIndexer(key), None


def as_variable(obj: Any, name: str | None = None) -> Variable:
    if isinstance(obj, Variable):
        return copy.copy(obj)
    elif isinstance(obj, tuple):
        try:
            return Variable(*obj)
        except (TypeError, ValueError) as e:
            raise e.__class__(
                f"Variable {name!r}: Could not convert tuple of form "
                f"(dims, data[, attrs, encoding]): {obj} to Variable."
            ) from e
    elif name is not None:
        data = as_compatible_data(obj)
        if ndim := len(data.shape) != 1:
            raise ValueError(
                f"cannot set variable {name!r} with {ndim!r}-dimensional data "
                "without explicit dimension names. Pass a tuple of "
                "(dims, data) instead."
            )
        return Variable(name, data)
    else:
        raise TypeError(
            f"Variable {name!r}: unable to convert object into a variable without an "
            f"explicit list of dimensions: {obj!r}"
        )
    raise NotImplementedError(f"cannot convert object to Variable: {obj}")
