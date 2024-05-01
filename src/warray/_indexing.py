from __future__ import annotations

import math
import operator
from typing import TYPE_CHECKING, Protocol

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from typing import Any, Iterable, SupportsIndex


integer_types = (int, np.integer)
BASIC_INDEXING_TYPES = (*integer_types, slice)


def as_integer_or_none(value: SupportsIndex) -> int | None:
    return None if value is None else operator.index(value)


def as_integer_slice(value: slice) -> slice:
    start = as_integer_or_none(value.start)
    stop = as_integer_or_none(value.stop)
    step = as_integer_or_none(value.step)
    return slice(start, stop, step)


def as_indexable(array: Any) -> ExplicitlyIndexed:
    """Always returns a ExplicitlyIndexed subclass...

    so that the vectorized indexing is always possible with the returned
    object.
    """
    if isinstance(array, ExplicitlyIndexed):
        return array
    if isinstance(array, np.ndarray):
        return NumpyIndexingAdapter(array)
    # if isinstance(array, pd.Index):
    #     return PandasIndexingAdapter(array)
    # if is_duck_dask_array(array):
    #     return DaskIndexingAdapter(array)
    # if hasattr(array, "__array_function__"):
    #     return NdArrayLikeIndexingAdapter(array)
    # if hasattr(array, "__array_namespace__"):
    #     return ArrayApiIndexingAdapter(array)

    raise NotImplementedError(f"Indexing with {type(array)} is not implemented")
    raise TypeError(f"Invalid array type: {type(array)}")


class ExplicitIndexer:
    """Base class for explicit indexer objects.

    ExplicitIndexer objects wrap a tuple of values given by their ``tuple``
    property. These tuples should always have length equal to the number of
    dimensions on the indexed array.

    Do not instantiate BaseIndexer objects directly: instead, use one of the
    sub-classes BasicIndexer, OuterIndexer or VectorizedIndexer.
    """

    __slots__ = ("_key",)

    def __init__(self, key: Iterable) -> None:
        if type(self) is ExplicitIndexer:
            raise TypeError("cannot instantiate base ExplicitIndexer objects")
        self._key = tuple(key)

    @property
    def tuple(self) -> tuple:
        return self._key

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.tuple})"


class BasicIndexer(ExplicitIndexer):
    """Tuple for basic indexing.

    All elements should be int or slice objects. Indexing follows NumPy's
    rules for basic indexing: each axis is independently sliced and axes
    indexed with an integer are dropped from the result.
    """

    __slots__ = ()

    def __init__(self, key: tuple) -> None:
        if not isinstance(key, tuple):
            raise TypeError(f"key must be a tuple: {key!r}")

        new_key = []
        for k in key:
            if isinstance(k, integer_types):
                k = int(k)
            elif isinstance(k, slice):
                k = as_integer_slice(k)
            else:
                raise TypeError(
                    f"unexpected indexer type for {type(self).__name__}: {k!r}"
                )
            new_key.append(k)

        super().__init__(new_key)


# --------------------


class HasShape(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...


class HasArray(Protocol):
    @property
    def array(self) -> np.ndarray: ...


class ExplicitlyIndexed:
    """Mixin to mark support for Indexer subclasses in indexing."""

    __slots__ = ()

    def __array__(self, dtype: npt.DTypeLike = None) -> np.ndarray:
        # Leave casting to an array up to the underlying array type.
        return np.asarray(self.get_duck_array(), dtype=dtype)  # type: ignore

    def get_duck_array(self: HasArray) -> np.ndarray:
        return self.array


class NdimSizeLenMixin:
    __slots__ = ()

    @property
    def ndim(self: HasShape) -> int:
        return len(self.shape)

    @property
    def size(self: HasShape) -> int:
        return math.prod(self.shape)

    def __len__(self: HasShape) -> int:
        try:
            return self.shape[0]
        except IndexError as e:
            raise TypeError("len() of unsized object") from e


class NDArrayMixin(NdimSizeLenMixin):
    """Mixin class for making wrappers of N-dimensional arrays that conform to
    the ndarray interface required for the data argument to Variable objects.

    A subclass should set the `array` property and override one or more of
    `dtype`, `shape` and `__getitem__`.
    """

    __slots__ = ()

    @property
    def dtype(self: HasArray) -> np.dtype:
        return self.array.dtype

    @property
    def shape(self: HasArray) -> tuple[int, ...]:
        return self.array.shape

    def __getitem__(self: HasArray, key: Any) -> Any:
        return self.array[key]

    def __repr__(self: HasArray) -> str:
        return f"{type(self).__name__}(array={self.array!r})"


class ExplicitlyIndexedNDArrayMixin(NDArrayMixin, ExplicitlyIndexed):
    __slots__ = ()

    def get_duck_array(self) -> np.ndarray:
        key = BasicIndexer((slice(None),) * self.ndim)
        return self[key]

    def __array__(self, dtype: np.typing.DTypeLike = None) -> np.ndarray:
        # This is necessary because we apply the indexing key in self.get_duck_array()
        # Note this is the base class for all lazy indexing classes
        return np.asarray(self.get_duck_array(), dtype=dtype)


class NumpyIndexingAdapter(ExplicitlyIndexedNDArrayMixin):
    """Wrap a NumPy array to use explicit indexing."""

    __slots__ = ("array",)

    def __init__(self, array: np.ndarray) -> None:
        # In NumpyIndexingAdapter we only allow to store bare np.ndarray
        if not isinstance(array, np.ndarray):
            raise TypeError(
                "NumpyIndexingAdapter only wraps np.ndarray. "
                f"Trying to wrap {type(array)}"
            )
        self.array = array

    def _indexing_array_and_key(self, key: Any) -> tuple[np.ndarray, Any]:
        # if isinstance(key, OuterIndexer):
        #     array = self.array
        #     key = _outer_to_numpy_indexer(key, self.array.shape)
        # elif isinstance(key, VectorizedIndexer):
        #     array = NumpyVIndexAdapter(self.array)
        #     key = key.tuple
        if isinstance(key, BasicIndexer):
            array = self.array
            # We want 0d slices rather than scalars. This is achieved by
            # appending an ellipsis (see
            # https://numpy.org/doc/stable/reference/arrays.indexing.html#detailed-notes).
            key = (*key.tuple, Ellipsis)
        else:
            raise TypeError(f"unexpected key type: {type(key)}")

        return array, key

    def transpose(self, order: tuple[int, ...]) -> np.ndarray:
        return self.array.transpose(order)

    def __getitem__(self, key: Any) -> Any:
        array, key = self._indexing_array_and_key(key)
        return array[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        array, key = self._indexing_array_and_key(key)
        try:
            array[key] = value
        except ValueError as e:
            # More informative exception if read-only view
            if not array.flags.writeable and not array.flags.owndata:
                raise ValueError(
                    "Assignment destination is a view.  "
                    "Do you want to .copy() array first?"
                ) from e
            else:
                raise
