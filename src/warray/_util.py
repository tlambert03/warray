from __future__ import annotations

import warnings
from typing import (
    Any,
    Hashable,
    Iterable,
    Literal,
    Mapping,
    Protocol,
    TypeVar,
    cast,
)

import numpy as np

ErrorOptionsWithWarn = Literal["raise", "warn", "ignore"]


class DuckArray(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def dtype(self) -> np.dtype: ...
    def __getitem__(self, key: Any) -> Any: ...


# Temporary placeholder for indicating an array api compliant type.
T_DuckArray = TypeVar("T_DuckArray", bound=DuckArray)
T = TypeVar("T")


def expanded_indexer(key: Any, ndim: int) -> tuple[Any, ...]:
    """Return tuple with length equal to the number of dimensions.

    The expansion is done by replacing all `Ellipsis` items with the right
    number of full slices and then padding the key with full slices so that it
    reaches the appropriate dimensionality.
    """
    if not isinstance(key, tuple):
        # numpy treats non-tuple keys equivalent to tuples of length 1
        key = (key,)

    new_key = []
    # handling Ellipsis right is a little tricky, see:
    # https://numpy.org/doc/stable/reference/arrays.indexing.html#advanced-indexing
    found_ellipsis = False
    for k in key:
        if k is Ellipsis:
            if not found_ellipsis:
                new_key.extend((ndim + 1 - len(key)) * [slice(None)])
                found_ellipsis = True
            else:
                new_key.append(slice(None))
        else:
            new_key.append(k)
    if len(new_key) > ndim:
        raise IndexError("too many indices")
    new_key.extend((ndim - len(new_key)) * [slice(None)])
    return tuple(new_key)


def either_dict_or_kwargs(
    pos_kwargs: Mapping[Any, T] | None,
    kw_kwargs: Mapping[str, T],
    func_name: str,
) -> Mapping[Hashable, T]:
    if pos_kwargs is None or pos_kwargs == {}:
        # Need an explicit cast to appease mypy due to invariance; see
        # https://github.com/python/mypy/issues/6228
        return cast(Mapping[Hashable, T], kw_kwargs)

    if not isinstance(pos_kwargs, Mapping):
        raise ValueError(f"the first argument to .{func_name} must be a dictionary")
    if kw_kwargs:
        raise ValueError(
            f"cannot specify both keyword and positional arguments to .{func_name}"
        )
    return pos_kwargs


def is_fancy_indexer(indexer: Any) -> bool:
    """Return False if indexer is a int, slice, a 1d list, or a 0-1dim ndarray."""
    if isinstance(indexer, (int, slice)):
        return False
    if isinstance(indexer, np.ndarray):
        return indexer.ndim > 1
    if isinstance(indexer, list):
        return bool(indexer) and not isinstance(indexer[0], int)
    return True


def drop_dims_from_indexers(
    indexers: Mapping[Any, Any],
    dims: Iterable[Hashable] | Mapping[Any, int],
    missing_dims: ErrorOptionsWithWarn,
) -> Mapping[Hashable, Any]:
    """Drop any dimensions from indexers that are not present in dims.

    Parameters
    ----------
    indexers : dict
        A dict with keys matching dimensions and values given
        by integers, slice objects or arrays.
    dims : sequence
        The dimensions that should be selected from.
    missing_dims : {"raise", "warn", "ignore"}
        What to do if dimensions that should be selected from are not present in the
        DataArray.
    """
    if missing_dims == "raise":
        invalid = indexers.keys() - set(dims)
        if invalid:
            raise ValueError(
                f"Dimensions {invalid} do not exist. Expected one or more of {dims}"
            )

        return indexers

    elif missing_dims == "warn":
        # don't modify input
        indexers = dict(indexers)

        invalid = indexers.keys() - set(dims)
        if invalid:
            warnings.warn(
                f"Dimensions {invalid} do not exist. Expected one or more of {dims}",
                stacklevel=3,
            )
        for key in invalid:
            indexers.pop(key)

        return indexers

    elif missing_dims == "ignore":
        return {key: val for key, val in indexers.items() if key in dims}

    else:
        raise ValueError(
            f"Unrecognised option {missing_dims} for missing_dims argument"
        )


def as_compatible_data(data: T_DuckArray | Any) -> T_DuckArray:
    from warray import DataArray, Variable

    if isinstance(data, (Variable, DataArray)):
        return cast("T_DuckArray", data.data)

    if not isinstance(data, np.ndarray) and (
        hasattr(data, "__array_function__") or hasattr(data, "__array_namespace__")
    ):
        return cast("T_DuckArray", data)

    return cast("T_DuckArray", np.asarray(data))
