from __future__ import annotations

import types
from collections.abc import Hashable, Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Protocol, overload

import numpy as np

if TYPE_CHECKING:
    from typing import Any, Union

    from numpy.typing import DTypeLike

    DTypeMaybeMapping = Union[DTypeLike, Mapping[Any, DTypeLike]]


class AryProto(Protocol):
    @property
    def values(self) -> np.ndarray: ...
    @property
    def ndim(self) -> int: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    def __len__(self) -> int: ...
    def __getitem__(self, key: Any) -> Any: ...
    def _iter(self) -> Iterator[Any]: ...


class AbstractArray:
    """Shared base class for DataArray and Variable."""

    __slots__ = ()

    def __bool__(self: AryProto) -> bool:
        return bool(self.values)

    def __float__(self: AryProto) -> float:
        return float(self.values)

    def __int__(self: AryProto) -> int:
        return int(self.values)

    def __complex__(self: AryProto) -> complex:
        return complex(self.values)

    def __array__(self: AryProto, dtype: DTypeLike | None = None) -> np.ndarray:
        return np.asarray(self.values, dtype=dtype)

    # def __repr__(self) -> str:
    # return formatting.array_repr(self)

    # def _repr_html_(self):
    #     if OPTIONS["display_style"] == "text":
    #         return f"<pre>{escape(repr(self))}</pre>"
    #     return formatting_html.array_repr(self)

    def __format__(self: AryProto, format_spec: str = "") -> str:
        if format_spec != "":
            if self.shape == ():
                # Scalar values might be ok use format_spec with instead of repr:
                return self.data.__format__(format_spec)  # type: ignore
            else:
                # TODO: If it's an array the formatting.array_repr(self) should
                # take format_spec as an input. If we'd only use self.data we
                # lose all the information about coords for example which is
                # important information:
                raise NotImplementedError(
                    "Using format_spec is only supported"
                    f" when shape is (). Got shape = {self.shape}."
                )
        else:
            return self.__repr__()

    def _iter(self: AryProto) -> Iterator[Any]:
        for n in range(len(self)):
            yield self[n]

    def __iter__(self: AryProto) -> Iterator[Any]:
        if len(self.shape) == 0:
            raise TypeError("iteration over a 0-d array")
        return self._iter()

    @overload
    def get_axis_num(self, dim: Iterable[Hashable]) -> tuple[int, ...]: ...

    @overload
    def get_axis_num(self, dim: Hashable) -> int: ...

    def get_axis_num(self, dim: Hashable | Iterable[Hashable]) -> int | tuple[int, ...]:
        """Return axis number(s) corresponding to dimension(s) in this array.

        Parameters
        ----------
        dim : str or iterable of str
            Dimension name(s) for which to lookup axes.

        Returns
        -------
        int or tuple of int
            Axis number or numbers corresponding to the given dimensions.
        """
        if not isinstance(dim, str) and isinstance(dim, Iterable):
            return tuple(self._get_axis_num(d) for d in dim)
        else:
            return self._get_axis_num(dim)

    def _get_axis_num(self: Any, dim: Hashable) -> int:
        _raise_if_any_duplicate_dimensions(self.dims)
        try:
            return self.dims.index(dim)  # type: ignore
        except ValueError as e:
            raise ValueError(
                f"{dim!r} not found in array dimensions {self.dims!r}"
            ) from e

    @property
    def sizes(self: Any) -> Mapping[Hashable, int]:
        """Ordered mapping from dimension names to lengths.

        Immutable.

        See Also
        --------
        Dataset.sizes
        """
        return types.MappingProxyType(dict(zip(self.dims, self.shape)))


def _raise_if_any_duplicate_dimensions(
    dims: tuple, err_context: str = "This function"
) -> None:
    if len(set(dims)) < len(dims):
        repeated_dims = {d for d in dims if dims.count(d) > 1}
        raise ValueError(
            f"{err_context} cannot handle duplicate dimensions, but dimensions "
            f"{repeated_dims} appear more than once on this object's dims: {dims}"
        )
