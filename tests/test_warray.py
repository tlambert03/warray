from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np
import pytest
import xarray as xr

from warray import DataArray


class Case(NamedTuple):
    data: np.ndarray
    expected_dims: tuple[str, ...]
    expected_coords: set[str]
    coords: list | dict | None = None
    dims: Sequence[str] | None = None


CASES = [
    Case(
        data=np.random.rand(4, 3),
        expected_dims=("dim_0", "dim_1"),
        expected_coords={"dim_0", "dim_1"},
        coords=[["a", "b", "c", "d"], [1, 2, 3]],
    ),
    Case(
        data=np.random.rand(4, 3),
        expected_dims=("d0", "d1"),
        expected_coords={"d0", "d1"},
        coords={"d0": ["a", "b", "c", "d"], "d1": [1, 2, 3]},
    ),
    Case(
        data=np.random.rand(4, 3),
        expected_dims=("d0", "d1"),
        expected_coords={"d0", "d1"},
        coords=[["a", "b", "c", "d"], [1, 2, 3]],
        dims=["d0", "d1"],
    ),
]


@pytest.mark.parametrize("case", CASES)
def test_basic_array(case: Case) -> None:
    data = case.data
    coords = case.coords
    dims = case.dims

    wa = DataArray(data, coords=coords, dims=dims)
    xa = xr.DataArray(data, coords=coords, dims=dims)
    assert set(wa.coords) == set(xa.coords) == case.expected_coords
    assert wa.dims == xa.dims == case.expected_dims
    assert np.array_equal(xa, data)
    assert np.array_equal(wa, data)
    assert np.array_equal(wa, xa)
    wa0 = wa[0]
    xa0 = xa[0]

    xa.isel({case.expected_dims[0]: 0})
    wa.isel({case.expected_dims[0]: 0})

    assert wa0.dims == xa0.dims == case.expected_dims[1:]
    assert set(wa0.coords) == set(xa0.coords) == case.expected_coords
    assert np.array_equal(xa0, data[0])
    assert np.array_equal(wa0, xa0)
