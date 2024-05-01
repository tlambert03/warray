import numpy as np

from warray import DataArray
from xarray import DataArray, Variable


def test_basic_array() -> None:
    data = np.random.rand(4, 3)
    da = DataArray(data, coords=[["a", "b", "c", "d"], [1, 2, 3]])
    assert set(da.coords) == {"dim_0", "dim_1"}
    assert da.dims == ("dim_0", "dim_1")
    assert np.array_equal(da.data, data)

    da0 = da[0]
