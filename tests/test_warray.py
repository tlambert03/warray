import numpy as np

from warray import DataArray


def test_basic_array() -> None:
    da = DataArray(np.random.rand(4, 3), coords=[["a", "b", "c", "d"], [1, 2, 3]])
