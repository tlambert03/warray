"""Package description."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("warray")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Talley Lambert"
__email__ = "talley.lambert@example.com"

__all__ = ["DataArray", "Variable"]

from ._dataarray import DataArray
from ._variable import Variable
