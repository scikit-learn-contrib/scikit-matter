"""
The :mod:`sklearn.sample selection` module will include FPS and CUR selection, each
with the optional PCov-flavor
"""

from .pcov_cur import PCovCUR
from .pcov_fps import PCovFPS
from .simple_cur import CUR
from .simple_fps import FPS

__all__ = ["PCovFPS", "PCovCUR", "FPS", "CUR"]
