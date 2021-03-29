"""
The :mod:`sklearn.sample selection` module will include FPS and CUR selection, each
with the optional PCov-flavor
"""

from .simple_fps import FPS
from .simple_cur import CUR
from .pcov_fps import PCovFPS
from .pcov_cur import PCovCUR


__all__ = ["PCovFPS", "PCovCUR", "FPS", "CUR"]
