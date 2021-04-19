"""
The :mod:`sklearn.sample selection` module will include FPS and CUR selection, each
with the optional PCov-flavor
"""

from ._base import (
    CUR,
    FPS,
    PCovCUR,
    PCovFPS,
)
from ._voronoi_fps import VoronoiFPS

__all__ = ["PCovFPS", "PCovCUR", "FPS", "CUR", "VoronoiFPS"]
