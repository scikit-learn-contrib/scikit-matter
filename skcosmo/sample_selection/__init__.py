"""
The :mod:`sklearn.sample selection` module will include FPS and CUR selection, each
with the optional PCov-flavor
"""

from ..selection.FPS import SampleFPS
from ..selection.CUR import SampleCUR
from .simple_fps import FPS
from .simple_cur import CUR

__all__ = ["SampleFPS", "SampleCUR", "CUR", "FPS"]
