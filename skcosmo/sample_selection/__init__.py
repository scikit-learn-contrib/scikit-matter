"""
The :mod:`sklearn.sample selection` module will include FPS and CUR selection, each
with the optional PCov-flavor
"""

from ..selection.FPS import SampleFPS
from ..selection.CUR import SampleCUR


__all__ = ["SampleFPS", "SampleCUR"]
