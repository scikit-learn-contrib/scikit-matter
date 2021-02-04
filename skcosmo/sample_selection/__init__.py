"""
The :mod:`sklearn.selection` module includes FPS and CUR selection, each
with the optional PCov-flavor
"""

from ..selection.FPS import SampleFPS
from ..selection.CUR import SampleCUR


__all__ = ["SampleFPS", "SampleCUR"]
