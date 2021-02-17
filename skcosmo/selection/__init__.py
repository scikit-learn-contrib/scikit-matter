"""
The :mod:`sklearn.selection` module includes FPS and CUR selection, each
with the optional PCov-flavor
"""

from .FPS import SampleFPS
from .CUR import SampleCUR


__all__ = ["SampleFPS", "SampleCUR"]
