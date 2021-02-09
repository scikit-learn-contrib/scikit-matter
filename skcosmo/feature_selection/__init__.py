"""
The :mod:`sklearn.selection` module includes FPS and CUR selection, each
with the optional PCov-flavor
"""

from ..selection.FPS import FeatureFPS
from ..selection.CUR import FeatureCUR
from .simple_fps import SimpleFPS


__all__ = ["FeatureFPS", "FeatureCUR", "SimpleFPS"]
