"""
The :mod:`sklearn.selection` module includes FPS and CUR selection, each
with the optional PCov-flavor
"""

from ..selection.FPS import FeatureFPS
from ..selection.CUR import FeatureCUR


__all__ = ["FeatureFPS", "FeatureCUR"]
