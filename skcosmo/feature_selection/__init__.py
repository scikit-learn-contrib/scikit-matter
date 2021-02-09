"""
The :mod:`sklearn.selection` module includes FPS and CUR selection, each
with the optional PCov-flavor
"""

from ..selection.FPS import FeatureFPS  # noqa
from ..selection.CUR import FeatureCUR  # noqa
from .simple_fps import SimpleFPS, CSimpleFPS  # noqa

__all__ = ["FeatureFPS", "FeatureCUR", "CSimpleFPS", "SimpleFPS"]
