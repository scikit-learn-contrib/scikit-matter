"""
The :mod:`sklearn.selection` module includes FPS and CUR selection, each
with the optional PCov-flavor
"""

from .FPS import FeatureFPS, SampleFPS
from .CUR import FeatureCUR, SampleCUR
from .VoronoiFPS import SimpleFPS, SimpleVoronoiFPS, SampleVoronoiFPS, FeatureVoronoiFPS


__all__ = [
    "FeatureFPS",
    "SampleFPS",
    "FeatureCUR",
    "SampleCUR",
    "SimpleFPS",
    "SimpleVoronoiFPS",
    "SampleVoronoiFPS",
    "FeatureVoronoiFPS",
]
