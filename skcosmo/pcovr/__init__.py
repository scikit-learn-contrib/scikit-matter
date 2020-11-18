"""
The :mod:`skcosmo.selection` module includes FPS and CUR selection, each
with the optional PCov-flavor
"""

from .pcovr_distances import get_Ct, get_Kt


__all__ = ["get_Ct", "get_Kt"]
