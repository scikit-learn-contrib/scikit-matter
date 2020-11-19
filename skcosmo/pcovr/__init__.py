"""
The :mod:`skcosmo.selection` module includes FPS and CUR selection, each
with the optional PCov-flavor
"""

from .pcovr_distances import pcovr_covariance, pcovr_kernel_distance


__all__ = ["pcovr_covariance", "pcovr_kernel_distance"]
