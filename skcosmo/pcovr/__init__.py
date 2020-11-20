"""
The :mod:`skcosmo.pcovr` module includes the two distance
measures, as defined by Principal Covariates Regression (PCovR)
"""

from .pcovr_distances import pcovr_covariance, pcovr_kernel_distance


__all__ = ["pcovr_covariance", "pcovr_kernel_distance"]
