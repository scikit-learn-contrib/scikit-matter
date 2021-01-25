"""
The :mod:`skcosmo.pcovr` module includes the two distance
measures, as defined by Principal Covariates Regression (PCovR)
"""

from .pcovr import PCovR, pcovr_covariance, pcovr_kernel


__all__ = ["pcovr_covariance", "pcovr_kernel", "PCovR"]
