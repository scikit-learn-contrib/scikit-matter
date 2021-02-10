"""
The :mod:`skcosmo.decomposition` module includes the two distance
measures, as defined by Principal Covariates Regression (PCovR)
"""

from .pcovr import PCovR, pcovr_covariance, pcovr_kernel
from .kpcovr import KPCovR

__all__ = ["pcovr_covariance", "pcovr_kernel", "PCovR", "KPCovR"]
