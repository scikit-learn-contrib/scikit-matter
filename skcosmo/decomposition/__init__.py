"""
The :mod:`skcosmo.decomposition` module includes the two distance
measures, as defined by Principal Covariates Regression (PCovR)
"""

from ._kernel_pcovr import KernelPCovR
from ._pcovr import (
    PCovR,
    pcovr_covariance,
    pcovr_kernel,
)

__all__ = ["pcovr_covariance", "pcovr_kernel", "PCovR", "KernelPCovR"]
