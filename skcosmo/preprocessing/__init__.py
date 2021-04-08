"""
The :mod:`sklearn.preprocessing` module includes scaling, centering and
normalization methods.
"""

from .flexible_scaler import (
    KernelNormalizer,
    SparseKernelCenterer,
    StandardFlexibleScaler,
)

__all__ = ["StandardFlexibleScaler", "KernelNormalizer", "SparseKernelCenterer"]
