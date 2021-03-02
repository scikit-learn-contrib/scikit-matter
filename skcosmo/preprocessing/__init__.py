"""
The :mod:`sklearn.preprocessing` module includes scaling, centering and
normalization methods.
"""

from .flexible_scaler import (
    StandardFlexibleScaler,
    KernelNormalizer,
    SparseKernelCenterer,
)


__all__ = ["StandardFlexibleScaler", "KernelNormalizer", "SparseKernelCenterer"]
