"""
The :mod:`sklearn.preprocessing` module includes scaling, centering and
normalization methods.
"""

from .flexible_scaler import (
    StandardFlexibleScaler,
    KernelFlexibleCenterer,
    SparseKernelCenterer,
)


__all__ = ["StandardFlexibleScaler", "KernelFlexibleCenterer", "SparseKernelCenterer"]
