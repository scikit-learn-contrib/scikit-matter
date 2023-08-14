"""This module includes scaling, centering and normalization methods."""

from ._data import (
    KernelNormalizer,
    SparseKernelCenterer,
    StandardFlexibleScaler,
)

__all__ = ["StandardFlexibleScaler", "KernelNormalizer", "SparseKernelCenterer"]
