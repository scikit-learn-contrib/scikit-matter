"""
The :mod:`sklearn.preprocessing` module includes scaling, centering and
normalization methods.
"""

from .flexible_scaler import StandardFlexibleScaler, KernelFlexibleCenterer


__all__ = ["StandardFlexibleScaler", "KernelFlexibleCenterer"]
