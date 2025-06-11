"""Classes for building linear models."""

from ._base import OrthogonalRegression
from ._ridge import Ridge2FoldCV

__all__ = ["OrthogonalRegression", "Ridge2FoldCV"]
