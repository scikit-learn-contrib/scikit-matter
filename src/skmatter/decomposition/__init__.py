r"""
Often, one wants to construct new ML features from their current representation
in order to compress data or visualise trends in the dataset. In the archetypal
method for this dimensionality reduction, principal components analysis (PCA),
features are transformed into the latent space which best preserves the
variance of the original data.

This module provides the Principal Covariates
Regression (PCovR), as introduced by [deJong1992]_, which is a modification to PCA
that incorporates target information, such that the resulting embedding could
be tuned using a mixing parameter α to improve performance in regression tasks
(:math:`\alpha = 0` corresponding to linear regression and :math:`\alpha = 1`
corresponding to PCA). Also provided is Principal Covariates Classification (PCovC),
proposed in [Jorgensen2025]_, which can similarly be used for classification problems.

[Helfrecht2020]_ introduced the non-linear version of PCovR,
Kernel Principal Covariates Regression (KPCovR), where the mixing parameter α
now interpolates between kernel ridge regression (:math:`\alpha = 0`) and
kernel principal components analysis (KPCA, :math:`\alpha = 1`).

The module includes:

* :ref:`PCovR-api` the standard Principal Covariates Regression. Utilises a
  combination between a PCA-like and an LR-like loss, and therefore attempts to find
  a low-dimensional projection of the feature vectors that simultaneously minimises
  information loss and error in predicting the target properties using only the
  latent space vectors :math:`\mathbf{T}`.
* :ref:`PCovC-api` the standard Principal Covariates Classification, proposed in
  [Jorgensen2025]_.
* :ref:`KPCovR-api` the Kernel Principal Covariates Regression.
  A kernel-based variation on the
  original PCovR method, proposed in [Helfrecht2020]_.
"""

from ._pcov import _BasePCov

from ._pcovr import PCovR
from ._pcovc import PCovC

from ._kernel_pcovr import KernelPCovR

__all__ = [
    "_BasePCov",
    "PCovR",
    "PCovC",
    "KernelPCovR",
]
