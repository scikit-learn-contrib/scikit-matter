"""
This module implements the sparse kernel density estimator.

A large dataset can be generated during the molecular dynamics sampling. The
distribution of the sampled data reflects the (free) energetic stability of molecular
patterns. The KDE model can be used to characterize the probability distribution, and
thus to identify the stable patterns in the system. However, the computational
cost of KDE is `O(N^2)` where `N` is the number of sampled points, which is very
expensive. Here we offer a sparse implementation of the KDE model with a
`O(MN)` computational cost, where `M` is the number of grid points generated from the
sampled data.

The following class is available:

* :ref:`sparse-kde-api` computes the kernel density estimator based on a set of grid
  points generated from the sampled data.

"""

from ._sparsekde import SparseKDE

__all__ = ["SparseKDE"]
