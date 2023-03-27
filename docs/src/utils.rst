Utility Classes
===============

.. _PCovR_dist-api:

.. currentmodule:: skmatter.utils._pcovr_utils

Modified Gram Matrix :math:`\mathbf{\tilde{K}}`
###############################################

.. autofunction:: pcovr_kernel


Modified Covariance Matrix :math:`\mathbf{\tilde{C}}`
#####################################################

.. autofunction:: pcovr_covariance

Orthogonalizers for CUR
#######################

.. currentmodule:: skmatter.utils._orthogonalizers

When computing non-iterative CUR, it is necessary to orthogonalize the input matrices after each selection. For this, we have supplied a feature and a sample orthogonalizer for feature and sample selection.

.. autofunction:: X_orthogonalizer
.. autofunction:: Y_feature_orthogonalizer
.. autofunction:: Y_sample_orthogonalizer


Random Partitioning with Overlaps
#################################

.. automodule:: skmatter.model_selection._split
