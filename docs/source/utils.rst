utils
=====

.. _PCovR_dist-api:

.. currentmodule:: skcosmo.utils.pcovr_utils

Modified Gram Matrix :math:`\mathbf{\tilde{K}}`
###############################################

.. autofunction:: pcovr_kernel


Modified Covariance Matrix :math:`\mathbf{\tilde{C}}`
#####################################################

.. autofunction:: pcovr_covariance

Orthogonalizers for CUR
#######################

.. currentmodule:: skcosmo.utils.orthogonalizers

When computing non-iterative CUR, it is necessary to orthogonalize the input matrices after each selection. For this, we have supplied a feature and a sample orthogonalizer for feature and sample selection.

.. autofunction:: feature_orthogonalizer
.. autofunction:: sample_orthogonalizer


Random Partitioning with Overlaps
#################################

.. automodule:: skcosmo.model_selection._split
