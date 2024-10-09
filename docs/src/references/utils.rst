Utility Classes
===============

.. _PCovR_dist-api:

Modified Gram Matrix :math:`\mathbf{\tilde{K}}`
-----------------------------------------------

.. autofunction:: skmatter.utils.pcovr_kernel


Modified Covariance Matrix :math:`\mathbf{\tilde{C}}`
-----------------------------------------------------

.. autofunction:: skmatter.utils.pcovr_covariance

Orthogonalizers for CUR
-----------------------

When computing non-iterative CUR, it is necessary to orthogonalize the input matrices
after each selection. For this, we have supplied a feature and a sample orthogonalizer
for feature and sample selection.

.. autofunction:: skmatter.utils.X_orthogonalizer
.. autofunction:: skmatter.utils.Y_feature_orthogonalizer
.. autofunction:: skmatter.utils.Y_sample_orthogonalizer


Random Partitioning with Overlaps
---------------------------------

.. autofunction:: skmatter.model_selection.train_test_split


Effective Dimension of Covariance Matrix
----------------------------------------

.. autofunction:: skmatter.utils.effdim

Oracle Approximating Shrinkage
------------------------------

.. autofunction:: skmatter.utils.oas
