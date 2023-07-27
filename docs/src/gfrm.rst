.. _gfrm:

Reconstruction Measures
=======================

.. marker-reconstruction-introduction-begin

A set of easily-interpretable error measures of the relative information capacity of
feature space `F` with respect to feature space `F'`. The methods returns a value
between 0 and 1, where 0 means that `F` and `F'` are completey distinct in terms of
linearly-decodable information, and where 1 means that `F'` is contained in `F`. All
methods are implemented as the root mean-square error for the regression of the
feature matrix `X_F'` (or sometimes called `Y` in the doc) from `X_F` (or sometimes
called `X` in the doc) for transformations with different constraints (linear,
orthogonal, locally-linear). By default a custom 2-fold cross-validation
:py:class:`skosmo.linear_model.RidgeRegression2FoldCV` is used to ensure the
generalization of the transformation and efficiency of the computation, since we deal
with a multi-target regression problem. Methods were applied to compare different
forms of featurizations through different hyperparameters and induced metrics and
kernels [Goscinski2021]_ .

.. marker-reconstruction-introduction-end

.. currentmodule:: skmatter.metrics

.. _GRE-api:

Global Reconstruction Error
---------------------------

.. autofunction:: pointwise_global_reconstruction_error
.. autofunction:: global_reconstruction_error

.. _GRD-api:

Global Reconstruction Distortion
--------------------------------

.. autofunction:: pointwise_global_reconstruction_distortion
.. autofunction:: global_reconstruction_distortion

.. _LRE-api:

Local Reconstruction Error
--------------------------

.. autofunction:: pointwise_local_reconstruction_error
.. autofunction:: local_reconstruction_error
