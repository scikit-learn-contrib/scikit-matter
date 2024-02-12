"""Set of metrics that can be used for an enhanced understanding of your machine
learning model.

First are the easily-interpretable error measures of the relative information
capacity of feature space `F` with respect to feature space `F'`. The methods
returns a value between 0 and 1, where 0 means that `F` and `F'` are completey
distinct in terms of linearly-decodable information, and where 1 means that `F'`
is contained in `F`. All methods are implemented as the root mean-square error
for the regression of the feature matrix `X_F'` (or sometimes called `Y` in the
doc) from `X_F` (or sometimes called `X` in the doc) for transformations with
different constraints (linear, orthogonal, locally-linear). By default a custom
2-fold cross-validation :py:class:`skosmo.linear_model.Ridge2FoldCV`
is used to ensure the generalization of the transformation and efficiency of the
computation, since we deal with a multi-target regression problem. Methods were
applied to compare different forms of featurizations through different
hyperparameters and induced metrics and kernels [Goscinski2021]_ .

These reconstruction measures are available:

* :ref:`GRE-api` (GRE) computes the amount of linearly-decodable information
  recovered through a global linear reconstruction.
* :ref:`GRD-api` (GRD) computes the amount of distortion contained in a global
  linear reconstruction.
* :ref:`LRE-api` (LRE) computes the amount of decodable information recovered
  through a local linear reconstruction for the k-nearest neighborhood of each
  sample.

Next, we offer a set of prediction rigidity metrics, which can be used to
quantify the robustness of the local or component-wise predictions that the
machine learning model has been trained to make, based on the training dataset
composition.

These prediction rigidities are available:

* :ref:`LPR-api` (LPR) computes the local prediction rigidity of a linear or
  kernel model.
* :ref:`CPR-api` (CPR) computes the component-wise prediction rigidity of a
  linear or kernel model.
"""

from ._reconstruction_measures import (
    check_global_reconstruction_measures_input,
    check_local_reconstruction_measures_input,
    global_reconstruction_distortion,
    global_reconstruction_error,
    local_reconstruction_error,
    pointwise_global_reconstruction_distortion,
    pointwise_global_reconstruction_error,
    pointwise_local_reconstruction_error,
)

from ._prediction_rigidities import (
    local_prediction_rigidity,
    componentwise_prediction_rigidity,
)

from .pairwise import (
    pairwise_euclidean_distances,
)

__all__ = [
    "pointwise_global_reconstruction_error",
    "global_reconstruction_error",
    "pointwise_global_reconstruction_distortion",
    "global_reconstruction_distortion",
    "pointwise_local_reconstruction_error",
    "local_reconstruction_error",
    "check_global_reconstruction_measures_input",
    "check_local_reconstruction_measures_input",
    "local_prediction_rigidity",
    "componentwise_prediction_rigidity",
    "pairwise_euclidean_distances",
]
