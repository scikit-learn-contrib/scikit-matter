Data Sub-Selection
============================

This module contains two feature and sample selection modules corresponding to CUR selection and Farthest Point Sampling.  Each method can be modified to include aspects of supervised methods using the `mixing` parameter (0 - 1), where `mixing = 1` corresponds to traditional, unsupervised CUR and FPS. These methods are the focus of `Imbalzano [2018] <https://aip.scitation.org/doi/10.1063/1.5024611>`_ and Cersonsky [2021] (forthcoming).

.. _CUR-api:

CUR
###

.. currentmodule:: skcosmo.selection.CUR

CUR decomposition begins by approximating a matrix :math:`{\mathbf{X}}` using a subset of columns and rows

.. math::
    \mathbf{\hat{X}} \approx \mathbf{X}_\mathbf{c} \left(\mathbf{X}_\mathbf{c}^- \mathbf{X} \mathbf{X}_\mathbf{r}^-\right) \mathbf{X}_\mathbf{r}.

These subsets of rows and columns, denoted :math:`\mathbf{X}_\mathbf{r}` and :math:`\mathbf{X}_\mathbf{c}`, respectively, are determined by iterative maximization of a leverage score :math:`\pi`, representative of the relative importance of each column or row.
In each iteration of CUR, we select the column or row that maximizes :math:`\pi` and orthogonalize the remaining columns or rows.
These steps are iterated until a sufficient number of features has been selected.
This iterative approach, albeit comparatively time consuming, is the most deterministic and efficient route in reducing the number of features needed to approximate :math:`\mathbf{X}` when compared to selecting all features in a single iteration based upon the relative :math:`\\pi` importance.

These selection methods can be modified to be semi-supervised by using augmented right or left singular vectors, as shown in [Cersonsky 2021].

.. autoclass:: FeatureCUR
    :show-inheritance:
    :special-members:

    .. automethod:: get_product
    .. automethod:: orthogonalize
    .. automethod:: select

.. autoclass:: SampleCUR
    :show-inheritance:
    :special-members:

    .. automethod:: get_product
    .. automethod:: orthogonalize
    .. automethod:: select

.. _FPS-api:

Farthest Point-Sampling
#######################

.. currentmodule:: skcosmo.selection.FPS

Farthest Point Sampling is a common selection technique intended to exploit the  diversity of the input space.

In FPS, the selection of the first point is made at random or by a separate metric.
Each subsequent selection is made to maximize the distance to the previous selections.
It is common to use the Euclidean distance, however other distance metrics may be employed.

These selection methods can be modified to be semi-supervised by using the PCovR covariance and Gram matrices to compute the distances, as shown in [Cersonsky 2021].

.. autoclass:: FeatureFPS
    :show-inheritance:

    .. automethod:: select

.. autoclass:: SampleFPS
    :show-inheritance:

    .. automethod:: select

Random Partitioning
###################

.. currentmodule:: skcosmo.model_selection

.. autofunction:: train_test_split


Orthogonalizers for CUR
#######################

.. currentmodule:: skcosmo.selection.orthogonalizers

When computing non-iterative CUR, it is necessary to orthogonalize the input matrices after each selection. For this, we have supplied a feature and a sample orthogonalizer for feature and sample selection.

.. autofunction:: feature_orthogonalizer
.. autofunction:: sample_orthogonalizer
