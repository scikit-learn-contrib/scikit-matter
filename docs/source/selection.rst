Feature and Sample Selection
============================

`scikit-COSMO` contains multiple feature and sample selection modules,
primarily corresponding to methods derived from CUR matrix decomposition
and Farthest Point Sampling.

In their classical form, these methods determine a data subset that maximizes the
variance (CUR) or distribution (FPS) of the features or samples. These methods
can be modified to combine supervised and unsupervised learning, in a formulation
denoted `PCov-CUR` and `PCov-FPS`.
For further reading, refer to [Imbalzano2018]_ and [Cersonsky2021]_.

.. _CUR-api:

CUR
###


CUR decomposition begins by approximating a matrix :math:`{\mathbf{X}}` using a subset of columns and rows

.. math::
    \mathbf{\hat{X}} \approx \mathbf{X}_\mathbf{c} \left(\mathbf{X}_\mathbf{c}^- \mathbf{X} \mathbf{X}_\mathbf{r}^-\right) \mathbf{X}_\mathbf{r}.

These subsets of rows and columns, denoted :math:`\mathbf{X}_\mathbf{r}` and
:math:`\mathbf{X}_\mathbf{c}`, respectively, can be determined by iterative
maximization of a leverage score :math:`\pi`, representative of the relative
importance of each column or row. From hereon, we will call selection methods
which are derived off of the CUR decomposition "CUR" as a shorthand for
"CUR-derived selection". In each iteration of CUR, we select the column or row
that maximizes :math:`\pi` and orthogonalize the remaining columns or rows.
These steps are iterated until a sufficient number of features has been selected.
This iterative approach, albeit comparatively time consuming, is the most
deterministic and efficient route in reducing the number of features needed to
approximate :math:`\mathbf{X}` when compared to selecting all features in a
single iteration based upon the relative :math:`\pi` importance.

These selection methods can be modified to be semi-supervised by using augmented
right or left singular vectors, as shown in [Cersonsky2021]_.

Feature Selection using CUR
----------------------------

.. currentmodule:: skcosmo.feature_selection
.. autoclass:: CUR
    :show-inheritance:
    :no-undoc-members:

    .. automethod:: fit

Sample Selection using CUR
----------------------------

.. currentmodule:: skcosmo.sample_selection
.. autoclass:: CUR
    :show-inheritance:
    :no-undoc-members:

    .. automethod:: fit


Feature Selection using CUR + PCovR
-----------------------------------

.. currentmodule:: skcosmo.feature_selection
.. autoclass:: PCovCUR
    :show-inheritance:
    :members:

Sample Selection using CUR + PCovR
----------------------------------

.. currentmodule:: skcosmo.sample_selection
.. autoclass:: PCovCUR
    :show-inheritance:
    :members:

.. _FPS-api:

Farthest Point-Sampling
#######################

Farthest Point Sampling is a common selection technique intended to exploit the
diversity of the input space.

In FPS, the selection of the first point is made at random or by a separate metric.
Each subsequent selection is made to maximize the distance to the previous selections.
It is common to use the Euclidean distance, however other distance metrics may be employed.


Feature Selection using FPS
---------------------------

.. currentmodule:: skcosmo.feature_selection
.. autoclass:: FPS
    :show-inheritance:
    :no-undoc-members:

    .. automethod:: fit

Sample Selection using FPS
--------------------------

.. currentmodule:: skcosmo.sample_selection
.. autoclass:: FPS
    :show-inheritance:
    :no-undoc-members:

    .. automethod:: fit


Feature Selection using FPS + PCovR
-----------------------------------

.. currentmodule:: skcosmo.feature_selection
.. autoclass:: PCovFPS
    :show-inheritance:
    :no-undoc-members:

    .. automethod:: fit


Sample Selection using FPS + PCovR
-----------------------------------

.. currentmodule:: skcosmo.sample_selection
.. autoclass:: PCovFPS
    :show-inheritance:
