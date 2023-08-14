.. _selection-api:

Feature and Sample Selection
============================

.. automodule:: skmatter._selection

.. _CUR-api:

CUR
---

CUR decomposition begins by approximating a matrix :math:`{\mathbf{X}}` using a subset
of columns and rows

.. math::
    \mathbf{\hat{X}} \approx \mathbf{X}_\mathbf{c} \left(\mathbf{X}_\mathbf{c}^-
    \mathbf{X} \mathbf{X}_\mathbf{r}^-\right) \mathbf{X}_\mathbf{r}.

These subsets of rows and columns, denoted :math:`\mathbf{X}_\mathbf{r}` and
:math:`\mathbf{X}_\mathbf{c}`, respectively, can be determined by iterative maximization
of a leverage score :math:`\pi`, representative of the relative importance of each
column or row. From hereon, we will call selection methods which are derived off of the
CUR decomposition "CUR" as a shorthand for "CUR-derived selection". In each iteration of
CUR, we select the column or row that maximizes :math:`\pi` and orthogonalize the
remaining columns or rows. These steps are iterated until a sufficient number of
features has been selected. This iterative approach, albeit comparatively time
consuming, is the most deterministic and efficient route in reducing the number of
features needed to approximate :math:`\mathbf{X}` when compared to selecting all
features in a single iteration based upon the relative :math:`\pi` importance.

The feature and sample selection versions of CUR differ only in the computation of
:math:`\pi`. In sample selection :math:`\pi` is computed using the left singular
vectors, versus in feature selection, :math:`\pi` is computed using the right singular
vectors.

.. autoclass:: skmatter.feature_selection.CUR
   :members:
   :private-members: _compute_pi
   :undoc-members:
   :inherited-members:

.. autoclass:: skmatter.sample_selection.CUR
   :members:
   :private-members: _compute_pi
   :undoc-members:
   :inherited-members:

.. _PCov-CUR-api:

PCov-CUR
--------

PCov-CUR extends upon CUR by using augmented right or left singular vectors inspired by
Principal Covariates Regression, as demonstrated in [Cersonsky2021]_. These methods
employ the modified kernel and covariance matrices introduced in :ref:`PCovR-api` and
available via the Utility Classes.

Again, the feature and sample selection versions of PCov-CUR differ only in the
computation of :math:`\pi`. S

.. autoclass:: skmatter.feature_selection.PCovCUR
   :members:
   :private-members: _compute_pi
   :undoc-members:
   :inherited-members:

.. autoclass:: skmatter.sample_selection.PCovCUR
   :members:
   :private-members: _compute_pi
   :undoc-members:
   :inherited-members:

.. _FPS-api:

Farthest Point-Sampling (FPS)
-----------------------------

Farthest Point Sampling is a common selection technique intended to exploit the
diversity of the input space.

In FPS, the selection of the first point is made at random or by a separate metric. Each
subsequent selection is made to maximize the Haussdorf distance, i.e. the minimum
distance between a point and all previous selections. It is common to use the Euclidean
distance, however other distance metrics may be employed.

Similar to CUR, the feature and selection versions of FPS differ only in the way
distance is computed (feature selection does so column-wise, sample selection does so
row-wise), and are built off of the same base class,

These selectors can be instantiated using :py:class:`skmatter.feature_selection.FPS` and
:py:class:`skmatter.sample_selection.FPS`.

.. autoclass:: skmatter.feature_selection.FPS
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: skmatter.sample_selection.FPS
   :members:
   :undoc-members:
   :inherited-members:

.. _PCov-FPS-api:

PCov-FPS
--------

PCov-FPS extends upon FPS much like PCov-CUR does to CUR. Instead of using the Euclidean
distance solely in the space of :math:`\mathbf{X}`, we use a combined distance in terms
of :math:`\mathbf{X}` and :math:`\mathbf{y}`.

.. autoclass:: skmatter.feature_selection.PCovFPS
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: skmatter.sample_selection.PCovFPS
   :members:
   :undoc-members:
   :inherited-members:

.. _Voronoi-FPS-api:

Voronoi FPS
-----------

.. autoclass:: skmatter.sample_selection.VoronoiFPS
   :members:
   :undoc-members:
   :inherited-members:


When *Not* to Use Voronoi FPS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In many cases, this algorithm may not increase upon the efficiency. For example, for
simple metrics (such as Euclidean distance), Voronoi FPS will likely not accelerate, and
may decelerate, computations when compared to FPS. The sweet spot for Voronoi FPS is
when the number of selectable samples is already enough to divide the space with Voronoi
polyhedrons, but not yet comparable to the total number of samples, when the cost of
bookkeeping significantly degrades the speed of work compared to FPS.

.. _DCH-api:

Directional Convex Hull (DCH)
-----------------------------

.. autoclass:: skmatter.sample_selection.DirectionalConvexHull
   :members:
   :undoc-members:
   :inherited-members:
