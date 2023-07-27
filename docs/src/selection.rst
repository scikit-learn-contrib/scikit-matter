.. _selection-api:

Feature and Sample Selection
============================

.. marker-selection-introduction-begin

Data sub-selection modules primarily corresponding to methods derived from
CUR matrix decomposition and Farthest Point Sampling. In their classical form,
CUR and FPS determine a data subset that maximizes the variance (CUR) or
distribution (FPS) of the features or samples.
These methods can be modified to combine supervised target information denoted by the
methods `PCov-CUR` and `PCov-FPS`.
For further reading, refer to [Imbalzano2018]_ and [Cersonsky2021]_.

These selectors can be used for both feature and sample selection, with similar
instantiations. All sub-selection methods  scores each feature or sample
(without an estimator)
and chooses that with the maximum score. As an simple example

.. doctest::

    >>> # feature selection
    >>> import numpy as np
    >>> from skmatter.feature_selection import CUR, FPS, PCovCUR, PCovFPS
    >>> selector = CUR(
    ...     # the number of selections to make
    ...     # if None, set to half the samples or features
    ...     # if float, fraction of the total dataset to select
    ...     # if int, absolute number of selections to make
    ...     n_to_select=2,
    ...     # option to use `tqdm <https://tqdm.github.io/>`_ progress bar
    ...     progress_bar=True,
    ...     # float, cutoff score to stop selecting
    ...     score_threshold=1e-12,
    ...     # boolean, whether to select randomly after non-redundant selections
    ...     # are exhausted
    ...     full=False,
    ... )
    >>> X = np.array(
    ...     [
    ...         [0.12, 0.21, 0.02],  # 3 samples, 3 features
    ...         [-0.09, 0.32, -0.10],
    ...         [-0.03, -0.53, 0.08],
    ...     ]
    ... )
    >>> y = np.array([0.0, 0.0, 1.0])  # classes of each sample
    >>> selector.fit(X)
    CUR(n_to_select=2, progress_bar=True, score_threshold=1e-12)
    >>> Xr = selector.transform(X)
    >>> print(Xr.shape)
    (3, 2)
    >>> selector = PCovCUR(n_to_select=2)
    >>> selector.fit(X, y)
    PCovCUR(n_to_select=2)
    >>> Xr = selector.transform(X)
    >>> print(Xr.shape)
    (3, 2)
    >>>
    >>> # Now sample selection
    >>> from skmatter.sample_selection import CUR, FPS, PCovCUR, PCovFPS
    >>> selector = CUR(n_to_select=2)
    >>> selector.fit(X)
    CUR(n_to_select=2)
    >>> Xr = X[selector.selected_idx_]
    >>> print(Xr.shape)
    (2, 3)

.. marker-selection-introduction-end

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
may decelerate, computations when compared to FPS.  The sweet spot for Voronoi FPS is
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
