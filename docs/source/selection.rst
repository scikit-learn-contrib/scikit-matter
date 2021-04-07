
Feature and Sample Selection
============================

`scikit-COSMO` contains multiple data sub-selection modules,
primarily corresponding to methods derived from CUR matrix decomposition
and Farthest Point Sampling. In their classical form, CUR and FPS determine
a data subset that maximizes the
variance (CUR) or distribution (FPS) of the features or samples. These methods
can be modified to combine supervised and unsupervised learning, in a formulation
denoted `PCov-CUR` and `PCov-FPS`.
For further reading, refer to [Imbalzano2018]_ and [Cersonsky2021]_.


These selectors can be used for both feature and sample selection, with similar
instantiations. Currently, all sub-selection methods extend :py:class:`GreedySelector`,
where at each iteration the model scores each
feature or sample (without an estimator) and chooses that with the maximum score.
This can be executed using:

.. code-block:: python

    selector = Selector(
                        # the number of selections to make
                        # if None, set to half the samples or features
                        # if float, fraction of the total dataset to select
                        # if int, absolute number of selections to make
                        n_to_select=4,

                        # option to use `tqdm <https://tqdm.github.io/>`_ progress bar
                        progress_bar=True,

                        # float, cutoff score to stop selecting
                        score_threshold=1E-12

                        # boolean, whether to select randomly after non-redundant selections are exhausted
                        full=False,
                        )
    selector.fit(X, y)

    Xr = selector.transform(X)

where `Selector` is one of the classes below that overwrites the method :py:func:`score`.

From :py:class:`GreedySelector`, selectors inherit these public methods:

.. currentmodule:: skcosmo._selection

.. class:: GreedySelector

  .. automethod:: fit
  .. automethod:: transform
  .. automethod:: get_support

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

The feature and sample selection versions of CUR differ only in the computation
of :math:`\pi`. In sample selection :math:`\pi` is computed using the left
singular vectors, versus in feature selection, :math:`\pi` is computed using the
right singular vectors. In addition to :py:class:`GreedySelector`, both instances
of CUR selection build off of :py:class:`skcosmo._selection._cur._CUR`, and inherit

.. currentmodule:: skcosmo._selection

.. automethod:: _CUR.score
.. automethod:: _CUR._compute_pi

They are instantiated using
:py:class:`skcosmo.feature_selection.CUR` and :py:class:`skcosmo.sample_selection.CUR`, e.g.

.. code-block:: python

    from skcosmo.feature_selection import CUR
    selector = CUR(
                        n_to_select=4,
                        progress_bar=True,
                        score_threshold=1E-12
                        full=False,

                        # int, number of eigenvectors to use in computing pi
                        k = 1,

                        # boolean, whether to orthogonalize after each selection, defaults to true
                        iterative = True,

                        # float, threshold below which scores will be considered 0, defaults to 1E-12
                        tolerance=1E-12,
                        )
    selector.fit(X)

    Xr = selector.transform(X)


PCov-CUR
########

PCov-CUR extends upon CUR by using augmented right or left singular vectors
inspired by Principal Covariates Regression, as demonstrated in [Cersonsky2021]_.
These methods employ the modified kernel and covariance matrices introduced in :ref:`PCovR-api`
and available via the Utility Classes.

Again, the feature and sample selection versions of PCov-CUR differ only in the computation
of :math:`\pi`. So, in addition to :py:class:`GreedySelector`, both instances
of PCov-CUR selection build off of :py:class:`skcosmo._selection._cur._PCovCUR`, inheriting

.. currentmodule:: skcosmo._selection

.. automethod:: _PCovCUR.score
.. automethod:: _PCovCUR._compute_pi

and are instantiated using
:py:class:`skcosmo.feature_selection.PCovCUR` and :py:class:`skcosmo.sample_selection.PCovCUR`.

.. code-block:: python

    from skcosmo.feature_selection import PCovCUR
    selector = PCovCUR(
                        n_to_select=4,
                        progress_bar=True,
                        score_threshold=1E-12
                        full=False,

                        # float, default=0.5
                        # The PCovR mixing parameter, as described in PCovR as alpha
                        mixing = 0.5,

                        # int, number of eigenvectors to use in computing pi
                        k = 1,

                        # boolean, whether to orthogonalize after each selection, defaults to true
                        iterative = True,

                        # float, threshold below which scores will be considered 0, defaults to 1E-12
                        tolerance=1E-12,
                        )
    selector.fit(X, y)

    Xr = selector.transform(X)

.. _FPS-api:

Farthest Point-Sampling (FPS)
#############################

Farthest Point Sampling is a common selection technique intended to exploit the
diversity of the input space.

In FPS, the selection of the first point is made at random or by a separate metric.
Each subsequent selection is made to maximize the Haussdorf distance,
i.e. the minimum distance between a point and all previous selections.
It is common to use the Euclidean distance, however other distance metrics may be employed.

Similar to CUR, the feature and selection versions of FPS differ only in the way
distance is computed (feature selection does so column-wise, sample selection does
so row-wise), and are built off of the same base class, :py:class:`skcosmo._selection._fps._FPS`,
in addition to GreedySelector, and inherit

.. currentmodule:: skcosmo._selection

.. automethod:: _FPS.score
.. automethod:: _FPS.get_distance
.. automethod:: _FPS.get_select_distance

These selectors can be instantiated using
:py:class:`skcosmo.feature_selection.FPS` and :py:class:`skcosmo.sample_selection.FPS`.

.. code-block:: python

    from skcosmo.feature_selection import FPS
    selector = FPS(
                        n_to_select=4,
                        progress_bar=True,
                        score_threshold=1E-12
                        full=False,

                        # int or 'random', default=0
                        # Index of the first selection.
                        # If ‘random’, picks a random value when fit starts.
                        initialize = 0,
                        )
    selector.fit(X)

    Xr = selector.transform(X)

PCov-FPS
########
PCov-FPS extends upon FPS much like PCov-CUR does to CUR. Instead of using the
Euclidean distance solely in the space of :math:`\mathbf{X}`, we use a combined
distance in terms of :math:`\mathbf{X}` and :math:`\mathbf{y}`.

Again, the feature and sample selection versions of PCov-FPS differ only in
computing the distances. So, in addition to :py:class:`GreedySelector`, both instances
of PCov-FPS selection build off of :py:class:`skcosmo._selection._fps._PCovFPS`, and inherit

.. currentmodule:: skcosmo._selection

.. automethod:: _PCovFPS.score
.. automethod:: _PCovFPS.get_distance
.. automethod:: _PCovFPS.get_select_distance


and can
be instantiated using
:py:class:`skcosmo.feature_selection.PCovFPS` and :py:class:`skcosmo.sample_selection.PCovFPS`.

.. code-block:: python

    from skcosmo.feature_selection import PCovFPS
    selector = PCovFPS(
                        n_to_select=4,
                        progress_bar=True,
                        score_threshold=1E-12
                        full=False,

                        # float, default=0.5
                        # The PCovR mixing parameter, as described in PCovR as alpha
                        mixing = 0.5,

                        # int or 'random', default=0
                        # Index of the first selection.
                        # If ‘random’, picks a random value when fit starts.
                        initialize = 0,
                        )
    selector.fit(X, y)

    Xr = selector.transform(X)
