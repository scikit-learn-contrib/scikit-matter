"""
Sequential sample selection
"""

import warnings

import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d
from scipy.interpolate.interpnd import _ndim_coords_from_arrays
from scipy.spatial import ConvexHull
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .._selection import _CUR, _FPS, _PCovCUR, _PCovFPS


def _linear_interpolator(points, values):
    """
    Returns linear interpolator for unstructured D-D data. Tessellate the input point
    set to N-D simplices, and interpolate linearly on each simplex. See
    ``LinearNDInterpolator`` for more details.

    points : 2-D ndarray of floats with shape (n, D), or length D tuple of 1-D ndarrays with shape (n,).
        Data point coordinates.
    values : ndarray of float or complex, shape (n,)
        Data values.


    Reference:
    ---------
    The code is an adapted excerpt from
    https://github.com/scipy/scipy/blob/dde50595862a4f9cede24b5d1c86935c30f1f88a/scipy/interpolate/_ndgriddata.py#L119-L273
    """  # NoQa: E501

    points = _ndim_coords_from_arrays(points)

    if points.ndim < 2:
        ndim = points.ndim
    else:
        ndim = points.shape[-1]

    if ndim == 1:
        points = points.ravel()
        # Sort points/values together, necessary as input for interp1d
        idx = np.argsort(points)
        points = points[idx]
        values = values[idx]
        return interp1d(
            points, values, kind="linear", axis=0, bounds_error=False, fill_value=np.nan
        )
    else:
        return LinearNDInterpolator(points, values, fill_value=np.nan, rescale=False)


class FPS(_FPS):
    """
    Transformer that performs Greedy Sample Selection using Farthest Point Sampling.

    Parameters
    ----------

    initialize: int, list of int, or 'random', default=0
        Index of the first selection(s). If 'random', picks a random
        value when fit starts. Stored in :py:attr:`self.initialize`.

    n_to_select : int or float, default=None
        The number of selections to make. If `None`, half of the samples are
        selected. If integer, the parameter is the absolute number of selections
        to make. If float between 0 and 1, it is the fraction of the total dataset to
        select. Stored in :py:attr:`self.n_to_select`.

    score_threshold : float, default=None
        Threshold for the score. If `None` selection will continue until the
        n_to_select is chosen. Otherwise will stop when the score falls below the
        threshold. Stored in :py:attr:`self.score_threshold`.

    score_threshold_type : str, default="absolute"
        How to interpret the ``score_threshold``. When "absolute", the score used by the
        selector is compared to the threshold directly. When "relative", at each
        iteration, the score used by the selector is compared proportionally to the
        score of the first selection, i.e. the selector quits when
        ``current_score / first_score < threshold``. Stored in
        :py:attr:`self.score_threshold_type`.

    progress_bar: bool, default=False
              option to use `tqdm <https://tqdm.github.io/>`_ progress bar to monitor
              selections. Stored in :py:attr:`self.report_progress`.

    full : bool, default=False
        In the case that all non-redundant selections are exhausted, choose
        randomly from the remaining samples. Stored in :py:attr:`self.full`.

    random_state: int or RandomState instance, default=0

    Attributes
    ----------

    n_selected_ : int
                  Counter tracking the number of selections that have been made

    X_selected_ : ndarray,
                  Matrix containing the selected samples, for use in fitting

    y_selected_ : ndarray,
                  In sample selection, the matrix containing the selected targets, for
                  use in fitting

    selected_idx_ : ndarray
                  indices of selected samples

    Examples
    --------
    >>> from skmatter.sample_selection import FPS
    >>> import numpy as np
    >>> selector = FPS(
    ...     n_to_select=2,
    ...     # int or 'random', default=0
    ...     # Index of the first selection.
    ...     # If ‘random’, picks a random value when fit starts.
    ...     initialize=0,
    ... )
    >>> X = np.array(
    ...     [
    ...         [0.12, 0.21, 0.02],  # 3 samples, 3 features
    ...         [-0.09, 0.32, -0.10],
    ...         [-0.03, -0.53, 0.08],
    ...     ]
    ... )
    >>> selector.fit(X)
    FPS(n_to_select=2)
    >>> selector.selected_idx_
    array([0, 2])
    """

    def __init__(
        self,
        initialize=0,
        n_to_select=None,
        score_threshold=None,
        score_threshold_type="absolute",
        progress_bar=False,
        full=False,
        random_state=0,
    ):
        super().__init__(
            selection_type="sample",
            initialize=initialize,
            n_to_select=n_to_select,
            score_threshold=score_threshold,
            score_threshold_type=score_threshold_type,
            progress_bar=progress_bar,
            full=full,
            random_state=random_state,
        )


class PCovFPS(_PCovFPS):
    """Transformer that performs Greedy Sample Selection using PCovR-weighted
    Farthest Point Sampling.

    Parameters
    ----------

    mixing: float, default=0.5
            The PCovR mixing parameter, as described in PCovR as
            :math:`{\\alpha}`

    initialize: int or 'random', default=0
        Index of the first selection. If 'random', picks a random
        value when fit starts.

    n_to_select : int or float, default=None
        The number of selections to make. If `None`, half of the samples are
        selected. If integer, the parameter is the absolute number of selections
        to make. If float between 0 and 1, it is the fraction of the total dataset to
        select. Stored in :py:attr:`self.n_to_select`.

    score_threshold : float, default=None
        Threshold for the score. If `None` selection will continue until the n_to_select
        is chosen. Otherwise will stop when the score falls below the threshold. Stored
        in :py:attr:`self.score_threshold`.

    score_threshold_type : str, default="absolute"
        How to interpret the ``score_threshold``. When "absolute", the score used by the
        selector is compared to the threshold directly. When "relative", at each
        iteration, the score used by the selector is compared proportionally to the
        score of the first selection, i.e. the selector quits when
        ``current_score / first_score < threshold``. Stored in
        :py:attr:`self.score_threshold_type`.

    progress_bar: bool, default=False
              option to use `tqdm <https://tqdm.github.io/>`_ progress bar to monitor
              selections. Stored in :py:attr:`self.report_progress`.

    full : bool, default=False
        In the case that all non-redundant selections are exhausted, choose
        randomly from the remaining samples. Stored in :py:attr:`self.full`.

    random_state: int or RandomState instance, default=0

    Attributes
    ----------

    n_selected_ : int
                  Counter tracking the number of selections that have been made

    X_selected_ : ndarray,
                  Matrix containing the selected samples, for use in fitting

    y_selected_ : ndarray,
                  In sample selection, the matrix containing the selected targets, for
                  use in fitting

    selected_idx_ : ndarray
                  indices of selected samples

    Examples
    --------
    >>> from skmatter.sample_selection import PCovFPS
    >>> import numpy as np
    >>> selector = PCovFPS(
    ...     n_to_select=2,
    ...     # int or 'random', default=0
    ...     # Index of the first selection.
    ...     # If ‘random’, picks a random value when fit starts.
    ...     initialize=0,
    ... )
    >>> X = np.array(
    ...     [
    ...         [0.12, 0.21, 0.02],  # 3 samples, 3 features
    ...         [-0.09, 0.32, -0.10],
    ...         [-0.03, -0.53, 0.08],
    ...     ]
    ... )
    >>> y = np.array([0.0, 0.0, 1.0])  # classes of each sample
    >>> selector.fit(X, y)
    PCovFPS(n_to_select=2)
    >>> selector.selected_idx_
    array([0, 2])
    """

    def __init__(
        self,
        mixing=0.5,
        initialize=0,
        n_to_select=None,
        score_threshold=None,
        score_threshold_type="absolute",
        progress_bar=False,
        full=False,
        random_state=0,
    ):
        super().__init__(
            selection_type="sample",
            mixing=mixing,
            initialize=initialize,
            n_to_select=n_to_select,
            score_threshold=score_threshold,
            score_threshold_type=score_threshold_type,
            progress_bar=progress_bar,
            full=full,
            random_state=random_state,
        )


class CUR(_CUR):
    """Transformer that performs Greedy Sample Selection by choosing samples
    which maximize the magnitude of the left singular vectors, consistent with
    classic CUR matrix decomposition.

    Parameters
    ----------
    recompute_every : int
                      number of steps after which to recompute the pi score
                      defaults to 1, if 0 no re-computation is done

    k : int
        number of eigenvectors to compute the importance score with, defaults to 1

    tolerance: float
         threshold below which scores will be considered 0, defaults to 1E-12

    n_to_select : int or float, default=None
        The number of selections to make. If `None`, half of the samples are
        selected. If integer, the parameter is the absolute number of selections
        to make. If float between 0 and 1, it is the fraction of the total dataset to
        select. Stored in :py:attr:`self.n_to_select`.

    score_threshold : float, default=None
        Threshold for the score. If `None` selection will continue until the
        n_to_select is chosen. Otherwise will stop when the score falls below the
        threshold. Stored in :py:attr:`self.score_threshold`.

    score_threshold_type : str, default="absolute"
        How to interpret the ``score_threshold``. When "absolute", the score used by
        the selector is compared to the threshold directly. When "relative", at each
        iteration, the score used by the selector is compared proportionally to the
        score of the first selection, i.e. the selector quits when
        ``current_score / first_score < threshold``. Stored in
        :py:attr:`self.score_threshold_type`.

    progress_bar: bool, default=False
              option to use `tqdm <https://tqdm.github.io/>`_ progress bar to monitor
              selections. Stored in :py:attr:`self.report_progress`.

    full : bool, default=False
        In the case that all non-redundant selections are exhausted, choose
        randomly from the remaining samples. Stored in :py:attr:`self.full`.

    random_state: int or RandomState instance, default=0

    Attributes
    ----------

    X_current_ : ndarray (n_samples, n_features)
                  The original matrix orthogonalized by previous selections

    n_selected_ : int
                  Counter tracking the number of selections that have been made

    X_selected_ : ndarray,
                  Matrix containing the selected samples, for use in fitting

    y_selected_ : ndarray,
                  In sample selection, the matrix containing the selected targets, for
                  use in fitting

    pi_ : ndarray (n_features),
                  the importance score see :func:`_compute_pi`

    selected_idx_ : ndarray
                  indices of selected features

    Examples
    --------
    >>> from skmatter.sample_selection import CUR
    >>> import numpy as np
    >>> selector = CUR(n_to_select=2, random_state=0)
    >>> X = np.array(
    ...     [
    ...         [0.12, 0.21, 0.02],  # 3 samples, 3 features
    ...         [-0.09, 0.32, -0.10],
    ...         [-0.03, -0.53, 0.08],
    ...     ]
    ... )
    >>> np.random.seed(0)  # there is a source of randomness in it
    >>> selector.fit(X)
    CUR(n_to_select=2)
    >>> np.round(selector.pi_, 2)  # importance scole
    array([0., 1., 0.])
    >>> selector.selected_idx_  # importance scole
    array([2, 0])
    >>> # selector.transform(X) cannot be used as sklearn API
    >>> # restricts the change of sample size using transformers
    >>> # So one has to do
    >>> X[selector.selected_idx_]
    array([[-0.03, -0.53,  0.08],
           [ 0.12,  0.21,  0.02]])
    """

    def __init__(
        self,
        recompute_every=1,
        k=1,
        tolerance=1e-12,
        n_to_select=None,
        score_threshold=None,
        score_threshold_type="absolute",
        progress_bar=False,
        full=False,
        random_state=0,
    ):
        super().__init__(
            selection_type="sample",
            recompute_every=recompute_every,
            k=k,
            tolerance=tolerance,
            n_to_select=n_to_select,
            score_threshold=score_threshold,
            score_threshold_type=score_threshold_type,
            progress_bar=progress_bar,
            full=full,
            random_state=random_state,
        )


class PCovCUR(_PCovCUR):
    r"""Transformer that performs Greedy Sample Selection by choosing samples
    which maximize the importance score :math:`\pi`, which is the sum over
    the squares of the first :math:`k` components of the PCovR-modified
    left singular vectors.

    Parameters
    ----------

    mixing: float, default=0.5
            The PCovR mixing parameter, as described in PCovR as
            :math:`{\\alpha}`. Stored in :py:attr:`self.mixing`.

    recompute_every : int
                      number of steps after which to recompute the pi score
                      defaults to 1, if 0 no re-computation is done


    k : int
        number of eigenvectors to compute the importance score with, defaults to 1

    tolerance: float
         threshold below which scores will be considered 0, defaults to 1E-12

    n_to_select : int or float, default=None
        The number of selections to make. If `None`, half of the samples are
        selected. If integer, the parameter is the absolute number of selections
        to make. If float between 0 and 1, it is the fraction of the total dataset to
        select. Stored in :py:attr:`self.n_to_select`.

    score_threshold : float, default=None
        Threshold for the score. If `None` selection will continue until the
        n_to_select is chosen. Otherwise will stop when the score falls below the
        threshold. Stored in :py:attr:`self.score_threshold`.

    score_threshold_type : str, default="absolute"
        How to interpret the ``score_threshold``. When "absolute", the score used by
        the selector is compared to the threshold directly. When "relative", at each
        iteration, the score used by the selector is compared proportionally to the
        score of the first selection, i.e. the selector quits when
        ``current_score / first_score < threshold``. Stored in
        :py:attr:`self.score_threshold_type`.

    progress_bar: bool, default=False
              option to use `tqdm <https://tqdm.github.io/>`_
              progress bar to monitor selections.
              Stored in :py:attr:`self.report_progress`.

    full : bool, default=False
        In the case that all non-redundant selections are exhausted, choose
        randomly from the remaining samples. Stored in :py:attr:`self.full`.

    random_state: int or RandomState instance, default=0

    Attributes
    ----------

    X_current_ : ndarray (n_samples, n_features)
                  The original matrix orthogonalized by previous selections

    y_current_ : ndarray (n_samples, n_properties)
                The targets orthogonalized by a regression on
                the previous selections.

    n_selected_ : int
                  Counter tracking the number of selections that have been made

    X_selected_ : ndarray,
                  Matrix containing the selected samples, for use in fitting

    y_selected_ : ndarray,
                  In sample selection, the matrix containing the selected targets, for
                  use in fitting

    pi_ : ndarray (n_features),
                  the importance score see :func:`_compute_pi`

    selected_idx_ : ndarray
                  indices of selected features

    Examples
    --------
    >>> from skmatter.sample_selection import PCovCUR
    >>> import numpy as np
    >>> selector = PCovCUR(n_to_select=2, random_state=0)
    >>> X = np.array(
    ...     [
    ...         [0.12, 0.21, 0.02],  # 3 samples, 3 features
    ...         [-0.09, 0.32, -0.10],
    ...         [-0.03, -0.53, 0.08],
    ...     ]
    ... )
    >>> y = np.array([0.0, 0.0, 1.0])  # classes of each sample
    >>> selector.fit(X, y)
    PCovCUR(n_to_select=2)
    >>> np.round(selector.pi_, 2)  # importance scole
    array([1., 0., 0.])
    >>> selector.selected_idx_  # importance scole
    array([2, 1])
    >>> # selector.transform(X) cannot be used as sklearn API
    >>> # restricts the change of sample size using transformers
    >>> # So one has to do
    >>> X[selector.selected_idx_]
    array([[-0.03, -0.53,  0.08],
           [-0.09,  0.32, -0.1 ]])
    """

    def __init__(
        self,
        mixing=0.5,
        recompute_every=1,
        k=1,
        tolerance=1e-12,
        n_to_select=None,
        score_threshold=None,
        score_threshold_type="absolute",
        progress_bar=False,
        full=False,
        random_state=0,
    ):
        super().__init__(
            selection_type="sample",
            mixing=mixing,
            recompute_every=recompute_every,
            k=k,
            tolerance=tolerance,
            n_to_select=n_to_select,
            score_threshold=score_threshold,
            score_threshold_type=score_threshold_type,
            progress_bar=progress_bar,
            full=full,
            random_state=random_state,
        )


def _directional_distance(equations, points):
    """
    Computes the distance of the points to the planes defined by the equations
    with respect to the direction of the first dimension.

    equations : ndarray of shape (n_facets, n_dim)
                each row contains the coefficienst for the plane equation of the form
                equations[i, 0]*x_1 + ...
                    + equations[i, -2]*x_{n_dim} = equations[i, -1]
                -equations[i, -1] is the offset
    points    : ndarray of shape (n_samples, n_dim)
                points to compute the directional distance from

    Returns
    -------
    directional_distance : ndarray of shape (nsamples, nequations)
                closest distance wrt. the first dimension of the point to the planes
                defined by the equations
    """
    orthogonal_distances = -(points @ equations[:, :-1].T) - equations[:, -1:].T
    return -orthogonal_distances / equations[:, :1].T


class DirectionalConvexHull:
    """
    Performs Sample Selection by constructing a Directional Convex Hull and determining
    the distance to the hull as outlined in the reference [dch]_.

    Parameters
    ----------
    low_dim_idx    : list of ints, default None
                   Indices of columns of X containing features to be used for the
                   directional convex hull construction (also known as the low-
                   dimensional (LD) hull). By default [0] is used.

    tolerance      : float, default=1.0E-12
                   Tolerance for the negative distances to the directional
                   convex hull to consider a point below the convex hull. Depending
                   if a point is below or above the convex hull the distance is
                   differently computed. A very low value can result in a completely
                   wrong distances. Distances cannot be distinguished from zero  up
                   to tolerance. It is recommended to leave the default setting.

    Attributes
    ----------
    high_dim_idx_   : list of ints
                    Indices of columns in data containing high-dimensional
                    features (i.e. those not used for the convex hull
                    construction)

    selected_idx_ : numpy.ndarray
                    Indices of datapoints that form the vertices of the
                    convex hull
    interpolator_high_dim_  : scipy.interpolate.interpnd.LinearNDInterpolator
                    Interpolator for the features in the high-
                    dimensional space

    Examples
    --------
    >>> from skmatter.sample_selection import DirectionalConvexHull
    >>> selector = DirectionalConvexHull(
    ...     # Indices of columns of X to use for fitting
    ...     # the convex hull
    ...     low_dim_idx=[0, 1],
    ... )
    >>> X = np.array(
    ...     [
    ...         [0.12, 0.21, 0.02],  # 3 samples, 3 features
    ...         [-0.09, 0.32, -0.10],
    ...         [-0.03, -0.53, 0.08],
    ...         [-0.41, 0.25, 0.34],
    ...     ]
    ... )
    >>> y = np.array([0.1, 1.0, 0.2, 0.4])  # classes of each sample
    >>> dch = selector.fit(X, y)
    >>> # Get the distance to the convex hull for samples used to fit the
    >>> # convex hull. This can also be called using other samples (X_new)
    >>> # and corresponding properties (y_new) that were not used to fit
    >>> # the hull. In this case they are alle one the conex hull so we
    >>> # zeros
    >>> np.allclose(dch.score_samples(X, y), [0.0, 0.0, 0.0, 0.0])
    True



    References
    ----------
    .. [dch] A. Anelli, E. A. Engel, C. J. Pickard and M. Ceriotti,
             Physical Review Materials, 2018.
    """

    def __init__(self, low_dim_idx=None, tolerance=1e-12):
        self.low_dim_idx = low_dim_idx

        if low_dim_idx is None:
            self.low_dim_idx = [0]
        else:
            self.low_dim_idx = low_dim_idx
        self.tolerance = tolerance

    def fit(self, X, y):
        """
        Learn the samples that form the convex hull.

        Parameters
        ----------
        X        : ndarray of shape (n_samples, n_features)
                   Feature matrix of samples to use for constructing the convex
                   hull.
        y        : ndarray of shape (n_samples,)
                   Target values (property on which the convex hull should be
                   constructed, e.g. Gibbs free energy)

        Returns
        -------
        self : object
            Fitted scorer.
        """

        X, y = self._check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        if len(y.shape) == 1:
            y = y.reshape((len(y), 1))

        if (max(np.abs(self.low_dim_idx)) > X.shape[1]) and (
            min(self.low_dim_idx) >= 0
        ):
            raise ValueError(
                "One or more columns indexed with low_dim_idx is"
                " out of bounds with the dimensions of X."
            )

        self.high_dim_idx_ = np.setdiff1d(np.arange(X.shape[1]), self.low_dim_idx)

        # get number of dimensions for the convex (lower dimensional) hull construction
        n_low_dim_idx = len(self.low_dim_idx)

        # append features and target property to the same data matrix (for the
        # convex hull construction)
        convex_hull_data = np.zeros((X.shape[0], n_low_dim_idx + 1))
        convex_hull_data[:, :1] = y
        convex_hull_data[:, 1:] = X[:, self.low_dim_idx].copy()

        # create high-dimensional feature matrix
        high_dim_feats = X[:, self.high_dim_idx_]
        # get scipy convex hull
        self.convex_hull_ = ConvexHull(convex_hull_data, incremental=True)
        # get normal equations to the hull simplices
        y_normal = self.convex_hull_.equations[:, 0]

        # get vertices_idx of the convex hull
        directional_facets_idx = np.where(y_normal < 0)[0]
        self.directional_simplices_ = self.convex_hull_.simplices[
            directional_facets_idx
        ]
        self._directional_equations_ = self.convex_hull_.equations[
            directional_facets_idx
        ]

        self.selected_idx_ = np.unique(self.directional_simplices_.flatten())
        self._directional_points_ = convex_hull_data[self.selected_idx_]

        # required for the score_feature_matrix function
        self.interpolator_high_dim_ = _linear_interpolator(
            points=convex_hull_data[self.selected_idx_, 1:],
            values=high_dim_feats[self.selected_idx_],
        )

        return self

    @property
    def directional_vertices_(self):
        return self.selected_idx_

    def _check_X_y(self, X, y):
        return check_X_y(X, y, ensure_min_features=1, multi_output=False)

    def _check_is_fitted(self, X):
        check_is_fitted(
            self,
            [
                "high_dim_idx_",
                "interpolator_high_dim_",
                "selected_idx_",
            ],
        )
        n_features = X.shape[1]
        if n_features != self.n_features_in_:
            raise ValueError(
                f"X has {n_features} features, but {self.__class__.__name__} "
                f"is expecting {self.n_features_in_} features as input."
            )
        return True

    def score_samples(self, X, y):
        """
        Calculate the distance of the samples to the convex hull in the target
        direction y. Samples with a distance > 0 lie above the convex surface.
        Samples with a distance of zero lie on the convex surface. Samples with
        a distance value < 0 lie below the convex surface.

        Parameters
        ----------
        X    : ndarray of shape (n_samples, n_features)
               Feature matrix of samples to use for determining distance
               to the convex hull. Please note that samples provided should
               have the same dimensions (features) as used during fitting
               of the convex hull. The same column indices will be used for
               the low- and high-dimensional features.

        y    : ndarray of shape (n_samples,)
               Target values (property on which the convex hull should be
               constructed, e.g. Gibbs free energy)

        Returns
        -------
        dch_distance : ndarray of shape (n_samples, len(high_dim_idx_))
            The distance (residuals)  of samples to the convex hull in
            the higher-dimensional space.
        """
        X, y = self._check_X_y(X, y)
        self._check_is_fitted(X)

        # features used for the convex hull construction
        low_dim_feats = X[:, self.low_dim_idx]
        hull_space_data = np.hstack((y.reshape(-1, 1), low_dim_feats))

        # the X points projected on the convex surface
        return self._directional_convex_hull_distance(hull_space_data).reshape(y.shape)

    def _directional_convex_hull_distance(self, points):
        """Distance to the fitted directional convex hull in the target dimension y.

        If a point lies above the convex hull, it will have a positive distance. If a
        point lies below the convex hull, it will have a negative distance - this should
        only occur if you pass a point that the convex hull has not been fitted on in
        the `fit` function. If a point lies on the convex hull, it will have a distance
        of zero.

        Parameters
        ----------
        points : The points for which you would like to calculate the distance to the
                 fitted directional convex hull.
        """
        # directional distances to all plane equations
        all_directional_distances = _directional_distance(
            self._directional_equations_, points
        )
        # we get negative distances for each plane to check if any distance is below the
        # threshold
        below_directional_convex_hull = np.any(
            all_directional_distances < -self.tolerance, axis=1
        )
        # directional distances to corresponding plane equation
        directional_distances = np.zeros(len(points))
        directional_distances[~below_directional_convex_hull] = np.min(
            all_directional_distances[~below_directional_convex_hull], axis=1
        )
        # some distances can be positive, so we take the max of all negative distances
        negative_directional_distances = all_directional_distances.copy()
        negative_directional_distances[all_directional_distances > 0] = -np.inf
        directional_distances[below_directional_convex_hull] = np.max(
            negative_directional_distances[below_directional_convex_hull], axis=1
        )
        return directional_distances

    def score_feature_matrix(self, X):
        """
        Calculate the distance (or more specifically, the residuals) of the
        samples to the convex hull in the high-dimensional space. Samples
        with a distance value of zero in all the higher dimensions lie on
        the convex hull.


        Parameters
        ----------
        X    : ndarray of shape (n_samples, n_features)
               Feature matrix of samples to use for determining distance
               to the convex hull. Please note that samples provided should
               have the same dimensions (features) as used during fitting
               of the convex hull. The same column indices will be used for
               the low- and high-dimensional features.

        Returns
        -------
        dch_distance : numpy.array of shape (n_samples, len(high_dim_idx_))
            The distance (residuals)  of samples to the convex hull in
            the higher-dimensional space.
        """
        X = check_array(X)
        self._check_is_fitted(X)

        # features used for the convex hull construction
        low_dim_feats = X[:, self.low_dim_idx]
        # HD features not used for the convex hull
        high_dim_feats = X[:, self.high_dim_idx_]

        if len(self.low_dim_idx) == 1:
            low_dim_feats = low_dim_feats.reshape(
                -1,
            )
        # interpolate features
        interpolated_high_dim_feats = self.interpolator_high_dim_(low_dim_feats)

        if np.any(np.isnan(interpolated_high_dim_feats)):
            warnings.warn(
                "There are samples in X with a low-dimensional part that is outside "
                "of the range of the convex surface. Distance will contain nans.",
                UserWarning,
                stacklevel=1,
            )

        # determine the distance between the original high-dimensional data and
        # interpolated high-dimensional data
        dch_distance = high_dim_feats - interpolated_high_dim_feats

        return dch_distance
