"""
Sequential sample selection
"""

import warnings

import numpy as np
from scipy.interpolate import (
    LinearNDInterpolator,
    interp1d,
)
from scipy.interpolate.interpnd import _ndim_coords_from_arrays
from scipy.spatial import ConvexHull
from sklearn.utils.validation import (
    check_array,
    check_is_fitted,
    check_X_y,
)

from .._selection import (
    _CUR,
    _FPS,
    _PCovCUR,
    _PCovFPS,
)


def _linear_interpolator(points, values):
    """
    Returns linear interpolater for unstructured D-D data. Tessellate the input point set to N-D simplices, and interpolate linearly on each simplex. See `LinearNDInterpolator` for more details.

    points : 2-D ndarray of floats with shape (n, D), or length D tuple of 1-D ndarrays with shape (n,).
        Data point coordinates.
    values : ndarray of float or complex, shape (n,)
        Data values.


    Reference:
    ---------
    The code is an adapted excerpt from
    https://github.com/scipy/scipy/blob/dde50595862a4f9cede24b5d1c86935c30f1f88a/scipy/interpolate/_ndgriddata.py#L119-L273
    """

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
        n_to_select is chosen. Otherwise will stop when the score falls below the threshold.
        Stored in :py:attr:`self.score_threshold`.

    score_threshold_type : str, default="absolute"
        How to interpret the ``score_threshold``. When "absolute", the score used by
        the selector is compared to the threshold directly. When "relative", at each iteration,
        the score used by the selector is compared proportionally to the score of the first
        selection, i.e. the selector quits when ``current_score / first_score < threshold``.
        Stored in :py:attr:`self.score_threshold_type`.

    progress_bar: bool, default=False
              option to use `tqdm <https://tqdm.github.io/>`_
              progress bar to monitor selections. Stored in :py:attr:`self.report_progress`.

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
                  In sample selection, the matrix containing the selected targets, for use in fitting

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
        Threshold for the score. If `None` selection will continue until the
        n_to_select is chosen. Otherwise will stop when the score falls below the threshold.
        Stored in :py:attr:`self.score_threshold`.

    score_threshold_type : str, default="absolute"
        How to interpret the ``score_threshold``. When "absolute", the score used by
        the selector is compared to the threshold directly. When "relative", at each iteration,
        the score used by the selector is compared proportionally to the score of the first
        selection, i.e. the selector quits when ``current_score / first_score < threshold``.
        Stored in :py:attr:`self.score_threshold_type`.

    progress_bar: bool, default=False
              option to use `tqdm <https://tqdm.github.io/>`_
              progress bar to monitor selections. Stored in :py:attr:`self.report_progress`.

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
                  In sample selection, the matrix containing the selected targets, for use in fitting


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
         n_to_select is chosen. Otherwise will stop when the score falls below the threshold.
         Stored in :py:attr:`self.score_threshold`.

     score_threshold_type : str, default="absolute"
         How to interpret the ``score_threshold``. When "absolute", the score used by
         the selector is compared to the threshold directly. When "relative", at each iteration,
         the score used by the selector is compared proportionally to the score of the first
         selection, i.e. the selector quits when ``current_score / first_score < threshold``.
         Stored in :py:attr:`self.score_threshold_type`.

     progress_bar: bool, default=False
               option to use `tqdm <https://tqdm.github.io/>`_
               progress bar to monitor selections. Stored in :py:attr:`self.report_progress`.

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
                   In sample selection, the matrix containing the selected targets, for use in fitting

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
        n_to_select is chosen. Otherwise will stop when the score falls below the threshold.
        Stored in :py:attr:`self.score_threshold`.

    score_threshold_type : str, default="absolute"
        How to interpret the ``score_threshold``. When "absolute", the score used by
        the selector is compared to the threshold directly. When "relative", at each iteration,
        the score used by the selector is compared proportionally to the score of the first
        selection, i.e. the selector quits when ``current_score / first_score < threshold``.
        Stored in :py:attr:`self.score_threshold_type`.

    progress_bar: bool, default=False
              option to use `tqdm <https://tqdm.github.io/>`_
              progress bar to monitor selections. Stored in :py:attr:`self.report_progress`.

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
                  In sample selection, the matrix containing the selected targets, for use in fitting

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


class DirectionalConvexHull:
    """
    Performs Sample Selection by constructing a Directional Convex Hull and determining the distance to the hull as outlined in the reference

    Parameters
    ----------

    low_dim_idx    : list of ints, default None
                   Indices of columns of X containing features to be used for the
                   directional convex hull construction (also known as the low-
                   dimensional (LD) hull). By default [0] is used.

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
                    Interpolater for the features in the high-
                    dimensional space

    interpolator_y_  : scipy.interpolate.interpnd.LinearNDInterpolator
                    Interpolater for the targets y

    References
    ----------
    .. [dch] A. Anelli, E. A. Engel, C. J. Pickard and M. Ceriotti,
             Physical Review Materials, 2018.
    """

    def __init__(self, low_dim_idx=None):
        self.low_dim_idx = low_dim_idx

        if low_dim_idx is None:
            self.low_dim_idx = [0]
        else:
            self.low_dim_idx = low_dim_idx

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
        convex_hull = ConvexHull(convex_hull_data)
        # get normal equations to the hull simplices
        y_normal = convex_hull.equations[:, 0]

        # get vertices_idx of the convex hull
        self.selected_idx_ = np.unique(
            convex_hull.simplices[np.where(y_normal < 0)[0]].flatten()
        )

        # required for the score_feature_matrix function
        self.interpolator_high_dim_ = _linear_interpolator(
            points=convex_hull_data[self.selected_idx_, 1:],
            values=high_dim_feats[self.selected_idx_],
        )

        # required to compute the distance of the low-dimensional feature to the convex hull
        self.interpolator_y_ = _linear_interpolator(
            points=convex_hull_data[self.selected_idx_, 1:],
            values=convex_hull_data[self.selected_idx_, 0],
        )

        return self

    def _check_X_y(self, X, y):
        return check_X_y(X, y, ensure_min_features=2, multi_output=False)

    def _check_is_fitted(self, X):
        check_is_fitted(
            self,
            [
                "high_dim_idx_",
                "interpolator_high_dim_",
                "interpolator_y_",
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
        dch_distance : numpy.array of shape (n_samples, len(high_dim_idx_))
            The distance (residuals)  of samples to the convex hull in
            the higher-dimensional space.
        """
        X, y = self._check_X_y(X, y)
        self._check_is_fitted(X)

        # features used for the convex hull construction
        low_dim_feats = X[:, self.low_dim_idx]

        # the X points projected on the convex surface
        interpolated_y = self.interpolator_y_(low_dim_feats).reshape(y.shape)

        if np.any(np.isnan(interpolated_y)):
            warnings.warn(
                "There are samples in X with a low-dimensional part that is outside of the range of the convex surface. Distance will contain nans.",
                UserWarning,
            )

        return y - interpolated_y

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
                "There are samples in X with a low-dimensional part that is outside of the range of the convex surface. Distance will contain nans.",
                UserWarning,
            )

        # determine the distance between the original high-dimensional data and
        # interpolated high-dimensional data
        dch_distance = high_dim_feats - interpolated_high_dim_feats

        return dch_distance
