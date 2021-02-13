"""
Sequential feature selection
"""
import numbers

import numpy as np

from sklearn.feature_selection._base import SelectorMixin
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.utils.validation import check_is_fitted, check_array


class GreedySelector(SelectorMixin, MetaEstimatorMixin, BaseEstimator):
    """Transformer that performs Greedy Feature Selection.

    This Greedy Selector adds (forward selection) features to form a
    feature subset in a greedy fashion. At each stage, the model scores each
    feature (without an estimator) and chooses the feature with the maximum score.

    Parameters
    ----------

    n_features_to_select : int or float, default=None
        The number of features to select. If `None`, half of the features are
        selected. If integer, the parameter is the absolute number of features
        to select. If float between 0 and 1, it is the fraction of features to
        select.

    score_thresh_to_select : float, default=None
        Threshold for the score. If `None` selection will continue until the
        number or fraction given by n_features_to_select is chosen. Otherwise
        will stop when the score falls below the threshold.

    scoring : str, callable, list/tuple or dict, default=None
        A single str (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the features. It is assumed that the
        next feature to select is that which maximizes the score.

        NOTE that when using custom scorers, each scorer should return a single
        value. Metric functions returning a list/array of values can be wrapped
        into multiple scorers that return one value each.

    Attributes
    ----------
    n_features_to_select_ : int
        The number of features that were selected.

    support_ : ndarray of shape (n_features,), dtype=bool
        The mask of selected features.

    """

    def __init__(
        self,
        scoring,
        n_features_to_select=None,
        score_thresh_to_select=None,
    ):

        self.n_features_to_select = n_features_to_select
        self.n_selected_ = 0
        self.scoring = scoring

    def fit(self, X, y=None, warm_start=False):
        """Learn the features to select.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target values.
        warm_start : bool
            Whether the fit should continue after having already
            run, after increasing n_features_to_select.
            Assumes it is called with the same X and y

        Returns
        -------
        self : object
        """
        tags = self._get_tags()

        if y is not None:
            X, y = self._validate_data(
                X,
                y,
                accept_sparse="csc",
                ensure_min_features=2,
                force_all_finite=not tags.get("allow_nan", True),
                multi_output=True,
            )
        else:
            X = check_array(
                X,
                accept_sparse="csc",
                ensure_min_features=2,
                force_all_finite=not tags.get("allow_nan", True),
            )

        n_features = X.shape[1]

        error_msg = (
            "n_features_to_select must be either None, an "
            "integer in [1, n_features - 1] "
            "representing the absolute "
            "number of features, or a float in (0, 1] "
            "representing a percentage of features to "
            f"select. Got {self.n_features_to_select}"
        )

        if self.n_features_to_select is None:
            n_iterations = n_features // 2
        elif isinstance(self.n_features_to_select, numbers.Integral):
            if not 0 < self.n_features_to_select < n_features:
                raise ValueError(error_msg)
            n_iterations = self.n_features_to_select
        elif isinstance(self.n_features_to_select, numbers.Real):
            if not 0 < self.n_features_to_select <= 1:
                raise ValueError(error_msg)
            n_iterations = int(n_features * self.n_features_to_select)
        else:
            raise ValueError(error_msg)

        if warm_start:
            if self.n_selected_ == 0:
                raise ValueError(
                    "Cannot fit with warm_start=True without having been previously initialized"
                )
            self._continue_greedy_search(X, y, n_iterations)
        else:
            self.n_selected_ = 0
            self._init_greedy_search(X, y, n_iterations)

        n_iterations -= self.n_selected_

        for _ in range(n_iterations):

            new_feature_idx = self._get_best_new_feature(self.scoring, X, y)
            self._update_post_selection(X, y, new_feature_idx)
            self._postprocess()

        self.support_ = np.zeros(X.shape[1], dtype=bool)
        self.support_[self.selected_idx_] = True
        return self

    def _init_greedy_search(self, X, y, n_to_select):
        """ Initializes the search. Prepares an array to store the selected features. """

        self.X_selected_ = np.zeros((X.shape[0], n_to_select), float)
        self.selected_idx_ = np.zeros((n_to_select), int)

    def _continue_greedy_search(self, X, y, n_to_select):
        """ Continues the search. Prepares an array to store the selected features. """

        self.X_selected_ = np.pad(
            self.X_selected_,
            [(0, 0), (0, n_to_select - self.n_selected_)],
            "constant",
            constant_values=0.0,
        )
        self.selected_idx_.resize(n_to_select)

    def _get_best_new_feature(self, scorer, X, y):

        scores = scorer(X, y)

        return np.argmax(scores)

    def _update_post_selection(self, X, y, last_selected):

        self.X_selected_[:, self.n_selected_] = X[:, last_selected]
        self.selected_idx_[self.n_selected_] = last_selected
        self.n_selected_ += 1

    def _get_support_mask(self):
        check_is_fitted(self, ["support_"])
        return self.support_

    def _postprocess(self):
        pass

    def _more_tags(self):
        return {
            "requires_y": False,
        }
