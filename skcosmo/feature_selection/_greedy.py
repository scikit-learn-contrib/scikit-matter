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
    ):

        self.n_features_to_select = n_features_to_select
        self.scoring = scoring

    def fit(self, X, y=None, current_mask=None):
        """Learn the features to select.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target values.
        current_mask : array-like of shape (n_features)
                       an initial mask to start with, if restarting

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
            self.n_features_to_select_ = n_features // 2
        elif isinstance(self.n_features_to_select, numbers.Integral):
            if not 0 < self.n_features_to_select < n_features:
                raise ValueError(error_msg)
            self.n_features_to_select_ = self.n_features_to_select
        elif isinstance(self.n_features_to_select, numbers.Real):
            if not 0 < self.n_features_to_select <= 1:
                raise ValueError(error_msg)
            self.n_features_to_select_ = int(n_features * self.n_features_to_select)
        else:
            raise ValueError(error_msg)

        # the current mask corresponds to the set of features:
        if current_mask is None:
            current_mask = np.zeros(shape=n_features, dtype=bool)

        n_iterations = self.n_features_to_select_ - np.count_nonzero(current_mask)

        for _ in range(n_iterations):
            new_feature_idx = self._get_best_new_feature(
                self.scoring, X, y, current_mask
            )
            current_mask[new_feature_idx] = True

            self.selected_ = np.array([*self.selected_, new_feature_idx])
            self.support_ = current_mask

        return self

    def _get_best_new_feature(self, scorer, X, y, current_mask):
        # Return the best new feature to add to the current_mask, i.e. return
        # the best new feature to add (resp. remove) when doing forward
        # selection (resp. backward selection)
        candidate_feature_indices = self._get_candidate_features(current_mask)
        scores = np.zeros(X.shape[1])

        for feature_idx in candidate_feature_indices:
            scores[feature_idx] = scorer(X, y, feature_idx)

        return np.argmax(scores)

    def _get_support_mask(self):
        check_is_fitted(self, ["support_"])
        return self.support_

    def _get_candidate_features(self, current_mask):
        return np.where(current_mask is False)[0]

    def _more_tags(self):
        return {
            "requires_y": False,
        }
