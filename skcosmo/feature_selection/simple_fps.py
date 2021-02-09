import numpy as np
from sklearn.utils.validation import check_is_fitted, check_array

from ._greedy import GreedySelector


class SimpleFPS(GreedySelector):
    """Transformer that performs Greedy Feature Selection using Farthest Point Sampling.


    Parameters
    ----------

    n_features_to_select : int or float, default=None
        The number of features to select. If `None`, half of the features are
        selected. If integer, the parameter is the absolute number of features
        to select. If float between 0 and 1, it is the fraction of features to
        select.

    Attributes
    ----------
    haussdorf_ : ndarray of shape (n_features,)
                 the minimum distance from each feature to the set of selected
                 features. once a feature is selected, the distance is not updated;
                 the final list will reflect the distances when selected.
    n_features_to_select_ : int
        The number of features that were selected.

    norms_ : ndarray of shape (n_features,)
        The self-covariances of each of the features

    selected_: ndarray of shape (n_features_to_select), dtype=int
               indices of the selected features

    support_ : ndarray of shape (n_features,), dtype=bool
        The mask of selected features.

    """

    def __init__(self, n_features_to_select=None):

        scoring = self.score
        self.selected_ = []

        super().__init__(scoring=scoring, n_features_to_select=n_features_to_select)

    def fit(self, X, y=None, initial=[0], haussdorfs=None):
        """Learn the features to select.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target values.
        initial : array-like, int
                  initial features to use in the selection
        haussdorfs : array-like of shape (n_features)
                     pre-computed haussdorf distances for each of the features

        Returns
        -------
        self : object
        """

        # while the super class checks this, because we are doing some pre-processing
        # before calling super().fit(), we should check as well
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
        self.support_ = np.zeros(shape=n_features, dtype=bool)
        self.support_[initial] = True
        self.selected_ = np.array(initial).flatten()

        self.norms_ = (X ** 2).sum(axis=0)

        if haussdorfs is None:
            self.haussdorf_ = (
                self.norms_
                + self.norms_[self.selected_[0]]
                - 2 * (X.T @ X[:, self.selected_[0]])
            )
            for i in self.selected_[1:]:
                new_haussdorf = self.norms_ + self.norms_[i] - 2 * (X.T @ X[:, i])
                self.haussdorf_ = np.minimum(self.haussdorf_, new_haussdorf)
        else:
            if len(haussdorfs) != n_features:
                raise ValueError(
                    "The number of pre-computed haussdorf distances"
                    "does not match the number of features."
                )
            self.haussdorf_ = haussdorfs

        super().fit(X, y, current_mask=self.support_)

    def score(self, X, y, candidate_feature_idx):

        self.haussdorf_[candidate_feature_idx] = min(
            self.get_distance(X, candidate_feature_idx, self.selected_[-1]),
            self.haussdorf_[candidate_feature_idx],
        )
        return self.haussdorf_[candidate_feature_idx]

    def get_distance(self, X, i, j):
        return self.norms_[i] + self.norms_[j] - 2 * np.dot(X[:, i], X[:, j])

    def get_select_distance(self, X):
        check_is_fitted(self)
        return np.array([self.haussdorf_[i] for i in self.selected_])
