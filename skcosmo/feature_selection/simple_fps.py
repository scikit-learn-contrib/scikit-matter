import numpy as np
import numbers
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

    initialize: int or 'random', default=0
        Index of the first feature to be selected. If 'random', picks a random
        value when fit starts.

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

    def __init__(self, n_features_to_select=None, initialize=0):

        scoring = self.score
        self.initialize = initialize
        self.selected_ = []

        super().__init__(scoring=scoring, n_features_to_select=n_features_to_select)

    def _get_best_new_feature(self, scorer, X, y):
        scores = scorer(X, y)

        return np.argmax(scores)

    def _init_greedy_search(self, X, y, n_to_select):

        super()._init_greedy_search(X, y, n_to_select)
        self.norms_ = (X ** 2).sum(axis=0)

        if self.initialize == "random":
            initialize = np.random.randint(X.shape[1])
        elif isinstance(self.initialize, numbers.Integral):
            initialize = self.initialize
        else:
            raise ValueError("Invalid value of the initialize parameter")

        self.selected_ = [initialize]
        self.haussdorf_ = np.full(X.shape[1], np.inf)
        self._update_post_selection(X, y, self.selected_[0])

    def score(self, X, y):
        return self.haussdorf_

    def _update_post_selection(self, X, y, last_selected):

        # distances of all points to the new point
        new_dist = (
            self.norms_ + self.norms_[last_selected] - 2 * X[:, last_selected] @ X
        )

        # update in-place the Haussdorf distance list
        np.minimum(self.haussdorf_, new_dist, self.haussdorf_)

        super()._update_post_selection(X, y, last_selected)

    def _get_distance(self, X, i, j):
        return self.norms_[i] + self.norms_[j] - 2 * np.dot(X[:, i], X[:, j])

    def get_select_distance(self, X):
        check_is_fitted(self)
        return np.array([self.haussdorf_[i] for i in self.selected_])
<<<<<<< HEAD
=======


from .fps import _c_fps_update


class CSimpleFPS(SimpleFPS):
    def _update_post_selection(self, X, y, last_selected):
        _c_fps_update(X, last_selected, self.haussdorf_, self.norms_)
        GreedySelector._update_post_selection(self, X, y, last_selected)
>>>>>>> fa1e522... black
