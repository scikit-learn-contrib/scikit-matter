import numpy as np
from sklearn.utils.validation import check_is_fitted, check_array

from ._greedy import GreedySelector
from sklearn.utils.validation import NotFittedError
import numbers


class VoronoiFPS(GreedySelector):
    """
    Base Class defined for Voronoi FPS methods

    :param initialize: predetermined index; if None provided, first index selected
                 is 0
    :type selected_: int, None
    """

    def __init__(
        self,
        n_features_to_select=None,
        initialize=0,
        tolerance=1e-12,
        progress_bar=False,
    ):

        scoring = self.score
        self.initialize = initialize

        super().__init__(
            scoring=scoring,
            n_features_to_select=n_features_to_select,
            progress_bar=progress_bar,
            score_thresh_to_select=tolerance,
        )

    def _init_greedy_search(self, X, y, n_to_select):

        super()._init_greedy_search(X, y, n_to_select)
        self.norms_ = (X ** 2).sum(axis=0)

        if self.initialize == "random":
            initialize = np.random.randint(X.shape[1])
        elif isinstance(self.initialize, numbers.Integral):
            initialize = self.initialize
        else:
            raise ValueError("Invalid value of the initialize parameter")

        self.selected_idx_[0] = initialize
        self.haussdorf_ = np.full(X.shape[1], np.inf)

        if self.initial_haussdorfs_ is None:
            self.haussdorf_ = (
                self.norms_
                + self.norms_[self.selected_idx_[0]]
                - 2 * (X.T @ X[:, self.selected_idx_[0]])
            )
        else:
            if len(self.initial_haussdorfs_) != n_features:
                raise ValueError(
                    "The number of pre-computed haussdorf distances"
                    "does not match the number of features."
                )
            self.haussdorf_ = self.initial_haussdorfs_
        # assignment points to Voronoi cell (initially we have 1 Voronoi cell)
        # this is the list of the index of the selected point that is the center of the
        # Voronoi cell to which each point in X belongs to
        self.voronoi_number = np.full(self.haussdorf_.shape[0], 0)
        # index of the maximum - d2 point in each voronoi cell
        # this is the index of the point which is farthest from the center in each
        # voronoi cell.
        self.voronoi_i_far = np.zeros(self.n_features_to_select, int)
        self.voronoi_i_far[0] = np.argmax(self.haussdorf_)
        # number of points in each voronoi_cell
        self.voronoi_np = np.zeros(self.n_features_to_select, int)
        self.voronoi_np[0] = X.shape[1]

        # define the voronoi_r2 for the selected_ point
        # this is the maximum distance from the center for each of the cells
        self.voronoi_r2 = np.zeros(self.n_features_to_select, float)
        self.voronoi_r2[0] = self.haussdorf_[self.voronoi_i_far[0]]

        # flag array: should we update this polyhedron or not
        self.f_active_ = np.zeros(self.n_features_to_select, bool)
        # quater of the square distance between new selected point and previously
        # selected points
        self.sel_d2q_ = np.zeros(self.n_features_to_select, float)
        # norms for the selected points
        self.nsel_ = np.zeros(self.n_features_to_select, float)
        self.nsel_[0] = self.norms_[initialize]
        self._update_post_selection(X, y, self.selected_idx_[0])

    def fit(self, X, y=None, warm_start=False, haussdorfs=None):
        """Learn the features to select.

        Parameters
        ----------
        haussdorfs : array-like of shape (n_features)
                     pre-computed haussdorf distances for each of the features

        Returns
        -------
        self : object
        """
        self.initial_haussdorfs_ = haussdorfs
        super().fit(X, y, warm_start)

    def _get_best_new_feature(self, scorer, X, y):
        c_new = super()._get_best_new_feature(scorer, X, y)
        return self.voronoi_i_far[c_new]

    def _postprocess(self, X, y, new_feature_idx):
        # the new farthest point must be one of the "farthest from its cell" points
        # so we don't need to loop over all points to find it
        i = self.n_selected_ - 1
        self.f_active_[:i] = False
        nsel = 0
        """must compute distance of the new point to all the previous FPS. Some
            of these might have been computed already, but bookkeeping could be
            worse that recomputing (TODO: verify!)
            """
        # calculation in a single block
        self.sel_d2q_[:i] = (
            self.nsel_[:i]
            + self.norms_[new_feature_idx]
            - 2 * (self.X_selected_[:, :i].T @ X[:, new_feature_idx])
        ) * 0.25
        for ic in range(i):
            # empty voronoi, no need to consider it
            if self.voronoi_np[ic] > 1 and self.sel_d2q_[ic] < self.voronoi_r2[ic]:
                # these voronoi cells need to be updated
                self.f_active_[ic] = True
                self.voronoi_r2[ic] = 0
                nsel += self.voronoi_np[ic]
        self.f_active_[i] = True
        if nsel > X.shape[1] // 6:
            # it's better to do a standard update....
            all_dist = (
                self.norms_
                + self.norms_[new_feature_idx]
                - 2 * (X.T @ X[:, new_feature_idx])
            )
            l_update = np.where(all_dist < self.haussdorf_)[0]
            self.haussdorf_[l_update] = all_dist[l_update]
            for j in l_update:
                self.voronoi_np[self.voronoi_number[j]] -= 1
            self.voronoi_number[l_update] = i
            self.voronoi_np[i] = len(l_update)

            for ic in np.where(self.f_active_)[0]:
                jc = np.where(self.voronoi_number == ic)[0]
                if len(jc) == 0:
                    continue
                self.voronoi_i_far[ic] = jc[np.argmax(self.haussdorf_[jc])]
                self.voronoi_r2[ic] = self.haussdorf_[self.voronoi_i_far[ic]]
        else:
            for j in range(self.haussdorf_.shape[0]):
                # check only "active" cells
                if self.f_active_[self.voronoi_number[j]]:
                    # check, can this point be in a new polyhedron or not
                    if self.sel_d2q_[self.voronoi_number[j]] < self.haussdorf_[j]:
                        d2_j = (
                            self.norms_[j]
                            + self.norms_[new_feature_idx]
                            - 2 * X[:, j].T @ X[:, new_feature_idx]
                        )
                        # assign a point to the new polyhedron
                        if self.haussdorf_[j] > d2_j:
                            self.haussdorf_[j] = d2_j
                            self.voronoi_np[self.voronoi_number[j]] -= 1
                            self.voronoi_number[j] = i
                            self.voronoi_np[i] += 1
                    # if this point was assigned to the new cell, we need update data for this polyhedron.
                    # vice versa, we need to update the data for the cell, because we set voronoi_r2 as zero
                    if self.haussdorf_[j] > self.voronoi_r2[self.voronoi_number[j]]:
                        self.voronoi_r2[self.voronoi_number[j]] = self.haussdorf_[j]
                        self.voronoi_i_far[self.voronoi_number[j]] = j

    def score(self, X, y):
        # choose the point on the voronoi surface
        i = self.n_selected_
        return self.voronoi_r2[:i]

    def _update_post_selection(self, X, y, last_selected):
        """
        Saves the most recently selected feature and increments the feature counter
        """
        super()._update_post_selection(X, y, last_selected)
        self.nsel_[self.n_selected_ - 1] = self.norms_[last_selected]

    def get_select_distance(self):
        if hasattr(self, "haussdorf_"):
            return self.haussdorf_[self._get_support_mask()]
        else:
            raise NotFittedError()

    def set_n_features_to_select(self, new_n):
        #function changes the size of the auxiliary arrays
        #if we want to increase the number of features.
        if isinstance(new_n, numbers.Integral) and (new_n > self.n_features_to_select):
            diff = new_n - self.n_features_to_select
            self.voronoi_i_far = np.hstack((self.voronoi_i_far, np.zeros(diff, int)))
            self.voronoi_np = np.hstack((self.voronoi_np, np.zeros(diff, int)))
            self.voronoi_r2 = np.hstack((self.voronoi_r2, np.zeros(diff, float)))
            self.f_active_ = np.hstack((self.f_active_, np.zeros(diff, bool)))
            self.sel_d2q_ = np.hstack((self.sel_d2q_, np.zeros(diff, float)))
            self.nsel_ = np.hstack((self.nsel_, np.zeros(diff, float)))
            self.n_features_to_select = new_n
        else:
            raise ValueError("Invalid value of the number of features")
