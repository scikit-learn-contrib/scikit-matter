import numpy as np

from .simple_fps import FPS


class VoronoiFPS(FPS):
    """
    Base Class defined for Voronoi FPS methods

    :param initialize: predetermined index; if None provided, first index selected
                 is 0
    :type selected_: int, None
    """

    def __init__(self, voronoi_cutoff_fraction=1.0 / 6.0, **kwargs):
        self.voronoi_cutoff_fraction = voronoi_cutoff_fraction
        self.n_poorly_chosen = 0
        super().__init__(**kwargs)

    def _init_greedy_search(self, X, y, n_to_select):


        self.n_dist_calc1 = [0,0,0]
        n_features = X.shape[1]

        # self.voronoi_number should coincide with self.selected_idx_
        # self.furthest_point = np.full(self.n_features_to_select, np.inf)

        self.voronoi_location = np.full(n_features, 0)
        self.idx_in_voronoi = [np.array([]) for n in range(self.n_features_to_select)]
        self.idx_in_voronoi[0] = np.arange(n_features, dtype=int)
        self.number_in_voronoi = np.full(self.n_features_to_select, 0)
        self.number_in_voronoi[0] = n_features

        # define the voronoi_r2 for the selected_ point
        # this is the maximum distance from the center for each of the cells
        # self.furthest_distance = np.zeros(self.n_features_to_select, float)

        # quarter of the square distance between new selected point and previously
        # selected points
        self.sel_d2q_ = np.zeros(self.n_features_to_select, float)

        super()._init_greedy_search(X, y, n_to_select)

        # self.voronoi_r2[0] = self.haussdorf_[self.voronoi_i_far[0]]

    def _get_active(self, X, last_selected):
        # because we haven't called super() yet, self.n_selected_ has not been
        # incremented yet

        f_active = np.zeros(self.n_features_to_select, bool)

        """must compute distance of the new point to all the previous FPS. Some
            of these might have been computed already, but bookkeeping could be
            worse that recomputing (TODO: verify!)
            """
        # calculation in a single block
        self.sel_d2q_[: self.n_selected_] = (
            self.norms_[self.selected_idx_[: self.n_selected_]]
            + self.norms_[last_selected]
            - 2 * (self.X_selected_[:, : self.n_selected_].T @ X[:, last_selected])
        ) * 0.25
        self.n_dist_calc1[1] += self.n_selected_
        self.n_dist_calc += self.n_selected_
        for ic in range(self.n_selected_):
            # empty voronoi, no need to consider it
            if self.number_in_voronoi[ic] > 1:
                r2 = max(self.haussdorf_[self.idx_in_voronoi[ic]])
                if self.sel_d2q_[ic] < r2:
                    # these voronoi cells need to be updated
                    f_active[ic] = True
        f_active[self.n_selected_] = True
        return f_active

    def _move_point(self, point, new_voronoi_idx):
        old_voronoi_loc = self.voronoi_location[point]
        self.voronoi_location[point] = new_voronoi_idx

        self.number_in_voronoi[old_voronoi_loc] -= 1
        self.idx_in_voronoi[old_voronoi_loc] = np.array(np.setdiff1d(
            self.idx_in_voronoi[old_voronoi_loc], [point]
        ), dtype=int)
        self.number_in_voronoi[new_voronoi_idx] += 1

        self.idx_in_voronoi[new_voronoi_idx] = np.array(np.concatenate(
            (self.idx_in_voronoi[new_voronoi_idx], [point])
        ), dtype=int)

    def _repartition(self, X, old_voronoi_idx, new_voronoi_idx):

        points_in_voronoi = self.idx_in_voronoi[old_voronoi_idx]
        true_idx = self.selected_idx_[new_voronoi_idx]

        new_dist = (
            self.norms_[points_in_voronoi]
            + self.norms_[true_idx]
            - 2 * X[:, points_in_voronoi].T @ X[:, true_idx]
        )
        self.n_dist_calc1[0] += len(points_in_voronoi)
        self.n_dist_calc += len(points_in_voronoi)

        updated_points = np.where(new_dist < self.haussdorf_[points_in_voronoi])[0]
        same_points = np.where(new_dist >= self.haussdorf_[points_in_voronoi])[0]
        l_update = points_in_voronoi[updated_points]
        if len(l_update)==0:
            self.n_poorly_chosen += 1

        self.haussdorf_[l_update] = new_dist[updated_points]

        for p in l_update:
            self._move_point(p, new_voronoi_idx)

    def _get_dist(self, X, last_selected):
        f_active = self._get_active(X, last_selected)
        self._move_point(last_selected, self.n_selected_)
        self.haussdorf_[last_selected] = 0
        if (
            np.sum(self.number_in_voronoi[f_active]) / X.shape[1]
            > self.voronoi_cutoff_fraction
        ):
            all_dist = super()._get_dist(X, last_selected)
            self.n_dist_calc1[2] += X.shape[-1]

            updated_points = np.where(all_dist < self.haussdorf_)[0]
            for p in updated_points:
                if self.voronoi_location[p] != last_selected:
                    self._move_point(int(p), self.n_selected_)

            self.haussdorf_[updated_points] = all_dist[updated_points]
        else:
            for v in np.where(f_active)[0]:
                self._repartition(X, v, self.n_selected_)
        return self.haussdorf_

    def _continue_greedy_search(self, X, y, n_to_select):
        """ Continues the search. Prepares an array to store the selected features. """

        super()._continue_greedy_search(X, y, n_to_select)

        n_pad = n_to_select - self.n_selected_
        for n in range(n_pad):
            self.idx_in_voronoi.append(np.array([]))
        self.number_in_voronoi = np.pad(
            self.number_in_voronoi, (0, n_pad), "constant", constant_values=0
        )
        self.sel_d2q_ = np.pad(self.sel_d2q_, (0, n_pad), "constant", constant_values=0)
