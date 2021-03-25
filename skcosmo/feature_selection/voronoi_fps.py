import numpy as np
from time import time
from .simple_fps import FPS
import numbers


class VoronoiFPS(FPS):
    """
    In FPS, points are selected based upon their Hausdorff distance to
    previous selections, i.e. the minimum distance between a given point and
    any previously selected points. This implicitly constructs a Voronoi
    tessellation which is updated with each new selection, as each unselected
    point "belongs" to the Voronoi polyhedron of the nearest previous selection.

    This implicit tessellation enabled a more efficient evaluation of the FPS --
    at each iteration, we need only consider for selection those points at the
    boundaries of the Voronoi polyhedra, and when updating the tessellation
    we need only consider moving those points whose Hausdorff distance is
    greater than half of the distance between the corresponding Voronoi center
    and the newly selected point, per the triangle equality.

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
        n_trial_calculation=4,
        full_fraction=None,
    ):

        self.n_trial_calculation_ = n_trial_calculation
        self.full_fraction = full_fraction

        super().__init__(
            n_features_to_select=n_features_to_select,
            progress_bar=progress_bar,
            initialize=initialize,
            tolerance=tolerance,
        )

    def _init_greedy_search(self, X, y, n_to_select):

        n_features = X.shape[1]
        """index of the voronoi cell associated with each of the columns of X"""
        self.vlocation_of_idx = np.full(n_features, -1)

        """quarter of the square distance between new selected point and previously
        selected points"""
        self.sel_d2q_ = np.zeros(self.n_features_to_select, float)
        self.new_dist_ = np.zeros(n_features)

        """ Determines the optimal switching point for full calculation.
        The point is that when calculating distances in Voronoi_FPS it is necessary
        to "jump" between indexes in the array. This results in a significant increase in time.
        Therefore, if you need to recalculate a large number of distances, it is more advantageous
        to run Simple_FPS"""
        if self.full_fraction is None:
            simple_fps_timing = -time()
            for i in range(self.n_trial_calculation_):
                dummy = X.T @ X[:, 0]
            simple_fps_timing += time()
            simple_fps_timing /= self.n_trial_calculation_

            lower_fraction = 0
            top_fraction = 1
            while top_fraction - lower_fraction > 0.01:
                voronoi_fps_times = np.zeros(self.n_trial_calculation_)
                self.full_fraction = (top_fraction + lower_fraction) / 2
                for i in range(self.n_trial_calculation_):
                    sel = np.random.randint(
                        n_features, size=int(n_features * self.full_fraction)
                    )
                    voronoi_fps_times[i] = -time()
                    dummy = X[:, 0] @ X[:, sel]
                    voronoi_fps_times[i] += time()
                voronoi_fps_timing = np.sum(voronoi_fps_times)
                voronoi_fps_timing /= self.n_trial_calculation_
                if voronoi_fps_timing < simple_fps_timing:
                    lower_fraction = self.full_fraction
                else:
                    top_fraction = self.full_fraction
                # print(f"optimal switching point = {self.full_fraction}, voronoi time = {voronoi_fps_timing},",
                #      f"simple fps timing =  {simple_fps_timing}")
            self.full_fraction = lower_fraction  # make sure we are on the "good" side
        else:
            if isinstance(self.full_fraction, numbers.Real):
                if not 0 < self.full_fraction <= 1:
                    raise ValueError(
                        "Switching point should be real and more than 0 and less than 1",
                        f"received {self.full_fraction}",
                    )
        self.stats = []

        super()._init_greedy_search(X, y, n_to_select)

    def _continue_greedy_search(self, X, y, n_to_select):
        """ Continues the search. Prepares an array to store the selected features. """

        super()._continue_greedy_search(X, y, n_to_select)

        self.sel_d2q_ = np.pad(self.sel_d2q_, (0, n_pad), "constant", constant_values=0)
        self.new_dist_ = np.zeros(X.shape[1])

    def _get_active(self, X, last_selected):
        """
        Finds the indices of the Voronoi cells that might change due to the fact
        that a new point has been selected (index is last_selected)
        """
        # because we haven't called super() yet, self.n_selected_ has not been
        # incremented yet

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

        """
        These are the points for which need to recompute distances.
        Let:
        L is the last point selected;
        S are the selected points from before this iteration;
        X are the candidates;
        The logic here is that we want to check if d(XL) can be smaller than
        min(d(XS)) (which is stored in self.haussdorf_)
        now, if a point belongs to the Voronoi cell of S then min(d(XS_i))=d(XS).
        Triangle inequality implies that d(XL)>=|d(XS) - d(SL)| so we just need to
        check if |d(XS) - d(SL)|>= d(XS) to know that we don't need to check X.
        but |d(XS) - d(SL)|^2>= d(XS)^2 if and only if d(SL)/2 > d(SX)
        """
        active_points = np.where(
            self.sel_d2q_[self.vlocation_of_idx] < self.haussdorf_
        )[0]

        return active_points

    def _calculate_distances(self, X, last_selected, **kwargs):

        # n_selected has not been incremented, so index of new voronoi is
        # n_selected

        start = time()
        if self.n_selected_ == 0:
            self.haussdorf_ = super()._calculate_distances(X, last_selected)
            # tracker of how many distances must be computed at each step
            self.number_calculated_dist = np.shape(self.haussdorf_)[0]
            updated_points = np.arange(X.shape[-1], dtype=int)
        else:
            active_points = self._get_active(X, last_selected)
            self.number_calculated_dist = self.n_selected_

            # we need to compute distances between the new point and all the points
            # in the active Voronoi cells.
            if len(active_points) > 0:
                if len(active_points) / X.shape[1] > self.full_fraction:
                    # if the number of distances we need to compute is large, it is
                    # better to switch to a full-matrix-algebra calculation.
                    self.new_dist_[:] = super()._calculate_distances(X, last_selected)
                    self.number_calculated_dist += np.shape(self.haussdorf_)[0]
                else:
                    # ... else we only iterate over the active points, although
                    # this involves more memory jumps, and can be more costly than
                    # computing all distances.
                    self.new_dist_[:] = self.haussdorf_
                    self.number_calculated_dist += np.shape(active_points)[0]

                    self.new_dist_[active_points] = (
                        self.norms_[active_points]
                        + self.norms_[last_selected]
                        - 2 * X[:, last_selected] @ X[:, active_points]
                    )
                    self.new_dist_[last_selected] = 0
                # updates haussdorf distances and keeps track of the updated points
                updated_points = np.where(self.new_dist_ < self.haussdorf_)[0]
                np.minimum(self.haussdorf_, self.new_dist_, self.haussdorf_)
            else:
                updated_points = np.array([])

        if len(updated_points) > 0:
            self.vlocation_of_idx[updated_points] = self.n_selected_

        assert self.vlocation_of_idx[last_selected] == self.n_selected_

        self.stats.append([self.number_calculated_dist, time() - start])
        return self.haussdorf_
