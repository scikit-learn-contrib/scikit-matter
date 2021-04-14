import numbers
from time import time

import numpy as np

from .._selection import _FPS


class _VoronoiFPS(_FPS):
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

    This is particularly appealing when using a non-Euclidean or
    computationally-intensive distance metric, for which the
    decrease in computational time due to the reduction in distance
    calculations outweighs the increase from book-keeping. For simple
    metrics (such as Euclidean distance), VoronoiFPS will likely not
    accelerate, and may decelerate, computations when compared
    to FPS.

    Parameters
    ----------

    n_to_select: int or float, default=None
        The number of features to select. If `None`,
        half of the features are selected. If integer, the parameter is the
        absolute number of features to select. If float between 0 and 1, it is
        the fraction of features to select.

    initialize: int or 'random', default=0
        Index of the first feature to be selected. If 'random', picks a random
        value when fit starts.

    score_threshold: float, defaults to 1E-12
        threshold below which distances will be considered 0

    progress_bar: boolean, default=False
        option to use `tqdm <https://tqdm.github.io/>`_
        progress bar to monitor selections

    n_trial_calculation: integer, default=4
        Number of calculations used for the switching point between Voronoi FPS
        and traditional FPS (for detail look at full_fraction).

    full_fraction: float, default=None
        Proportion of calculated distances from the total number of features at
        which the switch from Voronoi FPS to FPS occurs.
        At a certain number of distances to be calculated,
        the use of Voronoi FPS becomes unreasonably expensive due to the associated
        costs connected with reading data from the memory. The switching point
        depends on many conditions, and it is determined "in situ" for optimal
        use of the algorithm. Determination is done with a few test calculations
        and memory operations.
    """

    def __init__(
        self,
        n_to_select=None,
        initialize=0,
        score_threshold=1e-12,
        progress_bar=False,
        n_trial_calculation=4,
        full_fraction=None,
    ):

        self.n_trial_calculation = n_trial_calculation
        self.full_fraction = full_fraction

        super().__init__(
            n_to_select=n_to_select,
            progress_bar=progress_bar,
            selection_type="sample",
            initialize=initialize,
            score_threshold=score_threshold,
        )

    def _init_greedy_search(self, X, y, n_to_select):
        """Initializes the search. Prepares an array to store the selected
        features. This function also determines the switching point if it was not
        given during initialization."""

        n_to_select_from = X.shape[0]
        self.vlocation_of_idx = np.full(n_to_select_from, -1)
        """index of the voronoi cell associated with each of the columns of X"""

        self.sel_d2q_ = np.zeros(self.n_to_select, float)
        """quarter of the square distance between new selected point and previously
        selected points"""
        self.new_dist_ = np.zeros(n_to_select_from)

        """ Determines the optimal switching point for full calculation.
        The point is that when calculating distances in Voronoi_FPS it is necessary
        to "jump" between indexes in the array. This results in a significant increase in time.
        Therefore, if you need to recalculate a large number of distances, it is more advantageous
        to run Simple_FPS"""
        if self.full_fraction is None:
            simple_fps_timing = -time()
            if not isinstance(self.n_trial_calculation, numbers.Integral):
                raise TypeError("Number of trial calculation should be integer")
            if self.n_trial_calculation == 0:
                raise ValueError(
                    "Number of trial calculation should be more or equal to 1"
                )
            for i in range(self.n_trial_calculation):
                _ = X @ X[[0]].T
            simple_fps_timing += time()
            simple_fps_timing /= self.n_trial_calculation

            lower_fraction = 0
            top_fraction = 1
            while top_fraction - lower_fraction > 0.01:
                voronoi_fps_times = np.zeros(self.n_trial_calculation)
                self.full_fraction = (top_fraction + lower_fraction) / 2
                for i in range(self.n_trial_calculation):
                    sel = np.random.randint(
                        n_to_select_from,
                        size=int(n_to_select_from * self.full_fraction),
                    )
                    voronoi_fps_times[i] = -time()
                    _ = X[0] @ X[sel].T
                    voronoi_fps_times[i] += time()
                voronoi_fps_timing = np.sum(voronoi_fps_times)
                voronoi_fps_timing /= self.n_trial_calculation
                if voronoi_fps_timing < simple_fps_timing:
                    lower_fraction = self.full_fraction
                else:
                    top_fraction = self.full_fraction

            self.full_fraction = lower_fraction
        else:
            if isinstance(self.full_fraction, numbers.Real):
                if not 0 < self.full_fraction <= 1:
                    raise ValueError(
                        "Switching point should be real and more than 0 and less than 1",
                        f"received {self.full_fraction}",
                    )
            else:
                raise ValueError(
                    "Switching point should be real and more than 0 and less than 1",
                    f"received {self.full_fraction}",
                )

        super()._init_greedy_search(X, y, n_to_select)

    def _continue_greedy_search(self, X, y, n_to_select):
        """ Continues the search. Prepares an array to store the selected features. """

        super()._continue_greedy_search(X, y, n_to_select)

        n_pad = n_to_select - self.n_selected_

        self.sel_d2q_ = np.pad(self.sel_d2q_, (0, n_pad), "constant", constant_values=0)
        self.new_dist_ = np.zeros(X.shape[0])

    def _get_active(self, X, last_selected):
        """
        Finds the indices of the Voronoi cells that might change due to the fact
        that a new point has been selected (index is last_selected,
        because we haven't called super() yet, self.n_selected_ has not been
        incremented yet)

        must compute distance of the new point to all the previous FPS. Some
        of these might have been computed already, but bookkeeping is
        worse that recomputing

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

        if not hasattr(self, "n_selected_") or self.n_selected_ == 0:
            return np.arange(X.shape[0], dtype=int)

        else:
            self.sel_d2q_[: self.n_selected_] = (
                self.norms_[self.selected_idx_[: self.n_selected_]]
                + self.norms_[last_selected]
                - 2 * (self.X_selected_[: self.n_selected_] @ X[last_selected].T)
            ) * 0.25

            # calculation in a single block"""
            active_points = np.where(
                self.sel_d2q_[self.vlocation_of_idx] < self.haussdorf_
            )[0]

            return active_points

    def _calculate_distances(self, X, last_selected, **kwargs):

        """n_selected has not been incremented, so index of new voronoi is
        n_selected"""

        active_points = self._get_active(X, last_selected)

        if len(active_points) > 0:
            if len(active_points) / X.shape[0] > self.full_fraction:
                self.new_dist_ = (
                    self.norms_
                    + self.norms_[last_selected]
                    - 2 * X[last_selected] @ X.T
                )
                """if the number of distances we need to compute is large,
                it is better to switch to a full-matrix-algebra calculation.
                """
            else:
                """
                ... else we only iterate over the active points, although
                this involves more memory jumps, and can be more costly than
                computing all distances.
                """
                self.new_dist_[:] = self.haussdorf_

                self.new_dist_[active_points] = (
                    self.norms_[active_points]
                    + self.norms_[last_selected]
                    - 2 * X[last_selected] @ X[active_points].T
                )
                self.new_dist_[last_selected] = 0

            updated_points = np.where(self.new_dist_ < self.haussdorf_)[0]
            """updates haussdorf distances and keeps track of the updated
            points"""
            np.minimum(
                self.haussdorf_, self.new_dist_, self.haussdorf_, casting="unsafe"
            )
        else:
            updated_points = np.array([])

        if len(updated_points) > 0:
            self.vlocation_of_idx[updated_points] = self.n_selected_

        self.vlocation_of_idx[last_selected] = self.n_selected_

        return self.haussdorf_
