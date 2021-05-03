import numbers
from time import time

import numpy as np
from sklearn.utils import check_random_state

from .._selection import GreedySelector


class VoronoiFPS(GreedySelector):
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

    .. image:: VoronoiFPS-Schematic.pdf

    To demonstrate the algorithm behind Voronoi FPS, let :math:`*_{m+1}` be a new chosen point,
    :math:`v(j)` was chosen earlier, :math:`j` is a point in the polyhedron with center
    :math:`v(j)`. From the inequalities of the triangle one can easily see that if
    :math:`d(v(j),j)<d(*_{m+1}, j)/2`, then point :math:`j` is guaranteed not to fall
    into the formed polyhedron and the distance to it can be uncalculated.

    This algorithm is particularly appealing when using a non-Euclidean or
    computationally-intensive distance metric, for which the
    decrease in computational time due to the reduction in distance
    calculations outweighs the increase from book-keeping. For simple
    metrics (such as Euclidean distance), VoronoiFPS will likely not
    accelerate, and may decelerate, computations when compared
    to FPS.

    Parameters
    ----------

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
        self, n_trial_calculation=4, full_fraction=None, initialize=0, **kwargs
    ):

        self.n_trial_calculation = n_trial_calculation
        self.full_fraction = full_fraction
        self.initialize = initialize
        super().__init__(selection_type="sample", **kwargs)

    def score(self, X=None, y=None):
        """
        Returns the Haussdorf distances of all samples to previous selections

        NOTE: This function does not compute the importance score each time it
        is called, in order to avoid unnecessary computations. The haussdorf
        distance is updated in :py:func:`self._update_post_selection`

        Parameters
        ----------
        X : ignored
        y : ignored

        Returns
        -------
        haussdorf : Haussdorf distances
        """
        return self.haussdorf_

    def get_distance(self):
        """

        Traditional FPS employs a column-wise Euclidean
        distance for feature selection, which can be expressed using the covariance matrix
        :math:`\\mathbf{C} = \\mathbf{X} ^ T \\mathbf{X}`

        .. math::
            \\operatorname{d}_c(i, j) = C_{ii} - 2 C_{ij} + C_{jj}.

        For sample selection, this is a row-wise Euclidean distance, which can
        be expressed in terms of the Gram matrix :math:`\\mathbf{K} = \\mathbf{X} \\mathbf{X} ^ T`

        .. math::
            \\operatorname{d}_r(i, j) = K_{ii} - 2 K_{ij} + K_{jj}.

        Returns
        -------

        haussdorf : ndarray of shape (`n_to_select_from_`)
                     the minimum distance from each point to the set of selected
                     points. once a point is selected, the distance is not updated;
                     the final list will reflect the distances when selected.

        """
        return self.haussdorf_

    def get_select_distance(self):
        """

        Returns
        -------

        haussdorf_at_select : ndarray of shape (`n_to_select`)
                     at the time of selection, the minimum distance from each
                     selected point to the set of previously selected points.

        """
        mask = self.get_support(indices=True, ordered=True)
        return self.haussdorf_at_select_[mask]

    def _init_greedy_search(self, X, y, n_to_select):
        """
        Initializes the search. Prepares an array to store the selected
        features. This function also determines the switching point if it was not
        given during initialization.
        The point is that when calculating distances in Voronoi_FPS it is
        necessary to "jump" between indexes in the array. This results in a
        significant increase in time. Therefore, if you need to recalculate a
        large number of distances, it is more advantageous to run simple
        calculation along the whole matrix.
        """

        n_to_select_from = X.shape[0]
        self.vlocation_of_idx = np.full(n_to_select_from, 1)
        # index of the voronoi cell associated with each of the columns of X

        self.dSL_ = np.zeros(self.n_to_select, float)
        # distance between new selected point and previously
        # selected points

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
                    random_state = check_random_state(self.random_state)
                    sel = random_state.randint(
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

        self.norms_ = (X ** 2).sum(axis=abs(self._axis - 1))

        if self.initialize == "random":
            random_state = check_random_state(self.random_state)
            initialize = random_state.randint(X.shape[self._axis])
        elif isinstance(self.initialize, numbers.Integral):
            initialize = self.initialize
        else:
            raise ValueError("Invalid value of the initialize parameter")

        self.selected_idx_[0] = initialize
        self.haussdorf_ = np.full(X.shape[self._axis], np.inf)
        self.haussdorf_at_select_ = np.full(X.shape[self._axis], np.inf)
        self._update_post_selection(X, y, self.selected_idx_[0])

    def _continue_greedy_search(self, X, y, n_to_select):
        """Continues the search. Prepares an array to store the selected
        features."""

        super()._continue_greedy_search(X, y, n_to_select)

        n_pad = n_to_select - self.n_selected_

        self.dSL_ = np.pad(self.dSL_, (0, n_pad), "constant", constant_values=0)

    def _get_active(self, X, last_selected):
        """
        Finds the indices of the Voronoi cells that might change due to the fact
        that a new point has been selected (index is last_selected,
        because we haven't called super() yet, self.n_selected_ has not been
        incremented yet)

        This function calculates the distances between the last selected point
        and the previously selected points. Next, the list of points to which
        the distance is to be calculated is shortened.

        These are the points for which need to recompute distances.
        Let:
        L is the last point selected;
        S are the selected points from before this iteration;
        X are the candidates;
        The logic here is that we want to check if d(XL) can be smaller than
        min(d(X,S)) (which is stored in self.haussdorf_)
        now, if a point belongs to the Voronoi cell of S then
        min(d(X,S_i))=d(X,S). Triangle inequality implies that
        d(S,L) < |d(X,S) + d(L,X)| so we just need to check if
        |d(X,S) - d(S,L)|>= d(X,S) to know that we don't need to check X.
        but |d(X,S) - d(S,L)|^2>= d(X,S)^2 if and only if d(S,L)/2 > d(S,X)
        """

        if not hasattr(self, "n_selected_") or self.n_selected_ == 0:
            return np.arange(X.shape[0], dtype=int)

        else:
            self.dSL_[: self.n_selected_] = (
                self.norms_[self.selected_idx_[: self.n_selected_]]
                + self.norms_[last_selected]
                - 2 * (self.X_selected_[: self.n_selected_] @ X[last_selected].T)
            ) * 0.25
            # calculation in a single block

            active_points = np.where(
                self.dSL_[self.vlocation_of_idx] < self.haussdorf_
            )[0]

            return active_points

    def _update_post_selection(self, X, y, last_selected):
        """
        Saves the most recently selected feature, increments the feature counter
        and update the haussdorf distances
        Let:
        L is the last point selected;
        S are the selected points from before this iteration;
        X is the one active point;
        This function calculates d(L, X) and checks the condition
        d(L, X)< min d(X, S_i). If so, we move X to a new polyhedron.
        If the number of active points is too high, it is faster to calculate
        the distances between L and all the points in the dataset.
        """

        self.haussdorf_at_select_[last_selected] = self.haussdorf_[last_selected]
        active_points = self._get_active(X, last_selected)

        if len(active_points) > 0:
            if len(active_points) / X.shape[0] > self.full_fraction:
                self.new_dist_ = (
                    self.norms_
                    + self.norms_[last_selected]
                    - 2 * X[last_selected] @ X.T
                )
            else:
                self.new_dist_ = self.haussdorf_.copy()

                self.new_dist_[active_points] = (
                    self.norms_[active_points]
                    + self.norms_[last_selected]
                    - 2 * X[last_selected] @ X[active_points].T
                )
                self.new_dist_[last_selected] = 0

            updated_points = np.where(self.new_dist_ < self.haussdorf_)[0]
            np.minimum(
                self.haussdorf_, self.new_dist_, self.haussdorf_, casting="unsafe"
            )
        else:
            updated_points = np.array([])

        if len(updated_points) > 0:
            self.vlocation_of_idx[updated_points] = self.n_selected_

        self.vlocation_of_idx[last_selected] = self.n_selected_
        super()._update_post_selection(X, y, last_selected)
