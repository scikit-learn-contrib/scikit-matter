import numpy as np

from .simple_fps import FPS

VORONOI_CUTOFF_FRACTION = 1.0 / 6.0


class VoronoiFPS(FPS):
    """
    Base Class defined for Voronoi FPS methods. 
    The method selects columns of a feature matrix X based on how dissimilar they are
    from each other. At each point there is a set Xsel of columns, and each of the 
    selected columns is the center of a Voronoi polyhedron. 
    The method keeps track of which of the X columns falls within the Voronoi cell
    of the columns of Xsel, and uses this information, as well as the distances between
    the Xsel, to reduce the number of distances that must be computed.

    :param initialize: predetermined index; if None provided, first index selected
                 is 0
    :type selected_: int, None
    """

    def _init_greedy_search(self, X, y, n_to_select):

        n_features = X.shape[1]

        # index of the voronoi cell associated with each of the columns of X
        self.vlocation_of_idx = np.full(n_features, -1)

        # number of points in each voronoi polyhedron (VP)
        self.number_in_voronoi = np.full(self.n_features_to_select, 0)
        self.number_in_voronoi[0] = n_features

        # furthest point in from each VP center
        self.furthest_point = np.full(self.n_features_to_select, -1, dtype=int)

        # quarter of the square distance between new selected point and previously
        # selected points
        self.sel_d2q_ = np.zeros(self.n_features_to_select, float)

        super()._init_greedy_search(X, y, n_to_select)

    def _continue_greedy_search(self, X, y, n_to_select):
        """ Continues the search. Prepares an array to store the selected features. """

        super()._continue_greedy_search(X, y, n_to_select)

        n_pad = n_to_select - self.n_selected_
        self.number_in_voronoi = np.pad(
            self.number_in_voronoi, (0, n_pad), "constant", constant_values=0
        )
        self.sel_d2q_ = np.pad(self.sel_d2q_, (0, n_pad), "constant", constant_values=0)
        self.furthest_point = np.concatenate(
            (self.furthest_point, np.zeros(n_pad, dtype=int))
        )

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

        # these are the points for which need to recompute distances. 
        # L is the last point selected
        # S are the selected points from before this iteration
        # X are the candidates
        # the logic here is that we want to check if d(XL) can be smaller than 
        # min(d(XS)) (which is stored in self.haussdorf_)
        # now, if a point belongs to the Voronoi cell of S then min(d(XS_i))=d(XS).
        # Triangle inequality implies that d(XL)>=|d(XS) - d(SL)| so we just need to
        # check if |d(XS) - d(SL)|>= d(XS) to know that we don't need to check X.
        # but |d(XS) - d(SL)|^2>= d(XS)^2 if and only if d(SL)/2 > d(SX)
        active_points = np.where(
                        self.sel_d2q_[self.vlocation_of_idx]
                        <self.haussdorf_)[0]
                    
        return active_points

    def _calculate_distances(self, X, last_selected, **kwargs):
        
        # n_selected has not been incremented, so index of new voronoi is
        # n_selected
        self.eligible_[last_selected] = False

        if self.n_selected_ == 0:
            self.haussdorf_ = super()._calculate_distances(X, last_selected)
            # tracker of how many distances must be computed at each step                        
            self.number_calculated_dist = np.shape(self.haussdorf_)[0]
            updated_points = np.arange(X.shape[-1], dtype=int)
            old_voronoi_loc = []
        else:
            active_points = self._get_active(X, last_selected)
            self.number_calculated_dist = self.n_selected_
            
            # we need to compute distances between the new point and all the points
            # in the active Voronoi cells.
            if len(active_points) > 0:
                if (
                    len(active_points) / X.shape[1]
                    > VORONOI_CUTOFF_FRACTION
                ):
                    # if the number of distances we need to compute is large, it is
                    # better to switch to a full-matrix-algebra calculation.
                    new_dist = super()._calculate_distances(X, last_selected)
                    updated_points = np.where(new_dist < self.haussdorf_)[0]
                    self.number_calculated_dist += np.shape(self.haussdorf_)[0]
                    np.minimum(self.haussdorf_, new_dist, self.haussdorf_)
                else:
                    # ... else we only iterate over the active points, although
                    # this involves more memory jumps, and can be more costly than
                    # computing all distances.
                    new_dist = self.haussdorf_.copy()
                    self.number_calculated_dist += np.shape(active_points)[0]
                    
                    new_dist[active_points] = (
                        self.norms_[active_points]
                        + self.norms_[last_selected]
                        - 2 * X[:, last_selected].T @ X[:, active_points]
                    )                    
                    
                    # updates haussdorf distances and keeps track of the updated points
                    new_dist[last_selected] = 0
                    updated_points = np.where(new_dist < self.haussdorf_)[0]
                    np.minimum(self.haussdorf_, new_dist, self.haussdorf_)

                old_voronoi_loc = list(set(self.vlocation_of_idx[updated_points]))
            else:
                updated_points = np.array([])
                old_voronoi_loc = []

        self.vlocation_of_idx[last_selected] = self.n_selected_
        if len(updated_points) > 0:
            self.vlocation_of_idx[updated_points] = self.n_selected_
        for v in old_voronoi_loc:
            saved_points = np.where(v == self.vlocation_of_idx)[0]
            self.number_in_voronoi[v] = np.shape(saved_points)[0]
            furthest_point = np.argmax(self.haussdorf_[saved_points])
            self.furthest_point[v] = saved_points[furthest_point]
            
        self.number_in_voronoi[self.n_selected_] = np.shape(updated_points)[0]
        furthest_point = np.argmax(self.haussdorf_[updated_points])
        self.furthest_point[self.n_selected_] = updated_points[furthest_point]
        self.eligible_[:] = False
        self.eligible_[self.furthest_point[: self.n_selected_ + 1]] = True
        self.eligible_[self.selected_idx_[: self.n_selected_]] = False
        assert self.vlocation_of_idx[last_selected] == self.n_selected_
        return self.haussdorf_
