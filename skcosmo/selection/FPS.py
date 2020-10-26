import numpy as np
from sklearn.pcovr_distances import get_Ct, get_Kt
from ._base import _BaseSelection

def _calculate_pcov_distances_(points, ref_idx, idxs=None):
    if idxs is None:
        idxs = range(points.shape[0])
    return np.array([np.real(points[j][j] - 2*points[j][ref_idx] + points[ref_idx][ref_idx]) for j in idxs])


class _FPS(_BaseSelection):
    """
    Super-class defined for FPS selection methods

    Parameters
    ----------
    idxs : list of int, None
        predetermined indices
        if None provided, first index selected is random
    tolerance : float
        Threshold below which values will be considered 0
        stored in `self.tol`

    Attributes
    ----------
    distances : ndarray
        at the current iteration, minimum distances between each feature
        or sample with those selected
    distance_when_selected : ndarray
        distance for each feature or sample when it was selected, nan otherwise
    idx : list
        contains the indices of the feature or sample selections made
    product : ndarray
        shape is (m x m) (feature selection) or (n x n) (sample selection)
        inner or outer product of A and Y, as defined by PCov-CUR
    tol : float
        corresponds to `tolerance` passed in constructor

    # """

    def __init__(self, idxs=None, tolerance=1E-12, **kwargs):
        if idxs is not None:
            self.idx = idxs
        else:
            self.idx = [np.random.randint(self.product.shape[0])]
        self.distances = np.min([_calculate_pcov_distances_(
            self.product, idx) for idx in self.idx], axis=0)
        self.distance_when_selected = np.nan * np.zeros(self.product.shape[0])

    def select(self, n):
        """Method for FPS select based upon a product of the input matrices

        Parameters
        ----------
        n : number of selections to make

        Returns
        -------
        idx: list of n selections
        """
        if len(self.idx) > n:
            return self.idx[:n]

        # Loop over the remaining points...
        for i in self.progress(range(len(self.idx)-1, n-1)):
            for j in np.where(self.distances > 0)[0]:
                self.distances[j] = min(self.distances[j],
                                        _calculate_pcov_distances_(self.product,
                                                                 self.idx[i], [j]))
            self.idx.append(np.argmax(self.distances))
            self.distance_when_selected[self.idx[-1]
                                        ] = self.distances[self.idx[-1]]

            if np.abs(self.distances).max() < self.tol:
                return self.idx
        return self.idx


class sampleFPS(_FPS):
    def __init__(self, matrix, mixing=1.0, tolerance=1E-12, **kwargs):

        self.product = get_Kt(mixing, self.A, self.Y)
        super().__init__(matrix=matrix, tolerance=tolerance, **kwargs)

class featureFPS(_FPS):
    def __init__(self, matrix, mixing=1.0, tolerance=1E-12, **kwargs):

        self.product = get_Ct(mixing, self.A, self.Y)
        super().__init__(matrix=matrix, tolerance=tolerance, **kwargs)
