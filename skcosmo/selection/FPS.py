import numpy as np
from skcosmo.pcovr.pcovr_distances import get_Ct, get_Kt


def _calculate_pcov_distances_(points, ref_idx, idxs=None):
    if idxs is None:
        idxs = range(points.shape[0])
    return np.array(
        [
            np.real(points[j][j] - 2 * points[j][ref_idx] + points[ref_idx][ref_idx])
            for j in idxs
        ]
    )


class _FPS:
    """
    Base Class defined for FPS selection methods

    Parameters
    ----------
    idxs : list of int, None
        predetermined indices
        if None provided, first index selected is random

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
        defines the distances matrix between features/samples
    tol : float
        corresponds to `tolerance` passed in constructor


    """

    def __init__(self, idxs=None, **kwargs):
        if idxs is not None:
            self.idx = idxs
        else:
            self.idx = [np.random.randint(self.product.shape[0])]
        self.distances = np.min(
            [_calculate_pcov_distances_(self.product, idx) for idx in self.idx], axis=0
        )
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
        for i in range(len(self.idx) - 1, n - 1):
            for j in np.where(self.distances > 0)[0]:
                self.distances[j] = min(
                    self.distances[j],
                    _calculate_pcov_distances_(self.product, self.idx[i], [j]),
                )
            self.idx.append(np.argmax(self.distances))
            self.distance_when_selected[self.idx[-1]] = self.distances[self.idx[-1]]

            if np.abs(self.distances).max() < self.tol:
                return self.idx
        return self.idx


class sampleFPS(_FPS):
    """
    Instantiation of FPS for sample selection using Euclidean Distances
    When mixing < 1, this will use PCov-FPS, where the property and
    structure matrices are used to construct a combined distance

    Parameters
    ----------
    matrix : ndarray of shape (n x m)
        Data to select from -
        Samples selection will choose a subset of the `n` rows
        stored in `self.A`
    mixing : float
        mixing parameter, as described in PCovR as `alpha`
        stored in `self.mixing`
    tolerance : float
        Threshold below which values will be considered 0
        stored in `self.tol`
    Y (optional) : ndarray of shape (n x p)
        Array to include in biased selection when mixing < 1
        Required when mixing < 1, throws AssertionError otherwise
        stored in `self.Y`

    """

    def __init__(self, X, mixing=1.0, tolerance=1e-12, Y=None, **kwargs):

        self.mixing = mixing
        self.tol = tolerance

        self.A = matrix.copy()

        if mixing < 1:
            try:
                assert Y is not None
                self.Y = Y
            except AssertionError:
                raise Exception(r"For $\alpha < 1$, $Y$ must be supplied.")
        else:
            self.Y = None

        self.product = get_Kt(self.mixing, self.A, self.Y, rcond=self.tol)
        super().__init__(tolerance=tolerance, **kwargs)


class featureFPS(_FPS):
    """
    Instantiation of FPS for feature selection using Euclidean Distances
    When mixing < 1, this will use PCov-FPS, where the property and
    structure matrices are used to construct a combined distance

    Parameters
    ----------
    matrix : ndarray of shape (n x m)
        Data to select from -
        Feature selection will choose a subset of the `m` columns
        stored in `self.A`
    mixing : float
        mixing parameter, as described in PCovR as `alpha`
        stored in `self.mixing`
    tolerance : float
        Threshold below which values will be considered 0
        stored in `self.tol`
    Y (optional) : ndarray of shape (n x p)
        Array to include in biased selection when mixing < 1
        Required when mixing < 1, throws AssertionError otherwise
        stored in `self.Y`

    """

    def __init__(self, X, mixing=1.0, tolerance=1e-12, Y=None, **kwargs):

        self.mixing = mixing
        self.tol = tolerance

        self.A = X.copy()
        if mixing < 1:
            try:
                assert Y is not None
                self.Y = Y
            except AssertionError:
                raise Exception(r"For $\alpha < 1$, $Y$ must be supplied.")
        else:
            self.Y = None

        self.product = get_Ct(self.mixing, self.A, self.Y, rcond=self.tol)
        super().__init__(tolerance=tolerance, **kwargs)
