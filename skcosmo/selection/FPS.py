from abc import abstractmethod
import numpy as np
import scipy

from ._base import _BaseSelection

def calculate_pcov_distances(points, ref_idx, idxs=None):
    if idxs is None:
        idxs = range(points.shape[0])
    return np.array([np.real(points[j][j] - 2*points[j][ref_idx] + points[ref_idx][ref_idx]) for j in idxs])


class _FPS(_BaseSelection):
    """
    Super-class defined for FPS selection methods

    Parameters
    ----------
    mixing : float
        mixing parameter, as described in PCovR as `alpha`
        stored in `self.mixing`
    idxs : list of int, None
        predetermined indices
        if None provided, first index selected is random
    matrix : ndarray of shape (n x m)
        Data to select from -
        Feature selection will choose a subset of the `m` columns
        Samples selection will choose a subset of the `n` rows
        stored in `self.A`
    precompute : int
        Number of selections to precompute
    progress_bar : bool, callable
        Option to include tqdm progress bar or a callable progress bar
        implemented in `self.progress(iterable)`
    tolerance : float
        Threshold below which values will be considered 0
        stored in `self.tol`
    Y (optional) : ndarray of shape (n x p)
        Array to include in biased selection when mixing < 1
        Required when mixing < 1, throws AssertionError otherwise
        stored in `self.Y`

    Attributes
    ----------
    A : ndarray of shape (n x m)
        corresponds to `matrix` passed in constructor
    mixing : float
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
    Y_current : ndarray of shape (n x p)
        copy of `Y` which has been orthogonalized by previous selections
        only initialized when Y is specified

    # """

    def __init__(self, matrix, mixing=1.0, idxs=None, tolerance=1E-12, **kwargs):
        super().__init__(matrix=matrix, mixing=mixing, tolerance=tolerance, **kwargs)

        self.product = self.get_product()

        if idxs is not None:
            self.idx = idxs
        else:
            self.idx = [np.random.randint(self.product.shape[0])]
        self.distances = np.min([calculate_pcov_distances(
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
                                        calculate_pcov_distances(self.product,
                                                                 self.idx[i], [j]))
            self.idx.append(np.argmax(self.distances))
            self.distance_when_selected[self.idx[-1]
                                        ] = self.distances[self.idx[-1]]

            if np.abs(self.distances).max() < self.tol:
                return self.idx
        return self.idx

    @abstractmethod
    def get_product(self):
        """Abstract method for computing the product (C or K) of the input matrices
        """
        return


class sampleFPS(_FPS):
    def __init__(self, matrix, mixing=1.0, tolerance=1E-12, **kwargs):
        super().__init__(matrix=matrix, mixing=mixing, tolerance=tolerance, **kwargs)

    def get_product(self):
        """
            Creates the PCovR modified kernel distances
            ~K = (mixing) * X X^T +
                 (1-mixing) * Y Y^T

        """

        K = np.zeros((self.A.shape[0], self.A.shape[0]))
        if self.mixing < 1:
            K += (1 - self.mixing) * self.Y @ self.Y.T
        if self.mixing > 0:
            K += (self.mixing) * self.A @ self.A.T

        return K


class featureFPS(_FPS):
    def __init__(self, matrix, mixing=1.0, tolerance=1E-12, **kwargs):

        super().__init__(matrix=matrix, mixing=mixing, tolerance=tolerance, **kwargs)

    def get_product(self):
        """
            Creates the PCovR modified covariance
            ~C = (mixing) * X^T X +
                 (1-mixing) * (X^T X)^(-1/2) ~Y ~Y^T (X^T X)^(-1/2)

            where ~Y is the properties obtained by linear regression.
        """

        C = np.zeros((self.A.shape[1], self.A.shape[1]), dtype=np.float64)

        cov = self.A.T @ self.A

        if self.mixing < 1:
            # changing these next two lines can cause a LARGE error
            Cinv = np.linalg.pinv(cov)
            Cisqrt = scipy.linalg.sqrtm(Cinv)

            # parentheses speed up calculation greatly
            Y_hat = Cisqrt @ (self.A.T @ self.Y)
            Y_hat = Y_hat.reshape((C.shape[0], -1))
            Y_hat = np.real(Y_hat)

            C += (1 - self.mixing) * Y_hat @ Y_hat.T

        if self.mixing > 0:
            C += (self.mixing) * cov

        return C
