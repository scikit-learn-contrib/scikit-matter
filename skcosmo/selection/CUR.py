from abc import abstractmethod
import numpy as np
from scipy.sparse.linalg import eigs as speig

from skcosmo.pcovr.pcovr_distances import get_Ct, get_Kt
from .orthogonalizers import feature_orthogonalizer, sample_orthogonalizer


class _CUR:
    """
    Base class for CUR selection methods
    Requires a product, typically the gram or covariance matrix, \
    from which to compute the importance score

    If the model is iterative, the orthogonalize method must be overwritten
    to orthogonalize the input matrices after each iteration.

    Parameters
    ----------
    iterative: boolean
        whether to orthogonalize the matrices after each selection
    k : int
        number of eigenvectors to compute the importance score with
    tolerance : float
        Threshold below which values will be considered 0
        stored in `self.tol`

    # """

    def __init__(self, iterative=True, tolerance=1e-12, k=1, **kwargs):

        self.k = k
        self.iter = iterative
        self.tol = tolerance
        self.idx = []
        self.pi = []

    def select(self, n):
        """Method for CUR select based upon a product of the input matrices

        Parameters
        ----------
        n : number of selections to make

        Returns
        -------
        idx: list of n selections
        """

        if len(self.idx) > n:
            return self.idx[:n]

        for i in range(len(self.idx), n):
            # try:
            if self.iter:
                v, U = speig(self.product, k=self.k, tol=self.tol)
                U = U[:, np.flip(np.argsort(v))]
                pi = (np.real(U)[:, : self.k] ** 2.0).sum(axis=1)
            pi[self.idx] = 0.0
            self.idx.append(pi.argmax())
            self.pi.append(max(pi))

            self.orthogonalize()
            self.product = self.get_product()

            if np.isnan(self.product).any():
                print(
                    f"The product matrix has rank {i}. "
                    + f"n_select reduced from {n} to {i}."
                )
                return self.idx
        # except: # Left as bare except because exception is ArpackError
        #     print(f"The product matrix has rank {i}. " + \
        #           f"n_select reduced from {n} to {i}.")
        #     return self.idx

        return self.idx

    @abstractmethod
    def get_product(self):
        """Abstract method for computing the inner or outer product of the input matrices"""
        return

    @abstractmethod
    def orthogonalize(self):
        """Method to orthogonalize matrices"""
        return


class sampleCUR(_CUR):
    """
    Instantiation of CUR for sample selection using left singular vectors
    When mixing < 1, this will use PCov-CUR, where the property and
    structure matrices are used to construct augmented singular vectors

    Parameters
    ----------
    matrix : ndarray of shape (n x m)
        Data to select from -
        Samples selection will choose a subset of the `n` rows
        stored in `self.A`
    mixing : float
        mixing parameter, as described in PCovR as `alpha`
        stored in `self.mixing`
    iterative : boolean
        whether or not to compute CUR iteratively
    tolerance : float
        Threshold below which values will be considered 0
        stored in `self.tol`
    Y (optional) : ndarray of shape (n x p)
        Array to include in biased selection when mixing < 1
        Required when mixing < 1, throws AssertionError otherwise
        stored in `self.Y`

    """

    def __init__(self, X, mixing=1.0, iterative=True, tol=1e-12, Y=None, **kwargs):
        super().__init__(iterative=iterative, tol=tol, **kwargs)

        self.mixing = mixing

        self.A = matrix.copy()

        if mixing < 1:
            try:
                assert Y is not None
                self.Y = Y
            except AssertionError:
                raise Exception(r"For $\alpha < 1$, $Y$ must be supplied.")
        else:
            self.Y = None

        if not self.iter:
            self.A_current = None
            self.Y_current = None
        else:
            self.A_current = self.A.copy()
            if self.Y is not None:
                self.Y_current = self.Y.copy()
            else:
                self.Y_current = None

        self.product = self.get_product()

    def get_product(self):
        """Abstract method for computing the PCovR Gram Matrix"""
        return get_Kt(self.mixing, self.A_current, self.Y_current, self.tol)

    def orthogonalize(self):
        """Orthogonalizes the remaining samples by those already selected"""
        if self.iter:
            self.A_current, self.Y_current = sample_orthogonalizer(
                self.idx, self.A_current, self.Y_current, self.tol
            )


class featureCUR(_CUR):
    """
    Instantiation of CUR for feature selection using right singular vectors
    When mixing < 1, this will use PCov-CUR, where the property and
    structure matrices are used to construct augmented singular vectors

    Parameters
    ----------
    matrix : ndarray of shape (n x m)
        Data to select from -
        Samples selection will choose a subset of the `m` columns
        stored in `self.A`
    mixing : float
        mixing parameter, as described in PCovR as `alpha`
        stored in `self.mixing`
    iterative : boolean
        whether or not to compute CUR iteratively
    tolerance : float
        Threshold below which values will be considered 0
        stored in `self.tol`
    Y (optional) : ndarray of shape (n x p)
        Array to include in biased selection when mixing < 1
        Required when mixing < 1, throws AssertionError otherwise
        stored in `self.Y`
    """

    def __init__(self, X, mixing=1.0, iterative=True, tol=1e-12, Y=None, **kwargs):
        super().__init__(iterative=iterative, tol=tol, **kwargs)

        self.mixing = mixing

        self.A = matrix.copy()

        if mixing < 1:
            try:
                assert Y is not None
                self.Y = Y
            except AssertionError:
                raise Exception(r"For $\alpha < 1$, $Y$ must be supplied.")
        else:
            self.Y = None

        if not self.iter:
            self.A_current = None
            self.Y_current = None
        else:
            self.A_current = self.A.copy()
            if self.Y is not None:
                self.Y_current = self.Y.copy()

        self.product = self.get_product()

    def get_product(self):
        """Abstract method for computing the PCovR Covariance Matrix"""
        return get_Ct(self.mixing, self.A_current, self.Y_current, self.tol)

    def orthogonalize(self):
        """Orthogonalizes the remaining features by those already selected"""
        if self.iter:
            self.A_current, self.Y_current = feature_orthogonalizer(
                self.idx, self.A_current, self.Y_current, self.tol
            )
