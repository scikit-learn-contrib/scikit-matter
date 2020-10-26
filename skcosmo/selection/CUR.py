from abc import abstractmethod
import numpy as np
from scipy.sparse.linalg import eigs as speig

from sklearn.pcovr_distances import get_Ct, get_Kt
from .orthogonalizers import feature_orthogonalizer, sample_orthogonalizer
from ._base import _BaseSelection

class _CUR(_BaseSelection):
    """
    Base class for CUR selection methods

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

    def __init__(self, iterative=True,
                 tolerance=1E-12, k=1, **kwargs):

        self.k = k
        self.product = self.get_product()
        self.iter = iterative
        self.tol = tolerance

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

        for i in self.progress(range(len(self.idx), n)):
            try:
                if self.iter:
                    v, U = speig(self.product, k=self.k, tol=self.tol)
                    U = U[:, np.flip(np.argsort(v))]
                    pi = (np.real(U)[:, :self.k]**2.0).sum(axis=1)
                pi[self.idx] = 0.0
                self.idx.append(pi.argmax())

                self.orthogonalize()
                self.product = self.get_product()

                if np.isnan(self.product).any():
                    print(f"The product matrix has rank {i}. " + \
                          f"n_select reduced from {n} to {i}.")
                    return self.idx
            except: # Left as bare except because exception is ArpackError
                print(f"The product matrix has rank {i}. " + \
                      f"n_select reduced from {n} to {i}.")
                return self.idx

            return self.idx

    @abstractmethod
    def get_product(self):
        """Abstract method for computing the product (C or K) of the input matrices
        """
        return

    @abstractmethod
    def orthogonalize(self):
        """Method to orthogonalize matrices
        """
        return

class sampleCUR(_CUR):
    def __init__(self, matrix, mixing=1.0, iterative=True, tolerance=1E-12, k=1, **kwargs):
        super().__init__(self, iterative=iterative, tolerance=tolerance, k=k, **kwargs)

        self.mixing = mixing

        self.A = matrix.copy()
        self.Y = kwargs.get("Y")

        if(not self.iter):
            self.A_current = None
            self.Y_current = None
        else:
            self.A_current = self.A.copy()
            self.Y_current = self.Y.copy()

    def get_product(self):
        get_Kt(self.mixing, self.A_current, self.Y_current)

    def orthogonalize(self):
        if(self.iter):
            self.A_current, \
            self.Y_current = sample_orthogonalizer(self.mixing,
                                                   self.A_current,
                                                   self.Y_current,
                                                   self.tol
                                                    )


class featureCUR(_CUR):
    def __init__(self, matrix, mixing=1.0, iterative=True, tolerance=1E-12, k=1, **kwargs):
        super().__init__(self, iterative=iterative, tolerance=tolerance, k=k, **kwargs)

        self.mixing = mixing

        self.A = matrix.copy()
        self.Y = kwargs.get("Y")

        if(not self.iter):
            self.A_current = None
            self.Y_current = None
        else:
            self.A_current = self.A.copy()
            self.Y_current = self.Y.copy()

    def get_product(self):
        get_Ct(self.mixing, self.A_current, self.Y_current)

    def orthogonalize(self):
        if(self.iter):
            self.A_current, \
            self.Y_current = feature_orthogonalizer(self.mixing,
                                                    self.A_current,
                                                    self.Y_current,
                                                    self.tol
                                                    )
