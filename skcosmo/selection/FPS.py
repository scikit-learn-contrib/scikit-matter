# -*- coding: utf-8 -*-
"""

This module contains Farthest Point Sampling (FPS) classes for sub-selecting
features or samples from given datasets. Each class supports a Principal
Covariates Regression (PCov)-inspired variant, using a mixing parameter and
target values to bias the selections.

Authors: Rose K. Cersonsky
         Michele Ceriotti

"""

from abc import abstractmethod
import numpy as np
from skcosmo.pcovr.pcovr_distances import pcovr_covariance, pcovr_kernel
from sklearn.utils import check_X_y, check_array


def _calc_distances_(K, ref_idx, idxs=None):
    """
    Calculates the distance between points in ref_idx and idx

    Assumes

    .. math::
        d(i, j) = K_{i,i} - 2 * K_{i,j} + K_{j,j}

    : param K : distance matrix, must contain distances for ref_idx and idxs
    : type K : array

    : param ref_idx : index of reference points
    : type ref_idx : int

    : param idxs : indices of points to compute distance to ref_idx
                   defaults to all indices in K
    : type idxs : list of int, None
    """
    if idxs is None:
        idxs = range(K.shape[0])
    return np.array(
        [np.real(K[j][j] - 2 * K[j][ref_idx] + K[ref_idx][ref_idx]) for j in idxs]
    )


class _BaseFPS:
    """
    Base Class defined for FPS selection methods

    :param idxs: predetermined indices; if None provided, first index selected
                 is random
    :type idxs: list of int, None

    """

    def __init__(self, tol=1e-12, idxs=None):
        if idxs is not None:
            self.idx = idxs
        else:
            self.idx = [np.random.randint(self.product.shape[0])]
        self.distances = np.min([self.calc_distance(i) for i in self.idx], axis=0)
        self.distance_selected = np.nan * np.zeros(self.product.shape[0])

    def select(self, n):
        """Method for FPS select based upon a product of the input matrices

        Parameters
        ----------
        n : number of selections to make, must be > 0

        Returns
        -------
        idx: list of n selections
        """

        if n <= 0:
            raise ValueError("You must call select(n) with n > 0.")

        if len(self.idx) > n:
            return self.idx[:n]

        # Loop over the remaining points...
        for i in range(len(self.idx) - 1, n - 1):
            for j in np.where(self.distances > 0)[0]:
                self.distances[j] = min(
                    self.distances[j], self.calc_distance(self.idx[i], [j])
                )

            self.idx.append(np.argmax(self.distances))
            self.distance_selected[i + 1] = self.distances[i + 1]

            if np.abs(self.distances).max() < self.tol:
                return self.idx
        return self.idx

    @abstractmethod
    def calc_distance(self, idx_1, idx_2=None):
        """
            Abstract method to be used for calculating the distances
            between two indexed points

        : param idx_1 : index of first point to use
        : type idx_1 : int

        : param idx_2 : index of first point to use; if None, calculates the
                        distance between idx_1 and all points
        : type idx_2 : list of int or None
        """
        pass


class SampleFPS(_BaseFPS):
    """

    For sample selection, traditional FPS employs a row-wise Euclidean
    distance, which can be expressed using the Gram matrix
    :math:`\\mathbf{K} = \\mathbf{X} \\mathbf{X}^T`

    .. math::
        \\operatorname{d}_r(i, j) = K_{ii} - 2 K_{ij} + K_{jj}.

    When mixing < 1, this will use PCov-FPS, where a modified Gram matrix is
    used to express the distances

    .. math::
        \\mathbf{\\tilde{K}} = \\alpha \\mathbf{XX}^T +
        (1 - \\alpha)\\mathbf{\\hat{Y}\\hat{Y}}^T

    :param idxs: predetermined indices; if None provided, first index selected
                 is random
    :type idxs: list of int, None

    :param X: Data matrix :math:`\\mathbf{X}` from which to select a
                   subset of the `n` rows
    :type X: array of shape (n x m)

    :param mixing: mixing parameter, as described in PCovR as
                  :math:`{\\alpha}`, defaults to 1
    :type mixing: float

    :param tol: threshold below which values will be considered 0,
                      defaults to 1E-12
    :type tol: float

    :param Y: array to include in biased selection when mixing < 1;
              required when mixing < 1, throws AssertionError otherwise
    :type Y: array of shape (n x p), optional when :math:`{\\alpha = 1}`

    """

    def __init__(self, X, mixing=1.0, tol=1e-12, Y=None, **kwargs):

        self.mixing = mixing
        self.tol = tol

        if mixing < 1:
            try:
                self.A, self.Y = check_X_y(X, Y, copy=True, multi_output=True)
            except AssertionError:
                raise Exception(r"For $\alpha < 1$, $Y$ must be supplied.")
        else:
            self.A, self.Y = check_array(X, copy=True), None

        self.product = pcovr_kernel(self.mixing, self.A, self.Y)
        super().__init__(tol=tol, **kwargs)

    def calc_distance(self, idx_1, idx_2=None):
        return _calc_distances_(self.product, idx_1, idx_2)


class FeatureFPS(_BaseFPS):
    """

    For feature selection, traditional FPS employs a column-wise Euclidean
    distance, which can be expressed using the covariance matrix
    :math:`\\mathbf{C} = \\mathbf{X} ^ T \\mathbf{X}`

    .. math::
        \\operatorname{d}_c(i, j) = C_{ii} - 2 C_{ij} + C_{jj}.

    When mixing < 1, this will use PCov-FPS, where a modified covariance matrix
    is used to express the distances

    .. math::
        \\mathbf{\\tilde{C}} = \\alpha \\mathbf{X}^T\\mathbf{X} +
        (1 - \\alpha)(\\mathbf{X}^T\\mathbf{X})^{-1/2}\\mathbf{X}^T
        \\mathbf{\\hat{Y}\\hat{Y}}^T\\mathbf{X}(\\mathbf{X}^T\\mathbf{X})^{-1/2}

    :param idxs: predetermined indices; if None provided, first index selected
                 is random
    :type idxs: list of int, None

    :param X: Data matrix :math:`\\mathbf{X}` from which to select a
                   subset of the `n` columns
    :type X: array of shape (n x m)

    :param mixing: mixing parameter, as described in PCovR as
                   :math:`{\\alpha}`, defaults to 1
    :type mixing: float

    :param tol: threshold below which values will be considered 0,
                      defaults to 1E-12
    :type tol: float

    :param Y: array to include in biased selection when mixing < 1;
              required when mixing < 1, throws AssertionError otherwise
    :type Y: array of shape (n x p), optional when :math:`{\\alpha = 1}`

    """

    def __init__(self, X, mixing=1.0, tol=1e-12, Y=None, **kwargs):

        self.mixing = mixing
        self.tol = tol

        if mixing < 1:
            try:
                self.A, self.Y = check_X_y(X, Y, copy=True, multi_output=True)
            except AssertionError:
                raise Exception(r"For $\alpha < 1$, $Y$ must be supplied.")
        else:
            self.A, self.Y = check_array(X, copy=True), None

        self.product = pcovr_covariance(self.mixing, self.A, self.Y, rcond=self.tol)
        super().__init__(tol=tol, **kwargs)

    def calc_distance(self, idx_1, idx_2=None):
        return _calc_distances_(self.product, idx_1, idx_2)
