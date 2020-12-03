# -*- coding: utf-8 -*-
"""

This module contains classes for sub-selecting features or samples from given
datasets using the CUR decomposition method. Each class supports a Principal
Covariates Regression (PCov)-inspired variant, using a mixing parameter and
target values to bias the selections.

Authors: Rose K. Cersonsky
         Michele Ceriotti

"""

from abc import abstractmethod

import numpy as np
from scipy.sparse.linalg import eigs as speig

from sklearn.utils import check_X_y, check_array
from skcosmo.utils import get_progress_bar
from skcosmo.pcovr.pcovr_distances import pcovr_covariance, pcovr_kernel
from .orthogonalizers import feature_orthogonalizer, sample_orthogonalizer


class _BaseCUR:
    """
    Base class for CUR selection methods
    Requires a product, typically the gram or covariance matrix, \
    from which to compute the importance score

    If the model is iterative, the orthogonalize method must be overwritten
    to orthogonalize the input matrices after each iteration.

    :param iterative: whether to orthogonalize after each selection,
                      defaults to `true`
    :type iterative: boolean

    :param k: number of eigenvectors to compute the importance score with,
              defaults to 1
    :type k: int

    :param tol: threshold below which values will be considered 0,
                      defaults to 1E-12
    :type tol: float

    :param progress_bar: option to use `tqdm <https://tqdm.github.io/>`_
                         progress bar to monitor selections
    :type progress_bar: boolean

    """

    def __init__(self, iterative=True, tol=1e-12, k=1, progress_bar=False):

        self.k = k
        self.iter = iterative
        self.tol = tol
        self.idx = []
        self.pi = []

        if progress_bar:
            self.report_progress = get_progress_bar()
        else:
            self.report_progress = lambda x: x

    def select(self, n):
        """Method for CUR select based upon a product of the input matrices

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

        for i in self.report_progress(range(len(self.idx), n)):
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

        return self.idx

    @abstractmethod
    def get_product(self):
        """
        Abstract method for computing the inner or outer product of the
        input matrices
        """
        return

    @abstractmethod
    def orthogonalize(self):
        """Method to orthogonalize matrices"""
        return


class SampleCUR(_BaseCUR):
    """
    For sample selection, the importance score :math:`\\pi` is the sum over
    the squares of the first :math:`k` components of the left singular vectors

    .. math::

        \\pi_j =
        \\sum_i^k \\left(\\mathbf{U}_\\mathbf{\\tilde{K}}\\right)_{ij}^2.

    where :math:`{\\mathbf{\\tilde{K}} = \\alpha \\mathbf{XX}^T +
    (1 - \\alpha)\\mathbf{\\hat{Y}\\hat{Y}}^T}` for some mixing parameter
    :math:`{\\alpha}`. When :math:`{\\alpha = 1}`, this defaults to the Gram
    matrix :math:`{\\mathbf{K} = \\mathbf{X}\\mathbf{X}^T}`.

    :param iterative: whether to orthogonalize after each selection,
                      defaults to `true`
    :type iterative: boolean

    :param k: number of eigenvectors to compute the importance score with,
              defaults to 1
    :type k: int

    :param X: Data matrix :math:`\\mathbf{X}` from which to select a
                   subset of the `n` rows
    :type X: array of shape (n x m)

    :param mixing: mixing parameter,
                   as described in PCovR as :math:`{\\alpha}`, defaults to 1
    :type mixing: float

    :param progress_bar: option to use `tqdm <https://tqdm.github.io/>`_
                         progress bar to monitor selections
    :type progress_bar: boolean

    :param tol: threshold below which values will be considered 0,
                      defaults to 1E-12
    :type tol: float

    :param Y: array to include in biased selection when mixing < 1; required
              when mixing < 1, throws AssertionError otherwise
    :type Y: array of shape (n x p), optional when :math:`{\\alpha = 1}`

    """

    def __init__(self, X, mixing=1.0, Y=None, **kwargs):
        super().__init__(**kwargs)

        self.mixing = mixing

        if mixing < 1:
            try:
                self.A, self.Y = check_X_y(X, Y, copy=True, multi_output=True)
            except AssertionError:
                raise Exception(r"For $\alpha < 1$, $Y$ must be supplied.")
        else:
            self.A, self.Y = check_array(X, copy=True), None

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
        """
        Computes the modified PCovR Gram Matrix
        :math:`{\\mathbf{\\tilde{K}} = \\alpha \\mathbf{XX}^T +
        (1 - \\alpha)\\mathbf{\\hat{Y}\\hat{Y}}^T}`
        """
        return pcovr_kernel(self.mixing, self.A_current, self.Y_current)

    def orthogonalize(self):
        """
        Orthogonalizes the remaining samples by those already selected

        .. math::
            \\mathbf{X} \\leftarrow \\mathbf{X} -
            \\mathbf{X} \\left(\\frac{\\mathbf{x}_{r}^T \\mathbf{x}_{r}}
                                {\\lVert \\mathbf{x}_{r}\\rVert^2}\\right)`.

        When `mixing < 1`, this also includes orthogonalization of
        :math:`\\mathbf{Y}`

        .. math::
            \\mathbf{Y} \\leftarrow \\mathbf{Y} -
            \\mathbf{X} \\left(\\mathbf{X}_\\mathbf{r}^T
            \\mathbf{X}_\\mathbf{r}\\right)^{-1}
            \\mathbf{X}_\\mathbf{r}^T \\mathbf{Y}_\\mathbf{r}`.

        """
        if self.iter:
            self.A_current, self.Y_current = sample_orthogonalizer(
                self.idx, self.A_current, self.Y_current, self.tol
            )


class FeatureCUR(_BaseCUR):
    """
    For feature selection, the importance score :math:`\\pi` is the sum over
    the squares of the first :math:`k` components of the right singular vectors

    .. math::

        \\pi_j =
        \\sum_i^k \\left(\\mathbf{U}_\\mathbf{\\tilde{C}}\\right)_{ij}^2.

    where :math:`{\\mathbf{\\tilde{C}} = \\alpha \\mathbf{X}^T\\mathbf{X} +
    (1 - \\alpha)(\\mathbf{X}^T\\mathbf{X})^{-1/2}\\mathbf{X}^T
    \\mathbf{\\hat{Y}\\hat{Y}}^T\\mathbf{X}(\\mathbf{X}^T\\mathbf{X})^{-1/2}}`
    for some mixing parameter :math:`{\\alpha}`. When :math:`{\\alpha = 1}`,
    this defaults to the covariance matrix
    :math:`{\\mathbf{C} = \\mathbf{X}^T\\mathbf{X}}`.

    :param iterative: whether to orthogonalize after each selection,
                      defaults to `true`
    :type iterative: boolean

    :param k: number of eigenvectors to compute the importance score with,
              defaults to 1
    :type k: int

    :param X: Data matrix :math:`\\mathbf{X}` from which to select a
                   subset of the `n` columns
    :type X: array of shape (n x m)

    :param mixing: mixing parameter, as described in PCovR as
                   :math:`{\\alpha}`, defaults to 1
    :type mixing: float

    :param progress_bar: option to use `tqdm <https://tqdm.github.io/>`_
                         progress bar to monitor selections
    :type progress_bar: boolean

    :param tol: threshold below which values will be considered 0,
                      defaults to 1E-12
    :type tol: float

    :param Y: array to include in biased selection when mixing < 1;
              required when mixing < 1, throws AssertionError otherwise
    :type Y: array of shape (n x p), optional when :math:`{\\alpha = 1}`

    """

    def __init__(self, X, mixing=1.0, Y=None, **kwargs):
        super().__init__(**kwargs)

        self.mixing = mixing

        if mixing < 1:
            try:
                self.A, self.Y = check_X_y(X, Y, copy=True, multi_output=True)
            except AssertionError:
                raise Exception(r"For $\alpha < 1$, $Y$ must be supplied.")
        else:
            self.A, self.Y = check_array(X, copy=True), None

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
        """
        Computes the modified PCovR Covariance Matrix
        :math:`{\\mathbf{\\tilde{C}} = \\alpha \\mathbf{X}^T\\mathbf{X} +
        (1 - \\alpha)(\\mathbf{X}^T\\mathbf{X})^{-1/2}\\mathbf{X}^T
        \\mathbf{\\hat{Y}\\hat{Y}}^T\\mathbf{X}(\\mathbf{X}^T
        \\mathbf{X})^{-1/2}}`
        """
        return pcovr_covariance(self.mixing, self.A_current, self.Y_current, self.tol)

    def orthogonalize(self):
        """
        Orthogonalizes the remaining features by those already selected, such
        that

        :math:`{\\mathbf{X} \\leftarrow \\mathbf{X} -
        \\left(\\frac{\\mathbf{X}_{c}\\mathbf{X}_{c}^T\\cdot}
        {\\lVert\\mathbf{X}_{c}\\rVert^2}\\right)\\mathbf{X}}`.

        When `mixing < 1`, this also includes orthogonalization of
        :math:`\\mathbf{Y}`

        :math:`{\\mathbf{Y} \\leftarrow \\mathbf{Y} - \\mathbf{X}_\\mathbf{c}
        \\left(\\mathbf{X}_\\mathbf{c}^T \\mathbf{X}_\\mathbf{c}\\right)^{-1}
        \\mathbf{X}_\\mathbf{c}^T \\mathbf{Y}}`.

        """
        if self.iter:
            self.A_current, self.Y_current = feature_orthogonalizer(
                self.idx, self.A_current, self.Y_current, self.tol
            )
