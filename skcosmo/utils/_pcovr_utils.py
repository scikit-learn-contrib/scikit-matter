from copy import deepcopy

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.extmath import randomized_svd
from sklearn.utils.validation import check_is_fitted


def _check_coefs(fitted_regressor, X, y):
    r"""
    Checks that the coefficients of a fitted
    regression model is compatible with the shapes
    of X and y

    :param regressor: sklearn-style regressor
    :type regressor: object
    :param X: feature matrix with which to compare the
        regression coefficients
    :type X: array
    :param y: target values with which to compare the
        regression coefficients
    :type y: array

    """
    if fitted_regressor.coef_.ndim != y.ndim:
        raise ValueError(
            "The target regressor has a shape incompatible "
            "with the supplied target space"
        )
    elif fitted_regressor.coef_.ndim == 1:
        if fitted_regressor.coef_.shape[0] != X.shape[1]:
            raise ValueError(
                "The target regressor has a shape incompatible "
                "with the supplied feature space"
            )
    else:
        if fitted_regressor.coef_.shape[0] != y.shape[1]:
            raise ValueError(
                "The target regressor has a shape incompatible "
                "with the supplied target space"
            )
        elif fitted_regressor.coef_.shape[1] != X.shape[1]:
            raise ValueError(
                "The target regressor has a shape incompatible "
                "with the supplied feature space"
            )


def _check_dual_coefs(fitted_regressor, K, y):
    r"""
    Checks that the dual coefficients of a fitted
    regression model is compatible with the shapes
    of K and y

    :param regressor: sklearn-style regressor
    :type regressor: object
    :param K: kernel matrix with which to compare the
        regression coefficients
    :type K: array
    :param y: target values with which to compare the
        regression coefficients
    :type y: array

    """
    if fitted_regressor.dual_coef_.ndim != y.ndim:
        raise ValueError(
            "The target regressor has a shape incompatible "
            "with the supplied target space"
        )
    elif fitted_regressor.dual_coef_.ndim == 1:
        if fitted_regressor.dual_coef_.shape[0] != K.shape[0]:
            raise ValueError(
                "The target regressor has a shape incompatible "
                "with the supplied sample space"
            )
    else:
        if fitted_regressor.dual_coef_.shape[0] != K.shape[0]:
            raise ValueError(
                "The target regressor has a shape incompatible "
                "with the supplied sample space"
            )
        elif fitted_regressor.dual_coef_.shape[1] != y.shape[1]:
            raise ValueError(
                "The target regressor has a shape incompatible "
                "with the supplied target space"
            )


def check_lr_fit(regressor, X, y):
    r"""
    Checks that a (linear) regressor is fitted, and if not,
    fits it with the provided data

    :param regressor: sklearn-style regressor
    :type regressor: object
    :param X: feature matrix with which to fit the regressor
        if it is not already fitted
    :type X: array
    :param y: target values with which to fit the regressor
        if it is not already fitted
    :type y: array
    """
    try:
        check_is_fitted(regressor)
        fitted_regressor = deepcopy(regressor)
        _check_coefs(regressor, X, y)

    except NotFittedError:
        fitted_regressor = clone(regressor)
        fitted_regressor.fit(X, y=y)

    return fitted_regressor


def check_krr_fit(regressor, K, y):
    r"""
    Checks that a (kernel ridge) regressor is fitted, and if not,
    fits it with the provided data

    :param regressor: sklearn-style regressor
    :type regressor: object
    :param X: feature matrix with which to fit the regressor
        if it is not already fitted
    :type X: array
    :param y: target values with which to fit the regressor
        if it is not already fitted
    :type y: array
    """
    try:
        check_is_fitted(regressor)
        fitted_regressor = deepcopy(regressor)
        _check_dual_coefs(regressor, K, y)

    except NotFittedError:
        fitted_regressor = clone(regressor)

        # Use a precomputed kernel
        # to avoid re-computing K
        fitted_regressor.set_params(kernel="precomputed")
        fitted_regressor.fit(K, y=y)

    return fitted_regressor


def pcovr_covariance(
    mixing,
    X,
    Y,
    rcond=1e-12,
    return_isqrt=False,
    rank=None,
    random_state=0,
    iterated_power="auto",
):
    r"""
    Creates the PCovR modified covariance

    .. math::

        \mathbf{\tilde{C}} = \alpha \mathbf{X}^T \mathbf{X} +
        (1 - \alpha) \left(\left(\mathbf{X}^T
        \mathbf{X}\right)^{-\frac{1}{2}} \mathbf{X}^T
        \mathbf{\hat{Y}}\mathbf{\hat{Y}}^T \mathbf{X} \left(\mathbf{X}^T
        \mathbf{X}\right)^{-\frac{1}{2}}\right)

    where :math:`\mathbf{\hat{Y}}`` are the properties obtained by linear regression.

    Parameters
    ----------
    mixing : float
             mixing parameter, as described in PCovR as :math:`{\alpha}`,

    X : ndarray of shape (n x m)
        Data matrix :math:`\mathbf{X}`

    Y : ndarray of shape (n x p)
        Array to include in biased selection when mixing < 1

    rcond : float,  default=1E-12
            threshold below which eigenvalues will be considered 0,

    return_isqrt : bool, default=False
                    Whether to return the calculated inverse square root of
                    the covariance. Used when inverse square root is needed
                    and the pcovr_covariance has already been calculated

    rank : int, default=min(X.shape)
           number of eigenpairs to estimate the inverse square root with

    random_state : int, default=0
                   random seed to use for randomized svd

    """

    C = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)

    if mixing < 1 or return_isqrt:

        if rank is None:
            rank = min(X.shape)

        if rank >= min(X.shape):
            vC, UC = np.linalg.eigh(X.T @ X)

            vC = np.flip(vC)
            UC = np.flip(UC, axis=1)[:, vC > rcond]
            vC = np.sqrt(vC[vC > rcond])

        else:
            _, vC, UC = randomized_svd(
                X,
                n_components=rank,
                n_iter=iterated_power,
                flip_sign=True,
                random_state=random_state,
            )

            UC = UC.T[:, (vC ** 2) > rcond]
            vC = vC[(vC ** 2) > rcond]

        C_isqrt = UC @ np.diagflat(1.0 / vC) @ UC.T

        # parentheses speed up calculation greatly
        C_Y = C_isqrt @ (X.T @ Y)
        C_Y = C_Y.reshape((C.shape[0], -1))
        C_Y = np.real(C_Y)

        C += (1 - mixing) * C_Y @ C_Y.T

    if mixing > 0:
        C += (mixing) * (X.T @ X)

    if return_isqrt:
        return C, C_isqrt
    else:
        return C


def pcovr_kernel(mixing, X, Y, **kernel_params):
    r"""
    Creates the PCovR modified kernel distances

    .. math::

        \mathbf{\tilde{K}} = \alpha \mathbf{K} +
        (1 - \alpha) \mathbf{Y}\mathbf{Y}^T

    the default kernel is the linear kernel, such that:

    .. math::

        \mathbf{\tilde{K}} = \alpha \mathbf{X} \mathbf{X}^T +
        (1 - \alpha) \mathbf{Y}\mathbf{Y}^T

    Parameters
    ----------

    mixing : float
             mixing parameter, as described in PCovR as :math:`{\alpha}`

    X : ndarray of shape (n x m)
        Data matrix :math:`\mathbf{X}`

    Y : ndarray of shape (n x p)
        Array to include in biased selection when mixing < 1

    kernel_params : dict, optional
                    dictionary of arguments to pass to pairwise_kernels
                    if none are specified, assumes that the kernel is linear

    """

    K = np.zeros((X.shape[0], X.shape[0]))
    if mixing < 1:
        K += (1 - mixing) * Y @ Y.T
    if mixing > 0:
        if "kernel" not in kernel_params:
            K += (mixing) * X @ X.T
        elif kernel_params.get("kernel") != "precomputed":
            K += (mixing) * pairwise_kernels(X, **kernel_params)
        else:
            K += (mixing) * X

    return K
