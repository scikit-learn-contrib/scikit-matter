import numpy as np
import scipy


def pcovr_covariance(mixing, X_proxy, Y_proxy, rcond=1e-12, return_isqrt=False):
    """
    Creates the PCovR modified covariance
    ~C = (mixing) * X^T X +
         (1-mixing) * (X^T X)^(-1/2) ~Y ~Y^T (X^T X)^(-1/2)

    where ~Y is the properties obtained by linear regression.

    :param mixing: mixing parameter,
                   as described in PCovR as :math:`{\\alpha}`, defaults to 1
    :type mixing: float

    :param X_proxy: Data matrix :math:`\\mathbf{X}`
    :type X_proxy: array of shape (n x m)

    :param Y_proxy: array to include in biased selection when mixing < 1
    :type Y_proxy: array of shape (n x p)

    :param rcond: threshold below which eigenvalues will be considered 0,
                      defaults to 1E-12
    :type rcond: float

    """

    C = np.zeros((X_proxy.shape[1], X_proxy.shape[1]), dtype=np.float64)

    cov = X_proxy.T @ X_proxy

    if mixing < 1 or return_isqrt:
        # Do not try to approximate C_inv, it will affect results
        C_inv = np.linalg.pinv(cov, rcond=rcond)
        C_isqrt = scipy.linalg.sqrtm(C_inv)

        # parentheses speed up calculation greatly
        Y_hat = C_isqrt @ (X_proxy.T @ Y_proxy)
        Y_hat = Y_hat.reshape((C.shape[0], -1))
        Y_hat = np.real(Y_hat)

        C += (1 - mixing) * Y_hat @ Y_hat.T

    if mixing > 0:
        C += (mixing) * cov

    if return_isqrt:
        return C, C_isqrt
    else:
        return C


def pcovr_kernel(mixing, X_proxy, Y_proxy):
    """
    Creates the PCovR modified kernel distances
    ~K = (mixing) * X X^T +
         (1-mixing) * Y Y^T

    :param mixing: mixing parameter,
                   as described in PCovR as :math:`{\\alpha}`, defaults to 1
    :type mixing: float

    :param X_proxy: Data matrix :math:`\\mathbf{X}`
    :type X_proxy: array of shape (n x m)

    :param Y_proxy: array to include in biased selection when mixing < 1
    :type Y_proxy: array of shape (n x p)

    """

    K = np.zeros((X_proxy.shape[0], X_proxy.shape[0]))
    if mixing < 1:
        K += (1 - mixing) * Y_proxy @ Y_proxy.T
    if mixing > 0:
        K += (mixing) * X_proxy @ X_proxy.T

    return K
