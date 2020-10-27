import numpy as np
import scipy


def get_Ct(mixing, X_proxy, Y_proxy, rcond):
    """
    Creates the PCovR modified covariance
    ~C = (mixing) * X^T X +
         (1-mixing) * (X^T X)^(-1/2) ~Y ~Y^T (X^T X)^(-1/2)

    where ~Y is the properties obtained by linear regression.
    """

    C = np.zeros((X_proxy.shape[1], X_proxy.shape[1]), dtype=np.float64)

    cov = X_proxy.T @ X_proxy

    if mixing < 1:
        # Do not try to approximate C_inv, it will affect results
        C_inv = np.linalg.pinv(cov)
        C_isqrt = scipy.linalg.sqrtm(C_inv)

        # parentheses speed up calculation greatly
        Y_hat = C_isqrt @ (X_proxy.T @ Y_proxy)
        Y_hat = Y_hat.reshape((C.shape[0], -1))
        Y_hat = np.real(Y_hat)

        C += (1 - mixing) * Y_hat @ Y_hat.T

    if mixing > 0:
        C += (mixing) * cov

    return C


def get_Kt(mixing, X_proxy, Y_proxy, rcond):
    """
    Creates the PCovR modified kernel distances
    ~K = (mixing) * X X^T +
         (1-mixing) * Y Y^T

    """

    K = np.zeros((X_proxy.shape[0], X_proxy.shape[0]))
    if mixing < 1:
        K += (1 - mixing) * Y_proxy @ Y_proxy.T
    if mixing > 0:
        K += (mixing) * X_proxy @ X_proxy.T

    return K
