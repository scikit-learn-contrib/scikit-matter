from scipy.optimize import root_scalar
import numpy as np

EPSILON = 1e-10


def default_optimizer(fnc, interval, tol=1e-12):
    return np.real(root_scalar(fnc, bracket=interval, xtol=tol).root)


def BNS_val(Q, d, b, rho, i, z=None, optimizer=default_optimizer):
    r"""
    Computes the Bunch-Nielsen-Sorensen rank-one update to the symmetric eigenproblem

    .. math::

        B + \rho * b b^T

    where

    .. math::

        B + \rho * b b^T = Q (D + \rho z z^T) Q^T

    Parameters
    ----------
    Q - original eigenvectors of B
    d - original eigenvalues of B
    b - column vector update to B
    rho - scalar multiplier for rank-one update
    i - index of the eigenpair to return
    z - vector update decomposed along original eigenvectors
    optimizer - root finder which takes as arguments a function and an interval,
                returns the root

    """
    if z is None:
        z = np.linalg.lstsq(Q, b, rcond=None)[0]

    delta = (d - d[i]) / rho

    def wi(mu):
        return 1 + (z ** 2.0 / (delta - mu)).sum()

    interval = [(d[i] - d[i + 1]) - EPSILON, EPSILON]

    return optimizer(wi, interval=interval) * rho + d[i]


def BNS_vec(Q, d, b=None, rho=None, i=None, eig=None, z=None):
    r"""
    Computes the Bunch-Nielsen-Sorensen rank-one update to the symmetric eigenproblem (eigenvectors)

    .. math::

        B + \rho * b b^T

    where

    .. math::

        B + \rho * b b^T = Q (D + \rho z z^T) Q^T

    Parameters
    ----------
    Q - original eigenvectors of B
    d - original eigenvalues of B
    b - column vector update to B
    rho - scalar multiplier for rank-one update
    i - index of the eigenpair to return
    z - vector update decomposed along original eigenvectors

    """

    if z is None:
        z = np.linalg.lstsq(Q, b, rcond=None)[0]

    if eig is None:
        eig = BNS_val(Q, d, b, rho, i, z=z)

    Di = np.diagflat(1 / (d - eig))
    return (Q @ (Di @ z) / np.linalg.norm(Di @ z)).flatten()


def GuEisenstadt(dref, v, i):

    r"""
    .. math::

        B + \rho * v v^T


    Variables:
    D: matrix to be updated
    z: vector constituting the rank one update
    rho: constant within the rank one update

    """
    # from http://people.inf.ethz.ch/arbenz/ewp/Lnotes/2010/chapters4-5.pdf
    # d is the vector of eigenvalues of D, and v the rank-1 perturbation
    # return the approximation to the i-th eigenvalue and the
    # the n-vector [d(1..n) - lambda]
    d = dref.copy()  # we don't want to modify the array!
    n = len(d)
    di = d[i]
    v = v * v  # redefines v as the square. this is ugly AF, but will do for starters
    nv2 = v.sum()
    if i < n - 1:
        di1 = d[i + 1]
        lam = (di + di1) * 0.5
    else:
        di1 = d[n - 1] + nv2
        lam = di1
    eta = 1
    psi1 = (v[: i + 1] ** 2 / (d[: i + 1] - lam)).sum()
    psi2 = (v[i + 1 :] ** 2 / (d[i + 1 :] - lam)).sum()
    idlam = np.zeros(n)
    vi = np.zeros(n)
    vii = np.zeros(n)
    if 1 + psi1 + psi2 > 0:  # zero is on the left half of the interval
        d -= di
        lam = lam - di
        di1 = di1 - di
        di = 0
        while abs(eta) > 10 * EPSILON:
            idlam[:] = 1.0 / (d - lam)
            vi[:] = v * idlam
            vii[:] = vi * idlam
            psi1 = vi[: i + 1].sum()
            psi1s = vii[: i + 1].sum()
            psi2 = vi[i + 1 :].sum()
            psi2s = vii[i + 1 :].sum()
            # psi1 = (v[:i+1]/(d-lam[:i+1])).sum()
            # psi1s = (v[:i+1]/(d-lam[:i+1])**2).sum()
            # psi2 = (v[i+1:]/(d-lam[i+1:])).sum()
            # psi2s = (v[i+1:]/(d-lam[i+1:])**2).sum()
            # Solve for zero
            Di = -lam
            Di1 = di1 - lam
            a = (Di + Di1) * (1 + psi1 + psi2) - Di * Di1 * (psi1s + psi2s)
            b = Di * Di1 * (1 + psi1 + psi2)
            c = (1 + psi1 + psi2) - Di * psi1s - Di1 * psi2s
            if a > 0:
                eta = (2 * b) / (a + np.sqrt(a * a - 4 * b * c))
            else:
                eta = (a - np.sqrt(a * a - 4 * b * c)) / (2 * c)
            lam += eta
    else:  # zero is on the right half of the interval
        d -= di1
        lam -= di1
        di = di - di1
        di1 = 0
        while abs(eta) > 10 * EPSILON:
            idlam[:] = 1.0 / (d - lam)
            vi[:] = v * idlam
            vii[:] = vi * idlam
            psi1 = vi[: i + 1].sum()
            psi1s = vii[: i + 1].sum()
            psi2 = vi[i + 1 :].sum()
            psi2s = vii[i + 1 :].sum()
            # Solve for zero
            Di = di - lam
            Di1 = -lam
            a = (Di + Di1) * (1 + psi1 + psi2) - Di * Di1 * (psi1s + psi2s)
            b = Di * Di1 * (1 + psi1 + psi2)
            c = (1 + psi1 + psi2) - Di * psi1s - Di1 * psi2s
            if a > 0:
                eta = (2 * b) / (a + np.sqrt(a * a - 4 * b * c))
            else:
                eta = (a - np.sqrt(a * a - 4 * b * c)) / (2 * c)
            lam += eta
    return lam
