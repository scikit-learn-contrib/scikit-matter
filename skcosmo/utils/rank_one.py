import numpy as np


def eta_solve(Di, Di1, psi1, psi1s, psi2, psi2s):
    a = (Di + Di1) * (1 + psi1 + psi2) - Di * Di1 * (psi1s + psi2s)
    b = Di * Di1 * (1 + psi1 + psi2)
    c = (1 + psi1 + psi2) - Di * psi1s - Di1 * psi2s
    if a > 0:
        eta = (2 * b) / (a + np.sqrt(a * a - 4 * b * c))
    else:
        eta = (a - np.sqrt(a * a - 4 * b * c)) / (2 * c)
    return eta


def dlamdvec(dref, vref, i, eps=1e-10):
    # from http://people.inf.ethz.ch/arbenz/ewp/Lnotes/2010/chapters4-5.pdf
    # d is the vector of eigenvalues of D, and v the rank-1 perturbation
    # return the approximation to the i-th eigenvalue and the
    # the n-vector [d(1..n) - lambda]
    n = len(dref)

    d = dref.copy()
    v = vref.copy()
    dvec = np.zeros(n)

    di = d[i]
    nv2 = 0.0
    v *= v
    nv2 = v.sum()

    if i < n - 1:
        di1 = d[i + 1]
        lam = (di + di1) * 0.5
    else:
        di1 = d[n - 1] + nv2
        lam = di1
    eta = 1

    psi1 = 0
    psi2 = 0
    for k in range(i + 1):
        psi1 += v[k] / (d[k] - lam)
    for k in range(i + 1, n):
        psi2 += v[k] / (d[k] - lam)

    if 1 + psi1 + psi2 > 0:  # zero is on the left half of the interval
        for k in range(n):
            d[k] -= di
        lam = lam - di
        di1 = di1 - di
        di = 0
        while abs(eta) > 10 * eps:
            psi1 = 0
            psi2 = 0
            psi1s = 0
            psi2s = 0
            for k in range(i + 1):
                idlam = 1.0 / (d[k] - lam)
                psi1 += v[k] * idlam
                psi1s += v[k] * idlam * idlam
            for k in range(i + 1, n):
                idlam = 1.0 / (d[k] - lam)
                psi2 += v[k] * idlam
                psi2s += v[k] * idlam * idlam

            # Solve for zero
            Di = -lam
            Di1 = di1 - lam
            eta = eta_solve(Di, Di1, psi1, psi1s, psi2, psi2s)
            lam += eta
    else:  # zero is on the right half of the interval
        for k in range(n):
            d[k] -= di1
        lam -= di1
        di = di - di1
        di1 = 0
        it = 0
        while abs(eta) > 10 * eps and (d[-1] - lam != 0):
            it += 1
            psi1 = 0
            psi2 = 0
            psi1s = 0
            psi2s = 0
            for k in range(i + 1):
                idlam = 1.0 / (d[k] - lam)
                psi1 += v[k] * idlam
                psi1s += v[k] * idlam * idlam
            for k in range(i + 1, n):
                idlam = 1.0 / (d[k] - lam)
                psi2 += v[k] * idlam
                psi2s += v[k] * idlam * idlam

            # Solve for zero
            Di = di - lam
            Di1 = -lam
            eta = eta_solve(Di, Di1, psi1, psi1s, psi2, psi2s)
            lam += eta

    dvec = d - lam
    return lam, dvec


def rank1_update(d, v):

    sz = len(d)
    dlam = np.zeros(sz)
    lam = np.zeros(sz)
    dvec = np.zeros((sz, sz))
    Q = np.zeros((sz, sz))
    zt = np.zeros(sz)

    for k in range(sz):
        dlam[k], dvec[:, k] = dlamdvec(d, v, k)

    for k in range(sz):
        zt[k] = dvec[k, k]
        for j in range(k):
            zt[k] *= dvec[k, j] / (d[k] - d[j])
        for j in range(k + 1, sz):
            zt[k] *= dvec[k, j] / (d[j] - d[k])
        # the implementation I base this on assumes that v[k]>0; I found empirically this makes
        # it work when v[k]<0....
        zt[k] = np.sqrt(abs(zt[k]))
        if v[k] < 0:
            zt[k] *= -1

    nQ2 = 0
    for k in range(sz):
        nQ2 = 0
        for j in range(sz):
            if abs(dvec[j, k]) > 1e-100:
                Q[k, j] = zt[j] / dvec[j, k]
            else:
                Q[k, j] = 0
            nQ2 += Q[k, j] * Q[k, j]
        for j in range(sz):
            Q[k, j] /= np.sqrt(nQ2)

    for k in range(sz):
        lam[k] = d[(k if dlam[k] > 0 else k + 1)] + dlam[k]
    Q = Q.T
    return lam, Q
