cimport cython
import numpy as np
from libc.math cimport  sqrt, fabs
from libc.stdlib cimport malloc, free
from cython.parallel cimport prange, parallel, threadid

cdef double eps = 1e-10


cdef double eta_solve(double Di, double Di1, double psi1, double psi1s, double psi2, double psi2s) nogil:
    cdef double a, b, c, eta
    a = (Di + Di1)*(1 + psi1 + psi2) - Di*Di1*(psi1s + psi2s)
    b = Di*Di1*(1 + psi1 + psi2)
    c = (1 + psi1 + psi2) - Di*psi1s - Di1*psi2s
    if a > 0:
        eta = (2*b)/(a + sqrt(a*a - 4*b*c))
    else:
        eta = (a - sqrt(a*a - 4*b*c))/(2*c)
    return eta

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double dlamdvec(double [:] dref, double [:] vref, int i, double[:] dvec) nogil:
    # from http://people.inf.ethz.ch/arbenz/ewp/Lnotes/2010/chapters4-5.pdf
    # d is the vector of eigenvalues of D, and v the rank-1 perturbation
    # return the approximation to the i-th eigenvalue and the
    # the n-vector [d(1..n) - lambda]
    cdef int n = len(dref)
    
    cdef double *d = <double *> malloc(n*sizeof(double))
    cdef double *v = <double *> malloc(n*sizeof(double))
    cdef int k
    for k in range(n):
        d[k] = dref[k]
        v[k] = vref[k]

    cdef double di = d[i];        
    cdef double nv2=0.0
    cdef double lam, di1, Di, Di1, eta, psi1, psi1s, psi2, psi2s, idlam
    
    for k in range(n):
        v[k]*=v[k]
        nv2 += v[k]

    if i < n-1:
        di1 = d[i+1]
        lam = (di + di1)*0.5
    else:
        di1 = d[n-1] + nv2
        lam = di1
    eta = 1

    psi1 = 0
    psi2 = 0
    for k in range(i+1):
        psi1 += v[k]/(d[k]-lam)
    for k in range(i+1,n):
        psi2 += v[k]/(d[k]-lam)

    if 1 + psi1 + psi2 > 0: # zero is on the left half of the interval        
        for k in range(n):
            d[k] -= di
        lam = lam - di
        di1 = di1 - di
        di = 0
        while fabs(eta) > 10*eps:
            psi1 = 0; psi2 = 0; psi1s = 0; psi2s = 0
            for k in range(i+1):                
                idlam = 1.0/(d[k] - lam)
                psi1 += v[k] * idlam
                psi1s += v[k] * idlam*idlam
            for k in range(i+1,n):
                idlam = 1.0/(d[k] - lam)
                psi2 += v[k] * idlam
                psi2s += v[k] * idlam*idlam
                
            # Solve for zero
            Di = -lam
            Di1 = di1 - lam
            eta = eta_solve(Di, Di1, psi1, psi1s, psi2, psi2s)
            lam += eta
    else: # zero is on the right half of the interval
        for k in range(n):
            d[k] -= di1
        lam -=  di1
        di = di - di1
        di1 = 0
        while fabs(eta) > 10*eps and fabs(d[n-1]-lam)>0:
            psi1 = 0; psi2 = 0; psi1s = 0; psi2s = 0
            for k in range(i+1):
                idlam = 1.0/(d[k] - lam)
                psi1 += v[k] * idlam
                psi1s += v[k] * idlam*idlam
            for k in range(i+1,n):
                idlam = 1.0/(d[k] - lam)
                psi2 += v[k] * idlam
                psi2s += v[k] * idlam*idlam
                
            # Solve for zero
            Di = di - lam
            Di1 = -lam
            eta = eta_solve(Di, Di1, psi1, psi1s, psi2, psi2s)
            lam += eta;
            
    for k in range(n):
        dvec[k] = d[k] - lam
    free(d)
    free(v)
    return lam

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef rank1_update(double[:] d, double [:] v):
    
    cdef int sz = len(d)
    cdef int k
    cdef int j
    cdef double[:] dlam = np.zeros(sz)
    cdef double [:] lam = np.zeros(sz)
    cdef double[:,:] dvec = np.zeros((sz,sz))
    cdef double[:,:] Q = np.zeros((sz,sz))
    cdef double[:] zt = np.zeros(sz)
    
    with nogil, parallel():
        for k in prange(sz, schedule="static"):
            dlam[k] = dlamdvec(d, v, k, dvec[:,k])
            
    with nogil, parallel():
        for k in prange(sz, schedule="static"):
            zt[k] = dvec[k,k]
            for j in range(k):
                zt[k] *= dvec[k,j]/(d[k]-d[j])
            for j in range(k+1, sz):
                zt[k] *= dvec[k,j]/(d[j]-d[k])
            # the implementation I base this on assumes that v[k]>0; I found empirically this makes
            # it work when v[k]<0....
            zt[k] = sqrt(fabs(zt[k]))
            if v[k] < 0:
                zt[k] *= -1

    cdef double nQ2 = 0
    with nogil, parallel():                        
        for k in prange(sz, schedule="static"):
            nQ2 = 0
            for j in range(sz):                   
                if ( fabs(dvec[j,k]) >  1e-100 ):
                    Q[k,j] = zt[j]/dvec[j,k]
                else: 
                    Q[k,j] = 0
                nQ2 += Q[k,j]*Q[k,j]
            for j in range(sz):
                Q[k,j] /= sqrt(nQ2)
    
    for k in range(sz):
        lam[k] = d[(k if dlam[k]>0 else k+1)]+dlam[k]
    Q = Q.T
    return lam, Q
