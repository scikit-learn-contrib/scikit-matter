cimport cython
import numpy as np
cimport numpy as cnp

@cython.boundscheck(False)
@cython.wraparound(False)
def _c_fps_update(double[:,:] X, int last_selected, double[:] haussdorf, double[:] norms):
    
    new_dist = np.dot(X[:, last_selected], X)
    new_dist *= -2.0
    new_dist += norms
    new_dist += norms[last_selected]
    
    cdef int ndist = haussdorf.shape[0]
    for i in range(ndist):
        haussdorf[i] = min(haussdorf[i], new_dist[i])
