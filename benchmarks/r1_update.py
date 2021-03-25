from sklearn.datasets import make_low_rank_matrix
from skcosmo.feature_selection import CUR
from skcosmo.utils import X_orthogonalizer as x_orth
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
from tqdm import tqdm
from matplotlib import pyplot as plt
import scipy as sp

import pyximport
pyximport.install(reload_support=True)  # noqa
from skcosmo.utils.roupdate import rank1_update  # noqa
#from skcosmo.utils.rank_one import rank1_update

n_samples =  4000
n_features =  2000
n_select = 5
X = make_low_rank_matrix(n_samples=n_samples, n_features=n_features, effective_rank=100)
X = StandardScaler().fit_transform(X)

X_cur = X.copy()
cur = CUR(n_features_to_select=1)   # we want exact eigenvectors....
cur_times = np.nan * np.zeros(n_select, dtype=int)
cur_idx = np.zeros(n_select, dtype=int)


if True: # hard-coded skip of this part....
    start = time.time()
    for i in tqdm(range(1, n_select+1)):
        cur.n_features_to_select=i
        cur.fit(X_cur, warm_start=(i>1))
        cur_times[i-1] = time.time() - start;
        start = time.time()

    cur_idx = cur.selected_idx_

start = time.time()

ret = sp.sparse.linalg.svds( X, k=1, return_singular_vectors='vh' )
        
print("Timing for SVD: ", time.time() - start)

start = time.time()

X_r1 = X.copy()
C = X_r1.T @ X_r1
vC, UC = np.linalg.eigh(C)

print("Timing for eigh: ", time.time() - start)


r1_times = np.nan * np.zeros(n_select, dtype=int)
r1_idx = np.zeros(n_select, dtype=int)

X_current = X_r1.copy()
# sort eigenvalues in decreasing order, and use their negative, since we need this to match the expectations of G-E implementation of r1 update
vC_current = -vC[::-1].copy()
UC_current = UC[:,::-1].copy()
# lazy list to keep track of dropped features
idx_current = list(range(n_features)) 
for i in tqdm(range(1, n_select+1)):

    print("Minimum eigenvalue separation ", min(abs(vC_current[1:]- vC_current[:-1])))
    
    j1 = np.argmax(UC_current[:, 0] ** 2)
    
    r1_times[i-1] = time.time() - start;
    start = time.time()
    r1_idx[i-1] = idx_current[j1]
    print("CHECK ", r1_idx[i-1], cur_idx[i-1])

    xc = X_current[:, j1]
    xc = xc / np.sqrt(xc @ xc)
    

    v = X_current.T @ xc
    st1 = time.time()
    lam, Q = rank1_update(vC_current, UC_current.T @ v)
    print("r1u: ", time.time() - st1)
    
    st4 = time.time()
    vC_current[:] = lam    
    UC_current = UC_current@Q
    print("eigvec update: ", time.time() - st4)        
    
    st2 = time.time()
    X_current = x_orth(X_current, c=j1)
    print("ort: ", time.time() - st2)    
    
    # drops zero eigenvalues and zero rows
    st3 = time.time()
    X_current = np.hstack([X_current[:,:j1], X_current[:,j1+1:]])
    vC_current = vC_current[:-1]
    UC_current = np.vstack([UC_current[:j1,:-1], UC_current[j1+1:,:-1]])
    idx_current.pop(j1)
    
    print("bookkeeping: ", time.time() - st3)
    print("total: ", time.time() - st1)
    
print("CUR sel: ", cur_idx)
print("CUR r1:  ", r1_idx)

plt.semilogy(cur_times, 'b-')
plt.semilogy(r1_times, 'r-')
plt.show()

print(np.linalg.norm(X[:, cur_idx] - X[:, r1_idx]))
