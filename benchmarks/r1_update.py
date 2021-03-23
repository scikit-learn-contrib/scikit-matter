from sklearn.datasets import make_low_rank_matrix
from skcosmo.feature_selection import CUR
from skcosmo.utils import X_orthogonalizer as x_orth
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
from tqdm import tqdm
from matplotlib import pyplot as plt

import pyximport
pyximport.install(reload_support=True)  # noqa
from skcosmo.utils.roupdate import rank1_update  # noqa

n_samples = 1000
n_features = 100
n_select = 10
X = make_low_rank_matrix(n_samples=n_samples, n_features=n_features, effective_rank=100)
X = StandardScaler().fit_transform(X)

X_cur = X.copy()
cur = CUR(n_features_to_select=1)
cur_times = np.nan * np.zeros(n_select, dtype=int)
cur_idx = np.zeros(n_select, dtype=int)
start = time.time()

for i in tqdm(range(1, n_select+1)):
    cur.n_features_to_select=i
    cur.fit(X_cur, warm_start=(i>1))
    cur_times[i-1] = time.time() - start;
    start = time.time()

cur_idx = cur.selected_idx_

start = time.time()

X_r1 = X.copy()
C = X_r1.T @ X_r1
vC, UC = np.linalg.eigh(C)

r1_times = np.nan * np.zeros(n_select, dtype=int)
r1_idx = np.zeros(n_select, dtype=int)

X_current = X_r1.copy()
vC_current = vC.copy()
UC_current = UC.copy()
for i in tqdm(range(1, n_select+1)):
    j1 = np.argmax(UC_current[:, -1] ** 2)

    r1_times[i-1] = time.time() - start;
    start = time.time()
    r1_idx[i-1] = j1

    xc = X_current[:, j1]
    xc = xc / np.sqrt(xc @ xc)
    X1 = x_orth(X_current, c=j1)

    C1 = X1.T @ X1
    vC1, UC1 = np.linalg.eigh(C1)

    v = X_current.T @ xc
    lam, Q = rank1_update(-vC_current[::-1], UC_current[:, ::-1].T @ v)
    vC_current = -1.0 * np.asarray(lam)[::-1]
    UC_current = (UC_current[:, ::-1] @ Q)[:, ::-1]

    print(max(vC1), max(vC_current))

    if i > 2:
        break
plt.semilogy(cur_times)
plt.semilogy(r1_times)
plt.show()

print(np.linalg.norm(X[:, cur_idx] - X[:, r1_idx]))
