import pyximport  # noqa
import unittest
import numpy as np
from sklearn.preprocessing import StandardScaler
from skcosmo.utils import X_orthogonalizer as x_orth

pyximport.install(reload_support=True)  # noqa
from skcosmo.utils.roupdate import rank1_update  # noqa

EPSILON = 1e-8


class TestR1(unittest.TestCase):
    def test_accuracy(self):
        X = np.random.uniform(-1, 1, size=(1000, 100))
        X = StandardScaler().fit_transform(X)

        C = X.T @ X
        vC, UC = np.linalg.eigh(C)
        j1 = np.argmax(UC[:, -1] ** 2)

        xc = X[:, j1]
        xc = xc / np.sqrt(xc @ xc)
        X1 = x_orth(X, c=j1)

        C1 = X1.T @ X1
        vC1, UC1 = np.linalg.eigh(C1)

        v = X.T @ xc
        lam, Q = rank1_update(-vC[::-1], UC[:, ::-1].T @ v)
        r1vC1 = -1.0 * np.asarray(lam)[::-1]
        r1UC1 = (UC[:, ::-1] @ Q)[:, ::-1]

        self.assertTrue(np.allclose(r1vC1, vC1))
        self.assertTrue(np.allclose(r1UC1[:, -1] ** 2, UC1[:, -1] ** 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
