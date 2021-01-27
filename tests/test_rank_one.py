import unittest
import numpy as np
from sklearn.preprocessing import StandardScaler
from skcosmo.utils.rank_one_updates import BNS_val, GuEisenstadt
from skcosmo.utils import X_orthogonalizer as x_orth

EPSILON = 1e-8


class TestR1(unittest.TestCase):
    def setUp(self):
        X = np.random.uniform(size=(100, 500))
        X = StandardScaler().fit_transform(X)
        C = X.T @ X
        vC, UC = np.linalg.eig(C)
        self.vC = np.real(vC)
        self.UC = np.real(UC)

        i = np.random.choice(X.shape[1])

        Xp = x_orth(X.copy(), c=i)

        C_prime = Xp.T @ Xp

        real_eig, real_vec = np.linalg.eig(C_prime)

        real_vec = np.real(real_vec[:, np.argsort(-real_eig)])
        real_eig = np.real(real_eig[np.argsort(-real_eig)])

        self.c = C[:, [i]] / np.sqrt(C[i, i])

        self.i = np.argmax(vC)


class TestBNS(TestR1):
    def test_accuracy(self):
        new_eig = BNS_val(self.UC, self.vC, self.c, -1, self.i, z=None)

        print(new_eig, self.vC[self.i])

        self.assertTrue(new_eig == self.vC[self.i])


class TestGE(TestR1):
    def test_accuracy(self):
        new_eig = GuEisenstadt(self.vC.flatten(), self.c.flatten(), self.i)

        print(new_eig, self.vC[self.i])

        self.assertTrue(new_eig == self.vC[self.i])


if __name__ == "__main__":
    unittest.main(verbosity=2)
