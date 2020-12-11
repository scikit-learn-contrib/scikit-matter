import numpy as np
from sklearn import exceptions
import unittest
from sklearn.datasets import load_boston
from skcosmo.SparseMethods import SparseKPCA


class SparseKPCA_tests(unittest.TestCase):
    def rel_error(self, A, B):
        return np.linalg.norm(A - B) ** 2.0 / np.linalg.norm(A) ** 2.0

    def run_sparse(self, mixing=0.5, n_active=20, kernel="linear"):
        skpca = SparseKPCA(
            mixing=mixing,
            n_components=2,
            tol=1e-12,
            n_active=n_active,
            kernel=kernel,
        )
        if self.Knm is None:
            skpca.fit(self.X, self.Y)
            T = skpca.transform(self.X)
        else:
            skpca.fit(self.X, self.Y, Knm=self.Knm)
            T = skpca.transform(self.X, Knm=self.Knm)
        return T

    def setUp(self):
        X, y = load_boston(return_X_y=True)
        self.X = X
        self.Y = y
        self.Knm = None

        self.error_tol = 1e-3
        self.rounding = -int(round(np.log10(self.error_tol)))

        self.kernels = ["linear", "poly", "rbf", "cosine"]
        n_mixing = 11
        n_active = 10
        self.lr_errors = np.nan * np.zeros(n_mixing)
        self.lr_errors_active = np.nan * np.zeros(n_active)
        self.alphas = np.linspace(0, 1, n_mixing)
        self.n_active = [i for i in range(10, 21)]

    # Check work of the algorithm with different kernels
    def test_kernel(self):
        for i, kernel in enumerate(self.kernels):
            try:
                self.run_sparse(kernel=kernel)
            except:
                raise Exception(f"Kernel '{kernel}' doesn't work")

    # Checks that the model will not transform before fitting
    def test_transform_nonfitted_failure(self):
        model = SparseKPCA(mixing=0.5, n_components=2, tol=1e-12, n_active=20)
        with self.assertRaises(exceptions.NotFittedError):
            if self.Knm is not None:
                _ = model.transform(self.X, Knm=self.Knm)
            else:
                _ = model.transform(self.X)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SparseKPCA_tests)
    unittest.TextTestRunner().run(suite)
