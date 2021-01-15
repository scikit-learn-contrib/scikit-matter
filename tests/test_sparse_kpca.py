import unittest
import warnings

import numpy as np
from sklearn import exceptions
from sklearn.datasets import load_boston
from sklearn.utils.validation import check_is_fitted

from skcosmo.sparsified import SparseKPCA


def _linear_kernel(X, Y=None):
    if Y is None:
        Y = X.copy()
    return X @ Y.T


class SparseKPCA_tests(unittest.TestCase):
    def rel_error(self, A, B):
        return np.linalg.norm(A - B) ** 2.0 / np.linalg.norm(A) ** 2.0

    def setUp(self):
        X, y = load_boston(return_X_y=True)
        self.X = X
        self.Y = y

        self.error_tol = 1e-3
        self.rounding = -int(round(np.log10(self.error_tol)))

        self.kernels = ["linear", "poly", "rbf", "cosine", _linear_kernel]
        self.n_active = 10

    def check_kernel_types(self, kernel):
        skpca = SparseKPCA(
            n_components=2, tol=1e-12, n_active=self.n_active, kernel=kernel
        )
        _ = skpca.fit_transform(self.X)

    # Check work of the algorithm with different kernels
    def test_kernels(self):
        for i, kernel in enumerate(self.kernels):
            with self.subTest(kernel=kernel):
                try:
                    self.check_kernel_types(kernel=kernel)
                except exceptions.FitFailedWarning:
                    raise Exception(f"Kernel '{kernel}' doesn't work")

    # Checks that the model will not transform before fitting
    def test_transform_nonfitted_failure(self):
        model = SparseKPCA(n_components=2, tol=1e-12, n_active=self.n_active)
        with self.assertRaises(exceptions.NotFittedError):
            _ = model.transform(self.X)

    # Checks that the model will not inverse_transform before fitting
    def test_inverse_transform_nonfitted_failure(self):
        model = SparseKPCA(n_components=2, tol=1e-12, n_active=self.n_active)
        with self.assertRaises(exceptions.NotFittedError):
            _ = model.inverse_transform(self.X)

    # Checks that the model will not fit inverse transform when
    # _fit_inverse_transform is false
    def test_fit_inverse_transform_false_failure(self):
        model = SparseKPCA(
            n_components=2,
            tol=1e-12,
            fit_inverse_transform=False,
            n_active=self.n_active,
        )

        with self.assertRaises(exceptions.NotFittedError):
            model.fit(self.X)
            self.assertFalse(check_is_fitted(model, ["ptx_"]))

    # Checks that the model will not inverse transform when _fit_inverse_transform
    # is false
    def test_inverse_transform_false_failure(self):
        model = SparseKPCA(
            n_components=2,
            tol=1e-12,
            fit_inverse_transform=False,
            n_active=self.n_active,
        )
        with self.assertRaises(exceptions.NotFittedError) as cm:
            model.fit(self.X)
            _ = model.inverse_transform(model.transform(self.X))
            self.assertEquals(
                cm.message,
                "The fit_inverse_transform parameter was not"
                " set to True when instantiating and hence "
                "the inverse transform is not available.",
            )

    # Checks that the model will fail with a precomputed kernel of the wrong size
    def test_precomputed_kernel_shape(self):
        model = SparseKPCA(
            n_components=2, tol=1e-12, n_active=self.n_active, kernel="precomputed"
        )

        K = self.X @ self.X.T

        with self.assertRaises(ValueError) as cm:
            _ = model.fit(K)
            self.assertEqual(
                str(cm.exception.message),
                "The supplied kernel does not match n_active.",
            )

    # Checks that the model will result in the same projection with a precomputed
    # and not precomputed linear kernel
    def test_precomputed_kernel(self):
        args = dict(n_components=2, tol=0, n_active=self.n_active)

        precomputed_model = SparseKPCA(**args, kernel="precomputed")
        model = SparseKPCA(**args, kernel="linear")
        X_sparse = self.X[np.random.randint(self.X.shape[0], size=args["n_active"])]
        K_sparse = X_sparse @ X_sparse.T
        K = self.X @ X_sparse.T
        precomputed_model.fit(K, K_sparse)
        model.fit(self.X, X_sparse)
        precomputed_T = precomputed_model.transform(K)
        model_T = model.transform(self.X)

        self.assertLessEqual(
            self.rel_error(precomputed_T @ precomputed_T.T, model_T @ model_T.T),
            self.error_tol,
        )

    # Checks that the model will not fit the inverse transform in case of
    # a precomputed kernel
    def test_precomputed_no_inverse_transform_fit(self):
        args = dict(n_components=2, tol=0, n_active=self.n_active)

        with warnings.catch_warnings(record=True) as w:
            SparseKPCA(**args, kernel="precomputed", fit_inverse_transform=True)
            self.assertEqual(
                str(w[-1].message),
                "fit_inverse_transform not available for precomputed kernels.",
            )


if __name__ == "__main__":
    unittest.main()
