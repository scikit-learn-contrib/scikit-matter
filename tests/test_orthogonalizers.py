import unittest
import numpy as np
from skcosmo.utils import (
    X_orthogonalizer,
    Y_feature_orthogonalizer,
    Y_sample_orthogonalizer,
)
from skcosmo.datasets import load_csd_1000r
from sklearn.preprocessing import StandardScaler

EPSILON = 1e-8


class TestXOrth(unittest.TestCase):
    def setUp(self):

        self.n_samples = 100
        self.n_features = 100

    def test_null_column(self):
        # checks that the column passed to the orthogonalizer
        # is empty post-orthogonalization

        n_uncorrelated = self.n_features // 2

        X_random = np.random.uniform(-1, 1, size=(self.n_samples, self.n_features))
        X_correlated = np.zeros((self.n_samples, self.n_features))
        X_correlated[:, :n_uncorrelated] = np.random.uniform(
            -1, 1, size=(self.n_samples, n_uncorrelated)
        )

        for i in range(n_uncorrelated, self.n_features):
            X_correlated[:, i] = X_correlated[
                :, i - n_uncorrelated
            ] * np.random.uniform(-1, 1)

        feat_idx = np.arange(min(self.n_samples, self.n_features, n_uncorrelated))
        np.random.shuffle(feat_idx)

        for idx in feat_idx:
            with self.subTest(type="random X"):
                X_random = X_orthogonalizer(X_random, c=idx)
                self.assertLessEqual(np.linalg.norm(X_random[:, idx]), EPSILON)

            with self.subTest(type="correlated X"):
                X_correlated = X_orthogonalizer(X_correlated, c=idx)
                self.assertLessEqual(np.linalg.norm(X_correlated[:, idx]), EPSILON)
                self.assertLessEqual(
                    np.linalg.norm(X_correlated[:, idx + n_uncorrelated]), EPSILON
                )

    def test_null_row(self):
        # checks that the row passed to the orthogonalizer
        # is empty post-orthogonalization

        n_uncorrelated = self.n_samples // 2

        X_random = np.random.uniform(-1, 1, size=(self.n_samples, self.n_features))
        X_random2 = np.random.uniform(-1, 1, size=(self.n_samples, self.n_features))
        X_correlated = np.zeros((self.n_samples, self.n_features))
        X_correlated[:n_uncorrelated] = np.random.uniform(
            -1, 1, size=(n_uncorrelated, self.n_features)
        )

        for i in range(n_uncorrelated, self.n_samples):
            X_correlated[i] = X_correlated[i - n_uncorrelated] * np.random.uniform(
                -1, 1
            )

        feat_idx = np.arange(min(self.n_samples, self.n_features, n_uncorrelated))
        np.random.shuffle(feat_idx)

        for idx in feat_idx:
            with self.subTest(type="random X"):
                X_random = X_orthogonalizer(X_random.T, c=idx).T
                self.assertLessEqual(np.linalg.norm(X_random[idx]), EPSILON)

            with self.subTest(type="random X with column"):
                X_random2 = X_orthogonalizer(X_random2.T, x2=X_random2[idx].T).T
                self.assertLessEqual(np.linalg.norm(X_random2[idx]), EPSILON)

            with self.subTest(type="correlated X"):
                X_correlated = X_orthogonalizer(X_correlated.T, c=idx).T
                self.assertLessEqual(np.linalg.norm(X_correlated[idx]), EPSILON)
                self.assertLessEqual(
                    np.linalg.norm(X_correlated[idx + n_uncorrelated]), EPSILON
                )

    def test_multiple_orthogonalizations(self):
        # checks that the matrix is empty when orthogonalized simultaneously
        # by all uncorrelated columns

        n_uncorrelated = self.n_samples // 2

        X_correlated = np.zeros((self.n_samples, self.n_features))
        X_correlated[:, :n_uncorrelated] = np.random.uniform(
            -1, 1, size=(self.n_samples, n_uncorrelated)
        )

        for i in range(n_uncorrelated, self.n_features):
            X_correlated[:, i] = X_correlated[
                :, i - n_uncorrelated
            ] * np.random.uniform(-1, 1)

        X_correlated = X_orthogonalizer(
            X_correlated, x2=X_correlated[:, :n_uncorrelated]
        )

        self.assertLessEqual(np.linalg.norm(X_correlated), EPSILON)

    def test_multicolumn(self):
        # checks that an error is raised when x2 is the wrong shape for x1
        with self.assertRaises(ValueError) as cm:
            X_orthogonalizer(
                np.random.uniform(-3, 3, size=(self.n_samples, self.n_features)),
                x2=np.random.uniform(-3, 3, size=(self.n_samples + 4, self.n_features)),
            )
            self.assertEqual(
                str(cm.message),
                f"Orthogonalization only with a matrix containing {self.n_samples} entries.",
            )

    def test_warning(self):
        # checks that a warning is raised when trying to orthogonalize by
        # an empty vector
        with self.assertWarns(Warning, msg="Column vector contains only zeros."):
            X_orthogonalizer(np.zeros((self.n_samples, self.n_features)), 0)

    def test_copy(self):
        # checks that the X_orthogonalizer works in-place when copy=True

        X_random = np.random.uniform(-1, 1, size=(self.n_samples, self.n_features))

        idx = np.random.choice(X_random.shape[-1])

        new_X = X_orthogonalizer(X_random, idx, tol=EPSILON, copy=True)
        X_orthogonalizer(X_random, idx, tol=EPSILON, copy=False)
        self.assertTrue(np.allclose(X_random, new_X))


class TestYOrths(unittest.TestCase):
    def setUp(self):
        self.X, self.y = load_csd_1000r(return_X_y=True)
        self.X = StandardScaler().fit_transform(self.X)
        self.y = StandardScaler().fit_transform(self.y)

    def test_pass_feature(self):
        # checks that the Y_feature_orthogonalizer removes all targets
        # predictable by the given set of features

        Xc = self.X[:, np.random.choice(self.X.shape[-1], 3)]
        yhat = Xc @ np.linalg.pinv(Xc.T @ Xc, rcond=EPSILON) @ Xc.T @ self.y

        new_y = Y_feature_orthogonalizer(self.y, Xc, tol=EPSILON)
        self.assertTrue(np.allclose(self.y - new_y, yhat))

    def test_copy_feature(self):
        # checks the Y_feature_orthogonalizer operates in-place when copy=False

        Xc = self.X[:, np.random.choice(self.X.shape[-1], 3)]
        new_y = Y_feature_orthogonalizer(self.y, Xc, tol=EPSILON, copy=False)
        self.assertTrue(np.allclose(self.y, new_y))

    def test_pass_sample(self):
        # checks that the Y_samples_orthogonalizer removes all targets
        # predictable by the given set of samples

        r = np.random.choice(self.X.shape[0], 3)
        Xr = self.X[r]
        yr = self.y[r]

        yhat = self.X @ np.linalg.pinv(Xr.T @ Xr, rcond=EPSILON) @ Xr.T @ yr

        new_y = Y_sample_orthogonalizer(self.y, self.X, yr, Xr, tol=EPSILON)
        self.assertTrue(np.allclose(self.y - new_y, yhat))

    def test_copy_sample(self):
        # checks the Y_sample_orthogonalizer operates in-place when copy=False

        r = np.random.choice(self.X.shape[0], 3)
        Xr = self.X[r]
        yr = self.y[r]

        new_y = Y_sample_orthogonalizer(self.y, self.X, yr, Xr, tol=EPSILON, copy=False)
        self.assertTrue(np.allclose(self.y, new_y))


if __name__ == "__main__":
    unittest.main(verbosity=2)
