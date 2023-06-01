import unittest

import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

from skmatter.preprocessing import StandardFlexibleScaler


class ScalerTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_state = np.random.RandomState(0)

    def test_sample_weights(self):
        """Checks that sample weights of one are equal to the unweighted case.

        Also, that the nonuniform weights are different from the unweighted case"""
        X = self.random_state.uniform(0, 100, size=(3, 3))
        equal_wts = np.ones(len(X))
        nonequal_wts = self.random_state.uniform(0, 100, size=(len(X),))
        model = StandardFlexibleScaler()
        weighted_model = StandardFlexibleScaler()
        X_unweighted = model.fit_transform(X)
        X_equal_weighted = weighted_model.fit_transform(X, sample_weight=equal_wts)
        self.assertTrue((np.isclose(X_unweighted, X_equal_weighted, atol=1e-12)).all())
        X_nonequal_weighted = weighted_model.fit_transform(
            X, sample_weight=nonequal_wts
        )
        self.assertFalse(
            (np.isclose(X_unweighted, X_nonequal_weighted, atol=1e-12)).all()
        )

    def test_invalid_sample_weights(self):
        """Checks that weights must be 1D array with the same length as the number of
        samples"""
        X = self.random_state.uniform(0, 100, size=(3, 3))
        wts_len = np.ones(len(X) + 1)
        wts_dim = np.ones((len(X), 2))
        model = StandardFlexibleScaler()
        with self.assertRaises(ValueError):
            model.fit_transform(X, sample_weight=wts_len)
        with self.assertRaises(ValueError):
            model.fit_transform(X, sample_weight=wts_dim)

    def test_fit_transform_pf(self):
        """Checks that in the case of normalization by columns,
        the result is the same as in the case of using the package from sklearn
        """
        X = self.random_state.uniform(0, 100, size=(3, 3))
        model = StandardFlexibleScaler(column_wise=True)
        transformed_skmatter = model.fit_transform(X)
        transformed_sklearn = StandardScaler().fit_transform(X)
        self.assertTrue(
            (np.isclose(transformed_sklearn, transformed_skmatter, atol=1e-12)).all()
        )

    def test_fit_transform_npf(self):
        """Checks that the entire matrix is correctly normalized
        (not column-wise). Compare with the value calculated
        directly from the equation.
        """
        X = self.random_state.uniform(0, 100, size=(3, 3))
        model = StandardFlexibleScaler(column_wise=False)
        X_tr = model.fit_transform(X)
        mean = X.mean(axis=0)
        var = ((X - mean) ** 2).mean(axis=0)
        scale = np.sqrt(var.sum())
        X_ex = (X - mean) / scale
        self.assertTrue((np.isclose(X_ex, X_tr, atol=1e-12)).all())

    def test_transform(self):
        """Checks the transformation relative
        to the reference matrix.
        """
        X = self.random_state.uniform(0, 100, size=(3, 3))
        model = StandardFlexibleScaler(column_wise=True)
        model.fit(X)
        Y = self.random_state.uniform(0, 100, size=(3, 3))
        Y_tr = model.transform(Y)
        mean = X.mean(axis=0)
        var = ((X - mean) ** 2).mean(axis=0)
        scale = np.sqrt(var)
        Y_ex = (Y - mean) / scale
        self.assertTrue((np.isclose(Y_tr, Y_ex, atol=1e-12)).all())

    def test_inverse_transform(self):
        """Checks the inverse transformation with
        respect to the reference matrix.
        """
        X = self.random_state.uniform(0, 100, size=(3, 3))
        model = StandardFlexibleScaler(column_wise=True)
        model.fit(X)
        Y = self.random_state.uniform(0, 100, size=(3, 3))
        Y_tr = model.transform(Y)
        Y = np.around(Y, decimals=4)
        Y_inv = np.around((model.inverse_transform(Y_tr)), decimals=4)
        self.assertTrue((np.isclose(Y, Y_inv, atol=1e-12)).all())
        X = self.random_state.uniform(0, 100, size=(3, 3))
        model = StandardFlexibleScaler(column_wise=False)
        model.fit(X)
        Y = self.random_state.uniform(0, 100, size=(3, 3))
        Y_tr = model.transform(Y)
        Y = np.around(Y, decimals=4)
        Y_inv = np.around((model.inverse_transform(Y_tr)), decimals=4)
        self.assertTrue((np.isclose(Y, Y_inv, atol=1e-12)).all())

    def test_NotFittedError_transform(self):
        """Checks that an error is returned when
        trying to use the transform function
        before the fit function"""
        X = self.random_state.uniform(0, 100, size=(3, 3))
        model = StandardFlexibleScaler(column_wise=True)
        with self.assertRaises(sklearn.exceptions.NotFittedError):
            model.transform(X)

    def test_shape_inconsistent_transform(self):
        """Checks that an error is returned when attempting
        to use the transform function with mismatched matrix sizes."""
        X = self.random_state.uniform(0, 100, size=(3, 3))
        X_test = self.random_state.uniform(0, 100, size=(4, 4))
        model = StandardFlexibleScaler(column_wise=True)
        model.fit(X)
        with self.assertRaises(ValueError):
            model.transform(X_test)

    def test_shape_inconsistent_inverse(self):
        """Checks that an error is returned when attempting
        to use the inverse transform function with mismatched matrix sizes."""
        X = self.random_state.uniform(0, 100, size=(3, 3))
        X_test = self.random_state.uniform(0, 100, size=(4, 4))
        model = StandardFlexibleScaler(column_wise=True)
        model.fit(X)
        with self.assertRaises(ValueError):
            model.inverse_transform(X_test)

    def test_NotFittedError_inverse(self):
        """Checks that an error is returned when
        trying to use the inverse transform function
        before the fit function"""
        X = self.random_state.uniform(0, 100, size=(3, 3))
        model = StandardFlexibleScaler()
        with self.assertRaises(sklearn.exceptions.NotFittedError):
            model.inverse_transform(X)

    def test_ValueError_column_wise(self):
        """Checks that the matrix cannot be normalized
        across columns if there is a zero variation column."""
        X = self.random_state.uniform(0, 100, size=(3, 3))
        X[0][0] = X[1][0] = X[2][0] = 2
        model = StandardFlexibleScaler(column_wise=True)
        with self.assertRaises(ValueError):
            model.fit(X)

    def test_atol(self):
        """Checks that we can define absolute tolerance and it control the
        minimal variance of columns ot the whole matrix"""
        X = self.random_state.uniform(0, 100, size=(3, 3))
        atol = ((X[:, 0] - X[:, 0].mean(axis=0)) ** 2).mean(axis=0) + 1e-8
        model = StandardFlexibleScaler(column_wise=True, atol=atol, rtol=0)
        with self.assertRaises(ValueError):
            model.fit(X)
        atol = (X - X.mean(axis=0) ** 2).mean(axis=0) + 1e-8
        model = StandardFlexibleScaler(column_wise=False, atol=atol, rtol=0)
        with self.assertRaises(ValueError):
            model.fit(X)

    def test_rtol(self):
        """Checks that we can define relative tolerance and it control the
        minimal variance of columns or the whole matrix"""
        X = self.random_state.uniform(0, 100, size=(3, 3))
        mean = X[:, 0].mean(axis=0)
        rtol = ((X[:, 0] - mean) ** 2).mean(axis=0) / mean + 1e-8
        model = StandardFlexibleScaler(column_wise=True, atol=0, rtol=rtol)
        with self.assertRaises(ValueError):
            model.fit(X)
        mean = X.mean(axis=0)
        rtol = ((X - mean) ** 2).mean(axis=0) / mean + 1e-8
        model = StandardFlexibleScaler(column_wise=False, atol=0, rtol=rtol)
        with self.assertRaises(ValueError):
            model.fit(X)

    def test_ValueError_full(self):
        """Checks that the matrix cannot be normalized
        if there is a zero variation matrix."""
        X = np.array([2, 2, 2]).reshape(-1, 1)
        model = StandardFlexibleScaler(column_wise=False)
        with self.assertRaises(ValueError):
            model.fit(X)

    def test_not_w_mean(self):
        """Checks that the matrix normalized `with_mean=False`
        does not have a mean."""
        X = np.array([2, 2, 3]).reshape(-1, 1)
        model = StandardFlexibleScaler(with_mean=False)
        model.fit(X)
        self.assertTrue(np.allclose(model.mean_, 0))


if __name__ == "__main__":
    unittest.main()
