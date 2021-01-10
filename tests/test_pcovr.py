import unittest
from skcosmo.pcovr import PCovR
from sklearn.datasets import load_boston
import numpy as np
from sklearn import exceptions
from sklearn.utils.validation import check_X_y


def rel_error(A, B):
    return np.linalg.norm(A - B) ** 2.0 / np.linalg.norm(A) ** 2.0


class PCovRBaseTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = lambda mixing, **kwargs: PCovR(
            mixing, regularization=1e-8, **kwargs
        )
        self.error_tol = 1e-6

        self.X, self.Y = load_boston(return_X_y=True)

    def setUp(self):
        pass


class PCovRErrorTest(PCovRBaseTest):
    def test_lr_with_x_errors(self):
        """
        This test checks that PCovR returns a non-null property prediction
        and that the prediction error increases with `mixing`
        """
        prev_error = -1.0

        for i, mixing in enumerate(np.linspace(0, 1, 11)):

            pcovr = self.model(mixing=mixing, n_components=2, tol=1e-12)
            pcovr.fit(self.X, self.Y)

            Yp = pcovr.predict(self.X)
            error = rel_error(self.Y, Yp)

            with self.subTest(error=error):
                self.assertFalse(np.isnan(error))
            with self.subTest(error=error, alpha=mixing):
                self.assertGreaterEqual(error, prev_error - self.error_tol)

            prev_error = error

    def test_lr_with_t_errors(self):
        """
        This test checks that PCovR returns a non-null property prediction
        from the latent space projection and that the prediction error
        increases with `mixing`
        """

        prev_error = -1.0

        for i, mixing in enumerate(np.linspace(0, 1, 11)):
            pcovr = self.model(mixing=mixing, n_components=2, tol=1e-12)
            pcovr.fit(self.X, self.Y)

            Yp = pcovr.predict(T=pcovr.transform(self.X))
            error = rel_error(self.Y, Yp)

            with self.subTest(error=error):
                self.assertFalse(np.isnan(error))
            with self.subTest(error=error, alpha=mixing):
                self.assertGreaterEqual(error, prev_error - self.error_tol)

            prev_error = error

    def test_reconstruction_errors(self):
        """
        This test checks that PCovR returns a non-null reconstructed X
        and that the reconstruction error decreases with `mixing`
        """

        prev_error = 1.0

        for i, mixing in enumerate(np.linspace(0, 1, 11)):
            pcovr = self.model(mixing=mixing, n_components=2, tol=1e-12)
            pcovr.fit(self.X, self.Y)

            error = rel_error(self.X, pcovr.inverse_transform(pcovr.transform(self.X)))

            with self.subTest(error=error):
                self.assertFalse(np.isnan(error))
            with self.subTest(error=error, alpha=mixing):
                self.assertLessEqual(error, prev_error + self.error_tol)

            prev_error = error


class PCovRSpaceTest(PCovRBaseTest):
    def test_select_feature_space(self):
        """
        This test checks that PCovR implements the feature space version
        when :math:`n_{features} < n_{samples}``.
        """
        pcovr = self.model(mixing=0.5, n_components=2, tol=1e-12)
        pcovr.fit(self.X, self.Y)

        self.assertTrue(pcovr.space == "feature")

    def test_select_structure_space(self):
        """
        This test checks that PCovR implements the structure space version
        when :math:`n_{features} > n_{samples}``.
        """
        pcovr = self.model(mixing=0.5, n_components=2, tol=1e-12)

        n_structures = self.X.shape[1] - 1
        pcovr.fit(self.X[:n_structures], self.Y[:n_structures])

        self.assertTrue(pcovr.space == "structure")

    def test_bad_space(self):
        """
        This test checks that PCovR raises a ValueError when a non-valid
        space is designated.
        """
        with self.assertRaises(ValueError):
            pcovr = self.model(mixing=0.5, n_components=2, tol=1e-12, space="bad")
            pcovr.fit(self.X, self.Y)

    def test_override_space_selection(self):
        """
        This test checks that PCovR implements the space provided in the
        constructor, overriding that chosen by the input dimensions.
        """
        pcovr = self.model(mixing=0.5, n_components=2, tol=1e-12, space="structure")
        pcovr.fit(self.X, self.Y)

        self.assertTrue(pcovr.space == "structure")


class PCovRInfrastructureTest(PCovRBaseTest):
    def test_nonfitted_failure(self):
        """
        This test checks that PCovR will raise a `NonFittedError` if
        `transform` is called before the model is fitted
        """
        model = self.model(mixing=0.5, n_components=2, tol=1e-12)
        with self.assertRaises(exceptions.NotFittedError):
            _ = model.transform(self.X)

    def test_no_arg_predict(self):
        """
        This test checks that PCovR will raise a `ValueError` if
        `predict` is called without arguments
        """
        model = self.model(mixing=0.5, n_components=2, tol=1e-12)
        model.fit(self.X, self.Y)
        with self.assertRaises(ValueError):
            _ = model.predict()

    def test_T_shape(self):
        """
        This test checks that PCovR returns a latent space projection
        consistent with the shape of the input matrix
        """
        n_components = 5
        pcovr = self.model(mixing=0.5, n_components=n_components, tol=1e-12)
        pcovr.fit(self.X, self.Y)
        T = pcovr.transform(self.X)
        self.assertTrue(check_X_y(self.X, T, multi_output=True))
        self.assertTrue(T.shape[-1] == n_components)


if __name__ == "__main__":
    unittest.main(verbosity=2)
