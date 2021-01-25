import unittest
from skcosmo.pcovr import PCovR
from sklearn.datasets import load_boston
import numpy as np
from sklearn import exceptions
from sklearn.utils.validation import check_X_y


class PCovRBaseTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = lambda mixing=0.5, **kwargs: PCovR(
            mixing, regularization=1e-8, **kwargs
        )
        self.error_tol = 1e-5

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

            _ = pcovr.predict(X=self.X)
            _, error = pcovr.score(self.X, self.Y)

            with self.subTest(error=error):
                self.assertFalse(np.isnan(error))
            with self.subTest(error=error, alpha=round(mixing, 4)):
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

            T = pcovr.transform(self.X)
            _ = pcovr.predict(T=T)
            _, error = pcovr.score(self.X, self.Y, T=T)

            with self.subTest(error=error):
                self.assertFalse(np.isnan(error))
            with self.subTest(error=error, alpha=round(mixing, 4)):
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

            error, _ = pcovr.score(self.X, self.Y)

            with self.subTest(error=error):
                self.assertFalse(np.isnan(error))
            with self.subTest(error=error, alpha=round(mixing, 4)):
                self.assertLessEqual(error, prev_error + self.error_tol)

            prev_error = error


class PCovRSpaceTest(PCovRBaseTest):
    def test_select_feature_space(self):
        """
        This test checks that PCovR implements the feature space version
        when :math:`n_{features} < n_{samples}``.
        """
        pcovr = self.model(n_components=2, tol=1e-12)
        pcovr.fit(self.X, self.Y)

        self.assertTrue(pcovr.space == "feature")

    def test_select_structure_space(self):
        """
        This test checks that PCovR implements the structure space version
        when :math:`n_{features} > n_{samples}``.
        """
        pcovr = self.model(n_components=2, tol=1e-12)

        n_structures = self.X.shape[1] - 1
        pcovr.fit(self.X[:n_structures], self.Y[:n_structures])

        self.assertTrue(pcovr.space == "structure")

    def test_bad_space(self):
        """
        This test checks that PCovR raises a ValueError when a non-valid
        space is designated.
        """
        with self.assertRaises(ValueError):
            pcovr = self.model(n_components=2, tol=1e-12, space="bad")
            pcovr.fit(self.X, self.Y)

    def test_override_spaceselection(self):
        """
        This test checks that PCovR implements the space provided in the
        constructor, overriding that chosen by the input dimensions.
        """
        pcovr = self.model(n_components=2, tol=1e-12, space="structure")
        pcovr.fit(self.X, self.Y)

        self.assertTrue(pcovr.space == "structure")


class PCovRInfrastructureTest(PCovRBaseTest):
    def test_nonfitted_failure(self):
        """
        This test checks that PCovR will raise a `NonFittedError` if
        `transform` is called before the model is fitted
        """
        model = self.model(n_components=2, tol=1e-12)
        with self.assertRaises(exceptions.NotFittedError):
            _ = model.transform(self.X)

    def test_no_arg_predict(self):
        """
        This test checks that PCovR will raise a `ValueError` if
        `predict` is called without arguments
        """
        model = self.model(n_components=2, tol=1e-12)
        model.fit(self.X, self.Y)
        with self.assertRaises(ValueError):
            _ = model.predict()

    def test_T_shape(self):
        """
        This test checks that PCovR returns a latent space projection
        consistent with the shape of the input matrix
        """
        n_components = 5
        pcovr = self.model(n_components=n_components, tol=1e-12)
        pcovr.fit(self.X, self.Y)
        T = pcovr.transform(self.X)
        self.assertTrue(check_X_y(self.X, T, multi_output=True))
        self.assertTrue(T.shape[-1] == n_components)

    def test_default_regularization(self):
        model = PCovR(mixing=0.5)
        estimator_reg = getattr(model.estimator, "alpha")
        self.assertEqual(model.regularization, estimator_reg)

    def test_default_ncomponents(self):
        model = PCovR(mixing=0.5)
        model.fit(self.X, self.Y)

        self.assertEqual(model.n_components, min(self.X.shape))

    def test_Y_Shape(self):
        model = self.model()
        self.Y = np.vstack(self.Y)
        model.fit(self.X, self.Y)

        self.assertEqual(model.pxy_.shape[0], self.X.shape[1])
        self.assertEqual(model.pty_.shape[0], model.n_components)

    def test_Yhat_and_W(self):
        model = self.model(mixing=0.5)

        model.estimator.fit(self.X, self.Y)
        Yhat = model.estimator.predict(self.X).reshape(self.X.shape[0], -1)
        W = model.estimator.coef_.T.reshape(self.X.shape[1], -1)
        model.fit(self.X, self.Y, Yhat, W)


if __name__ == "__main__":
    unittest.main(verbosity=2)
