import unittest
import warnings

import numpy as np
from sklearn import exceptions
from sklearn.datasets import load_diabetes as get_dataset
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y

from skmatter.decomposition import PCovR


class PCovRBaseTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = lambda mixing=0.5, regressor=Ridge(
            alpha=1e-8, fit_intercept=False, tol=1e-12
        ), **kwargs: PCovR(mixing, regressor=regressor, **kwargs)
        self.error_tol = 1e-5

        self.X, self.Y = get_dataset(return_X_y=True)
        self.X = StandardScaler().fit_transform(self.X)
        self.Y = StandardScaler().fit_transform(np.vstack(self.Y)).ravel()

    def setUp(self):
        pass


class PCovRErrorTest(PCovRBaseTest):
    def test_against_pca(self):
        """Tests that mixing = 1.0 corresponds to PCA."""
        pcovr = PCovR(
            mixing=1.0, n_components=3, space="sample", svd_solver="full"
        ).fit(self.X, self.Y)
        pca = PCA(n_components=3, svd_solver="full").fit(self.X)

        # tests that the SVD is equivalent
        self.assertTrue(np.allclose(pca.singular_values_, pcovr.singular_values_))
        self.assertTrue(np.allclose(pca.explained_variance_, pcovr.explained_variance_))

        T_pcovr = pcovr.transform(self.X)
        T_pca = pca.transform(self.X)

        # tests that the projections are equivalent
        self.assertLessEqual(
            np.linalg.norm(T_pcovr @ T_pcovr.T - T_pca @ T_pca.T), 1e-8
        )

    def test_simple_reconstruction(self):
        """Check that PCovR with a full eigendecomposition at mixing=1 can fully
        reconstruct the input matrix.
        """
        for space in ["feature", "sample", "auto"]:
            with self.subTest(space=space):
                pcovr = self.model(
                    mixing=1.0, n_components=self.X.shape[-1], space=space
                )
                pcovr.fit(self.X, self.Y)
                Xr = pcovr.inverse_transform(pcovr.transform(self.X))
                self.assertLessEqual(
                    np.linalg.norm(self.X - Xr) ** 2.0 / np.linalg.norm(self.X) ** 2.0,
                    self.error_tol,
                )

    def test_simple_prediction(self):
        """
        Check that PCovR with a full eigendecomposition at mixing=0
        can reproduce a linear regression result.
        """
        for space in ["feature", "sample", "auto"]:
            with self.subTest(space=space):
                pcovr = self.model(mixing=0.0, n_components=1, space=space)

                pcovr.regressor.fit(self.X, self.Y)
                Yhat = pcovr.regressor.predict(self.X)

                pcovr.fit(self.X, self.Y)
                Yp = pcovr.predict(self.X)
                self.assertLessEqual(
                    np.linalg.norm(Yp - Yhat) ** 2.0 / np.linalg.norm(Yhat) ** 2.0,
                    self.error_tol,
                )

    def test_lr_with_x_errors(self):
        """
        Check that PCovR returns a non-null property prediction
        and that the prediction error increases with `mixing`
        """
        prev_error = -1.0

        for mixing in np.linspace(0, 1, 11):
            pcovr = self.model(mixing=mixing, n_components=2, tol=1e-12)
            pcovr.fit(self.X, self.Y)

            Yp = pcovr.predict(X=self.X)
            error = np.linalg.norm(self.Y - Yp) ** 2.0 / np.linalg.norm(self.Y) ** 2.0

            with self.subTest(error=error):
                self.assertFalse(np.isnan(error))
            with self.subTest(error=error, alpha=round(mixing, 4)):
                self.assertGreaterEqual(error, prev_error - self.error_tol)

            prev_error = error

    def test_lr_with_t_errors(self):
        """Check that PCovR returns a non-null property prediction from the latent space
        projection and that the prediction error increases with `mixing`.
        """
        prev_error = -1.0

        for mixing in np.linspace(0, 1, 11):
            pcovr = self.model(mixing=mixing, n_components=2, tol=1e-12)
            pcovr.fit(self.X, self.Y)

            T = pcovr.transform(self.X)
            Yp = pcovr.predict(T=T)
            error = np.linalg.norm(self.Y - Yp) ** 2.0 / np.linalg.norm(self.Y) ** 2.0

            with self.subTest(error=error):
                self.assertFalse(np.isnan(error))
            with self.subTest(error=error, alpha=round(mixing, 4)):
                self.assertGreaterEqual(error, prev_error - self.error_tol)

            prev_error = error

    def test_reconstruction_errors(self):
        """Check that PCovR returns a non-null reconstructed X and that the
        reconstruction error decreases with `mixing`.
        """
        prev_error = 1.0

        for mixing in np.linspace(0, 1, 11):
            pcovr = self.model(mixing=mixing, n_components=2, tol=1e-12)
            pcovr.fit(self.X, self.Y)

            Xr = pcovr.inverse_transform(pcovr.transform(self.X))
            error = np.linalg.norm(self.X - Xr) ** 2.0 / np.linalg.norm(self.X) ** 2.0

            with self.subTest(error=error):
                self.assertFalse(np.isnan(error))
            with self.subTest(error=error, alpha=round(mixing, 4)):
                self.assertLessEqual(error, prev_error + self.error_tol)

            prev_error = error


class PCovRSpaceTest(PCovRBaseTest):
    def test_select_feature_space(self):
        """
        Check that PCovR implements the feature space version
        when :math:`n_{features} < n_{samples}``.
        """
        pcovr = self.model(n_components=2, tol=1e-12)
        pcovr.fit(self.X, self.Y)

        self.assertTrue(pcovr.space_ == "feature")

    def test_select_sample_space(self):
        """
        Check that PCovR implements the sample space version
        when :math:`n_{features} > n_{samples}``.
        """
        pcovr = self.model(n_components=2, tol=1e-12)

        n_samples = self.X.shape[1] - 1
        pcovr.fit(self.X[:n_samples], self.Y[:n_samples])

        self.assertTrue(pcovr.space_ == "sample")

    def test_bad_space(self):
        """
        Check that PCovR raises a ValueError when a non-valid
        space is designated.
        """
        with self.assertRaises(ValueError):
            pcovr = self.model(n_components=2, tol=1e-12, space="bad")
            pcovr.fit(self.X, self.Y)

    def test_override_spaceselection(self):
        """
        Check that PCovR implements the space provided in the
        constructor, overriding that chosen by the input dimensions.
        """
        pcovr = self.model(n_components=2, tol=1e-12, space="sample")
        pcovr.fit(self.X, self.Y)

        self.assertTrue(pcovr.space_ == "sample")

    def test_spaces_equivalent(self):
        """
        Check that the results from PCovR, regardless of the space,
        are equivalent.
        """
        for alpha in np.linspace(0.01, 0.99, 11):
            with self.subTest(alpha=alpha, type="prediction"):
                pcovr_ss = self.model(
                    n_components=2, mixing=alpha, tol=1e-12, space="sample"
                )
                pcovr_ss.fit(self.X, self.Y)

                pcovr_fs = self.model(
                    n_components=2, mixing=alpha, tol=1e-12, space="feature"
                )
                pcovr_fs.fit(self.X, self.Y)

                self.assertTrue(
                    np.allclose(
                        pcovr_ss.predict(self.X),
                        pcovr_fs.predict(self.X),
                        self.error_tol,
                    )
                )

            with self.subTest(alpha=alpha, type="reconstruction"):
                pcovr_ss = self.model(
                    n_components=2, mixing=alpha, tol=1e-12, space="sample"
                )
                pcovr_ss.fit(self.X, self.Y)

                pcovr_fs = self.model(
                    n_components=2, mixing=alpha, tol=1e-12, space="feature"
                )
                pcovr_fs.fit(self.X, self.Y)

                self.assertTrue(
                    np.allclose(
                        pcovr_ss.inverse_transform(pcovr_ss.transform(self.X)),
                        pcovr_fs.inverse_transform(pcovr_fs.transform(self.X)),
                        self.error_tol,
                    )
                )


class PCovRTestSVDSolvers(PCovRBaseTest):
    def test_svd_solvers(self):
        """
        Check that PCovR works with all svd_solver modes and assigns
        the right n_components
        """
        for solver in ["arpack", "full", "randomized", "auto"]:
            with self.subTest(solver=solver):
                pcovr = self.model(tol=1e-12, svd_solver=solver)
                pcovr.fit(self.X, self.Y)

                if solver == "arpack":
                    self.assertTrue(pcovr.n_components_ == min(self.X.shape) - 1)
                else:
                    self.assertTrue(pcovr.n_components_ == min(self.X.shape))

    def test_bad_solver(self):
        """
        Check that PCovR will not work with a solver that isn't in
        ['arpack', 'full', 'randomized', 'auto']
        """
        for space in ["feature", "sample"]:
            with self.assertRaises(ValueError) as cm:
                pcovr = self.model(svd_solver="bad", space=space)
                pcovr.fit(self.X, self.Y)

            self.assertEqual(str(cm.exception), "Unrecognized svd_solver='bad'")

    def test_good_n_components(self):
        """Check that PCovR will work with any allowed values of n_components."""
        # this one should pass
        pcovr = self.model(n_components=0.5, svd_solver="full")
        pcovr.fit(self.X, self.Y)

        for svd_solver in ["auto", "full"]:
            # this one should pass
            pcovr = self.model(n_components=2, svd_solver=svd_solver)
            pcovr.fit(self.X, self.Y)

            # this one should pass
            pcovr = self.model(n_components="mle", svd_solver=svd_solver)
            pcovr.fit(self.X, self.Y)

    def test_bad_n_components(self):
        """Check that PCovR will not work with any prohibited values of n_components."""
        with self.assertRaises(ValueError) as cm:
            pcovr = self.model(n_components="mle", svd_solver="full")
            pcovr.fit(self.X[:2], self.Y[:2])
        self.assertEqual(
            str(cm.exception),
            "n_components='mle' is only supported if n_samples >= n_features",
        )

        with self.subTest(type="negative_ncomponents"):
            with self.assertRaises(ValueError) as cm:
                pcovr = self.model(n_components=-1, svd_solver="auto")
                pcovr.fit(self.X, self.Y)

            self.assertEqual(
                str(cm.exception),
                "n_components=%r must be between 1 and "
                "min(n_samples, n_features)=%r with "
                "svd_solver='%s'"
                % (
                    pcovr.n_components_,
                    min(self.X.shape),
                    pcovr.svd_solver,
                ),
            )
        with self.subTest(type="0_ncomponents"):
            with self.assertRaises(ValueError) as cm:
                pcovr = self.model(n_components=0, svd_solver="randomized")
                pcovr.fit(self.X, self.Y)

            self.assertEqual(
                str(cm.exception),
                "n_components=%r must be between 1 and "
                "min(n_samples, n_features)=%r with "
                "svd_solver='%s'"
                % (
                    pcovr.n_components_,
                    min(self.X.shape),
                    pcovr.svd_solver,
                ),
            )
        with self.subTest(type="arpack_X_ncomponents"):
            with self.assertRaises(ValueError) as cm:
                pcovr = self.model(n_components=min(self.X.shape), svd_solver="arpack")
                pcovr.fit(self.X, self.Y)
            self.assertEqual(
                str(cm.exception),
                "n_components=%r must be strictly less than "
                "min(n_samples, n_features)=%r with "
                "svd_solver='%s'"
                % (
                    pcovr.n_components_,
                    min(self.X.shape),
                    pcovr.svd_solver,
                ),
            )

        for svd_solver in ["auto", "full"]:
            with self.subTest(type="pi_ncomponents"):
                with self.assertRaises(ValueError) as cm:
                    pcovr = self.model(n_components=np.pi, svd_solver=svd_solver)
                    pcovr.fit(self.X, self.Y)
                self.assertEqual(
                    str(cm.exception),
                    "n_components=%r must be of type int "
                    "when greater than or equal to 1, was of type=%r"
                    % (pcovr.n_components_, type(pcovr.n_components_)),
                )


class PCovRInfrastructureTest(PCovRBaseTest):
    def test_nonfitted_failure(self):
        """
        Check that PCovR will raise a `NonFittedError` if
        `transform` is called before the pcovr is fitted
        """
        pcovr = self.model(n_components=2, tol=1e-12)
        with self.assertRaises(exceptions.NotFittedError):
            _ = pcovr.transform(self.X)

    def test_no_arg_predict(self):
        """
        Check that PCovR will raise a `ValueError` if
        `predict` is called without arguments
        """
        pcovr = self.model(n_components=2, tol=1e-12)
        pcovr.fit(self.X, self.Y)
        with self.assertRaises(ValueError):
            _ = pcovr.predict()

    def test_centering(self):
        """
        Check that PCovR raises a warning if
        given uncentered data.
        """
        pcovr = self.model(n_components=2, tol=1e-12)
        X = self.X.copy() + np.random.uniform(-1, 1, self.X.shape[1])
        with warnings.catch_warnings(record=True) as w:
            pcovr.fit(X, self.Y)
            self.assertEqual(
                str(w[0].message),
                "This class does not automatically center data, and your data mean is "
                "greater than the supplied tolerance.",
            )

    def test_T_shape(self):
        """Check that PCovR returns a latent space projection consistent with the shape
        of the input matrix.
        """
        n_components = 5
        pcovr = self.model(n_components=n_components, tol=1e-12)
        pcovr.fit(self.X, self.Y)
        T = pcovr.transform(self.X)
        self.assertTrue(check_X_y(self.X, T, multi_output=True))
        self.assertTrue(T.shape[-1] == n_components)

    def test_default_ncomponents(self):
        pcovr = PCovR(mixing=0.5)
        pcovr.fit(self.X, self.Y)

        self.assertEqual(pcovr.n_components_, min(self.X.shape))

    def test_Y_Shape(self):
        pcovr = self.model()
        self.Y = np.vstack(self.Y)
        pcovr.fit(self.X, self.Y)

        self.assertEqual(pcovr.pxy_.shape[0], self.X.shape[1])
        self.assertEqual(pcovr.pty_.shape[0], pcovr.n_components_)

    def test_prefit_regressor(self):
        regressor = Ridge(alpha=1e-8, fit_intercept=False, tol=1e-12)
        regressor.fit(self.X, self.Y)
        pcovr = self.model(mixing=0.5, regressor=regressor)
        pcovr.fit(self.X, self.Y)

        Yhat_regressor = regressor.predict(self.X).reshape(self.X.shape[0], -1)
        W_regressor = regressor.coef_.T.reshape(self.X.shape[1], -1)

        Yhat_pcovr = pcovr.regressor_.predict(self.X).reshape(self.X.shape[0], -1)
        W_pcovr = pcovr.regressor_.coef_.T.reshape(self.X.shape[1], -1)

        self.assertTrue(np.allclose(Yhat_regressor, Yhat_pcovr))
        self.assertTrue(np.allclose(W_regressor, W_pcovr))

    def test_prefit_regression(self):
        regressor = Ridge(alpha=1e-8, fit_intercept=False, tol=1e-12)
        regressor.fit(self.X, self.Y)
        Yhat = regressor.predict(self.X)
        W = regressor.coef_.reshape(self.X.shape[1], -1)

        pcovr1 = self.model(mixing=0.5, regressor="precomputed", n_components=1)
        pcovr1.fit(self.X, Yhat, W)
        t1 = pcovr1.transform(self.X)

        pcovr2 = self.model(mixing=0.5, regressor=regressor, n_components=1)
        pcovr2.fit(self.X, self.Y)
        t2 = pcovr2.transform(self.X)

        self.assertTrue(np.linalg.norm(t1 - t2) < self.error_tol)

    def test_regressor_modifications(self):
        regressor = Ridge(alpha=1e-8)
        pcovr = self.model(mixing=0.5, regressor=regressor)

        # PCovR regressor matches the original
        self.assertTrue(regressor.get_params() == pcovr.regressor.get_params())

        # PCovR regressor updates its parameters
        # to match the original regressor
        regressor.set_params(alpha=1e-6)
        self.assertTrue(regressor.get_params() == pcovr.regressor.get_params())

        # Fitting regressor outside PCovR fits the PCovR regressor
        regressor.fit(self.X, self.Y)
        self.assertTrue(hasattr(pcovr.regressor, "coef_"))

        # PCovR regressor doesn't change after fitting
        pcovr.fit(self.X, self.Y)
        regressor.set_params(alpha=1e-4)
        self.assertTrue(hasattr(pcovr.regressor_, "coef_"))
        self.assertTrue(regressor.get_params() != pcovr.regressor_.get_params())

    def test_incompatible_regressor(self):
        regressor = KernelRidge(alpha=1e-8, kernel="linear")
        regressor.fit(self.X, self.Y)
        pcovr = self.model(mixing=0.5, regressor=regressor)

        with self.assertRaises(ValueError) as cm:
            pcovr.fit(self.X, self.Y)
        self.assertEqual(
            str(cm.exception),
            "Regressor must be an instance of `LinearRegression`, `Ridge`, `RidgeCV`, "
            "or `precomputed`",
        )

    def test_none_regressor(self):
        pcovr = PCovR(mixing=0.5, regressor=None)
        pcovr.fit(self.X, self.Y)
        self.assertTrue(pcovr.regressor is None)
        self.assertTrue(pcovr.regressor_ is not None)

    def test_incompatible_coef_dim(self):
        # self.Y is 1D with one target
        # Don't need to test X shape, since this should
        # be caught by sklearn's validate_data
        Y_2D = np.column_stack((self.Y, self.Y))
        regressor = Ridge(alpha=1e-8, fit_intercept=False, tol=1e-12)
        regressor.fit(self.X, Y_2D)
        pcovr = self.model(mixing=0.5, regressor=regressor)

        # Dimension mismatch
        with self.assertRaises(ValueError) as cm:
            pcovr.fit(self.X, self.Y)
        self.assertEqual(
            str(cm.exception),
            "The regressor coefficients have a dimension incompatible with the "
            "supplied target space. The coefficients have dimension 2 and the targets "
            "have dimension 1",
        )

    def test_incompatible_coef_shape(self):
        # Shape mismatch (number of targets)
        Y_double = np.column_stack((self.Y, self.Y))
        Y_triple = np.column_stack((Y_double, self.Y))

        regressor = Ridge(alpha=1e-8, fit_intercept=False, tol=1e-12)
        regressor.fit(self.X, Y_double)

        pcovr = self.model(mixing=0.5, regressor=regressor)

        with self.assertRaises(ValueError) as cm:
            pcovr.fit(self.X, Y_triple)
        self.assertEqual(
            str(cm.exception),
            "The regressor coefficients have a shape incompatible with the supplied "
            "target space. The coefficients have shape %r and the targets have shape %r"
            % (regressor.coef_.shape, Y_triple.shape),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
