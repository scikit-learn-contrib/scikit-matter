import unittest
from skcosmo.decomposition import KPCovR, PCovR
from sklearn.datasets import load_boston
import numpy as np
from sklearn import exceptions
from sklearn.linear_model import RidgeCV
from sklearn.utils.validation import check_X_y
from skcosmo.preprocessing import StandardFlexibleScaler as SFS


class KPCovRBaseTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.error_tol = 1e-6

        self.X, self.Y = load_boston(return_X_y=True)

        # artificial second property
        self.Y = np.array(
            [self.Y, self.X @ np.random.randint(-2, 2, (self.X.shape[-1],))]
        ).T
        self.Y = self.Y.reshape(self.X.shape[0], -1)

        self.X = SFS().fit_transform(self.X)
        self.Y = SFS(column_wise=True).fit_transform(self.Y)

        self.model = lambda mixing=0.5, **kwargs: KPCovR(mixing, alpha=1e-8, **kwargs)

    def setUp(self):
        pass


class KPCovRErrorTest(KPCovRBaseTest):
    def test_lr_with_x_errors(self):
        """
        This test checks that KPCovR returns a non-null property prediction
        and that the prediction error increases with `mixing`
        """
        prev_error = -1.0

        for i, mixing in enumerate(np.linspace(0, 1, 11)):

            pcovr = KPCovR(mixing=mixing, n_components=2, tol=1e-12)
            pcovr.fit(self.X, self.Y)

            error = (
                np.linalg.norm(self.Y - pcovr.predict(self.X)) ** 2.0
                / np.linalg.norm(self.Y) ** 2.0
            )

            with self.subTest(error=error):
                self.assertFalse(np.isnan(error))
            with self.subTest(error=error, alpha=round(mixing, 4)):
                self.assertGreaterEqual(error, prev_error - self.error_tol)

            prev_error = error

    def test_reconstruction_errors(self):
        """
        This test checks that KPCovR returns a non-null reconstructed X
        and that the reconstruction error decreases with `mixing`
        """

        prev_error = 10.0

        for i, mixing in enumerate(np.linspace(0, 1, 11)):
            kpcovr = KPCovR(mixing=mixing, n_components=2, tol=1e-12)
            kpcovr.fit(self.X, self.Y)

            t = kpcovr.transform(self.X)
            K = kpcovr._get_kernel(self.X)

            error = np.linalg.norm(K - t @ t.T) ** 2.0 / np.linalg.norm(K) ** 2.0

            with self.subTest(error=error):
                self.assertFalse(np.isnan(error))
            with self.subTest(error=error, alpha=round(mixing, 4)):
                self.assertLessEqual(error, prev_error + self.error_tol)

            prev_error = error

    def test_kpcovr_error(self):
        for i, mixing in enumerate(np.linspace(0, 1, 11)):
            kpcovr = self.model(mixing=mixing, kernel="rbf", gamma=1.0, center=False)

            kpcovr.fit(self.X, self.Y)
            K = kpcovr._get_kernel(self.X)

            y = kpcovr.predict(self.X)
            Lkrr = np.linalg.norm(self.Y - y) ** 2 / np.linalg.norm(self.Y) ** 2

            t = kpcovr.transform(self.X)

            w = t @ np.linalg.pinv(t.T @ t, rcond=kpcovr.alpha) @ t.T
            Lkpca = np.trace(K - K @ w) / np.trace(K)

            self.assertTrue(
                np.isclose(
                    kpcovr.score(self.X, self.Y), sum([Lkpca, Lkrr]), self.error_tol
                )
            )


class KPCovRInfrastructureTest(KPCovRBaseTest):
    def test_nonfitted_failure(self):
        """
        This test checks that KPCovR will raise a `NonFittedError` if
        `transform` is called before the model is fitted
        """
        kpcovr = KPCovR(mixing=0.5, n_components=2, tol=1e-12)
        with self.assertRaises(exceptions.NotFittedError):
            _ = kpcovr.transform(self.X)

    def test_no_arg_predict(self):
        """
        This test checks that KPCovR will raise a `ValueError` if
        `predict` is called without arguments
        """
        kpcovr = KPCovR(mixing=0.5, n_components=2, tol=1e-12)
        kpcovr.fit(self.X, self.Y)
        with self.assertRaises(ValueError):
            _ = kpcovr.predict()

    def test_T_shape(self):
        """
        This test checks that KPCovR returns a latent space projection
        consistent with the shape of the input matrix
        """
        n_components = 5
        kpcovr = KPCovR(mixing=0.5, n_components=n_components, tol=1e-12)
        kpcovr.fit(self.X, self.Y)
        T = kpcovr.transform(self.X)
        self.assertTrue(check_X_y(self.X, T, multi_output=True))
        self.assertTrue(T.shape[-1] == n_components)

    def test_no_centerer(self):
        kpcovr = self.model(center=False)
        kpcovr.fit(self.X, self.Y)

        with self.assertRaises(AttributeError):
            _ = getattr(kpcovr, "centerer_")


class KernelTests(KPCovRBaseTest):
    def test_kernel_types(self):
        """
        This test checks that KPCovR can handle all kernels passable to
        sklearn kernel classes, including callable kernels
        """

        def _linear_kernel(X, Y):
            return X @ Y.T

        kernel_params = {
            "poly": {"degree": 2},
            "rbf": {"gamma": 3.0},
            "sigmoid": {"gamma": 3.0, "coef0": 0.5},
        }
        for kernel in ["linear", "poly", "rbf", "sigmoid", "cosine", _linear_kernel]:
            with self.subTest(kernel=kernel):
                kpcovr = KPCovR(
                    mixing=0.5,
                    n_components=2,
                    kernel=kernel,
                    **kernel_params.get(kernel, {})
                )
                kpcovr.fit(self.X, self.Y)

    def test_linear_matches_pcovr(self):
        """
        This test checks that KPCovR returns the same results as PCovR when
        using a linear kernel
        """

        # making a common Yhat so that the models are working off the same values
        ridge = RidgeCV(fit_intercept=False, alphas=np.logspace(-8, 2))
        Yhat = ridge.fit(self.X, self.Y).predict(self.X)

        # common instantiation parameters for the two models
        hypers = dict(
            mixing=0.5,
            n_components=1,
            alpha=1e-8,
        )

        # computing projection and predicton loss with linear KPCovR
        kpcovr = KPCovR(kernel="linear", fit_inverse_transform=True, **hypers)
        kpcovr.fit(self.X, self.Y, Yhat=Yhat)
        ly = (
            np.linalg.norm(self.Y - kpcovr.predict(self.X)) ** 2.0
            / np.linalg.norm(self.Y) ** 2.0
        )

        # computing projection and predicton loss with PCovR
        ref_pcovr = PCovR(**hypers, space="sample")
        ref_pcovr.fit(self.X, self.Y, Yhat=Yhat)
        ly_ref = (
            np.linalg.norm(self.Y - ref_pcovr.predict(self.X)) ** 2.0
            / np.linalg.norm(self.Y) ** 2.0
        )

        t_ref = ref_pcovr.transform(self.X)
        t = kpcovr.transform(self.X)

        K = kpcovr._get_kernel(self.X)

        k_ref = t_ref @ t_ref.T
        k = t @ t.T

        lk_ref = np.linalg.norm(K - k_ref) ** 2.0 / np.linalg.norm(K) ** 2.0
        lk = np.linalg.norm(K - k) ** 2.0 / np.linalg.norm(K) ** 2.0

        rounding = 3
        self.assertEqual(
            round(ly, rounding),
            round(ly_ref, rounding),
        )

        self.assertEqual(
            round(lk, rounding),
            round(lk_ref, rounding),
        )


class KPCovRTestSVDSolvers(KPCovRBaseTest):
    def test_svd_solvers(self):
        """
        This test checks that PCovR works with all svd_solver modes and assigns
        the right n_components
        """
        for solver in ["arpack", "full", "randomized", "auto"]:
            with self.subTest(solver=solver):
                pcovr = self.model(tol=1e-12, svd_solver=solver)
                pcovr.fit(self.X, self.Y)

                if solver == "arpack":
                    self.assertTrue(pcovr.n_components == self.X.shape[0] - 1)
                else:
                    self.assertTrue(pcovr.n_components == self.X.shape[0])

    def test_bad_solver(self):
        """
        This test checks that PCovR will not work with a solver that isn't in
        ['arpack', 'full', 'randomized', 'auto']
        """
        with self.assertRaises(ValueError) as cm:
            pcovr = self.model(svd_solver="bad")
            pcovr.fit(self.X, self.Y)

            self.assertTrue(str(cm.message), "Unrecognized svd_solver='bad'" "")

    def test_good_n_components(self):
        """
        This test checks that PCovR will work with any allowed values of
        n_components.
        """

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
        """
        This test checks that PCovR will not work with any prohibited values of
        n_components.
        """

        with self.subTest(type="negative_ncomponents"):
            with self.assertRaises(ValueError) as cm:
                pcovr = self.model(n_components=-1, svd_solver="auto")
                pcovr.fit(self.X, self.Y)

                self.assertTrue(
                    str(cm.message),
                    "self.n_components=%r must be between 0 and "
                    "min(n_samples, n_features)=%r with "
                    "svd_solver='%s'"
                    % (
                        pcovr.n_components,
                        self.X.shape[0],
                        pcovr.svd_solver,
                    ),
                )
        with self.subTest(type="0_ncomponents"):
            with self.assertRaises(ValueError) as cm:
                pcovr = self.model(n_components=0, svd_solver="randomized")
                pcovr.fit(self.X, self.Y)

                self.assertTrue(
                    str(cm.message),
                    "self.n_components=%r must be between 1 and "
                    "min(n_samples, n_features)=%r with "
                    "svd_solver='%s'"
                    % (
                        pcovr.n_components,
                        self.X.shape[0],
                        pcovr.svd_solver,
                    ),
                )
        with self.subTest(type="arpack_X_ncomponents"):
            with self.assertRaises(ValueError) as cm:
                pcovr = self.model(n_components=self.X.shape[0], svd_solver="arpack")
                pcovr.fit(self.X, self.Y)
                self.assertTrue(
                    str(cm.message),
                    "self.n_components=%r must be strictly less than "
                    "min(n_samples, n_features)=%r with "
                    "svd_solver='%s'"
                    % (
                        pcovr.n_components,
                        self.X.shape[0],
                        pcovr.svd_solver,
                    ),
                )

        for svd_solver in ["auto", "full"]:
            with self.subTest(type="pi_ncomponents"):
                with self.assertRaises(ValueError) as cm:
                    pcovr = self.model(n_components=np.pi, svd_solver=svd_solver)
                    pcovr.fit(self.X, self.Y)
                    self.assertTrue(
                        str(cm.message),
                        "self.n_components=%r must be of type int "
                        "when greater than or equal to 1, was of type=%r"
                        % (pcovr.n_components, type(pcovr.n_components)),
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
