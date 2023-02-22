import unittest

import numpy as np
from sklearn import exceptions
from sklearn.datasets import load_diabetes as get_dataset
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    Ridge,
    RidgeCV,
)
from sklearn.utils.validation import check_X_y

from skcosmo.decomposition import (
    KernelPCovR,
    PCovR,
)
from skcosmo.preprocessing import StandardFlexibleScaler as SFS


class KernelPCovRBaseTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_state = np.random.RandomState(0)

        self.error_tol = 1e-6

        self.X, self.Y = get_dataset(return_X_y=True)

        # for the sake of expedience, only use a subset of the dataset
        idx = self.random_state.choice(len(self.X), 1000)
        self.X = self.X[idx]
        self.Y = self.Y[idx]

        # artificial second property
        self.Y = np.array(
            [self.Y, self.X @ self.random_state.randint(-2, 2, (self.X.shape[-1],))]
        ).T
        self.Y = self.Y.reshape(self.X.shape[0], -1)

        self.X = SFS().fit_transform(self.X)
        self.Y = SFS(column_wise=True).fit_transform(self.Y)

        self.model = lambda mixing=0.5, regressor=KernelRidge(
            alpha=1e-8
        ), n_components=4, **kwargs: KernelPCovR(
            mixing,
            regressor=regressor,
            n_components=n_components,
            svd_solver=kwargs.pop("svd_solver", "full"),
            **kwargs
        )

    def setUp(self):
        pass


class KernelPCovRErrorTest(KernelPCovRBaseTest):
    def test_lr_with_x_errors(self):
        """
        This test checks that KernelPCovR returns a non-null property prediction
        and that the prediction error increases with `mixing`
        """
        prev_error = -1.0

        for i, mixing in enumerate(np.linspace(0, 1, 6)):
            kpcovr = KernelPCovR(mixing=mixing, n_components=2, tol=1e-12)
            kpcovr.fit(self.X, self.Y)

            error = (
                np.linalg.norm(self.Y - kpcovr.predict(self.X)) ** 2.0
                / np.linalg.norm(self.Y) ** 2.0
            )

            with self.subTest(error=error):
                self.assertFalse(np.isnan(error))
            with self.subTest(error=error, alpha=round(mixing, 4)):
                self.assertGreaterEqual(error, prev_error - self.error_tol)

            prev_error = error

    def test_reconstruction_errors(self):
        """
        This test checks that KernelPCovR returns a non-null reconstructed X
        and that the reconstruction error decreases with `mixing`
        """

        prev_error = 10.0
        prev_x_error = 10.0

        for i, mixing in enumerate(np.linspace(0, 1, 6)):
            kpcovr = KernelPCovR(
                mixing=mixing, n_components=2, fit_inverse_transform=True, tol=1e-12
            )
            kpcovr.fit(self.X, self.Y)

            t = kpcovr.transform(self.X)
            K = kpcovr._get_kernel(self.X)
            x = kpcovr.inverse_transform(t)

            error = np.linalg.norm(K - t @ t.T) ** 2.0 / np.linalg.norm(K) ** 2.0
            x_error = np.linalg.norm(self.X - x) ** 2.0 / np.linalg.norm(self.X) ** 2.0

            with self.subTest(error=error):
                self.assertFalse(np.isnan(error))
            with self.subTest(error=error, alpha=round(mixing, 4)):
                self.assertLessEqual(error, prev_error + self.error_tol)

            with self.subTest(error=x_error):
                self.assertFalse(np.isnan(x_error))
            with self.subTest(error=x_error, alpha=round(mixing, 4)):
                self.assertLessEqual(x_error, prev_x_error + self.error_tol)

            prev_error = error
            prev_x_error = x_error

    def test_kpcovr_error(self):
        for i, mixing in enumerate(np.linspace(0, 1, 6)):
            kpcovr = self.model(
                mixing=mixing,
                regressor=KernelRidge(kernel="rbf", gamma=1.0),
                kernel="rbf",
                gamma=1.0,
                center=False,
            )

            kpcovr.fit(self.X, self.Y)
            K = kpcovr._get_kernel(self.X)

            y = kpcovr.predict(self.X)
            Lkrr = np.linalg.norm(self.Y - y) ** 2 / np.linalg.norm(self.Y) ** 2

            t = kpcovr.transform(self.X)

            w = t @ np.linalg.pinv(t.T @ t, rcond=kpcovr.tol) @ t.T
            Lkpca = np.trace(K - K @ w) / np.trace(K)

            # this is only true for in-sample data
            self.assertTrue(
                np.isclose(
                    kpcovr.score(self.X, self.Y), -sum([Lkpca, Lkrr]), self.error_tol
                )
            )


class KernelPCovRInfrastructureTest(KernelPCovRBaseTest):
    def test_nonfitted_failure(self):
        """
        This test checks that KernelPCovR will raise a `NonFittedError` if
        `transform` is called before the model is fitted
        """
        kpcovr = KernelPCovR(mixing=0.5, n_components=2, tol=1e-12)
        with self.assertRaises(exceptions.NotFittedError):
            _ = kpcovr.transform(self.X)

    def test_no_arg_predict(self):
        """
        This test checks that KernelPCovR will raise a `ValueError` if
        `predict` is called without arguments
        """
        kpcovr = KernelPCovR(mixing=0.5, n_components=2, tol=1e-12)
        kpcovr.fit(self.X, self.Y)
        with self.assertRaises(ValueError):
            _ = kpcovr.predict()

    def test_T_shape(self):
        """
        This test checks that KernelPCovR returns a latent space projection
        consistent with the shape of the input matrix
        """
        n_components = 5
        kpcovr = KernelPCovR(mixing=0.5, n_components=n_components, tol=1e-12)
        kpcovr.fit(self.X, self.Y)
        T = kpcovr.transform(self.X)
        self.assertTrue(check_X_y(self.X, T, multi_output=True))
        self.assertTrue(T.shape[-1] == n_components)

    def test_no_centerer(self):
        """
        tests that when center=False, no centerer exists
        """
        kpcovr = self.model(center=False)
        kpcovr.fit(self.X, self.Y)

        with self.assertRaises(AttributeError):
            _ = getattr(kpcovr, "centerer_")

    def test_centerer(self):
        """
        tests that all functionalities that rely on the centerer work properly
        """

        kpcovr = self.model(center=True)
        kpcovr.fit(self.X, self.Y)

        self.assertTrue(hasattr(kpcovr, "centerer_"))
        _ = kpcovr.predict(self.X)
        _ = kpcovr.transform(self.X)
        _ = kpcovr.score(self.X, self.Y)

    def test_prefit_regressor(self):
        regressor = KernelRidge(alpha=1e-8, kernel="rbf", gamma=0.1)
        regressor.fit(self.X, self.Y)
        kpcovr = self.model(mixing=0.5, regressor=regressor, kernel="rbf", gamma=0.1)
        kpcovr.fit(self.X, self.Y)

        Yhat_regressor = regressor.predict(self.X).reshape(self.X.shape[0], -1)
        W_regressor = regressor.dual_coef_.reshape(self.X.shape[0], -1)

        Yhat_kpcovr = kpcovr.regressor_.predict(self.X).reshape(self.X.shape[0], -1)
        W_kpcovr = kpcovr.regressor_.dual_coef_.reshape(self.X.shape[0], -1)

        self.assertTrue(np.allclose(Yhat_regressor, Yhat_kpcovr))
        self.assertTrue(np.allclose(W_regressor, W_kpcovr))

    def test_regressor_modifications(self):
        regressor = KernelRidge(alpha=1e-8, kernel="rbf", gamma=0.1)
        kpcovr = self.model(mixing=0.5, regressor=regressor, kernel="rbf", gamma=0.1)

        # KPCovR regressor matches the original
        self.assertTrue(regressor.get_params() == kpcovr.regressor.get_params())

        # KPCovR regressor updates its parameters
        # to match the original regressor
        regressor.set_params(gamma=0.2)
        self.assertTrue(regressor.get_params() == kpcovr.regressor.get_params())

        # Fitting regressor outside KPCovR fits the KPCovR regressor
        regressor.fit(self.X, self.Y)
        self.assertTrue(hasattr(kpcovr.regressor, "dual_coef_"))

        # Raise error during KPCovR fit since regressor and KPCovR
        # kernel parameters now inconsistent
        with self.assertRaises(ValueError) as cm:
            kpcovr.fit(self.X, self.Y)
        self.assertTrue(
            str(cm.exception),
            "Kernel parameter mismatch: the regressor has kernel parameters "
            "{kernel: linear, gamma: 0.2, degree: 3, coef0: 1, kernel_params: None}"
            " and KernelPCovR was initialized with kernel parameters "
            "{kernel: linear, gamma: 0.1, degree: 3, coef0: 1, kernel_params: None}",
        )

    def test_incompatible_regressor(self):
        regressor = Ridge(alpha=1e-8)
        regressor.fit(self.X, self.Y)
        kpcovr = self.model(mixing=0.5, regressor=regressor)

        with self.assertRaises(ValueError) as cm:
            kpcovr.fit(self.X, self.Y)
        self.assertTrue(
            str(cm.exception),
            "Regressor must be an instance of `KernelRidge`",
        )

    def test_none_regressor(self):
        kpcovr = KernelPCovR(mixing=0.5, regressor=None)
        kpcovr.fit(self.X, self.Y)
        self.assertTrue(kpcovr.regressor is None)
        self.assertTrue(kpcovr.regressor_ is not None)

    def test_incompatible_coef_shape(self):
        # self.Y is 2D with two targets
        # Don't need to test X shape, since this should
        # be caught by sklearn's _validate_data
        regressor = KernelRidge(alpha=1e-8, kernel="linear")
        regressor.fit(self.X, self.Y[:, 0][:, np.newaxis])
        kpcovr = self.model(mixing=0.5, regressor=regressor)

        # Dimension mismatch
        with self.assertRaises(ValueError) as cm:
            kpcovr.fit(self.X, self.Y[:, 0])
        self.assertTrue(
            str(cm.exception),
            "The regressor coefficients have a dimension incompatible "
            "with the supplied target space. "
            "The coefficients have dimension %d and the targets "
            "have dimension %d" % (regressor.dual_coef_.ndim, self.Y[:, 0].ndim),
        )

        # Shape mismatch (number of targets)
        with self.assertRaises(ValueError) as cm:
            kpcovr.fit(self.X, self.Y)
        self.assertTrue(
            str(cm.exception),
            "The regressor coefficients have a shape incompatible "
            "with the supplied target space. "
            "The coefficients have shape %r and the targets "
            "have shape %r" % (regressor.dual_coef_.shape, self.Y.shape),
        )

    def test_precomputed_regression(self):
        regressor = KernelRidge(alpha=1e-8, kernel="rbf", gamma=0.1)
        regressor.fit(self.X, self.Y)
        Yhat = regressor.predict(self.X)
        W = regressor.dual_coef_.reshape(self.X.shape[0], -1)

        kpcovr1 = self.model(
            mixing=0.5, regressor="precomputed", kernel="rbf", gamma=0.1, n_components=1
        )
        kpcovr1.fit(self.X, Yhat, W)
        t1 = kpcovr1.transform(self.X)

        kpcovr2 = self.model(
            mixing=0.5, regressor=regressor, kernel="rbf", gamma=0.1, n_components=1
        )
        kpcovr2.fit(self.X, self.Y)
        t2 = kpcovr2.transform(self.X)

        self.assertTrue(np.linalg.norm(t1 - t2) < self.error_tol)


class KernelTests(KernelPCovRBaseTest):
    def test_kernel_types(self):
        """
        This test checks that KernelPCovR can handle all kernels passable to
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
                kpcovr = KernelPCovR(
                    mixing=0.5,
                    n_components=2,
                    regressor=KernelRidge(
                        kernel=kernel, **kernel_params.get(kernel, {})
                    ),
                    kernel=kernel,
                    **kernel_params.get(kernel, {})
                )
                kpcovr.fit(self.X, self.Y)

    def test_linear_matches_pcovr(self):
        """
        This test checks that KernelPCovR returns the same results as PCovR when
        using a linear kernel
        """

        ridge = RidgeCV(fit_intercept=False, alphas=np.logspace(-8, 2))
        ridge.fit(self.X, self.Y)

        # common instantiation parameters for the two models
        hypers = dict(
            mixing=0.5,
            n_components=1,
        )

        # computing projection and predicton loss with linear KernelPCovR
        # and use the alpha from RidgeCV for level regression comparisons
        kpcovr = KernelPCovR(
            regressor=KernelRidge(alpha=ridge.alpha_, kernel="linear"),
            kernel="linear",
            fit_inverse_transform=True,
            **hypers
        )
        kpcovr.fit(self.X, self.Y)
        ly = (
            np.linalg.norm(self.Y - kpcovr.predict(self.X)) ** 2.0
            / np.linalg.norm(self.Y) ** 2.0
        )

        # computing projection and predicton loss with PCovR
        ref_pcovr = PCovR(**hypers, regressor=ridge, space="sample")
        ref_pcovr.fit(self.X, self.Y)
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


class KernelPCovRTestSVDSolvers(KernelPCovRBaseTest):
    def test_svd_solvers(self):
        """
        This test checks that PCovR works with all svd_solver modes and assigns
        the right n_components
        """
        for solver in ["arpack", "full", "randomized", "auto"]:
            with self.subTest(solver=solver):
                kpcovr = self.model(tol=1e-12, n_components=None, svd_solver=solver)
                kpcovr.fit(self.X, self.Y)

                if solver == "arpack":
                    self.assertTrue(kpcovr.n_components == self.X.shape[0] - 1)
                else:
                    self.assertTrue(kpcovr.n_components == self.X.shape[0])

    def test_bad_solver(self):
        """
        This test checks that PCovR will not work with a solver that isn't in
        ['arpack', 'full', 'randomized', 'auto']
        """
        with self.assertRaises(ValueError) as cm:
            kpcovr = self.model(svd_solver="bad")
            kpcovr.fit(self.X, self.Y)

        self.assertTrue(str(cm.exception), "Unrecognized svd_solver='bad'" "")

    def test_good_n_components(self):
        """
        This test checks that PCovR will work with any allowed values of
        n_components.
        """

        # this one should pass
        kpcovr = self.model(n_components=0.5, svd_solver="full")
        kpcovr.fit(self.X, self.Y)

        for svd_solver in ["auto", "full"]:
            # this one should pass
            kpcovr = self.model(n_components=2, svd_solver=svd_solver)
            kpcovr.fit(self.X, self.Y)

            # this one should pass
            kpcovr = self.model(n_components="mle", svd_solver=svd_solver)
            kpcovr.fit(self.X, self.Y)

    def test_bad_n_components(self):
        """
        This test checks that PCovR will not work with any prohibited values of
        n_components.
        """

        with self.subTest(type="negative_ncomponents"):
            with self.assertRaises(ValueError) as cm:
                kpcovr = self.model(n_components=-1, svd_solver="auto")
                kpcovr.fit(self.X, self.Y)

            self.assertTrue(
                str(cm.exception),
                "self.n_components=%r must be between 0 and "
                "min(n_samples, n_features)=%r with "
                "svd_solver='%s'"
                % (
                    kpcovr.n_components,
                    self.X.shape[0],
                    kpcovr.svd_solver,
                ),
            )
        with self.subTest(type="0_ncomponents"):
            with self.assertRaises(ValueError) as cm:
                kpcovr = self.model(n_components=0, svd_solver="randomized")
                kpcovr.fit(self.X, self.Y)

            self.assertTrue(
                str(cm.exception),
                "self.n_components=%r must be between 1 and "
                "min(n_samples, n_features)=%r with "
                "svd_solver='%s'"
                % (
                    kpcovr.n_components,
                    self.X.shape[0],
                    kpcovr.svd_solver,
                ),
            )
        with self.subTest(type="arpack_X_ncomponents"):
            with self.assertRaises(ValueError) as cm:
                kpcovr = self.model(n_components=self.X.shape[0], svd_solver="arpack")
                kpcovr.fit(self.X, self.Y)
            self.assertTrue(
                str(cm.exception),
                "self.n_components=%r must be strictly less than "
                "min(n_samples, n_features)=%r with "
                "svd_solver='%s'"
                % (
                    kpcovr.n_components,
                    self.X.shape[0],
                    kpcovr.svd_solver,
                ),
            )

        for svd_solver in ["auto", "full"]:
            with self.subTest(type="pi_ncomponents"):
                with self.assertRaises(ValueError) as cm:
                    kpcovr = self.model(n_components=np.pi, svd_solver=svd_solver)
                    kpcovr.fit(self.X, self.Y)
                self.assertTrue(
                    str(cm.exception),
                    "self.n_components=%r must be of type int "
                    "when greater than or equal to 1, was of type=%r"
                    % (kpcovr.n_components, type(kpcovr.n_components)),
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
