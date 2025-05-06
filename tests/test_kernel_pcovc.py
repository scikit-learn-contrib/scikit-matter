import unittest

import numpy as np
from sklearn import exceptions
from sklearn.calibration import LinearSVC
from sklearn.datasets import load_breast_cancer as get_dataset
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.utils.validation import check_X_y
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier

from skmatter.decomposition import PCovC, KernelPCovC

class KernelPCovCBaseTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_state = np.random.RandomState(0)

        self.error_tol = 1e-6

        self.X, self.Y = get_dataset(return_X_y=True)

        # for the sake of expedience, only use a subset of the dataset
        idx = self.random_state.choice(len(self.X), 100)
        self.X = self.X[idx]
        self.Y = self.Y[idx]

        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

        self.model = lambda mixing=0.5, classifier=LinearSVC(), n_components=4, **kwargs: KernelPCovC(
            mixing=mixing,
            classifier=classifier,
            n_components=n_components,
            svd_solver=kwargs.pop("svd_solver", "full"),
            **kwargs,
        )

    def setUp(self):
        pass


class KernelPCovCErrorTest(KernelPCovCBaseTest):
    def test_cl_with_x_errors(self):
        """
        Check that KernelPCovC returns a non-null property prediction
        and that the prediction error increases with `mixing`
        """
        prev_error = -1.0

        for mixing in np.linspace(0, 1, 6):
            kpcovc = KernelPCovC(mixing=mixing, n_components=4, tol=1e-12)
            kpcovc.fit(self.X, self.Y)

            error = (
                np.linalg.norm(self.Y - kpcovc.predict(self.X)) ** 2.0
                / np.linalg.norm(self.Y) ** 2.0
            )

            with self.subTest(error=error):
                self.assertFalse(np.isnan(error))
            with self.subTest(error=error, alpha=round(mixing, 4)):
                self.assertGreaterEqual(error, prev_error - self.error_tol)

            prev_error = error

    def test_reconstruction_errors(self):
        """Check that KernelPCovC returns a non-null reconstructed X and that the
        reconstruction error decreases with `mixing`.
        """
        prev_error = 10.0
        prev_x_error = 10.0

        for mixing in np.linspace(0, 1, 6):
            kpcovc = KernelPCovC(
                mixing=mixing, n_components=4, fit_inverse_transform=True, tol=1e-12
            )
            kpcovc.fit(self.X, self.Y)

            t = kpcovc.transform(self.X)
            K = kpcovc._get_kernel(self.X)
            x = kpcovc.inverse_transform(t)

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

    def test_kpcovc_error(self):
        for mixing in np.linspace(0, 1, 6):
            kpcovc = self.model(
                mixing=mixing,
                classifier=LinearSVC(),
                kernel="rbf",
                gamma=1.0,
                center=False,
            )

            kpcovc.fit(self.X, self.Y)
            K = kpcovc._get_kernel(self.X)

            y = kpcovc.predict(self.X)
            Lkrr = np.linalg.norm(self.Y - y) ** 2 / np.linalg.norm(self.Y) ** 2

            t = kpcovc.transform(self.X)

            w = t @ np.linalg.pinv(t.T @ t, rcond=kpcovc.tol) @ t.T
            Lkpca = np.trace(K - K @ w) / np.trace(K)

            # this is only true for in-sample data
            self.assertTrue(
                np.isclose(
                    kpcovc.score(self.X, self.Y), -sum([Lkpca, Lkrr]), self.error_tol
                )
            )


class KernelPCovCInfrastructureTest(KernelPCovCBaseTest):
    def test_nonfitted_failure(self):
        """
        Check that KernelPCovC will raise a `NonFittedError` if
        `transform` is called before the model is fitted
        """
        kpcovc = KernelPCovC(mixing=0.5, n_components=4, tol=1e-12)
        with self.assertRaises(exceptions.NotFittedError):
            _ = kpcovc.transform(self.X)

    def test_no_arg_predict(self):
        """
        Check that KernelPCovC will raise a `ValueError` if
        `predict` is called without arguments
        """
        kpcovc = KernelPCovC(mixing=0.5, n_components=4, tol=1e-12)
        kpcovc.fit(self.X, self.Y)
        with self.assertRaises(ValueError):
            _ = kpcovc.predict()

    def test_T_shape(self):
        """
        Check that KernelPCovC returns a latent space projection
        consistent with the shape of the input matrix
        """
        n_components = 5
        kpcovc = KernelPCovC(mixing=0.5, n_components=n_components, tol=1e-12)
        kpcovc.fit(self.X, self.Y)
        T = kpcovc.transform(self.X)
        self.assertTrue(check_X_y(self.X, T, multi_output=True))
        self.assertTrue(T.shape[-1] == n_components)

    def test_no_centerer(self):
        """Tests that when center=False, no centerer exists."""
        kpcovc = self.model(center=False)
        kpcovc.fit(self.X, self.Y)

        with self.assertRaises(AttributeError):
            kpcovc.centerer_

    def test_centerer(self):
        """Tests that all functionalities that rely on the centerer work properly."""
        kpcovc = self.model(center=True)
        kpcovc.fit(self.X, self.Y)

        self.assertTrue(hasattr(kpcovc, "centerer_"))
        _ = kpcovc.predict(self.X)
        _ = kpcovc.transform(self.X)
        _ = kpcovc.score(self.X, self.Y)

    def test_prefit_classifier(self):
        classifier = LinearSVC()
        #this fails since we are trying to call decision_function(K) on a classifier fitted with X
        #see line 340 of kernel_pcovr
        classifier.fit(self.X, self.Y)
        print(classifier.n_features_in_)
        kpcovc = self.model(mixing=0.5, classifier=classifier, kernel="rbf", gamma=0.1)
        kpcovc.fit(self.X, self.Y)

        Yhat_classifier = classifier.predict(self.X).reshape(self.X.shape[0], -1)
        W_classifier = classifier.coef_.reshape(self.X.shape[1], -1)

        Yhat_kpcovc = kpcovc.classifier_.predict(self.X).reshape(self.X.shape[0], -1)
        W_kpcovc = kpcovc.classifier_.coef_.reshape(self.X.shape[1], -1)

        self.assertTrue(np.allclose(Yhat_classifier, Yhat_kpcovc))
        self.assertTrue(np.allclose(W_classifier, W_kpcovc))

    def test_classifier_modifications(self):
        classifier = LinearSVC()
        kpcovc = self.model(mixing=0.5, classifier=classifier, kernel="rbf", gamma=0.1)

        # KPCovC classifier matches the original
        self.assertTrue(classifier.get_params() == kpcovc.classifier.get_params())

        # KPCovC classifier updates its parameters
        # to match the original classifier
        classifier.set_params(random_state=3)
        self.assertTrue(classifier.get_params() == kpcovc.classifier.get_params())

        # Fitting classifier outside KPCovC fits the KPCovC classifier
        classifier.fit(self.X, self.Y)
        self.assertTrue(hasattr(kpcovc.classifier, "coef_"))

        # Raise error during KPCovC fit since classifier and KPCovC
        # kernel parameters now inconsistent
        with self.assertRaises(ValueError) as cm:
            kpcovc.fit(self.X, self.Y)
        self.assertTrue(
            str(cm.exception),
            "Kernel parameter mismatch: the regressor has kernel parameters "
            "{kernel: linear, gamma: 0.2, degree: 3, coef0: 1, kernel_params: None}"
            " and KernelPCovR was initialized with kernel parameters "
            "{kernel: linear, gamma: 0.1, degree: 3, coef0: 1, kernel_params: None}",
        )

    def test_incompatible_classifier(self):
        classifier = RidgeClassifier()
        classifier.fit(self.X, self.Y)
        kpcovc = self.model(mixing=0.5, classifier=classifier)

        with self.assertRaises(ValueError) as cm:
            kpcovc.fit(self.X, self.Y)
        self.assertTrue(
            str(cm.exception),
            "Regressor must be an instance of `KernelRidge`",
        )

    def test_none_classifier(self):
        kpcovc = KernelPCovC(mixing=0.5, classifier=None)
        kpcovc.fit(self.X, self.Y)
        self.assertTrue(kpcovc.classifier is None)
        self.assertTrue(kpcovc.classifier_ is not None)

    def test_incompatible_coef_shape(self):
        classifier1 = LinearSVC()

        # Modify Y to be multiclass
        Y_multiclass = self.Y.copy()
        Y_multiclass[0] = 2

        classifier1.fit(self.X, Y_multiclass)
        kpcovc1 = self.model(mixing=0.5, classifier=classifier1, kernel="rbf")

        # Binary classification shape mismatch
        with self.assertRaises(ValueError) as cm:
            kpcovc1.fit(self.X, self.Y)
        self.assertEqual(
            str(cm.exception),
            "For binary classification, expected classifier coefficients "
            "to have shape (1, %d) but got shape %r" 
            % (self.X.shape[1], classifier1.coef_.shape)
        )
        
        classifier2 = LinearSVC()
        classifier2.fit(self.X, self.Y)
        kpcovc2 = self.model(mixing=0.5, classifier=classifier2, kernel="rbf")

        # Multiclass classification shape mismatch 
        with self.assertRaises(ValueError) as cm:
            kpcovc2.fit(self.X, Y_multiclass)
        self.assertEqual(
            str(cm.exception),
            "For multiclass classification, expected classifier coefficients "
            "to have shape (%d, %d) but got shape %r" 
            % (len(np.unique(Y_multiclass)), self.X.shape[1], classifier2.coef_.shape)
        )

    def test_precomputed_classification(self):
        classifier = LinearSVC()
        classifier.fit(self.X, self.Y)
        Yhat = classifier.predict(self.X)
        W = classifier.coef_.reshape(self.X.shape[1], -1)

        kpcovc1 = self.model(
            mixing=0.5, classifier="precomputed", kernel="rbf", gamma=0.1, n_components=1
        )
        kpcovc1.fit(self.X, Yhat, W)
        t1 = kpcovc1.transform(self.X)

        kpcovc2 = self.model(
            mixing=0.5, classifier=classifier, kernel="rbf", gamma=0.1, n_components=1
        )
        kpcovc2.fit(self.X, self.Y)
        t2 = kpcovc2.transform(self.X)

        self.assertTrue(np.linalg.norm(t1 - t2) < self.error_tol)

class KernelTests(KernelPCovCBaseTest):
    def test_kernel_types(self):
        """Check that KernelPCovC can handle all kernels passable to sklearn
        kernel classes, including callable kernels
        """

        def _linear_kernel(X, Y):
            return X @ Y.T

        # kernel_params = {
        #     "poly": {"degree": 2},
        #     "rbf": {"gamma": 3.0},
        #     "sigmoid": {"gamma": 3.0, "coef0": 0.5},
        # }
        for kernel in ["linear", "poly", "rbf", "sigmoid", "cosine", _linear_kernel]:
            with self.subTest(kernel=kernel):
                kpcovc = KernelPCovC(
                    mixing=0.5,
                    n_components=2,
                    classifier=LinearSVC(),
                    kernel=kernel,
                    degree=2,
                    gamma=3.0,
                    coef0=0.5
                )
                kpcovc.fit(self.X, self.Y)

    def test_linear_matches_pcovc(self):
        """Check that KernelPCovC returns the same results as PCovC when using a linear
        kernel.
        """
        svc = LinearSVC()
        svc.fit(self.X, self.Y)

        # common instantiation parameters for the two models
        hypers = dict(
            mixing=0.5,
            n_components=1,
        )

        # computing projection and predicton loss with linear KernelPCovC
        # and use the alpha from RidgeCV for level regression comparisons
        kpcovc = KernelPCovC(
            classifier=LinearSVC(),
            kernel="linear",
            gamma='scale',
            **hypers,
        )
        kpcovc.fit(self.X, self.Y)
        ly = (
            np.linalg.norm(self.Y - kpcovc.predict(self.X)) ** 2.0
            / np.linalg.norm(self.Y) ** 2.0
        )

        # computing projection and predicton loss with PCovC
        ref_pcovc = PCovC(**hypers, classifier=svc)
        ref_pcovc.fit(self.X, self.Y)
        ly_ref = (
            np.linalg.norm(self.Y - ref_pcovc.predict(self.X)) ** 2.0
            / np.linalg.norm(self.Y) ** 2.0
        )

        t_ref = ref_pcovc.transform(self.X)
        t = kpcovc.transform(self.X)

        K = kpcovc._get_kernel(self.X)

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

class KernelPCovCTestSVDSolvers(KernelPCovCBaseTest):
    def test_svd_solvers(self):
        """
        Check that KPCovC works with all svd_solver modes and assigns
        the right n_components
        """
        for solver in ["arpack", "full", "randomized", "auto"]:
            with self.subTest(solver=solver):
                kpcovc = self.model(tol=1e-12, n_components=None, svd_solver=solver)
                kpcovc.fit(self.X, self.Y)

                if solver == "arpack":
                    self.assertTrue(kpcovc.n_components_ == self.X.shape[0] - 1)
                else:
                    self.assertTrue(kpcovc.n_components_ == self.X.shape[0])

        n_component_solvers = {
            "mle": "full",
            int(0.75 * max(self.X.shape)): "randomized",
            0.1: "full",
        }
        for n_components, solver in n_component_solvers.items():
            with self.subTest(solver=solver, n_components=n_components):
                kpcovc = self.model(
                    tol=1e-12, n_components=n_components, svd_solver="auto"
                )
                if solver == "randomized":
                    n_copies = (501 // max(self.X.shape)) + 1
                    X = np.hstack(np.repeat(self.X.copy(), n_copies)).reshape(
                        self.X.shape[0] * n_copies, -1
                    )
                    Y = np.hstack(np.repeat(self.Y.copy(), n_copies)).reshape(
                        self.X.shape[0] * n_copies, -1
                    )
                    kpcovc.fit(X, Y)
                else:
                    kpcovc.fit(self.X, self.Y)

                self.assertTrue(kpcovc.fit_svd_solver_ == solver)

    def test_bad_solver(self):
        """
        Check that KPCovC will not work with a solver that isn't in
        ['arpack', 'full', 'randomized', 'auto']
        """
        with self.assertRaises(ValueError) as cm:
            kpcovc = self.model(svd_solver="bad")
            kpcovc.fit(self.X, self.Y)

        self.assertTrue(str(cm.exception), "Unrecognized svd_solver='bad'" "")

    def test_good_n_components(self):
        """Check that KPCovC will work with any allowed values of n_components."""
        # this one should pass
        kpcovc = self.model(n_components=0.5, svd_solver="full")
        kpcovc.fit(self.X, self.Y)

        for svd_solver in ["auto", "full"]:
            # this one should pass
            kpcovc = self.model(n_components=2, svd_solver=svd_solver)
            kpcovc.fit(self.X, self.Y)

            # this one should pass
            kpcovc = self.model(n_components="mle", svd_solver=svd_solver)
            kpcovc.fit(self.X, self.Y)

    def test_bad_n_components(self):
        """Check that KPCovC will not work with any prohibited values of n_components."""
        with self.subTest(type="negative_ncomponents"):
            with self.assertRaises(ValueError) as cm:
                kpcovc = self.model(n_components=-1, svd_solver="auto")
                kpcovc.fit(self.X, self.Y)

            self.assertTrue(
                str(cm.exception),
                "self.n_components=%r must be between 0 and "
                "min(n_samples, n_features)=%r with "
                "svd_solver='%s'"
                % (
                    kpcovc.n_components,
                    self.X.shape[0],
                    kpcovc.svd_solver,
                ),
            )
        with self.subTest(type="0_ncomponents"):
            with self.assertRaises(ValueError) as cm:
                kpcovc = self.model(n_components=0, svd_solver="randomized")
                kpcovc.fit(self.X, self.Y)

            self.assertTrue(
                str(cm.exception),
                "self.n_components=%r must be between 1 and "
                "min(n_samples, n_features)=%r with "
                "svd_solver='%s'"
                % (
                    kpcovc.n_components,
                    self.X.shape[0],
                    kpcovc.svd_solver,
                ),
            )
        with self.subTest(type="arpack_X_ncomponents"):
            with self.assertRaises(ValueError) as cm:
                kpcovc = self.model(n_components=self.X.shape[0], svd_solver="arpack")
                kpcovc.fit(self.X, self.Y)
            self.assertTrue(
                str(cm.exception),
                "self.n_components=%r must be strictly less than "
                "min(n_samples, n_features)=%r with "
                "svd_solver='%s'"
                % (
                    kpcovc.n_components,
                    self.X.shape[0],
                    kpcovc.svd_solver,
                ),
            )

        for svd_solver in ["auto", "full"]:
            with self.subTest(type="pi_ncomponents"):
                with self.assertRaises(ValueError) as cm:
                    kpcovc = self.model(n_components=np.pi, svd_solver=svd_solver)
                    kpcovc.fit(self.X, self.Y)
                self.assertTrue(
                    str(cm.exception),
                    "self.n_components=%r must be of type int "
                    "when greater than or equal to 1, was of type=%r"
                    % (kpcovc.n_components, type(kpcovc.n_components)),
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
