import unittest

import numpy as np
from sklearn import exceptions
from sklearn.calibration import LinearSVC
from sklearn.datasets import load_breast_cancer as get_dataset
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.validation import check_X_y
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics.pairwise import pairwise_kernels

from skmatter.decomposition import KernelPCovC


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

        self.model = (
            lambda mixing=0.5,
            classifier=LogisticRegression(),
            n_components=4,
            **kwargs: KernelPCovC(
                mixing=mixing,
                classifier=classifier,
                n_components=n_components,
                svd_solver=kwargs.pop("svd_solver", "full"),
                **kwargs,
            )
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

    def test_cl_with_t_errors(self):
        """Check that KernelPCovC returns a non-null property prediction from
        the latent space projection and that the prediction error increases with
        `mixing`.
        """
        prev_error = -1.0

        for mixing in np.linspace(0, 1, 6):
            kpcovc = self.model(mixing=mixing, n_components=2, tol=1e-12)
            kpcovc.fit(self.X, self.Y)

            T = kpcovc.transform(self.X)

            error = (
                np.linalg.norm(self.Y - kpcovc.predict(T=T)) ** 2.0
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
        prev_error = 1.0

        for mixing in np.linspace(0, 1, 11):
            kpcovc = self.model(
                mixing=mixing, n_components=2, tol=1e-12, fit_inverse_transform=True
            )
            kpcovc.fit(self.X, self.Y)

            Xr = kpcovc.inverse_transform(kpcovc.transform(self.X))
            error = np.linalg.norm(self.X - Xr) ** 2.0 / np.linalg.norm(self.X) ** 2.0

            with self.subTest(error=error):
                self.assertFalse(np.isnan(error))
            with self.subTest(error=error, alpha=round(mixing, 4)):
                self.assertLessEqual(error, prev_error + self.error_tol)

            prev_error = error


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
        self.assertTrue(check_X_y(self.X, T, multi_output=True) == (self.X, T))
        self.assertTrue(T.shape[-1] == n_components)

    def test_Z_shape(self):
        """Check that KPCovC returns an evidence matrix consistent with the number
        of samples and the number of classes.
        """
        n_components = 5
        kpcovc = self.model(n_components=n_components, tol=1e-12)
        kpcovc.fit(self.X, self.Y)

        # Shape (n_samples, ) for binary classifcation
        Z_binary = kpcovc.decision_function(self.X)

        self.assertEqual(Z_binary.ndim, 1)
        self.assertEqual(Z_binary.shape[0], self.X.shape[0])

        # Shape (n_samples, n_classes) for multiclass classification
        kpcovc.fit(self.X, np.random.randint(0, 3, size=self.X.shape[0]))
        Z_multi = kpcovc.decision_function(self.X)

        self.assertEqual(Z_multi.ndim, 2)
        self.assertEqual(Z_multi.shape, (self.X.shape[0], len(kpcovc.classes_)))

    def test_decision_function(self):
        """Check that KPCovC's decision_function works when only T is
        provided and throws an error when appropriate.
        """
        kpcovc = self.model(center=True)
        kpcovc.fit(self.X, self.Y)

        with self.assertRaises(ValueError) as cm:
            _ = kpcovc.decision_function()
        self.assertEqual(
            str(cm.exception),
            "Either X or T must be supplied.",
        )

        _ = kpcovc.decision_function(self.X)
        T = kpcovc.transform(self.X)
        _ = kpcovc.decision_function(T=T)

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
        # in KPCovC, our classifiers don't compute the kernel for us, hence we only
        # allow prefit classifiers on K and y
        kernel_params = {"kernel": "rbf", "gamma": 0.1, "degree": 3, "coef0": 0}
        K = pairwise_kernels(self.X, metric="rbf", filter_params=True, **kernel_params)

        classifier = LinearSVC()
        classifier.fit(K, self.Y)

        kpcovc = KernelPCovC(mixing=0.5, classifier=classifier, **kernel_params)
        kpcovc.fit(self.X, self.Y)

        Z_classifier = classifier.decision_function(K)
        W_classifier = classifier.coef_.T

        Z_kpcovc = kpcovc.z_classifier_.decision_function(K)
        W_kpcovc = kpcovc.z_classifier_.coef_.T

        self.assertTrue(np.allclose(Z_classifier, Z_kpcovc))
        self.assertTrue(np.allclose(W_classifier, W_kpcovc))

    def test_classifier_modifications(self):
        classifier = RidgeClassifier()
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

    def test_incompatible_classifier(self):
        classifier = GaussianNB()
        classifier.fit(self.X, self.Y)
        kpcovc = self.model(mixing=0.5, classifier=classifier)

        with self.assertRaises(ValueError) as cm:
            kpcovc.fit(self.X, self.Y)
        self.assertEqual(
            str(cm.exception),
            "Classifier must be an instance of "
            "`LogisticRegression`, `LogisticRegressionCV`, `LinearSVC`, "
            "`LinearDiscriminantAnalysis`, `RidgeClassifier`, "
            "`RidgeClassifierCV`, `SGDClassifier`, `Perceptron`, "
            "or `precomputed`",
        )

    def test_none_classifier(self):
        kpcovc = KernelPCovC(mixing=0.5, classifier=None)
        kpcovc.fit(self.X, self.Y)
        self.assertTrue(kpcovc.classifier is None)
        self.assertTrue(kpcovc.classifier_ is not None)

    def test_incompatible_coef_shape(self):
        kernel_params = {"kernel": "sigmoid", "gamma": 0.1, "degree": 3, "coef0": 0}
        K = pairwise_kernels(
            self.X, metric="sigmoid", filter_params=True, **kernel_params
        )

        cl_multi = LinearSVC()
        cl_multi.fit(K, np.random.randint(0, 3, size=self.X.shape[0]))
        kpcovc_binary = self.model(mixing=0.5, classifier=cl_multi)

        # Binary classification shape mismatch
        with self.assertRaises(ValueError) as cm:
            kpcovc_binary.fit(self.X, self.Y)
        self.assertEqual(
            str(cm.exception),
            "For binary classification, expected classifier coefficients "
            "to have shape (1, %d) but got shape %r"
            % (K.shape[1], cl_multi.coef_.shape),
        )

        cl_binary = LinearSVC()
        cl_binary.fit(K, self.Y)
        kpcovc_multi = self.model(mixing=0.5, classifier=cl_binary)

        # Multiclass classification shape mismatch
        with self.assertRaises(ValueError) as cm:
            kpcovc_multi.fit(self.X, np.random.randint(0, 3, size=self.X.shape[0]))
        self.assertEqual(
            str(cm.exception),
            "For multiclass classification, expected classifier coefficients "
            "to have shape (%d, %d) but got shape %r"
            % (len(kpcovc_multi.classes_), K.shape[1], cl_binary.coef_.shape),
        )

    def test_precomputed_classification(self):
        kernel_params = {"kernel": "rbf", "gamma": 0.1, "degree": 3, "coef0": 0}
        K = pairwise_kernels(self.X, metric="rbf", filter_params=True, **kernel_params)

        classifier = LogisticRegression()
        classifier.fit(K, self.Y)

        W = classifier.coef_.T
        kpcovc1 = self.model(mixing=0.5, classifier="precomputed", **kernel_params)
        kpcovc1.fit(self.X, self.Y, W)
        t1 = kpcovc1.transform(self.X)

        kpcovc2 = self.model(mixing=0.5, classifier=classifier, **kernel_params)
        kpcovc2.fit(self.X, self.Y)
        t2 = kpcovc2.transform(self.X)

        self.assertTrue(np.linalg.norm(t1 - t2) < self.error_tol)

        # Now check for match when W is not passed:
        kpcovc3 = self.model(mixing=0.5, classifier="precomputed", **kernel_params)
        kpcovc3.fit(self.X, self.Y)
        t3 = kpcovc3.transform(self.X)

        self.assertTrue(np.linalg.norm(t3 - t2) < self.error_tol)
        self.assertTrue(np.linalg.norm(t3 - t1) < self.error_tol)

    def test_scale_z_parameter(self):
        """Check that changing scale_z changes the eigendecomposition."""
        kpcovc_scaled = self.model(scale_z=True)
        kpcovc_scaled.fit(self.X, self.Y)

        kpcovc_unscaled = self.model(scale_z=False)
        kpcovc_unscaled.fit(self.X, self.Y)
        assert not np.allclose(kpcovc_scaled.pkt_, kpcovc_unscaled.pkt_)


class KernelTests(KernelPCovCBaseTest):
    def test_kernel_types(self):
        """Check that KernelPCovC can handle all kernels passable to sklearn
        kernel classes, including callable kernels
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
                kpcovc = KernelPCovC(
                    mixing=0.5,
                    n_components=2,
                    classifier=LogisticRegression(),
                    kernel=kernel,
                    **kernel_params.get(kernel, {}),
                )
                kpcovc.fit(self.X, self.Y)


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

        self.assertEqual(str(cm.exception), "Unrecognized svd_solver='bad'")

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
        """Check that KPCovC will not work with any prohibited values of n_components"""
        with self.subTest(type="negative_ncomponents"):
            with self.assertRaises(ValueError) as cm:
                kpcovc = self.model(n_components=-1, svd_solver="auto")
                kpcovc.fit(self.X, self.Y)

            self.assertEqual(
                str(cm.exception),
                "n_components=%r must be between 1 and "
                "n_samples=%r with "
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

            self.assertEqual(
                str(cm.exception),
                "n_components=%r must be between 1 and "
                "n_samples=%r with "
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
            self.assertEqual(
                str(cm.exception),
                "n_components=%r must be strictly less than "
                "n_samples=%r with "
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
                self.assertEqual(
                    str(cm.exception),
                    "n_components=%r must be of type int "
                    "when greater than or equal to 1, was of type=%r"
                    % (kpcovc.n_components, type(kpcovc.n_components)),
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
