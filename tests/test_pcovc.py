import unittest
import warnings

import numpy as np
from sklearn import exceptions
from sklearn.datasets import load_breast_cancer as get_dataset
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y

from skmatter.decomposition import PCovC


class PCovCBaseTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = (
            lambda mixing=0.5, classifier=LogisticRegression(), **kwargs: PCovC(
                mixing=mixing, classifier=classifier, **kwargs
            )
        )

        self.error_tol = 1e-5

        self.X, self.Y = get_dataset(return_X_y=True)

        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

    def setUp(self):
        pass


class PCovCErrorTest(PCovCBaseTest):
    def test_against_pca(self):
        """Tests that mixing = 1.0 corresponds to PCA."""
        pcovc = PCovC(
            mixing=1.0, n_components=2, space="feature", svd_solver="full"
        ).fit(self.X, self.Y)

        pca = PCA(n_components=2, svd_solver="full").fit(self.X)

        # tests that the SVD is equivalent
        self.assertTrue(np.allclose(pca.singular_values_, pcovc.singular_values_))
        self.assertTrue(np.allclose(pca.explained_variance_, pcovc.explained_variance_))

        T_pcovc = pcovc.transform(self.X)
        T_pca = pca.transform(self.X)

        # tests that the projections are equivalent
        self.assertLessEqual(
            np.linalg.norm(T_pcovc @ T_pcovc.T - T_pca @ T_pca.T), 1e-8
        )

    def test_simple_reconstruction(self):
        """Check that PCovC with a full eigendecomposition at mixing=1 can fully
        reconstruct the input matrix.
        """
        for space in ["feature", "sample", "auto"]:
            with self.subTest(space=space):
                pcovc = self.model(
                    mixing=1.0, n_components=self.X.shape[-1], space=space
                )
                pcovc.fit(self.X, self.Y)
                Xr = pcovc.inverse_transform(pcovc.transform(self.X))
                self.assertLessEqual(
                    np.linalg.norm(self.X - Xr) ** 2.0 / np.linalg.norm(self.X) ** 2.0,
                    self.error_tol,
                )

    def test_simple_prediction(self):
        """
        Check that PCovC with a full eigendecomposition at mixing=0
        can fully reconstruct the input properties.
        """
        for space in ["feature", "sample", "auto"]:
            with self.subTest(space=space):
                pcovc = self.model(mixing=0.0, n_components=2, space=space)

                pcovc.classifier.fit(self.X, self.Y)
                Yhat = pcovc.classifier.predict(self.X)

                pcovc.fit(self.X, self.Y)
                Yp = pcovc.predict(self.X)
                self.assertLessEqual(
                    np.linalg.norm(Yp - Yhat) ** 2.0 / np.linalg.norm(Yhat) ** 2.0,
                    self.error_tol,
                )

    def test_cl_with_x_errors(self):
        """
        Check that PCovC returns a non-null property prediction
        and that the prediction error increases with `mixing`
        """
        prev_error = -1.0

        for mixing in np.linspace(0, 1, 11):
            pcovc = self.model(mixing=mixing, n_components=2, tol=1e-12)
            pcovc.fit(self.X, self.Y)

            Yp = pcovc.predict(X=self.X)
            error = np.linalg.norm(self.Y - Yp) ** 2.0 / np.linalg.norm(self.Y) ** 2.0

            with self.subTest(error=error):
                self.assertFalse(np.isnan(error))
            with self.subTest(error=error, alpha=round(mixing, 4)):
                self.assertGreaterEqual(error, prev_error - self.error_tol)

            prev_error = error

    def test_cl_with_t_errors(self):
        """Check that PCovc returns a non-null property prediction from the latent space
        projection and that the prediction error increases with `mixing`.
        """
        prev_error = -1.0

        for mixing in np.linspace(0, 1, 11):
            pcovc = self.model(mixing=mixing, n_components=2, tol=1e-12)
            pcovc.fit(self.X, self.Y)

            T = pcovc.transform(self.X)
            Yp = pcovc.predict(T=T)
            error = np.linalg.norm(self.Y - Yp) ** 2.0 / np.linalg.norm(self.Y) ** 2.0

            with self.subTest(error=error):
                self.assertFalse(np.isnan(error))
            with self.subTest(error=error, alpha=round(mixing, 4)):
                self.assertGreaterEqual(error, prev_error - self.error_tol)

            prev_error = error

    def test_reconstruction_errors(self):
        """Check that PCovC returns a non-null reconstructed X and that the
        reconstruction error decreases with `mixing`.
        """
        prev_error = 1.0

        for mixing in np.linspace(0, 1, 11):
            pcovc = self.model(mixing=mixing, n_components=2, tol=1e-12)
            pcovc.fit(self.X, self.Y)

            Xr = pcovc.inverse_transform(pcovc.transform(self.X))
            error = np.linalg.norm(self.X - Xr) ** 2.0 / np.linalg.norm(self.X) ** 2.0

            with self.subTest(error=error):
                self.assertFalse(np.isnan(error))
            with self.subTest(error=error, alpha=round(mixing, 4)):
                self.assertLessEqual(error, prev_error + self.error_tol)

            prev_error = error


class PCovCSpaceTest(PCovCBaseTest):
    def test_select_feature_space(self):
        """
        Check that PCovC implements the feature space version
        when :math:`n_{features} < n_{samples}``.
        """
        pcovc = self.model(n_components=2, tol=1e-12)
        pcovc.fit(self.X, self.Y)

        self.assertTrue(pcovc.space_ == "feature")

    def test_select_sample_space(self):
        """
        Check that PCovC implements the sample space version
        when :math:`n_{features} > n_{samples}``.
        """
        pcovc = self.model(n_components=2, tol=1e-12)

        n_samples = self.X.shape[1] - 1
        pcovc.fit(self.X[:n_samples], self.Y[:n_samples])

        self.assertTrue(pcovc.space_ == "sample")

    def test_bad_space(self):
        """
        Check that PCovC raises a ValueError when a non-valid
        space is designated.
        """
        with self.assertRaises(ValueError):
            pcovc = self.model(n_components=2, tol=1e-12, space="bad")
            pcovc.fit(self.X, self.Y)

    def test_override_spaceselection(self):
        """
        Check that PCovC implements the space provided in the
        constructor, overriding that chosen by the input dimensions.
        """
        pcovc = self.model(n_components=2, tol=1e-12, space="sample")
        pcovc.fit(self.X, self.Y)

        self.assertTrue(pcovc.space_ == "sample")

    def test_spaces_equivalent(self):
        """
        Check that the results from PCovC, regardless of the space,
        are equivalent.
        """
        for alpha in np.linspace(0.01, 0.99, 11):
            with self.subTest(alpha=alpha, type="prediction"):
                pcovc_ss = self.model(
                    n_components=2, mixing=alpha, tol=1e-12, space="sample"
                )
                pcovc_ss.fit(self.X, self.Y)

                pcovc_fs = self.model(
                    n_components=2, mixing=alpha, tol=1e-12, space="feature"
                )
                pcovc_fs.fit(self.X, self.Y)

                self.assertTrue(
                    np.allclose(
                        pcovc_ss.predict(self.X),
                        pcovc_fs.predict(self.X),
                        self.error_tol,
                    )
                )

            with self.subTest(alpha=alpha, type="reconstruction"):
                pcovc_ss = self.model(
                    n_components=2, mixing=alpha, tol=1e-12, space="sample"
                )
                pcovc_ss.fit(self.X, self.Y)

                pcovc_fs = self.model(
                    n_components=2, mixing=alpha, tol=1e-12, space="feature"
                )
                pcovc_fs.fit(self.X, self.Y)

                # if(alpha > 0.5):
                #     print(np.isclose(
                #             pcovc_ss.transform(self.X),
                #             pcovc_fs.transform(self.X),
                #             self.error_tol
                #         ))

                # failing for all alpha values
                # so these are similar (within approximately 0.001), but not exactly the same.
                # I think this is because transform and inverse_transform depend on Pxt and Ptx,
                # which in turn depend on Z, which is a matrix of class likelihoods (so maybe there is some rounding problems)
                self.assertTrue(
                    np.allclose(
                        pcovc_ss.inverse_transform(pcovc_ss.transform(self.X)),
                        pcovc_fs.inverse_transform(pcovc_fs.transform(self.X)),
                        self.error_tol,
                    )
                )


class PCovCTestSVDSolvers(PCovCBaseTest):
    def test_svd_solvers(self):
        """
        Check that PCovC works with all svd_solver modes and assigns
        the right n_components
        """
        for solver in ["arpack", "full", "randomized", "auto"]:
            with self.subTest(solver=solver):
                pcovc = self.model(tol=1e-12, svd_solver=solver)
                pcovc.fit(self.X, self.Y)

                if solver == "arpack":
                    self.assertTrue(pcovc.n_components_ == min(self.X.shape) - 1)
                else:
                    self.assertTrue(pcovc.n_components_ == min(self.X.shape))

    def test_bad_solver(self):
        """
        Check that PCovC will not work with a solver that isn't in
        ['arpack', 'full', 'randomized', 'auto']
        """
        for space in ["feature", "sample"]:
            with self.assertRaises(ValueError) as cm:
                pcovc = self.model(svd_solver="bad", space=space)
                pcovc.fit(self.X, self.Y)

            self.assertEqual(str(cm.exception), "Unrecognized svd_solver='bad'" "")

    def test_good_n_components(self):
        """Check that PCovC will work with any allowed values of n_components."""
        # this one should pass
        pcovc = self.model(n_components=0.5, svd_solver="full")
        pcovc.fit(self.X, self.Y)

        for svd_solver in ["auto", "full"]:
            # this one should pass
            pcovc = self.model(n_components=2, svd_solver=svd_solver)
            pcovc.fit(self.X, self.Y)

            # this one should pass
            pcovc = self.model(n_components="mle", svd_solver=svd_solver)
            pcovc.fit(self.X, self.Y)

    def test_bad_n_components(self):
        """Check that PCovC will not work with any prohibited values of n_components."""
        with self.assertRaises(ValueError) as cm:
            pcovc = self.model(
                n_components="mle", classifier=LogisticRegression(), svd_solver="full"
            )
            # changed X[:2], Y[:2] to X[:20], Y[:20] since first two rows of classes only had class 1 as target,
            # thus error was thrown
            pcovc.fit(self.X[:20], self.Y[:20])
        self.assertEqual(
            str(cm.exception),
            "n_components='mle' is only supported " "if n_samples >= n_features",
        )

        with self.subTest(type="negative_ncomponents"):
            with self.assertRaises(ValueError) as cm:
                pcovc = self.model(n_components=-1, svd_solver="auto")
                pcovc.fit(self.X, self.Y)

            self.assertEqual(
                str(cm.exception),
                "n_components=%r must be between 1 and "
                "min(n_samples, n_features)=%r with "
                "svd_solver='%s'"
                % (
                    pcovc.n_components_,
                    min(self.X.shape),
                    pcovc.svd_solver,
                ),
            )
        with self.subTest(type="0_ncomponents"):
            with self.assertRaises(ValueError) as cm:
                pcovc = self.model(n_components=0, svd_solver="randomized")
                pcovc.fit(self.X, self.Y)

            self.assertEqual(
                str(cm.exception),
                "n_components=%r must be between 1 and "
                "min(n_samples, n_features)=%r with "
                "svd_solver='%s'"
                % (
                    pcovc.n_components_,
                    min(self.X.shape),
                    pcovc.svd_solver,
                ),
            )
        with self.subTest(type="arpack_X_ncomponents"):
            with self.assertRaises(ValueError) as cm:
                pcovc = self.model(n_components=min(self.X.shape), svd_solver="arpack")
                pcovc.fit(self.X, self.Y)
            self.assertEqual(
                str(cm.exception),
                "n_components=%r must be strictly less than "
                "min(n_samples, n_features)=%r with "
                "svd_solver='%s'"
                % (
                    pcovc.n_components_,
                    min(self.X.shape),
                    pcovc.svd_solver,
                ),
            )

        for svd_solver in ["auto", "full"]:
            with self.subTest(type="pi_ncomponents"):
                with self.assertRaises(ValueError) as cm:
                    pcovc = self.model(n_components=np.pi, svd_solver=svd_solver)
                    pcovc.fit(self.X, self.Y)
                self.assertEqual(
                    str(cm.exception),
                    "n_components=%r must be of type int "
                    "when greater than or equal to 1, was of type=%r"
                    % (pcovc.n_components_, type(pcovc.n_components_)),
                )


class PCovCInfrastructureTest(PCovCBaseTest):
    def test_nonfitted_failure(self):
        """
        Check that PCovC will raise a `NonFittedError` if
        `transform` is called before the pcovc is fitted
        """
        pcovc = self.model(n_components=2, tol=1e-12)
        with self.assertRaises(exceptions.NotFittedError):
            _ = pcovc.transform(self.X)

    def test_no_arg_predict(self):
        """
        Check that PCovC will raise a `ValueError` if
        `predict` is called without arguments
        """
        pcovc = self.model(n_components=2, tol=1e-12)
        pcovc.fit(self.X, self.Y)
        with self.assertRaises(ValueError):
            _ = pcovc.predict()

    def test_centering(self):
        """
        Check that PCovC raises a warning if
        given uncentered data.
        """
        pcovc = self.model(n_components=2, tol=1e-12)
        X = self.X.copy() + np.random.uniform(-1, 1, self.X.shape[1])
        with warnings.catch_warnings(record=True) as w:
            pcovc.fit(X, self.Y)
            self.assertEqual(
                str(w[0].message),
                "This class does not automatically center data, and your data mean is "
                "greater than the supplied tolerance.",
            )

    def test_T_shape(self):
        """Check that PCovC returns a latent space projection consistent with the shape
        of the input matrix.
        """
        n_components = 5
        pcovc = self.model(n_components=n_components, tol=1e-12)
        pcovc.fit(self.X, self.Y)
        T = pcovc.transform(self.X)
        self.assertTrue(check_X_y(self.X, T, multi_output=True))
        self.assertTrue(T.shape[-1] == n_components)

    def test_Z_shape(self):
        """Check that PCovC returns an evidence matrix consistent with the number of samples
        and the number of classes.
        """
        n_components = 5
        pcovc = self.model(n_components=n_components, tol=1e-12)
        pcovc.fit(self.X, self.Y)

        # Shape (n_samples, ) for binary classifcation
        Z = pcovc.decision_function(self.X)

        self.assertTrue(Z.ndim == 1)
        self.assertTrue(Z.shape[0] == self.X.shape[0])

        # Modify Y so that it now contains three classes
        Y_multiclass = self.Y.copy()
        Y_multiclass[0] = 2
        pcovc.fit(self.X, Y_multiclass)
        n_classes = len(np.unique(Y_multiclass))

        # Shape (n_samples, n_classes) for multiclass classification
        Z = pcovc.decision_function(self.X)

        self.assertTrue(Z.ndim == 2)
        self.assertTrue((Z.shape[0], Z.shape[1]) == (self.X.shape[0], n_classes))

    def test_default_ncomponents(self):
        pcovc = PCovC(mixing=0.5)
        pcovc.fit(self.X, self.Y)

        self.assertEqual(pcovc.n_components_, min(self.X.shape))

    def test_Y_Shape(self):
        pcovc = self.model()
        Y = np.vstack(self.Y)
        pcovc.fit(self.X, Y)

        self.assertEqual(pcovc.pxz_.shape[0], self.X.shape[1])
        self.assertEqual(pcovc.ptz_.shape[0], pcovc.n_components_)

    def test_prefit_classifier(self):
        classifier = LogisticRegression()
        classifier.fit(self.X, self.Y)
        pcovc = self.model(mixing=0.5, classifier=classifier)
        pcovc.fit(self.X, self.Y)

        Z_classifier = classifier.decision_function(self.X).reshape(self.X.shape[0], -1)
        W_classifier = classifier.coef_.T.reshape(self.X.shape[1], -1)

        Z_pcovc = pcovc.z_classifier_.decision_function(self.X).reshape(
            self.X.shape[0], -1
        )
        W_pcovc = pcovc.z_classifier_.coef_.T.reshape(self.X.shape[1], -1)

        self.assertTrue(np.allclose(Z_classifier, Z_pcovc))
        self.assertTrue(np.allclose(W_classifier, W_pcovc))

    def test_precomputed_classification(self):
        classifier = LogisticRegression()
        classifier.fit(self.X, self.Y)
        Yhat = classifier.predict(self.X)
        W = classifier.coef_.T.reshape(self.X.shape[1], -1)
        pcovc1 = self.model(mixing=0.5, classifier="precomputed", n_components=1)
        pcovc1.fit(self.X, Yhat, W)
        t1 = pcovc1.transform(self.X)

        pcovc2 = self.model(mixing=0.5, classifier=classifier, n_components=1)
        pcovc2.fit(self.X, self.Y)
        t2 = pcovc2.transform(self.X)

        self.assertTrue(np.linalg.norm(t1 - t2) < self.error_tol)

    def test_classifier_modifications(self):
        classifier = LogisticRegression()
        pcovc = self.model(mixing=0.5, classifier=classifier)

        # PCovC classifier matches the original
        self.assertTrue(classifier.get_params() == pcovc.classifier.get_params())

        # PCovC classifier updates its parameters
        # to match the original classifier
        classifier.set_params(random_state=2)
        self.assertTrue(classifier.get_params() == pcovc.classifier.get_params())

        # Fitting classifier outside PCovC fits the PCovC classifier
        classifier.fit(self.X, self.Y)
        self.assertTrue(hasattr(pcovc.classifier, "coef_"))

        # PCovC classifier doesn't change after fitting
        pcovc.fit(self.X, self.Y)
        classifier.set_params(random_state=3)
        self.assertTrue(hasattr(pcovc.classifier_, "coef_"))
        self.assertTrue(classifier.get_params() != pcovc.classifier_.get_params())

    def test_incompatible_classifier(self):
        self.maxDiff = None
        classifier = GaussianNB()
        classifier.fit(self.X, self.Y)
        pcovc = self.model(mixing=0.5, classifier=classifier)

        with self.assertRaises(ValueError) as cm:
            pcovc.fit(self.X, self.Y)
        self.assertEqual(
            str(cm.exception),
            "Classifier must be an instance of "
            "`LinearDiscriminantAnalysis`, `LinearSVC`, `LogisticRegression`, "
            "`LogisticRegressionCV`, `MultiOutputClassifier`, `Perceptron`, "
            "`RidgeClassifier`, `RidgeClassifierCV`, `SGDClassifier`, or `precomputed`",
        )

    def test_none_classifier(self):
        pcovc = PCovC(mixing=0.5, classifier=None)

        pcovc.fit(self.X, self.Y)
        self.assertTrue(pcovc.classifier is None)
        self.assertTrue(pcovc.classifier_ is not None)

    def test_incompatible_coef_shape(self):
        classifier1 = LogisticRegression()

        # Modify Y to be multiclass
        Y_multiclass = self.Y.copy()
        Y_multiclass[0] = 2

        classifier1.fit(self.X, Y_multiclass)
        pcovc1 = self.model(mixing=0.5, classifier=classifier1)

        # Binary classification shape mismatch
        with self.assertRaises(ValueError) as cm:
            pcovc1.fit(self.X, self.Y)
        self.assertEqual(
            str(cm.exception),
            "For binary classification, expected classifier coefficients "
            "to have shape (1, %d) but got shape %r"
            % (self.X.shape[1], classifier1.coef_.shape),
        )

        classifier2 = LogisticRegression()
        classifier2.fit(self.X, self.Y)
        pcovc2 = self.model(mixing=0.5, classifier=classifier2)

        # Multiclass classification shape mismatch
        with self.assertRaises(ValueError) as cm:
            pcovc2.fit(self.X, Y_multiclass)
        self.assertEqual(
            str(cm.exception),
            "For multiclass classification, expected classifier coefficients "
            "to have shape (%d, %d) but got shape %r"
            % (len(np.unique(Y_multiclass)), self.X.shape[1], classifier2.coef_.shape),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
