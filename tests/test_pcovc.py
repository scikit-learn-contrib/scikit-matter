import unittest
import warnings

import numpy as np
from sklearn import exceptions
from sklearn.datasets import load_iris as get_dataset
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y
import pytest

from skmatter.decomposition import PCovC


class PCovCBaseTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = (
            lambda mixing=0.5,
            classifier=LogisticRegression(),
            scale_z=True,
            **kwargs: PCovC(
                mixing=mixing, classifier=classifier, scale_z=scale_z, **kwargs
            )
        )

        self.error_tol = 1e-5
        self.X, self.Y = get_dataset(return_X_y=True)

        # n_samples > 500 to ensure our svd_solver tests catch all cases
        X_stacked = np.tile(self.X, (4, 1))
        Y_stacked = np.tile(self.Y, 4)
        self.X, self.Y = X_stacked, Y_stacked

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
        can reproduce a linear classification result.
        """
        for space in ["feature", "sample", "auto"]:
            with self.subTest(space=space):
                pcovc = self.model(
                    mixing=0.0,
                    classifier=RidgeClassifier(),
                    n_components=2,
                    space=space,
                )

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
        """Check that PCovC returns a non-null property prediction from the latent space
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
        pcovc = self.model(n_components=1, tol=1e-12, svd_solver="arpack")
        n_samples = 2

        # select range where there are at least 2 classes in Y
        with pytest.warns(match="class does not automatically center data"):
            pcovc.fit(self.X[49 : 49 + n_samples], self.Y[49 : 49 + n_samples])

        assert pcovc.space_ == "sample"

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
                        pcovc_ss.decision_function(self.X),
                        pcovc_fs.decision_function(self.X),
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

            self.assertEqual(str(cm.exception), "Unrecognized svd_solver='bad'")

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
                n_components="mle", classifier=LinearSVC(), svd_solver="full"
            )
            # select range where there are at least 2 classes in Y
            pcovc.fit(self.X[49:51], self.Y[49:51])
        self.assertEqual(
            str(cm.exception),
            "n_components='mle' is only supported if n_samples >= n_features",
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
        m = (
            "This class does not automatically center data, and your data mean is "
            "greater than the supplied tolerance."
        )
        with pytest.warns(match=m):
            pcovc.fit(X, self.Y)

    def test_z_scaling(self):
        """
        Check that PCovC raises a warning if Z is not of scale, and does not
        if it is.
        """
        pcovc = self.model(n_components=2, scale_z=True)
        pcovc.fit(self.X, self.Y)

        pcovc = self.model(n_components=2, scale_z=False, z_mean_tol=0, z_var_tol=0)

        with warnings.catch_warnings(record=True) as w:
            pcovc.fit(self.X, self.Y)
            self.assertEqual(
                str(w[0].message),
                "This class does not automatically center Z, and the column means "
                "of Z are greater than the supplied tolerance. We recommend scaling "
                "Z (and the weights) by setting `scale_z=True`.",
            )
            self.assertEqual(
                str(w[1].message),
                "This class does not automatically scale Z, and the column variances "
                "of Z are greater than the supplied tolerance. We recommend scaling "
                "Z (and the weights) by setting `scale_z=True`.",
            )

    def test_T_shape(self):
        """Check that PCovC returns a latent space projection consistent with
        the shape of the input matrix.
        """
        n_components = 4
        pcovc = self.model(n_components=n_components, tol=1e-12)
        pcovc.fit(self.X, self.Y)
        T = pcovc.transform(self.X)
        self.assertTrue(check_X_y(self.X, T, multi_output=True))
        self.assertTrue(T.shape[-1] == n_components)

    def test_Y_Shape(self):
        pcovc = self.model()
        Y = np.vstack(self.Y)
        pcovc.fit(self.X, Y)

        self.assertEqual(pcovc.pxz_.shape[0], self.X.shape[1])
        self.assertEqual(pcovc.ptz_.shape[0], pcovc.n_components_)

    def test_Z_shape(self):
        """Check that PCovC returns an evidence matrix consistent with the
        number of samples and the number of classes.
        """
        n_components = 2
        pcovc = self.model(n_components=n_components, tol=1e-12)
        pcovc.fit(self.X, np.random.randint(0, 2, size=self.X.shape[0]))

        # Shape (n_samples, ) for binary classifcation
        Z_binary = pcovc.decision_function(self.X)
        self.assertEqual(Z_binary.ndim, 1)
        self.assertEqual(Z_binary.shape[0], self.X.shape[0])

        # Shape (n_samples, n_classes) for multiclass classification
        pcovc.fit(self.X, self.Y)
        Z_multi = pcovc.decision_function(self.X)

        self.assertEqual(Z_multi.ndim, 2)
        self.assertEqual(Z_multi.shape, (self.X.shape[0], len(pcovc.classes_)))

    def test_decision_function(self):
        """Check that PCovC's decision_function works when only T is
        provided and throws an error when appropriate.
        """
        pcovc = self.model()
        pcovc.fit(self.X, self.Y)
        with self.assertRaises(ValueError) as cm:
            _ = pcovc.decision_function()
        self.assertEqual(
            str(cm.exception),
            "Either X or T must be supplied.",
        )

        T = pcovc.transform(self.X)
        _ = pcovc.decision_function(T=T)

    def test_default_ncomponents(self):
        pcovc = PCovC(mixing=0.5)
        pcovc.fit(self.X, self.Y)

        self.assertEqual(pcovc.n_components_, min(self.X.shape))

    def test_prefit_classifier(self):
        """Check that a passed prefit classifier is not modified in
        PCovC's `fit` call.
        """
        classifier = LinearSVC()
        classifier.fit(self.X, self.Y)
        pcovc = self.model(mixing=0.5, classifier=classifier)
        pcovc.fit(self.X, self.Y)

        Z_classifier = classifier.decision_function(self.X)
        W_classifier = classifier.coef_.T

        Z_pcovc = pcovc.z_classifier_.decision_function(self.X)
        W_pcovc = pcovc.z_classifier_.coef_.T

        self.assertTrue(np.allclose(Z_classifier, Z_pcovc))
        self.assertTrue(np.allclose(W_classifier, W_pcovc))

    def test_precomputed_classification(self):
        classifier = LogisticRegression()
        classifier.fit(self.X, self.Y)

        W = classifier.coef_.T
        pcovc1 = self.model(mixing=0.5, classifier="precomputed", n_components=1)
        pcovc1.fit(self.X, self.Y, W)
        t1 = pcovc1.transform(self.X)

        pcovc2 = self.model(mixing=0.5, classifier=classifier, n_components=1)
        pcovc2.fit(self.X, self.Y)
        t2 = pcovc2.transform(self.X)

        self.assertTrue(np.linalg.norm(t1 - t2) < self.error_tol)

        # Now check for match when W is not passed:
        pcovc3 = self.model(mixing=0.5, classifier="precomputed", n_components=1)
        pcovc3.fit(self.X, self.Y)
        t3 = pcovc3.transform(self.X)

        self.assertTrue(np.linalg.norm(t3 - t2) < self.error_tol)
        self.assertTrue(np.linalg.norm(t3 - t1) < self.error_tol)

    def test_classifier_modifications(self):
        classifier = LinearSVC()
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
        classifier = GaussianNB()
        classifier.fit(self.X, self.Y)
        pcovc = self.model(mixing=0.5, classifier=classifier)

        with self.assertRaises(ValueError) as cm:
            pcovc.fit(self.X, self.Y)
        self.assertEqual(
            str(cm.exception),
            "Classifier must be an instance of "
            "`LogisticRegression`, `LogisticRegressionCV`, `LinearSVC`, "
            "`LinearDiscriminantAnalysis`, `RidgeClassifier`, `RidgeClassifierCV`, "
            "`SGDClassifier`, `Perceptron`, `MultiOutputClassifier`, "
            "or `precomputed`.",
        )

    def test_none_classifier(self):
        pcovc = PCovC(mixing=0.5, classifier=None)

        with pytest.warns(match="class does not automatically scale Z"):
            pcovc.fit(self.X, self.Y)

        assert pcovc.classifier is None
        assert pcovc.classifier_ is not None

    def test_incompatible_coef_shape(self):
        cl_multi = LogisticRegression()
        cl_multi.fit(self.X, self.Y)
        pcovc_binary = self.model(mixing=0.5, classifier=cl_multi)

        # Binary classification shape mismatch
        with self.assertRaises(ValueError) as cm:
            pcovc_binary.fit(self.X, np.random.randint(0, 2, size=self.X.shape[0]))
        self.assertEqual(
            str(cm.exception),
            "For binary classification, expected classifier coefficients "
            "to have shape (1, %d) but got shape %r"
            % (self.X.shape[1], cl_multi.coef_.shape),
        )

        cl_binary = LogisticRegression()
        cl_binary.fit(self.X, np.random.randint(0, 2, size=self.X.shape[0]))
        pcovc_multi = self.model(mixing=0.5, classifier=cl_binary)

        # Multiclass classification shape mismatch
        with self.assertRaises(ValueError) as cm:
            pcovc_multi.fit(self.X, self.Y)
        self.assertEqual(
            str(cm.exception),
            "For multiclass classification, expected classifier coefficients "
            "to have shape (%d, %d) but got shape %r"
            % (len(pcovc_multi.classes_), self.X.shape[1], cl_binary.coef_.shape),
        )

    def test_scale_z_parameter(self):
        """Check that changing scale_z changes the eigendecomposition."""
        pcovc_scaled = self.model(scale_z=True)
        pcovc_scaled.fit(self.X, self.Y)

        pcovc_unscaled = self.model(scale_z=False)
        pcovc_unscaled.fit(self.X, self.Y)

        assert not np.allclose(
            pcovc_scaled.singular_values_, pcovc_unscaled.singular_values_
        )


class PCovCMultiOutputTest(PCovCBaseTest):
    def test_prefit_multioutput(self):
        """Check that PCovC works if a prefit classifier
        is passed when `n_outputs > 1`.
        """
        classifier = MultiOutputClassifier(estimator=LogisticRegression())
        Y_double = np.column_stack((self.Y, self.Y))

        classifier.fit(self.X, Y_double)
        pcovc = self.model(mixing=0.25, classifier=classifier)
        pcovc.fit(self.X, Y_double)

        W_classifier = np.hstack([est_.coef_.T for est_ in classifier.estimators_])
        Z_classifier = self.X @ W_classifier

        W_pcovc = np.hstack([est_.coef_.T for est_ in pcovc.z_classifier_.estimators_])
        Z_pcovc = self.X @ W_pcovc

        self.assertTrue(np.allclose(Z_classifier, Z_pcovc))
        self.assertTrue(np.allclose(W_classifier, W_pcovc))

    def test_precomputed_multioutput(self):
        """Check that PCovC works if classifier=`precomputed` and `n_outputs > 1`."""
        classifier = MultiOutputClassifier(estimator=LogisticRegression())
        Y_double = np.column_stack((self.Y, self.Y))

        classifier.fit(self.X, Y_double)
        W = np.hstack([est_.coef_.T for est_ in classifier.estimators_])
        print(W.shape)
        pcovc1 = self.model(mixing=0.5, classifier="precomputed", n_components=1)
        pcovc1.fit(self.X, Y_double, W)
        t1 = pcovc1.transform(self.X)

        pcovc2 = self.model(mixing=0.5, classifier=classifier, n_components=1)
        pcovc2.fit(self.X, Y_double)
        t2 = pcovc2.transform(self.X)

        self.assertTrue(np.linalg.norm(t1 - t2) < self.error_tol)

        # Now check for match when W is not passed:
        pcovc3 = self.model(mixing=0.5, classifier="precomputed", n_components=1)
        pcovc3.fit(self.X, Y_double)
        t3 = pcovc3.transform(self.X)

        self.assertTrue(np.linalg.norm(t3 - t2) < self.error_tol)
        self.assertTrue(np.linalg.norm(t3 - t1) < self.error_tol)

    def test_Z_shape_multioutput(self):
        """Check that PCovC returns the evidence Z in the
        desired form when `n_outputs > 1`.
        """
        pcovc = PCovC()

        Y_double = np.column_stack((self.Y, self.Y))
        pcovc.fit(self.X, Y_double)

        Z = pcovc.decision_function(self.X)

        # list of (n_samples, n_classes) arrays when each column of Y is multiclass
        self.assertEqual(len(Z), Y_double.shape[1])

        for est, z_slice in zip(pcovc.z_classifier_.estimators_, Z):
            with self.subTest(type="z_arrays"):
                # each array is shape (n_samples, n_classes):
                self.assertEqual(self.X.shape[0], z_slice.shape[0])
                self.assertEqual(est.coef_.shape[0], z_slice.shape[1])

    def test_decision_function_multioutput(self):
        """Check that PCovC's decision_function works in edge
        cases when `n_outputs_ > 1`.
        """
        pcovc = self.model(
            classifier=MultiOutputClassifier(estimator=LogisticRegression())
        )
        pcovc.fit(self.X, np.column_stack((self.Y, self.Y)))
        with self.assertRaises(ValueError) as cm:
            _ = pcovc.decision_function()
        self.assertEqual(
            str(cm.exception),
            "Either X or T must be supplied.",
        )

        T = pcovc.transform(self.X)
        _ = pcovc.decision_function(T=T)

    def test_score(self):
        """Check that PCovC's score behaves properly with multiple labels."""
        pcovc_multi = self.model(
            classifier=MultiOutputClassifier(estimator=LogisticRegression())
        )
        pcovc_multi.fit(self.X, np.column_stack((self.Y, self.Y)))
        score_multi = pcovc_multi.score(self.X, np.column_stack((self.Y, self.Y)))

        pcovc_single = self.model().fit(self.X, self.Y)
        score_single = pcovc_single.score(self.X, self.Y)
        self.assertEqual(score_single, score_multi)

    def test_bad_multioutput_estimator(self):
        """Check that PCovC returns an error when a MultiOutputClassifier
        is improperly constructed.
        """
        with self.assertRaises(ValueError) as cm:
            pcovc = self.model(classifier=MultiOutputClassifier(estimator=GaussianNB()))
            pcovc.fit(self.X, np.column_stack((self.Y, self.Y)))
        self.assertEqual(
            str(cm.exception),
            "The instance of MultiOutputClassifier passed as the PCovC classifier "
            "contains `GaussianNB`, which is not supported. The MultiOutputClassifier "
            "must contain an instance of `LogisticRegression`, `LogisticRegressionCV`, "
            "`LinearSVC`, `LinearDiscriminantAnalysis`, `RidgeClassifier`, "
            "`RidgeClassifierCV`, `SGDClassifier`, `Perceptron`, or `precomputed`.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
