import warnings
import numpy as np
import pytest
from sklearn import exceptions
from sklearn.calibration import LinearSVC
from sklearn.datasets import load_iris as get_dataset
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y

from skmatter.decomposition import PCovC


@pytest.fixture(scope="module")
def pcovc_model():
    def _model(mixing=0.5, classifier=LogisticRegression(), scale_z=True, **kwargs):
        return PCovC(mixing=mixing, classifier=classifier, scale_z=scale_z, **kwargs)

    return _model


@pytest.fixture(scope="module")
def error_tol():
    return 1e-5


@pytest.fixture(scope="module")
def X():
    X, _ = get_dataset(return_X_y=True)
    X_stacked = np.tile(X, (4, 1))
    scaler = StandardScaler()
    return scaler.fit_transform(X_stacked)


@pytest.fixture(scope="module")
def Y():
    _, Y = get_dataset(return_X_y=True)
    return np.tile(Y, 4)


# PCovCErrorTest


def test_against_pca(X, Y):
    pcovc = PCovC(mixing=1.0, n_components=2, space="feature", svd_solver="full").fit(
        X, Y
    )
    pca = PCA(n_components=2, svd_solver="full").fit(X)
    np.testing.assert_allclose(pca.singular_values_, pcovc.singular_values_)
    np.testing.assert_allclose(pca.explained_variance_, pcovc.explained_variance_)
    T_pcovc = pcovc.transform(X)
    T_pca = pca.transform(X)
    assert np.linalg.norm(T_pcovc @ T_pcovc.T - T_pca @ T_pca.T) <= 1e-8


@pytest.mark.parametrize("space", ["feature", "sample", "auto"])
def test_simple_reconstruction(pcovc_model, X, Y, error_tol, space):
    pcovc = pcovc_model(mixing=1.0, n_components=X.shape[-1], space=space)
    pcovc.fit(X, Y)
    Xr = pcovc.inverse_transform(pcovc.transform(X))
    assert np.linalg.norm(X - Xr) ** 2.0 / np.linalg.norm(X) ** 2.0 <= error_tol


@pytest.mark.parametrize("space", ["feature", "sample", "auto"])
def test_simple_prediction(pcovc_model, X, Y, error_tol, space):
    pcovc = pcovc_model(
        mixing=0.0, classifier=RidgeClassifier(), n_components=2, space=space
    )
    pcovc.classifier.fit(X, Y)
    Yhat = pcovc.classifier.predict(X)
    pcovc.fit(X, Y)
    Yp = pcovc.predict(X)
    assert np.linalg.norm(Yp - Yhat) ** 2.0 / np.linalg.norm(Yhat) ** 2.0 <= error_tol


def test_cl_with_x_errors(pcovc_model, X, Y, error_tol):
    prev_error = -1.0
    for mixing in np.linspace(0, 1, 11):
        pcovc = pcovc_model(mixing=mixing, n_components=2, tol=1e-12)
        pcovc.fit(X, Y)
        Yp = pcovc.predict(X=X)
        error = np.linalg.norm(Y - Yp) ** 2.0 / np.linalg.norm(Y) ** 2.0
        assert not np.isnan(error)
        assert error >= prev_error - error_tol
        prev_error = error


def test_cl_with_t_errors(pcovc_model, X, Y, error_tol):
    prev_error = -1.0
    for mixing in np.linspace(0, 1, 11):
        pcovc = pcovc_model(mixing=mixing, n_components=2, tol=1e-12)
        pcovc.fit(X, Y)
        T = pcovc.transform(X)
        Yp = pcovc.predict(T=T)
        error = np.linalg.norm(Y - Yp) ** 2.0 / np.linalg.norm(Y) ** 2.0
        assert not np.isnan(error)
        assert error >= prev_error - error_tol
        prev_error = error


def test_reconstruction_errors(pcovc_model, X, Y, error_tol):
    prev_error = 1.0
    for mixing in np.linspace(0, 1, 11):
        pcovc = pcovc_model(mixing=mixing, n_components=2, tol=1e-12)
        pcovc.fit(X, Y)
        Xr = pcovc.inverse_transform(pcovc.transform(X))
        error = np.linalg.norm(X - Xr) ** 2.0 / np.linalg.norm(X) ** 2.0
        assert not np.isnan(error)
        assert error <= prev_error + error_tol
        prev_error = error


# PCovCSpaceTest


def test_select_feature_space(pcovc_model, X, Y):
    pcovc = pcovc_model(n_components=2, tol=1e-12)
    pcovc.fit(X, Y)
    assert pcovc.space_ == "feature"


def test_select_sample_space(pcovc_model, X, Y):
    pcovc = pcovc_model(n_components=1, tol=1e-12, svd_solver="arpack")
    n_samples = 2
    with pytest.warns(match="class does not automatically center data"):
        pcovc.fit(X[49 : 49 + n_samples], Y[49 : 49 + n_samples])
    assert pcovc.space_ == "sample"


def test_bad_space(pcovc_model, X, Y):
    match = "Only feature and sample space are supported"
    with pytest.raises(ValueError, match=match):
        pcovc = pcovc_model(n_components=2, tol=1e-12, space="bad")
        pcovc.fit(X, Y)


def test_override_spaceselection(pcovc_model, X, Y):
    pcovc = pcovc_model(n_components=2, tol=1e-12, space="sample")
    pcovc.fit(X, Y)
    assert pcovc.space_ == "sample"


def test_spaces_equivalent(pcovc_model, X, Y, error_tol):
    for alpha in np.linspace(0.01, 0.99, 11):
        pcovc_ss = pcovc_model(n_components=2, mixing=alpha, tol=1e-12, space="sample")
        pcovc_ss.fit(X, Y)
        pcovc_fs = pcovc_model(n_components=2, mixing=alpha, tol=1e-12, space="feature")
        pcovc_fs.fit(X, Y)
        np.testing.assert_allclose(
            pcovc_ss.decision_function(X), pcovc_fs.decision_function(X), atol=error_tol
        )
        np.testing.assert_allclose(
            pcovc_ss.inverse_transform(pcovc_ss.transform(X)),
            pcovc_fs.inverse_transform(pcovc_fs.transform(X)),
            atol=error_tol,
        )


# PCovCTestSVDSolvers


def test_svd_solvers(pcovc_model, X, Y):
    for solver in ["arpack", "full", "randomized", "auto"]:
        pcovc = pcovc_model(tol=1e-12, svd_solver=solver)
        pcovc.fit(X, Y)
        if solver == "arpack":
            assert pcovc.n_components_ == min(X.shape) - 1
        else:
            assert pcovc.n_components_ == min(X.shape)


def test_bad_solver(pcovc_model, X, Y):
    for space in ["feature", "sample"]:
        with pytest.raises(ValueError, match="Unrecognized svd_solver='bad'"):
            pcovc = pcovc_model(svd_solver="bad", space=space)
            pcovc.fit(X, Y)


def test_good_n_components(pcovc_model, X, Y):
    pcovc = pcovc_model(n_components=0.5, svd_solver="full")
    pcovc.fit(X, Y)
    for svd_solver in ["auto", "full"]:
        pcovc = pcovc_model(n_components=2, svd_solver=svd_solver)
        pcovc.fit(X, Y)
        pcovc = pcovc_model(n_components="mle", svd_solver=svd_solver)
        pcovc.fit(X, Y)


def test_bad_n_components(pcovc_model, X, Y):
    match = "n_components='mle' is only supported if n_samples >= n_features"
    with pytest.raises(ValueError, match=match):
        pcovc = pcovc_model(
            n_components="mle", classifier=LinearSVC(), svd_solver="full"
        )
        pcovc.fit(X[49:51], Y[49:51])

    with pytest.raises(ValueError, match="n_components=.*must be between"):
        pcovc = pcovc_model(n_components=-1, svd_solver="auto")
        pcovc.fit(X, Y)

    with pytest.raises(ValueError, match="n_components=.*must be between"):
        pcovc = pcovc_model(n_components=0, svd_solver="randomized")
        pcovc.fit(X, Y)

    with pytest.raises(ValueError, match="n_components=.*strictly less than"):
        pcovc = pcovc_model(n_components=min(X.shape), svd_solver="arpack")
        pcovc.fit(X, Y)

    for svd_solver in ["auto", "full"]:
        with pytest.raises(ValueError, match="must be of type int"):
            pcovc = pcovc_model(n_components=np.pi, svd_solver=svd_solver)
            pcovc.fit(X, Y)


# PCovCInfrastructureTest


def test_nonfitted_failure(pcovc_model, X):
    pcovc = pcovc_model(n_components=2, tol=1e-12)
    with pytest.raises(exceptions.NotFittedError, match="instance is not fitted"):
        pcovc.transform(X)


def test_no_arg_predict(pcovc_model, X, Y):
    pcovc = pcovc_model(n_components=2, tol=1e-12)
    pcovc.fit(X, Y)
    with pytest.raises(ValueError, match="Either X or T must be supplied"):
        pcovc.predict()


def test_centering(pcovc_model, X, Y):
    pcovc = pcovc_model(n_components=2, tol=1e-12)
    X_mod = X.copy() + np.random.uniform(-1, 1, X.shape[1])
    m = (
        "This class does not automatically center data, and your data mean is "
        "greater than the supplied tolerance."
    )
    with pytest.warns(match=m):
        pcovc.fit(X_mod, Y)


def test_z_scaling(pcovc_model, X, Y):
    pcovc = pcovc_model(n_components=2, scale_z=True)
    pcovc.fit(X, Y)
    pcovc = pcovc_model(n_components=2, scale_z=False, z_mean_tol=0, z_var_tol=0)
    with warnings.catch_warnings(record=True) as w:
        pcovc.fit(X, Y)
        msg0 = str(w[0].message)
        msg1 = str(w[1].message)
        assert "does not automatically center Z" in msg0
        assert "does not automatically scale Z" in msg1


def test_T_shape(pcovc_model, X, Y):
    n_components = 4
    pcovc = pcovc_model(n_components=n_components, tol=1e-12)
    pcovc.fit(X, Y)
    T = pcovc.transform(X)
    check_X_y(X, T, multi_output=True)
    assert T.shape[-1] == n_components


def test_Y_Shape(pcovc_model, X, Y):
    pcovc = pcovc_model()
    Y2 = np.vstack(Y)
    pcovc.fit(X, Y2)
    assert pcovc.pxz_.shape[0] == X.shape[1]
    assert pcovc.ptz_.shape[0] == pcovc.n_components_


def test_Z_shape(pcovc_model, X, Y):
    n_components = 2
    pcovc = pcovc_model(n_components=n_components, tol=1e-12)
    pcovc.fit(X, np.random.randint(0, 2, size=X.shape[0]))
    Z_binary = pcovc.decision_function(X)
    assert Z_binary.ndim == 1
    assert Z_binary.shape[0] == X.shape[0]
    pcovc.fit(X, Y)
    Z_multi = pcovc.decision_function(X)
    assert Z_multi.ndim == 2
    assert Z_multi.shape == (X.shape[0], len(pcovc.classes_))


def test_decision_function(pcovc_model, X, Y):
    pcovc = pcovc_model()
    pcovc.fit(X, Y)
    with pytest.raises(ValueError, match="Either X or T must be supplied."):
        pcovc.decision_function()
    T = pcovc.transform(X)
    pcovc.decision_function(T=T)


def test_default_ncomponents(X, Y):
    pcovc = PCovC(mixing=0.5)
    pcovc.fit(X, Y)
    assert pcovc.n_components_ == min(X.shape)


def test_prefit_classifier(pcovc_model, X, Y):
    classifier = LinearSVC()
    classifier.fit(X, Y)
    pcovc = pcovc_model(mixing=0.5, classifier=classifier)
    pcovc.fit(X, Y)
    Z_classifier = classifier.decision_function(X)
    W_classifier = classifier.coef_.T
    Z_pcovc = pcovc.z_classifier_.decision_function(X)
    W_pcovc = pcovc.z_classifier_.coef_.T
    np.testing.assert_allclose(Z_classifier, Z_pcovc)
    np.testing.assert_allclose(W_classifier, W_pcovc)


def test_precomputed_classification(pcovc_model, X, Y, error_tol):
    classifier = LogisticRegression()
    classifier.fit(X, Y)
    W = classifier.coef_.T
    pcovc1 = pcovc_model(mixing=0.5, classifier="precomputed", n_components=1)
    pcovc1.fit(X, Y, W)
    t1 = pcovc1.transform(X)
    pcovc2 = pcovc_model(mixing=0.5, classifier=classifier, n_components=1)
    pcovc2.fit(X, Y)
    t2 = pcovc2.transform(X)
    assert np.linalg.norm(t1 - t2) < error_tol
    pcovc3 = pcovc_model(mixing=0.5, classifier="precomputed", n_components=1)
    pcovc3.fit(X, Y)
    t3 = pcovc3.transform(X)
    assert np.linalg.norm(t3 - t2) < error_tol
    assert np.linalg.norm(t3 - t1) < error_tol


def test_classifier_modifications(pcovc_model, X, Y):
    classifier = LinearSVC()
    pcovc = pcovc_model(mixing=0.5, classifier=classifier)
    assert classifier.get_params() == pcovc.classifier.get_params()
    classifier.set_params(random_state=2)
    assert classifier.get_params() == pcovc.classifier.get_params()
    classifier.fit(X, Y)
    assert hasattr(pcovc.classifier, "coef_")
    pcovc.fit(X, Y)
    classifier.set_params(random_state=3)
    assert hasattr(pcovc.classifier_, "coef_")
    assert classifier.get_params() != pcovc.classifier_.get_params()


def test_incompatible_classifier(pcovc_model, X, Y):
    classifier = GaussianNB()
    classifier.fit(X, Y)
    pcovc = pcovc_model(mixing=0.5, classifier=classifier)
    expected_msg = (
        "Classifier must be an instance of "
        "`LogisticRegression`, `LogisticRegressionCV`, `LinearSVC`, "
        "`LinearDiscriminantAnalysis`, `RidgeClassifier`, `RidgeClassifierCV`, "
        "`SGDClassifier`, `Perceptron`, or `precomputed`"
    )
    with pytest.raises(ValueError, match=expected_msg):
        pcovc.fit(X, Y)


def test_none_classifier(X, Y):
    pcovc = PCovC(mixing=0.5, classifier=None)
    with pytest.warns(match="class does not automatically scale Z"):
        pcovc.fit(X, Y)
    assert pcovc.classifier is None
    assert pcovc.classifier_ is not None


def test_incompatible_coef_shape(pcovc_model, X, Y):
    cl_multi = LogisticRegression()
    cl_multi.fit(X, Y)
    pcovc_binary = pcovc_model(mixing=0.5, classifier=cl_multi)
    with pytest.raises(ValueError, match="For binary classification"):
        pcovc_binary.fit(X, np.random.randint(0, 2, size=X.shape[0]))
    cl_binary = LogisticRegression()
    cl_binary.fit(X, np.random.randint(0, 2, size=X.shape[0]))
    pcovc_multi = pcovc_model(mixing=0.5, classifier=cl_binary)
    with pytest.raises(ValueError, match="For multiclass classification"):
        pcovc_multi.fit(X, Y)


def test_scale_z_parameter(pcovc_model, X, Y):
    pcovc_scaled = pcovc_model(scale_z=True)
    pcovc_scaled.fit(X, Y)
    pcovc_unscaled = pcovc_model(scale_z=False)
    pcovc_unscaled.fit(X, Y)
    assert not np.allclose(
        pcovc_scaled.singular_values_, pcovc_unscaled.singular_values_
    )
