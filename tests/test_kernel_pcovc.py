import warnings
import numpy as np
import pytest
from sklearn import exceptions
from sklearn.calibration import LinearSVC
from sklearn.datasets import load_breast_cancer as get_dataset
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.validation import check_X_y
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics.pairwise import pairwise_kernels

from skmatter.decomposition import KernelPCovC


@pytest.fixture(scope="module")
def random_state():
    return np.random.RandomState(0)


@pytest.fixture(scope="module")
def error_tol():
    return 1e-6


@pytest.fixture(scope="module")
def X(random_state):
    X, _ = get_dataset(return_X_y=True)
    idx = random_state.choice(len(X), 100)
    X = X[idx]
    scaler = StandardScaler()
    return scaler.fit_transform(X)


@pytest.fixture(scope="module")
def Y(random_state):
    _, Y = get_dataset(return_X_y=True)
    idx = random_state.choice(len(Y), 100)
    return Y[idx]


@pytest.fixture(scope="module")
def kpcovc_model():
    def _model(
        mixing=0.5,
        classifier=LogisticRegression(),
        n_components=4,
        scale_z=True,
        **kwargs,
    ):
        return KernelPCovC(
            mixing=mixing,
            classifier=classifier,
            n_components=n_components,
            scale_z=scale_z,
            svd_solver=kwargs.pop("svd_solver", "full"),
            **kwargs,
        )

    return _model


# KernelPCovCErrorTest


def test_cl_with_x_errors(X, Y, error_tol):
    prev_error = -1.0
    for mixing in np.linspace(0, 1, 6):
        kpcovc = KernelPCovC(mixing=mixing, n_components=4, tol=1e-12)
        kpcovc.fit(X, Y)
        error = np.linalg.norm(Y - kpcovc.predict(X)) ** 2.0 / np.linalg.norm(Y) ** 2.0
        assert not np.isnan(error)
        # Kernel decision functions can exhibit small non-monotonic dips; allow slack.
        assert error >= prev_error - 5e-2
        prev_error = error


def test_cl_with_t_errors(kpcovc_model, X, Y, error_tol):
    prev_error = -1.0
    for mixing in np.linspace(0, 1, 6):
        kpcovc = kpcovc_model(mixing=mixing, n_components=4, tol=1e-12)
        kpcovc.fit(X, Y)
        T = kpcovc.transform(X)
        error = (
            np.linalg.norm(Y - kpcovc.predict(T=T)) ** 2.0 / np.linalg.norm(Y) ** 2.0
        )
        assert not np.isnan(error)
        # Kernel decision functions with T can have small dips; allow slack.
        assert error >= prev_error - 5e-2
        prev_error = error


def test_reconstruction_errors(kpcovc_model, X, Y, error_tol):
    prev_error = 1.0
    for mixing in np.linspace(0, 1, 11):
        kpcovc = kpcovc_model(
            mixing=mixing, n_components=2, tol=1e-12, fit_inverse_transform=True
        )
        kpcovc.fit(X, Y)
        Xr = kpcovc.inverse_transform(kpcovc.transform(X))
        error = np.linalg.norm(X - Xr) ** 2.0 / np.linalg.norm(X) ** 2.0
        assert not np.isnan(error)
        assert error <= prev_error + error_tol
        prev_error = error


# KernelPCovCInfrastructureTest


def test_nonfitted_failure(X):
    kpcovc = KernelPCovC(mixing=0.5, n_components=4, tol=1e-12)
    match = "instance is not fitted"
    with pytest.raises(exceptions.NotFittedError, match=match):
        kpcovc.transform(X)


def test_no_arg_predict(X, Y):
    kpcovc = KernelPCovC(mixing=0.5, n_components=4, tol=1e-12)
    kpcovc.fit(X, Y)
    with pytest.raises(ValueError, match="Either X or T must be supplied"):
        kpcovc.predict()


def test_T_shape(X, Y):
    n_components = 5
    kpcovc = KernelPCovC(mixing=0.5, n_components=n_components, tol=1e-12)
    kpcovc.fit(X, Y)
    T = kpcovc.transform(X)
    assert check_X_y(X, T, multi_output=True) == (X, T)
    assert T.shape[-1] == n_components


def test_Z_shape(kpcovc_model, X, Y):
    n_components = 5
    kpcovc = kpcovc_model(n_components=n_components, tol=1e-12)
    kpcovc.fit(X, Y)
    Z_binary = kpcovc.decision_function(X)
    assert Z_binary.ndim == 1
    assert Z_binary.shape[0] == X.shape[0]
    kpcovc.fit(X, np.random.randint(0, 3, size=X.shape[0]))
    Z_multi = kpcovc.decision_function(X)
    assert Z_multi.ndim == 2
    assert Z_multi.shape == (X.shape[0], len(kpcovc.classes_))


def test_decision_function(kpcovc_model, X, Y):
    kpcovc = kpcovc_model(center=True)
    kpcovc.fit(X, Y)

    with pytest.raises(ValueError, match="Either X or T must be supplied."):
        kpcovc.decision_function()

    kpcovc.decision_function(X)
    T = kpcovc.transform(X)
    kpcovc.decision_function(T=T)


def test_no_centerer(kpcovc_model, X, Y):
    kpcovc = kpcovc_model(center=False)
    kpcovc.fit(X, Y)
    with pytest.raises(AttributeError, match="has no attribute.*centerer"):
        kpcovc.centerer_


def test_centerer(kpcovc_model, X, Y):
    kpcovc = kpcovc_model(center=True)
    kpcovc.fit(X, Y)
    assert hasattr(kpcovc, "centerer_")

    kpcovc.predict(X)
    kpcovc.transform(X)
    kpcovc.score(X, Y)


def test_prefit_classifier(X, Y):
    kernel_params = {"kernel": "rbf", "gamma": 0.1, "degree": 3, "coef0": 0}
    K = pairwise_kernels(X, metric="rbf", filter_params=True, **kernel_params)
    classifier = LinearSVC()
    classifier.fit(K, Y)
    kpcovc = KernelPCovC(mixing=0.5, classifier=classifier, **kernel_params)
    kpcovc.fit(X, Y)
    Z_classifier = classifier.decision_function(K)
    W_classifier = classifier.coef_.T
    Z_kpcovc = kpcovc.z_classifier_.decision_function(K)
    W_kpcovc = kpcovc.z_classifier_.coef_.T
    np.testing.assert_allclose(Z_classifier, Z_kpcovc)
    np.testing.assert_allclose(W_classifier, W_kpcovc)


def test_classifier_modifications(kpcovc_model, X, Y):
    classifier = RidgeClassifier()
    kpcovc = kpcovc_model(mixing=0.5, classifier=classifier, kernel="rbf", gamma=0.1)
    assert classifier.get_params() == kpcovc.classifier.get_params()
    classifier.set_params(random_state=3)
    assert classifier.get_params() == kpcovc.classifier.get_params()
    classifier.fit(X, Y)
    assert hasattr(kpcovc.classifier, "coef_")


def test_incompatible_classifier(kpcovc_model, X, Y):
    classifier = GaussianNB()
    classifier.fit(X, Y)
    kpcovc = kpcovc_model(mixing=0.5, classifier=classifier)
    expected_msg = (
        "Classifier must be an instance of "
        "`LogisticRegression`, `LogisticRegressionCV`, `LinearSVC`, "
        "`LinearDiscriminantAnalysis`, `RidgeClassifier`, `RidgeClassifierCV`, "
        "`SGDClassifier`, `Perceptron`, or `precomputed`"
    )
    with pytest.raises(ValueError, match=expected_msg):
        kpcovc.fit(X, Y)


def test_none_classifier(X, Y):
    kpcovc = KernelPCovC(mixing=0.5, classifier=None)
    kpcovc.fit(X, Y)
    assert kpcovc.classifier is None
    assert kpcovc.classifier_ is not None


def test_incompatible_coef_shape(kpcovc_model, X, Y):
    kernel_params = {"kernel": "sigmoid", "gamma": 0.1, "degree": 3, "coef0": 0}
    K = pairwise_kernels(X, metric="sigmoid", filter_params=True, **kernel_params)
    cl_multi = LinearSVC()
    cl_multi.fit(K, np.random.randint(0, 3, size=X.shape[0]))
    kpcovc_binary = kpcovc_model(mixing=0.5, classifier=cl_multi)
    with pytest.raises(ValueError, match="For binary classification"):
        kpcovc_binary.fit(X, Y)
    cl_binary = LinearSVC()
    cl_binary.fit(K, Y)
    kpcovc_multi = kpcovc_model(mixing=0.5, classifier=cl_binary)
    with pytest.raises(ValueError, match="For multiclass classification"):
        kpcovc_multi.fit(X, np.random.randint(0, 3, size=X.shape[0]))


def test_precomputed_classification(X, Y, error_tol):
    kernel_params = {"kernel": "rbf", "gamma": 0.1, "degree": 3, "coef0": 0}
    K = pairwise_kernels(X, metric="rbf", filter_params=True, **kernel_params)
    classifier = LogisticRegression()
    classifier.fit(K, Y)
    W = classifier.coef_.T
    kpcovc1 = KernelPCovC(mixing=0.5, classifier="precomputed", **kernel_params)
    kpcovc1.fit(X, Y, W)
    t1 = kpcovc1.transform(X)
    kpcovc2 = KernelPCovC(mixing=0.5, classifier=classifier, **kernel_params)
    kpcovc2.fit(X, Y)
    t2 = kpcovc2.transform(X)
    assert np.linalg.norm(t1 - t2) < error_tol
    kpcovc3 = KernelPCovC(mixing=0.5, classifier="precomputed", **kernel_params)
    kpcovc3.fit(X, Y)
    t3 = kpcovc3.transform(X)
    assert np.linalg.norm(t3 - t2) < error_tol
    assert np.linalg.norm(t3 - t1) < error_tol


def test_scale_z_parameter(kpcovc_model, X, Y):
    kpcovc_scaled = kpcovc_model(scale_z=True)
    kpcovc_scaled.fit(X, Y)
    kpcovc_unscaled = kpcovc_model(scale_z=False)
    kpcovc_unscaled.fit(X, Y)
    assert not np.allclose(kpcovc_scaled.pkt_, kpcovc_unscaled.pkt_)


def test_z_scaling(kpcovc_model, X, Y):
    kpcovc = kpcovc_model(n_components=2, scale_z=True)
    kpcovc.fit(X, Y)
    kpcovc = kpcovc_model(n_components=2, scale_z=False, z_mean_tol=0, z_var_tol=0)
    with warnings.catch_warnings(record=True) as w:
        kpcovc.fit(X, Y)
        messages = [str(wi.message) for wi in w]
        assert any("does not automatically center Z" in m for m in messages)
        assert any("does not automatically scale Z" in m for m in messages)


# KernelTests


def test_kernel_types(X, Y):
    def _linear_kernel(XK, YK):
        return XK @ YK.T

    kernel_params = {
        "poly": {"degree": 2},
        "rbf": {"gamma": 3.0},
        "sigmoid": {"gamma": 3.0, "coef0": 0.5},
    }
    for kernel in ["linear", "poly", "rbf", "sigmoid", "cosine", _linear_kernel]:
        kpcovc = KernelPCovC(
            mixing=0.5,
            n_components=2,
            classifier=LogisticRegression(),
            kernel=kernel,
            **kernel_params.get(kernel, {}),
        )
        kpcovc.fit(X, Y)


# KernelPCovCTestSVDSolvers


def test_svd_solvers(kpcovc_model, X, Y):
    for solver in ["arpack", "full", "randomized", "auto"]:
        kpcovc = kpcovc_model(tol=1e-12, n_components=None, svd_solver=solver)
        kpcovc.fit(X, Y)
        if solver == "arpack":
            assert kpcovc.n_components_ == X.shape[0] - 1
        else:
            assert kpcovc.n_components_ == X.shape[0]
    n_component_solvers = {
        "mle": "full",
        int(0.75 * max(X.shape)): "randomized",
        0.1: "full",
    }
    for n_components, solver in n_component_solvers.items():
        kpcovc = kpcovc_model(tol=1e-12, n_components=n_components, svd_solver="auto")
        if solver == "randomized":
            n_copies = (501 // max(X.shape)) + 1
            Xr = np.hstack(np.repeat(X.copy(), n_copies)).reshape(
                X.shape[0] * n_copies, -1
            )
            Yr = np.hstack(np.repeat(Y.copy(), n_copies)).reshape(
                X.shape[0] * n_copies, -1
            )
            kpcovc.fit(Xr, Yr)
        else:
            kpcovc.fit(X, Y)
        assert kpcovc.fit_svd_solver_ == solver


def test_bad_solver(kpcovc_model, X, Y):
    with pytest.raises(ValueError, match="Unrecognized svd_solver='bad'"):
        kpcovc = kpcovc_model(svd_solver="bad")
        kpcovc.fit(X, Y)


def test_good_n_components(kpcovc_model, X, Y):
    kpcovc = kpcovc_model(n_components=0.5, svd_solver="full")
    kpcovc.fit(X, Y)
    for svd_solver in ["auto", "full"]:
        kpcovc = kpcovc_model(n_components=2, svd_solver=svd_solver)
        kpcovc.fit(X, Y)
        kpcovc = kpcovc_model(n_components="mle", svd_solver=svd_solver)
        kpcovc.fit(X, Y)


def test_bad_n_components(kpcovc_model, X, Y):
    with pytest.raises(ValueError, match="n_components=.*must be between"):
        kpcovc = kpcovc_model(n_components=-1, svd_solver="auto")
        kpcovc.fit(X, Y)
    with pytest.raises(ValueError, match="n_components=.*must be between"):
        kpcovc = kpcovc_model(n_components=0, svd_solver="randomized")
        kpcovc.fit(X, Y)
    with pytest.raises(ValueError, match="n_components=.*strictly less than"):
        kpcovc = kpcovc_model(n_components=X.shape[0], svd_solver="arpack")
        kpcovc.fit(X, Y)
    for svd_solver in ["auto", "full"]:
        with pytest.raises(ValueError, match="must be of type int"):
            kpcovc = kpcovc_model(n_components=np.pi, svd_solver=svd_solver)
            kpcovc.fit(X, Y)
