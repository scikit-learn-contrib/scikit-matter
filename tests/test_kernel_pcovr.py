import numpy as np
import pytest
from sklearn import exceptions
from sklearn.datasets import load_diabetes as get_dataset
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.utils.validation import check_X_y

from skmatter.decomposition import PCovR, KernelPCovR
from skmatter.preprocessing import StandardFlexibleScaler as SFS


@pytest.fixture(scope="module")
def random_state():
    """Random state for reproducible tests."""
    return np.random.RandomState(0)


@pytest.fixture(scope="module")
def error_tol():
    """Error tolerance for tests."""
    return 1e-6


@pytest.fixture(scope="module")
def X(random_state):
    """Feature matrix."""
    X, _ = get_dataset(return_X_y=True)
    # for the sake of expedience, only use a subset of the dataset
    idx = random_state.choice(len(X), 100)
    X = X[idx]
    return SFS().fit_transform(X)


@pytest.fixture(scope="module")
def Y(random_state, X):
    """Target matrix (2D with artificial second property)."""
    X_full, Y = get_dataset(return_X_y=True)
    idx = random_state.choice(len(X_full), 100)
    X_full = X_full[idx]
    Y = Y[idx]

    # artificial second property
    Y = np.array([Y, X_full @ random_state.randint(-2, 2, (X_full.shape[-1],))]).T
    Y = Y.reshape(X_full.shape[0], -1)
    return SFS(column_wise=True).fit_transform(Y)


@pytest.fixture(scope="module")
def kpcovr_model():
    """Factory fixture for KernelPCovR model."""

    def _model(mixing=0.5, regressor=KernelRidge(alpha=1e-8), n_components=4, **kwargs):
        return KernelPCovR(
            mixing,
            regressor=regressor,
            n_components=n_components,
            svd_solver=kwargs.pop("svd_solver", "full"),
            **kwargs,
        )

    return _model


def test_lr_with_x_errors(X, Y, error_tol):
    """
    Check that KernelPCovR returns a non-null property prediction
    and that the prediction error increases with `mixing`
    """
    prev_error = -1.0

    for mixing in np.linspace(0, 1, 6):
        kpcovr = KernelPCovR(mixing=mixing, n_components=2, tol=1e-12)
        kpcovr.fit(X, Y)
        error = np.linalg.norm(Y - kpcovr.predict(X)) ** 2.0 / np.linalg.norm(Y) ** 2.0

        assert not np.isnan(error), f"Error is NaN for mixing={mixing}"
        assert error >= prev_error - error_tol, (
            f"Error decreased unexpectedly at mixing={round(mixing, 4)}"
        )

        prev_error = error


def test_reconstruction_errors(X, Y, error_tol):
    """Check that KernelPCovR returns a non-null reconstructed X and that the
    reconstruction error decreases with `mixing`.
    """
    prev_error = 10.0
    prev_x_error = 10.0

    for mixing in np.linspace(0, 1, 6):
        kpcovr = KernelPCovR(
            mixing=mixing, n_components=2, fit_inverse_transform=True, tol=1e-12
        )
        kpcovr.fit(X, Y)

        t = kpcovr.transform(X)
        K = kpcovr._get_kernel(X)
        x = kpcovr.inverse_transform(t)

        error = np.linalg.norm(K - t @ t.T) ** 2.0 / np.linalg.norm(K) ** 2.0
        x_error = np.linalg.norm(X - x) ** 2.0 / np.linalg.norm(X) ** 2.0

        assert not np.isnan(error), f"Error is NaN for mixing={mixing}"
        assert error <= prev_error + error_tol, (
            f"Error increased unexpectedly at mixing={round(mixing, 4)}"
        )

        assert not np.isnan(x_error), f"X error is NaN for mixing={mixing}"
        assert x_error <= prev_x_error + error_tol, (
            f"X error increased unexpectedly at mixing={round(mixing, 4)}"
        )

        prev_error = error
        prev_x_error = x_error


def test_kpcovr_error(kpcovr_model, X, Y, error_tol):
    for mixing in np.linspace(0, 1, 6):
        kpcovr = kpcovr_model(
            mixing=mixing,
            regressor=KernelRidge(kernel="rbf", gamma=1.0),
            kernel="rbf",
            gamma=1.0,
            center=False,
        )

        kpcovr.fit(X, Y)
        K = kpcovr._get_kernel(X)

        y = kpcovr.predict(X)
        Lkrr = np.linalg.norm(Y - y) ** 2 / np.linalg.norm(Y) ** 2

        t = kpcovr.transform(X)

        w = t @ np.linalg.pinv(t.T @ t, rcond=kpcovr.tol) @ t.T
        Lkpca = np.trace(K - K @ w) / np.trace(K)

        # this is only true for in-sample data
        assert np.isclose(kpcovr.score(X, Y), -sum([Lkpca, Lkrr]), error_tol)


def test_nonfitted_failure(X):
    """
    Check that KernelPCovR will raise a `NonFittedError` if
    `transform` is called before the model is fitted
    """
    kpcovr = KernelPCovR(mixing=0.5, n_components=2, tol=1e-12)
    with pytest.raises(exceptions.NotFittedError):
        _ = kpcovr.transform(X)


def test_no_arg_predict(X, Y):
    """
    Check that KernelPCovR will raise a `ValueError` if
    `predict` is called without arguments
    """
    kpcovr = KernelPCovR(mixing=0.5, n_components=2, tol=1e-12)
    kpcovr.fit(X, Y)
    with pytest.raises(ValueError):
        _ = kpcovr.predict()


def test_T_shape(X, Y):
    """
    Check that KernelPCovR returns a latent space projection
    consistent with the shape of the input matrix
    """
    n_components = 5
    kpcovr = KernelPCovR(mixing=0.5, n_components=n_components, tol=1e-12)
    kpcovr.fit(X, Y)
    T = kpcovr.transform(X)
    assert check_X_y(X, T, multi_output=True) == (X, T)
    assert T.shape[-1] == n_components


def test_no_centerer(kpcovr_model, X, Y):
    """Tests that when center=False, no centerer exists."""
    kpcovr = kpcovr_model(center=False)
    kpcovr.fit(X, Y)

    with pytest.raises(AttributeError):
        kpcovr.centerer_


def test_centerer(kpcovr_model, X, Y):
    """Tests that all functionalities that rely on the centerer work properly."""
    kpcovr = kpcovr_model(center=True)
    kpcovr.fit(X, Y)

    assert hasattr(kpcovr, "centerer_")
    _ = kpcovr.predict(X)
    _ = kpcovr.transform(X)
    _ = kpcovr.score(X, Y)


def test_prefit_regressor(kpcovr_model, X, Y):
    regressor = KernelRidge(alpha=1e-8, kernel="rbf", gamma=0.1)
    regressor.fit(X, Y)
    kpcovr = kpcovr_model(mixing=0.5, regressor=regressor, kernel="rbf", gamma=0.1)
    kpcovr.fit(X, Y)

    Yhat_regressor = regressor.predict(X).reshape(X.shape[0], -1)
    W_regressor = regressor.dual_coef_.reshape(X.shape[0], -1)

    Yhat_kpcovr = kpcovr.regressor_.predict(X).reshape(X.shape[0], -1)
    W_kpcovr = kpcovr.regressor_.dual_coef_.reshape(X.shape[0], -1)

    np.testing.assert_allclose(Yhat_regressor, Yhat_kpcovr)
    np.testing.assert_allclose(W_regressor, W_kpcovr)


def test_regressor_modifications(kpcovr_model, X, Y):
    regressor = KernelRidge(alpha=1e-8, kernel="rbf", gamma=0.1)
    kpcovr = kpcovr_model(mixing=0.5, regressor=regressor, kernel="rbf", gamma=0.1)

    # KPCovR regressor matches the original
    assert regressor.get_params() == kpcovr.regressor.get_params()

    # KPCovR regressor updates its parameters
    # to match the original regressor
    regressor.set_params(gamma=0.2)
    assert regressor.get_params() == kpcovr.regressor.get_params()

    # Fitting regressor outside KPCovR fits the KPCovR regressor
    regressor.fit(X, Y)
    assert hasattr(kpcovr.regressor, "dual_coef_")

    # Raise error during KPCovR fit since regressor and KPCovR
    # kernel parameters now inconsistent
    with pytest.raises(ValueError) as context:
        kpcovr.fit(X, Y)
    assert str(context.value) == (
        "Kernel parameter mismatch: the regressor has kernel parameters "
        "{kernel: 'rbf', gamma: 0.2, degree: 3, coef0: 1, kernel_params: None}"
        " and KernelPCovR was initialized with kernel parameters "
        "{kernel: 'rbf', gamma: 0.1, degree: 3, coef0: 1, kernel_params: None}"
    )


def test_incompatible_regressor(kpcovr_model, X, Y):
    regressor = Ridge(alpha=1e-8)
    regressor.fit(X, Y)
    kpcovr = kpcovr_model(mixing=0.5, regressor=regressor)

    with pytest.raises(ValueError) as context:
        kpcovr.fit(X, Y)
    assert str(context.value) == "Regressor must be an instance of `KernelRidge`"


def test_none_regressor(X, Y):
    kpcovr = KernelPCovR(mixing=0.5, regressor=None)
    kpcovr.fit(X, Y)
    assert kpcovr.regressor is None
    assert kpcovr.regressor_ is not None


def test_incompatible_coef_shape(kpcovr_model, X, Y):
    # Y is 2D with two targets
    # Don't need to test X shape, since this should
    # be caught by sklearn's _validate_data
    regressor = KernelRidge(alpha=1e-8, kernel="linear")
    regressor.fit(X, Y[:, 0])
    kpcovr = kpcovr_model(mixing=0.5, regressor=regressor)

    # Dimension mismatch
    with pytest.raises(ValueError) as context:
        kpcovr.fit(X, Y)
    assert str(context.value) == (
        "The regressor coefficients have a dimension incompatible "
        "with the supplied target space. "
        "The coefficients have dimension %d and the targets "
        "have dimension %d" % (regressor.dual_coef_.ndim, Y.ndim)
    )

    Y_double = np.column_stack((Y, Y))
    Y_triple = np.column_stack((Y_double, Y))
    regressor.fit(X, Y_double)

    # Shape mismatch (number of targets)
    with pytest.raises(ValueError) as context:
        kpcovr.fit(X, Y_triple)
    assert str(context.value) == (
        "The regressor coefficients have a shape incompatible "
        "with the supplied target space. "
        "The coefficients have shape %r and the targets "
        "have shape %r" % (regressor.dual_coef_.shape, Y_triple.shape)
    )


def test_precomputed_regression(kpcovr_model, X, Y, error_tol):
    regressor = KernelRidge(alpha=1e-8, kernel="rbf", gamma=0.1)
    regressor.fit(X, Y)
    Yhat = regressor.predict(X)
    W = regressor.dual_coef_.reshape(X.shape[0], -1)

    kpcovr1 = kpcovr_model(
        mixing=0.5, regressor="precomputed", kernel="rbf", gamma=0.1, n_components=1
    )
    kpcovr1.fit(X, Yhat, W)
    t1 = kpcovr1.transform(X)

    kpcovr2 = kpcovr_model(
        mixing=0.5, regressor=regressor, kernel="rbf", gamma=0.1, n_components=1
    )
    kpcovr2.fit(X, Y)
    t2 = kpcovr2.transform(X)

    assert np.linalg.norm(t1 - t2) < error_tol


@pytest.mark.parametrize(
    "kernel,kernel_params",
    [
        ("linear", {}),
        ("poly", {"degree": 2}),
        ("rbf", {"gamma": 3.0}),
        ("sigmoid", {"gamma": 3.0, "coef0": 0.5}),
        ("cosine", {}),
    ],
)
def test_kernel_types(X, Y, kernel, kernel_params):
    """Check that KernelPCovR can handle all kernels passable to sklearn
    kernel classes.
    """
    kpcovr = KernelPCovR(
        mixing=0.5,
        n_components=2,
        regressor=KernelRidge(kernel=kernel, **kernel_params),
        kernel=kernel,
        **kernel_params,
    )
    kpcovr.fit(X, Y)


def test_kernel_types_callable(X, Y):
    """Test callable kernel."""

    def _linear_kernel(X, Y):
        return X @ Y.T

    kpcovr = KernelPCovR(
        mixing=0.5,
        n_components=2,
        regressor=KernelRidge(kernel=_linear_kernel),
        kernel=_linear_kernel,
    )
    kpcovr.fit(X, Y)


def test_linear_matches_pcovr(X, Y):
    """Check that KernelPCovR returns the same results as PCovR when using a linear
    kernel.
    """
    ridge = RidgeCV(fit_intercept=False, alphas=np.logspace(-8, 2))
    ridge.fit(X, Y)

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
        **hypers,
    )
    kpcovr.fit(X, Y)
    ly = np.linalg.norm(Y - kpcovr.predict(X)) ** 2.0 / np.linalg.norm(Y) ** 2.0

    # computing projection and predicton loss with PCovR
    ref_pcovr = PCovR(**hypers, regressor=ridge, space="sample")
    ref_pcovr.fit(X, Y)
    ly_ref = np.linalg.norm(Y - ref_pcovr.predict(X)) ** 2.0 / np.linalg.norm(Y) ** 2.0

    t_ref = ref_pcovr.transform(X)
    t = kpcovr.transform(X)

    K = kpcovr._get_kernel(X)

    k_ref = t_ref @ t_ref.T
    k = t @ t.T

    lk_ref = np.linalg.norm(K - k_ref) ** 2.0 / np.linalg.norm(K) ** 2.0
    lk = np.linalg.norm(K - k) ** 2.0 / np.linalg.norm(K) ** 2.0

    rounding = 3
    assert round(ly, rounding) == round(ly_ref, rounding)
    assert round(lk, rounding) == round(lk_ref, rounding)


@pytest.mark.parametrize("solver", ["arpack", "full", "randomized", "auto"])
def test_svd_solvers(kpcovr_model, X, Y, solver):
    """
    Check that PCovR works with all svd_solver modes and assigns
    the right n_components
    """
    kpcovr = kpcovr_model(tol=1e-12, n_components=None, svd_solver=solver)
    kpcovr.fit(X, Y)

    if solver == "arpack":
        assert kpcovr.n_components_ == X.shape[0] - 1
    else:
        assert kpcovr.n_components_ == X.shape[0]


@pytest.mark.parametrize(
    "n_components,expected_solver",
    [
        ("mle", "full"),
        (0.1, "full"),
    ],
)
def test_svd_solver_selection(kpcovr_model, X, Y, n_components, expected_solver):
    """Test automatic SVD solver selection."""
    kpcovr = kpcovr_model(tol=1e-12, n_components=n_components, svd_solver="auto")
    kpcovr.fit(X, Y)
    assert kpcovr.fit_svd_solver_ == expected_solver


def test_svd_solver_randomized(kpcovr_model, X, Y, random_state):
    """Test randomized solver with large n_components."""
    n_components = int(0.75 * max(X.shape))
    expected_solver = "randomized"

    kpcovr = kpcovr_model(tol=1e-12, n_components=n_components, svd_solver="auto")
    n_copies = (501 // max(X.shape)) + 1
    X_large = np.hstack(np.repeat(X.copy(), n_copies)).reshape(
        X.shape[0] * n_copies, -1
    )
    Y_large = np.hstack(np.repeat(Y.copy(), n_copies)).reshape(
        X.shape[0] * n_copies, -1
    )
    kpcovr.fit(X_large, Y_large)
    assert kpcovr.fit_svd_solver_ == expected_solver


def test_bad_solver(kpcovr_model, X, Y):
    """
    Check that PCovR will not work with a solver that isn't in
    ['arpack', 'full', 'randomized', 'auto']
    """
    with pytest.raises(ValueError) as context:
        kpcovr = kpcovr_model(svd_solver="bad")
        kpcovr.fit(X, Y)

    assert str(context.value) == "Unrecognized svd_solver='bad'"


def test_good_n_components(kpcovr_model, X, Y):
    """Check that PCovR will work with any allowed values of n_components."""
    # this one should pass
    kpcovr = kpcovr_model(n_components=0.5, svd_solver="full")
    kpcovr.fit(X, Y)

    for svd_solver in ["auto", "full"]:
        # this one should pass
        kpcovr = kpcovr_model(n_components=2, svd_solver=svd_solver)
        kpcovr.fit(X, Y)

        # this one should pass
        kpcovr = kpcovr_model(n_components="mle", svd_solver=svd_solver)
        kpcovr.fit(X, Y)


def test_bad_n_components_negative(kpcovr_model, X, Y):
    """Check that PCovR rejects negative n_components."""
    with pytest.raises(ValueError) as context:
        kpcovr = kpcovr_model(n_components=-1, svd_solver="auto")
        kpcovr.fit(X, Y)

    assert str(context.value) == (
        "n_components=%r must be between 1 and "
        "n_samples=%r with "
        "svd_solver='%s'" % (-1, X.shape[0], "auto")
    )


def test_bad_n_components_zero(kpcovr_model, X, Y):
    """Check that PCovR rejects zero n_components."""
    with pytest.raises(ValueError) as context:
        kpcovr = kpcovr_model(n_components=0, svd_solver="randomized")
        kpcovr.fit(X, Y)

    assert str(context.value) == (
        "n_components=%r must be between 1 and "
        "n_samples=%r with "
        "svd_solver='%s'" % (0, X.shape[0], "randomized")
    )


def test_bad_n_components_arpack(kpcovr_model, X, Y):
    """Check that PCovR rejects n_components >= n_samples with arpack."""
    with pytest.raises(ValueError) as context:
        kpcovr = kpcovr_model(n_components=X.shape[0], svd_solver="arpack")
        kpcovr.fit(X, Y)

    assert str(context.value) == (
        "n_components=%r must be strictly less than "
        "n_samples=%r with "
        "svd_solver='%s'" % (X.shape[0], X.shape[0], "arpack")
    )


@pytest.mark.parametrize("svd_solver", ["auto", "full"])
def test_bad_n_components_float(kpcovr_model, X, Y, svd_solver):
    """Check that PCovR rejects non-integer n_components >= 1."""
    with pytest.raises(ValueError) as context:
        kpcovr = kpcovr_model(n_components=np.pi, svd_solver=svd_solver)
        kpcovr.fit(X, Y)

    assert str(context.value) == (
        "n_components=%r must be of type int "
        "when greater than or equal to 1, was of type=%r" % (np.pi, type(np.pi))
    )
