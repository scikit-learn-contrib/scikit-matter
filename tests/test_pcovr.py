import numpy as np
import pytest
from sklearn import exceptions
from sklearn.datasets import load_diabetes as get_dataset
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y

from skmatter.decomposition import PCovR


@pytest.fixture(scope="module")
def pcovr_model():
    """Factory fixture for PCovR model."""

    def _model(
        mixing=0.5,
        regressor=Ridge(alpha=1e-8, fit_intercept=False, tol=1e-12),
        **kwargs,
    ):
        return PCovR(mixing, regressor=regressor, **kwargs)

    return _model


@pytest.fixture(scope="module")
def error_tol():
    """Error tolerance for tests."""
    return 1e-5


@pytest.fixture(scope="module")
def X():
    """Feature matrix."""
    X, _ = get_dataset(return_X_y=True)
    return StandardScaler().fit_transform(X)


@pytest.fixture(scope="module")
def Y():
    """Target vector."""
    _, Y = get_dataset(return_X_y=True)
    return StandardScaler().fit_transform(np.vstack(Y)).ravel()


def test_against_pca(X, Y):
    """Tests that mixing = 1.0 corresponds to PCA."""
    pcovr = PCovR(mixing=1.0, n_components=3, space="sample", svd_solver="full").fit(
        X, Y
    )
    pca = PCA(n_components=3, svd_solver="full").fit(X)

    # tests that the SVD is equivalent
    np.testing.assert_allclose(pca.singular_values_, pcovr.singular_values_)
    np.testing.assert_allclose(pca.explained_variance_, pcovr.explained_variance_)

    T_pcovr = pcovr.transform(X)
    T_pca = pca.transform(X)

    # tests that the projections are equivalent
    assert np.linalg.norm(T_pcovr @ T_pcovr.T - T_pca @ T_pca.T) <= 1e-8


@pytest.mark.parametrize("space", ["feature", "sample", "auto"])
def test_simple_reconstruction(pcovr_model, X, Y, error_tol, space):
    """Check that PCovR with a full eigendecomposition at mixing=1 can fully
    reconstruct the input matrix.
    """
    pcovr = pcovr_model(mixing=1.0, n_components=X.shape[-1], space=space)
    pcovr.fit(X, Y)
    Xr = pcovr.inverse_transform(pcovr.transform(X))
    assert np.linalg.norm(X - Xr) ** 2.0 / np.linalg.norm(X) ** 2.0 <= error_tol, (
        f"Reconstruction error too high for space={space}"
    )


@pytest.mark.parametrize("space", ["feature", "sample", "auto"])
def test_simple_prediction(pcovr_model, X, Y, error_tol, space):
    """
    Check that PCovR with a full eigendecomposition at mixing=0
    can reproduce a linear regression result.
    """
    pcovr = pcovr_model(mixing=0.0, n_components=1, space=space)

    pcovr.regressor.fit(X, Y)
    Yhat = pcovr.regressor.predict(X)

    pcovr.fit(X, Y)
    Yp = pcovr.predict(X)
    assert (
        np.linalg.norm(Yp - Yhat) ** 2.0 / np.linalg.norm(Yhat) ** 2.0 <= error_tol
    ), f"Prediction error too high for space={space}"


def test_lr_with_x_errors(pcovr_model, X, Y, error_tol):
    """
    Check that PCovR returns a non-null property prediction
    and that the prediction error increases with `mixing`
    """
    prev_error = -1.0

    for mixing in np.linspace(0, 1, 11):
        pcovr = pcovr_model(mixing=mixing, n_components=2, tol=1e-12)
        pcovr.fit(X, Y)

        Yp = pcovr.predict(X=X)
        error = np.linalg.norm(Y - Yp) ** 2.0 / np.linalg.norm(Y) ** 2.0

        assert not np.isnan(error), f"Error is NaN for mixing={mixing}"
        assert error >= prev_error - error_tol, (
            f"Error decreased unexpectedly at mixing={round(mixing, 4)}"
        )

        prev_error = error


def test_lr_with_t_errors(pcovr_model, X, Y, error_tol):
    """Check that PCovR returns a non-null property prediction from the latent space
    projection and that the prediction error increases with `mixing`.
    """
    prev_error = -1.0

    for mixing in np.linspace(0, 1, 11):
        pcovr = pcovr_model(mixing=mixing, n_components=2, tol=1e-12)
        pcovr.fit(X, Y)

        T = pcovr.transform(X)
        Yp = pcovr.predict(T=T)
        error = np.linalg.norm(Y - Yp) ** 2.0 / np.linalg.norm(Y) ** 2.0

        assert not np.isnan(error), f"Error is NaN for mixing={mixing}"
        assert error >= prev_error - error_tol, (
            f"Error decreased unexpectedly at mixing={round(mixing, 4)}"
        )

        prev_error = error


def test_reconstruction_errors(pcovr_model, X, Y, error_tol):
    """Check that PCovR returns a non-null reconstructed X and that the
    reconstruction error decreases with `mixing`.
    """
    prev_error = 1.0

    for mixing in np.linspace(0, 1, 11):
        pcovr = pcovr_model(mixing=mixing, n_components=2, tol=1e-12)
        pcovr.fit(X, Y)

        Xr = pcovr.inverse_transform(pcovr.transform(X))
        error = np.linalg.norm(X - Xr) ** 2.0 / np.linalg.norm(X) ** 2.0

        assert not np.isnan(error), f"Error is NaN for mixing={mixing}"
        assert error <= prev_error + error_tol, (
            f"Error increased unexpectedly at mixing={round(mixing, 4)}"
        )

        prev_error = error


def test_select_feature_space(pcovr_model, X, Y):
    """
    Check that PCovR implements the feature space version
    when :math:`n_{features} < n_{samples}``.
    """
    pcovr = pcovr_model(n_components=2, tol=1e-12)
    pcovr.fit(X, Y)

    assert pcovr.space_ == "feature"


def test_select_sample_space(pcovr_model, X, Y):
    """
    Check that PCovR implements the sample space version
    when :math:`n_{features} > n_{samples}``.
    """
    pcovr = pcovr_model(n_components=2, tol=1e-12)

    n_samples = X.shape[1] - 1

    with pytest.warns(match="class does not automatically center data"):
        pcovr.fit(X[:n_samples], Y[:n_samples])

    assert pcovr.space_ == "sample"


def test_bad_space(pcovr_model, X, Y):
    """
    Check that PCovR raises a ValueError when a non-valid
    space is designated.
    """
    with pytest.raises(ValueError):
        pcovr = pcovr_model(n_components=2, tol=1e-12, space="bad")
        pcovr.fit(X, Y)


def test_override_spaceselection(pcovr_model, X, Y):
    """
    Check that PCovR implements the space provided in the
    constructor, overriding that chosen by the input dimensions.
    """
    pcovr = pcovr_model(n_components=2, tol=1e-12, space="sample")
    pcovr.fit(X, Y)

    assert pcovr.space_ == "sample"


@pytest.mark.parametrize("alpha", np.linspace(0.01, 0.99, 11))
def test_spaces_equivalent_prediction(pcovr_model, X, Y, error_tol, alpha):
    """
    Check that the results from PCovR, regardless of the space,
    are equivalent for prediction.
    """
    pcovr_ss = pcovr_model(n_components=2, mixing=alpha, tol=1e-12, space="sample")
    pcovr_ss.fit(X, Y)

    pcovr_fs = pcovr_model(n_components=2, mixing=alpha, tol=1e-12, space="feature")
    pcovr_fs.fit(X, Y)

    np.testing.assert_allclose(pcovr_ss.predict(X), pcovr_fs.predict(X), atol=error_tol)


@pytest.mark.parametrize("alpha", np.linspace(0.01, 0.99, 11))
def test_spaces_equivalent_reconstruction(pcovr_model, X, Y, error_tol, alpha):
    """
    Check that the results from PCovR, regardless of the space,
    are equivalent for reconstruction.
    """
    pcovr_ss = pcovr_model(n_components=2, mixing=alpha, tol=1e-12, space="sample")
    pcovr_ss.fit(X, Y)

    pcovr_fs = pcovr_model(n_components=2, mixing=alpha, tol=1e-12, space="feature")
    pcovr_fs.fit(X, Y)

    np.testing.assert_allclose(
        pcovr_ss.inverse_transform(pcovr_ss.transform(X)),
        pcovr_fs.inverse_transform(pcovr_fs.transform(X)),
        atol=error_tol,
    )


@pytest.mark.parametrize("solver", ["arpack", "full", "randomized", "auto"])
def test_svd_solvers(pcovr_model, X, Y, solver):
    """
    Check that PCovR works with all svd_solver modes and assigns
    the right n_components
    """
    pcovr = pcovr_model(tol=1e-12, svd_solver=solver)
    pcovr.fit(X, Y)

    if solver == "arpack":
        assert pcovr.n_components_ == min(X.shape) - 1
    else:
        assert pcovr.n_components_ == min(X.shape)


@pytest.mark.parametrize("space", ["feature", "sample"])
def test_bad_solver(pcovr_model, X, Y, space):
    """
    Check that PCovR will not work with a solver that isn't in
    ['arpack', 'full', 'randomized', 'auto']
    """
    with pytest.raises(ValueError) as context:
        pcovr = pcovr_model(svd_solver="bad", space=space)
        pcovr.fit(X, Y)

    assert str(context.value) == "Unrecognized svd_solver='bad'"


def test_good_n_components(pcovr_model, X, Y):
    """Check that PCovR will work with any allowed values of n_components."""
    # this one should pass
    pcovr = pcovr_model(n_components=0.5, svd_solver="full")
    pcovr.fit(X, Y)

    for svd_solver in ["auto", "full"]:
        # this one should pass
        pcovr = pcovr_model(n_components=2, svd_solver=svd_solver)
        pcovr.fit(X, Y)

        # this one should pass
        pcovr = pcovr_model(n_components="mle", svd_solver=svd_solver)
        pcovr.fit(X, Y)


def test_bad_n_components_mle(pcovr_model, X, Y):
    """Check that PCovR will not work with mle when n_samples < n_features."""
    pcovr = pcovr_model(n_components="mle", svd_solver="full")
    m = "n_components='mle' is only supported if n_samples >= n_features"
    with pytest.raises(ValueError, match=m):
        with pytest.warns(match="class does not automatically center data"):
            pcovr.fit(X[:2], Y[:2])


def test_bad_n_components_negative(pcovr_model, X, Y):
    """Check that PCovR rejects negative n_components."""
    with pytest.raises(ValueError) as context:
        pcovr = pcovr_model(n_components=-1, svd_solver="auto")
        pcovr.fit(X, Y)

    assert str(context.value) == (
        "n_components=%r must be between 1 and "
        "min(n_samples, n_features)=%r with "
        "svd_solver='%s'" % (-1, min(X.shape), "auto")
    )


def test_bad_n_components_zero(pcovr_model, X, Y):
    """Check that PCovR rejects zero n_components."""
    with pytest.raises(ValueError) as context:
        pcovr = pcovr_model(n_components=0, svd_solver="randomized")
        pcovr.fit(X, Y)

    assert str(context.value) == (
        "n_components=%r must be between 1 and "
        "min(n_samples, n_features)=%r with "
        "svd_solver='%s'" % (0, min(X.shape), "randomized")
    )


def test_bad_n_components_arpack(pcovr_model, X, Y):
    """Check that PCovR rejects n_components >= min(shape) with arpack."""
    with pytest.raises(ValueError) as context:
        pcovr = pcovr_model(n_components=min(X.shape), svd_solver="arpack")
        pcovr.fit(X, Y)

    assert str(context.value) == (
        "n_components=%r must be strictly less than "
        "min(n_samples, n_features)=%r with "
        "svd_solver='%s'" % (min(X.shape), min(X.shape), "arpack")
    )


@pytest.mark.parametrize("svd_solver", ["auto", "full"])
def test_bad_n_components_float(pcovr_model, X, Y, svd_solver):
    """Check that PCovR rejects non-integer n_components >= 1."""
    with pytest.raises(ValueError) as context:
        pcovr = pcovr_model(n_components=np.pi, svd_solver=svd_solver)
        pcovr.fit(X, Y)

    assert str(context.value) == (
        "n_components=%r must be of type int "
        "when greater than or equal to 1, was of type=%r" % (np.pi, type(np.pi))
    )


def test_nonfitted_failure(pcovr_model, X):
    """
    Check that PCovR will raise a `NonFittedError` if
    `transform` is called before the pcovr is fitted
    """
    pcovr = pcovr_model(n_components=2, tol=1e-12)
    with pytest.raises(exceptions.NotFittedError):
        _ = pcovr.transform(X)


def test_no_arg_predict(pcovr_model, X, Y):
    """
    Check that PCovR will raise a `ValueError` if
    `predict` is called without arguments
    """
    pcovr = pcovr_model(n_components=2, tol=1e-12)
    pcovr.fit(X, Y)
    with pytest.raises(ValueError):
        _ = pcovr.predict()


def test_centering(pcovr_model, X, Y):
    """
    Check that PCovR raises a warning if
    given uncentered data.
    """
    pcovr = pcovr_model(n_components=2, tol=1e-12)
    X_uncentered = X.copy() + np.random.uniform(-1, 1, X.shape[1])
    m = (
        "This class does not automatically center data, and your data mean is "
        "greater than the supplied tolerance."
    )
    with pytest.warns(match=m):
        pcovr.fit(X_uncentered, Y)


def test_T_shape(pcovr_model, X, Y):
    """Check that PCovR returns a latent space projection consistent with the shape
    of the input matrix.
    """
    n_components = 5
    pcovr = pcovr_model(n_components=n_components, tol=1e-12)
    pcovr.fit(X, Y)
    T = pcovr.transform(X)
    assert check_X_y(X, T, multi_output=True) == (X, T)
    assert T.shape[-1] == n_components


def test_default_ncomponents(X, Y):
    pcovr = PCovR(mixing=0.5)
    pcovr.fit(X, Y)

    assert pcovr.n_components_ == min(X.shape)


def test_Y_shape(pcovr_model, X, Y):
    pcovr = pcovr_model()
    Y_2d = np.vstack(Y)
    pcovr.fit(X, Y_2d)

    assert pcovr.pxy_.shape[0] == X.shape[1]
    assert pcovr.pty_.shape[0] == pcovr.n_components_


def test_prefit_regressor(pcovr_model, X, Y):
    regressor = Ridge(alpha=1e-8, fit_intercept=False, tol=1e-12)
    regressor.fit(X, Y)
    pcovr = pcovr_model(mixing=0.5, regressor=regressor)
    pcovr.fit(X, Y)

    Yhat_regressor = regressor.predict(X).reshape(X.shape[0], -1)
    W_regressor = regressor.coef_.T.reshape(X.shape[1], -1)

    Yhat_pcovr = pcovr.regressor_.predict(X).reshape(X.shape[0], -1)
    W_pcovr = pcovr.regressor_.coef_.T.reshape(X.shape[1], -1)

    np.testing.assert_allclose(Yhat_regressor, Yhat_pcovr)
    np.testing.assert_allclose(W_regressor, W_pcovr)


def test_prefit_regression(pcovr_model, X, Y, error_tol):
    regressor = Ridge(alpha=1e-8, fit_intercept=False, tol=1e-12)
    regressor.fit(X, Y)
    Yhat = regressor.predict(X)
    W = regressor.coef_.reshape(X.shape[1], -1)

    pcovr1 = pcovr_model(mixing=0.5, regressor="precomputed", n_components=1)
    pcovr1.fit(X, Yhat, W)
    t1 = pcovr1.transform(X)

    pcovr2 = pcovr_model(mixing=0.5, regressor=regressor, n_components=1)
    pcovr2.fit(X, Y)
    t2 = pcovr2.transform(X)

    assert np.linalg.norm(t1 - t2) < error_tol


def test_regressor_modifications(pcovr_model, X, Y):
    regressor = Ridge(alpha=1e-8)
    pcovr = pcovr_model(mixing=0.5, regressor=regressor)

    # PCovR regressor matches the original
    assert regressor.get_params() == pcovr.regressor.get_params()

    # PCovR regressor updates its parameters
    # to match the original regressor
    regressor.set_params(alpha=1e-6)
    assert regressor.get_params() == pcovr.regressor.get_params()

    # Fitting regressor outside PCovR fits the PCovR regressor
    regressor.fit(X, Y)
    assert hasattr(pcovr.regressor, "coef_")

    # PCovR regressor doesn't change after fitting
    pcovr.fit(X, Y)
    regressor.set_params(alpha=1e-4)
    assert hasattr(pcovr.regressor_, "coef_")
    assert regressor.get_params() != pcovr.regressor_.get_params()


def test_incompatible_regressor(pcovr_model, X, Y):
    regressor = KernelRidge(alpha=1e-8, kernel="linear")
    regressor.fit(X, Y)
    pcovr = pcovr_model(mixing=0.5, regressor=regressor)

    with pytest.raises(ValueError) as context:
        pcovr.fit(X, Y)

    assert str(context.value) == (
        "Regressor must be an instance of `LinearRegression`, `Ridge`, `RidgeCV`, "
        "or `precomputed`"
    )


def test_none_regressor(X, Y):
    pcovr = PCovR(mixing=0.5, regressor=None)
    pcovr.fit(X, Y)
    assert pcovr.regressor is None
    assert pcovr.regressor_ is not None


def test_incompatible_coef_dim(pcovr_model, X, Y):
    # Y is 1D with one target
    # Don't need to test X shape, since this should
    # be caught by sklearn's validate_data
    Y_2D = np.column_stack((Y, Y))
    regressor = Ridge(alpha=1e-8, fit_intercept=False, tol=1e-12)
    regressor.fit(X, Y_2D)
    pcovr = pcovr_model(mixing=0.5, regressor=regressor)

    # Dimension mismatch
    with pytest.raises(ValueError) as context:
        pcovr.fit(X, Y)

    assert str(context.value) == (
        "The regressor coefficients have a dimension incompatible with the "
        "supplied target space. The coefficients have dimension 2 and the targets "
        "have dimension 1"
    )


def test_incompatible_coef_shape(pcovr_model, X, Y):
    # Shape mismatch (number of targets)
    Y_double = np.column_stack((Y, Y))
    Y_triple = np.column_stack((Y_double, Y))

    regressor = Ridge(alpha=1e-8, fit_intercept=False, tol=1e-12)
    regressor.fit(X, Y_double)

    pcovr = pcovr_model(mixing=0.5, regressor=regressor)

    with pytest.raises(ValueError) as context:
        pcovr.fit(X, Y_triple)

    assert str(context.value) == (
        "The regressor coefficients have a shape incompatible with the supplied "
        "target space. The coefficients have shape %r and the targets have shape %r"
        % (regressor.coef_.shape, Y_triple.shape)
    )
