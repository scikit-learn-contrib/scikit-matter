import numpy as np
import pytest

from skmatter.feature_selection import FPS
from skmatter.neighbors import SparseKDE
from skmatter.neighbors._sparsekde import _covariance
from skmatter.utils import effdim, oas


@pytest.fixture(scope="module")
def sparse_kde_data():
    np.random.seed(0)
    n_samples_per_cov = 10000
    samples = np.concatenate(
        [
            np.random.multivariate_normal(
                [0, 0], [[1, 0.5], [0.5, 1]], n_samples_per_cov
            ),
            np.random.multivariate_normal(
                [4, 4], [[1, 0.5], [0.5, 0.5]], n_samples_per_cov
            ),
        ]
    )
    sample_results = np.array([[4.56393465, 4.20566218], [0.73562454, 1.11116178]])
    selector = FPS(n_to_select=int(np.sqrt(2 * n_samples_per_cov)))
    grids = selector.fit_transform(samples.T).T
    expect_score_fp = -759.831
    expect_score_fs = -781.567
    cell = np.array([4, 4])
    expect_score_periodic = -456.744
    return {
        "samples": samples,
        "sample_results": sample_results,
        "grids": grids,
        "expect_score_fp": expect_score_fp,
        "expect_score_fs": expect_score_fs,
        "cell": cell,
        "expect_score_periodic": expect_score_periodic,
    }


def test_sparse_kde(sparse_kde_data):
    estimator = SparseKDE(sparse_kde_data["samples"], None, fpoints=0.5)
    estimator.fit(sparse_kde_data["grids"])
    assert (
        round(estimator.score(sparse_kde_data["grids"]), 3)
        == sparse_kde_data["expect_score_fp"]
    )
    np.testing.assert_allclose(estimator.sample(2), sparse_kde_data["sample_results"])


def test_sparce_kde_fs(sparse_kde_data):
    estimator = SparseKDE(sparse_kde_data["samples"], None, fspread=0.5)
    estimator.fit(sparse_kde_data["grids"])
    assert (
        round(estimator.score(sparse_kde_data["grids"]), 3)
        == sparse_kde_data["expect_score_fs"]
    )


def test_sparse_kde_periodic(sparse_kde_data):
    estimator = SparseKDE(
        sparse_kde_data["samples"],
        None,
        metric_params={"cell_length": sparse_kde_data["cell"]},
        fpoints=0.5,
    )
    estimator.fit(sparse_kde_data["grids"])
    assert (
        round(estimator.score(sparse_kde_data["grids"]), 3)
        == sparse_kde_data["expect_score_periodic"]
    )


def test_dimension_check(sparse_kde_data):
    estimator = SparseKDE(
        sparse_kde_data["samples"],
        None,
        metric_params={"cell_length": sparse_kde_data["cell"]},
        fpoints=0.5,
    )
    with pytest.raises(ValueError, match="Cell dimension.*does not match"):
        estimator.fit(np.array([[4]]))


def test_fs_fp_imcompatibility(sparse_kde_data):
    estimator = SparseKDE(
        sparse_kde_data["samples"],
        None,
        metric_params={"cell_length": sparse_kde_data["cell"]},
        fspread=2,
        fpoints=0.5,
    )
    assert estimator.fpoints == -1


@pytest.fixture(scope="module")
def covariance_data():
    X = np.array([[1, 2], [3, 3], [4, 6]])
    expected_cov = np.array([[2.33333333, 2.83333333], [2.83333333, 4.33333333]])
    expected_cov_periodic = np.array(
        [[1.12597216, 0.45645371], [0.45645371, 0.82318948]]
    )
    cell = np.array([3, 3])
    return {
        "X": X,
        "expected_cov": expected_cov,
        "expected_cov_periodic": expected_cov_periodic,
        "cell": cell,
    }


def test_covariance(covariance_data):
    cov = _covariance(
        covariance_data["X"],
        np.full(len(covariance_data["X"]), 1 / len(covariance_data["X"])),
        None,
    )
    np.testing.assert_allclose(cov, covariance_data["expected_cov"])


def test_covariance_periodic(covariance_data):
    cov = _covariance(
        covariance_data["X"],
        np.full(len(covariance_data["X"]), 1 / len(covariance_data["X"])),
        covariance_data["cell"],
    )
    np.testing.assert_allclose(cov, covariance_data["expected_cov_periodic"])


def test_effdim():
    cov = np.array([[1, 1, 0], [1, 1.5, 0], [0, 0, 1]], dtype=np.float64)
    expected_effdim = 2.24909102090124
    np.testing.assert_allclose(effdim(cov), expected_effdim)


def test_oas():
    cov = np.array([[0.5, 1.0], [0.7, 0.4]])
    n = 10
    D = 2
    expected_oas = np.array([[0.48903924, 0.78078484], [0.54654939, 0.41096076]])
    np.testing.assert_allclose(oas(cov, n, D), expected_oas)
