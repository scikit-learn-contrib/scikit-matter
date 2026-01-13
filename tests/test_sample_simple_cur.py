import numpy as np
import pytest
from sklearn.datasets import load_diabetes as load

from skmatter.sample_selection import CUR, FPS


@pytest.fixture
def X_and_n_select():
    X, _ = load(return_X_y=True)
    X = X[FPS(n_to_select=100).fit(X).selected_idx_]
    n_select = min(20, min(X.shape) // 2)
    return X, n_select


def test_sample_transform(X_and_n_select):
    """
    Check that an error is raised when the transform function is used,
    because sklearn does not support well transformers that change the number
    of samples with other classes like Pipeline
    """
    X, _ = X_and_n_select
    selector = CUR(n_to_select=1)
    selector.fit(X)
    with pytest.raises(ValueError) as error:
        selector.transform(X)

    assert "Transform is not currently supported for sample selection." == str(
        error.value
    )


def test_restart(X_and_n_select):
    """Check that the model can be restarted with a new instance"""
    X, n_select = X_and_n_select
    ref_selector = CUR(n_to_select=n_select)
    ref_idx = ref_selector.fit(X).selected_idx_

    selector = CUR(n_to_select=1)
    selector.fit(X)

    for i in range(len(ref_idx) - 2):
        selector.n_to_select += 1
        selector.fit(X, warm_start=True)
        assert selector.selected_idx_[i] == ref_idx[i]


def test_non_it(X_and_n_select):
    """Check that the model can be run non-iteratively."""
    X, n_select = X_and_n_select
    K = X @ X.T
    _, UK = np.linalg.eigh(K)
    ref_idx = np.argsort(-(UK[:, -1] ** 2.0))[:n_select]

    selector = CUR(n_to_select=len(ref_idx), recompute_every=0)
    selector.fit(X)

    np.testing.assert_allclose(selector.selected_idx_, ref_idx)


def test_unique_selected_idx_zero_score():
    """
    Tests that the selected idxs are unique, which may not be the
    case when the score is numerically zero.
    """
    np.random.seed(0)
    n_samples = 10
    n_features = 15
    X = np.random.rand(n_samples, n_features)
    X[1] = X[0]
    X[2] = X[0]
    X[3] = X[0]
    selector_problem = CUR(n_to_select=len(X)).fit(X)
    assert len(selector_problem.selected_idx_) == len(
        set(selector_problem.selected_idx_)
    )
