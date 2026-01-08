import numpy as np
import pytest
from sklearn import exceptions

from skmatter.datasets import load_csd_1000r as load
from skmatter.feature_selection import CUR, FPS


@pytest.fixture
def X():
    X, _ = load(return_X_y=True)
    return FPS(n_to_select=10).fit(X).transform(X)


def test_bad_transform(X):
    selector = CUR(n_to_select=2)
    with pytest.raises(exceptions.NotFittedError):
        _ = selector.transform(X)


def test_restart(X):
    """Check that the model can be restarted with a new instance."""
    ref_selector = CUR(n_to_select=X.shape[-1] - 3).fit(X=X)
    ref_idx = ref_selector.selected_idx_

    selector = CUR(n_to_select=1)
    selector.fit(X)

    for i in range(X.shape[-1] - 3):
        selector.n_to_select += 1
        selector.fit(X, warm_start=True)
        assert selector.selected_idx_[i] == ref_idx[i]


def test_non_it(X):
    """Check that the model can be run non-iteratively."""
    C = X.T @ X
    _, UC = np.linalg.eigh(C)
    ref_idx = np.argsort(-(UC[:, -1] ** 2.0))[:-1]

    selector = CUR(n_to_select=X.shape[-1] - 1, recompute_every=0)
    selector.fit(X)

    assert np.allclose(selector.selected_idx_, ref_idx)


def test_unique_selected_idx_zero_score():
    """
    Tests that the selected idxs are unique, which may not be the
    case when the score is numerically zero
    """
    np.random.seed(0)
    n_samples = 10
    n_features = 15
    X = np.random.rand(n_samples, n_features)
    X[:, 1] = X[:, 0]
    X[:, 2] = X[:, 0]
    selector_problem = CUR(n_to_select=len(X.T)).fit(X)
    assert len(selector_problem.selected_idx_) == len(
        set(selector_problem.selected_idx_)
    )
