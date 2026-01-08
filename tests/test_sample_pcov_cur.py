import numpy as np
import pytest
from sklearn.datasets import load_diabetes as get_dataset

from skmatter.sample_selection import PCovCUR


EPSILON = 1e-6


@pytest.fixture
def X_y_idx():
    X, y = get_dataset(return_X_y=True)
    X = X[:, :4]
    idx = [256, 304, 41, 408, 311, 364, 152, 78, 359, 102]
    return X, y, idx


def test_known(X_y_idx):
    """Check that the model returns a known set of indices."""
    X, y, idx = X_y_idx
    selector = PCovCUR(n_to_select=10, mixing=0.5)
    selector.fit(X, y)

    np.testing.assert_allclose(selector.selected_idx_, idx)


def test_restart(X_y_idx):
    """Check that the model can be restarted with a new instance."""
    X, y, idx = X_y_idx
    selector = PCovCUR(n_to_select=1, mixing=0.5)
    selector.fit(X, y)

    for i in range(len(idx) - 2):
        selector.n_to_select += 1
        selector.fit(X, y, warm_start=True)
        assert selector.selected_idx_[i] == idx[i]

        assert np.linalg.norm(selector.X_current_[idx[i]]) <= EPSILON

        for j in range(X.shape[0]):
            assert (
                np.dot(selector.X_current_[idx[i]], selector.X_current_[j]) <= EPSILON
            )


def test_non_it(X_y_idx):
    """Check that the model can be run non-iteratively."""
    X, y, _ = X_y_idx
    idx = [256, 32, 138, 290, 362, 141, 359, 428, 254, 9]
    selector = PCovCUR(n_to_select=10, recompute_every=0)
    selector.fit(X, y)

    np.testing.assert_allclose(selector.selected_idx_, idx)


def test_multiple_k(X_y_idx):
    """Check that the model can be run with multiple k's."""
    X, y, _ = X_y_idx
    for k in list(set(np.logspace(0, np.log10(min(X.shape)), 4, dtype=int))):
        selector = PCovCUR(n_to_select=10, k=k)
        selector.fit(X, y)
