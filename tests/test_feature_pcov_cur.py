import numpy as np
import pytest
from sklearn.datasets import load_diabetes as get_dataset

from skmatter.feature_selection import PCovCUR


@pytest.fixture
def X_y_idx():
    X, y = get_dataset(return_X_y=True)
    idx = [2, 8, 3, 4, 1, 7, 5, 9, 6]
    return X, y, idx


def test_known(X_y_idx):
    """Check that the model returns a known set of indices."""
    X, y, idx = X_y_idx
    selector = PCovCUR(n_to_select=9)
    selector.fit(X, y)

    assert np.allclose(selector.selected_idx_, idx)


def test_restart(X_y_idx):
    """Check that the model can be restarted with a new instance."""
    X, y, idx = X_y_idx
    selector = PCovCUR(n_to_select=1)
    selector.fit(X, y)

    for i in range(len(idx) - 2):
        selector.n_to_select += 1
        selector.fit(X, y, warm_start=True)
        assert selector.selected_idx_[i] == idx[i]


def test_non_it(X_y_idx):
    """Check that the model can be run non-iteratively."""
    X, y, _ = X_y_idx
    idx = [2, 8, 3, 6, 7, 9, 1, 0, 5]
    selector = PCovCUR(n_to_select=9, recompute_every=0)
    selector.fit(X, y)
    assert np.allclose(selector.selected_idx_, idx)
