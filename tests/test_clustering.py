import numpy as np
import pytest

from skmatter.clustering import QuickShift


@pytest.fixture(scope="module")
def test_data():
    points = np.array(
        [
            [-1.72779275, -1.32763554],
            [-4.44991964, -2.13474901],
            [0.54817734, -2.43319467],
            [3.19881307, -0.49547222],
            [-1.1335991, 2.33478428],
            [0.55437388, 0.18745963],
        ]
    )
    cuts = np.array(
        [6.99485011, 8.80292681, 7.68486852, 9.5115009, 8.07736919, 6.22057056]
    )
    weights = np.array(
        [
            -3.94008092,
            -12.68095664,
            -7.07512499,
            -9.03064023,
            -8.26529849,
            -2.61132267,
        ]
    )
    qs_labels_ = np.array([0, 0, 0, 5, 5, 5])
    qs_cluster_centers_idx_ = np.array([0, 5])
    gabriel_labels_ = np.array([5, 5, 5, 5, 5, 5])
    gabriel_cluster_centers_idx_ = np.array([5])
    cell = [3, 3]
    gabriel_shell = 2

    return {
        "points": points,
        "cuts": cuts,
        "weights": weights,
        "qs_labels_": qs_labels_,
        "qs_cluster_centers_idx_": qs_cluster_centers_idx_,
        "gabriel_labels_": gabriel_labels_,
        "gabriel_cluster_centers_idx_": gabriel_cluster_centers_idx_,
        "cell": cell,
        "gabriel_shell": gabriel_shell,
    }


def test_fit_qs(test_data):
    model = QuickShift(dist_cutoff_sq=test_data["cuts"])
    model.fit(test_data["points"], samples_weight=test_data["weights"])
    assert np.all(model.labels_ == test_data["qs_labels_"])
    assert np.all(model.cluster_centers_idx_ == test_data["qs_cluster_centers_idx_"])


def test_fit_garbriel(test_data):
    model = QuickShift(gabriel_shell=test_data["gabriel_shell"])
    model.fit(test_data["points"], samples_weight=test_data["weights"])
    assert np.all(model.labels_ == test_data["gabriel_labels_"])
    assert np.all(
        model.cluster_centers_idx_ == test_data["gabriel_cluster_centers_idx_"]
    )


def test_dimension_check(test_data):
    model = QuickShift(
        test_data["cuts"], metric_params={"cell_length": test_data["cell"]}
    )
    with pytest.raises(ValueError):
        model.fit(np.array([[2]]))
