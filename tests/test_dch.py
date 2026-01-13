import numpy as np
import pytest
import warnings
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.utils.validation import NotFittedError

from skmatter.sample_selection import DirectionalConvexHull


@pytest.fixture
def test_data():
    X, y = load_diabetes(return_X_y=True)
    T = PCA(n_components=4).fit_transform(X)
    idx = [57, 123, 156, 187]
    y_distance_100 = 83.69441145645924
    feature_residuals_100 = [0.05926369, 0.03557203, 0.02328013]
    below_hull_point = np.array([[10, 0.03406231, -0.00834545, 0.01799892, 0.08001716]])
    return {
        "X": X,
        "y": y,
        "T": T,
        "idx": idx,
        "y_distance_100": y_distance_100,
        "feature_residuals_100": feature_residuals_100,
        "below_hull_point": below_hull_point,
    }


def test_selected_idx_and_scores(test_data):
    """Regression test that checks that DCH selects correct vertices and gets
    correct distances from the `score_feature_matrix` and `score_samples` functions.
    """
    T = test_data["T"]
    y = test_data["y"]
    idx = test_data["idx"]
    feature_residuals_100 = test_data["feature_residuals_100"]
    y_distance_100 = test_data["y_distance_100"]

    selector = DirectionalConvexHull()
    selector.fit(T, y)
    np.testing.assert_allclose(selector.selected_idx_, idx)

    # takes abs to avoid numerical noise changing the sign of PCA projections
    feature_residuals = np.abs(selector.score_feature_matrix(T))
    val = np.max(
        np.abs((feature_residuals_100 - feature_residuals[100]) / feature_residuals_100)
    )
    (
        np.testing.assert_allclose(
            feature_residuals_100, feature_residuals[100], rtol=1e-6
        ),
        (f"Maximum relative error 1e-6 < {val}"),
    )

    y_distances = selector.score_samples(T, y)
    val = np.max(np.abs((y_distance_100 - y_distances[100]) / y_distance_100))
    (
        np.testing.assert_allclose(y_distance_100, y_distances[100], rtol=1e-6),
        (f"Maximum relative error 1e-6 < {val}"),
    )


def test_cols(test_data):
    """
    Check that correct HD column indices are selected from given LD
    columns
    """
    X = test_data["X"]
    y = test_data["y"]
    selector = DirectionalConvexHull(low_dim_idx=[1, 4, 7])
    selector.fit(X, y)
    assert all(selector.high_dim_idx_ == [0, 2, 3, 5, 6, 8, 9])


def test_shapes(test_data):
    """
    Check that shapes of arrays returned from `score_feature_matrix` are
    consistent with the number of high-dimensional columns.
    """
    T = test_data["T"]
    X = test_data["X"]
    y = test_data["y"]

    selector = DirectionalConvexHull()
    selector.fit(T, y)
    assert selector.score_feature_matrix(T).shape == (442, 3)

    selector = DirectionalConvexHull(low_dim_idx=[2, 4, 8])
    selector.fit(X, y)
    assert selector.score_feature_matrix(X).shape == (442, 7)


def test_residual_features_without_fit(test_data):
    """
    Ensure that calling `score_feature_matrix` without fitting the DCH first raises
    an error.
    """
    T = test_data["T"]
    selector = DirectionalConvexHull()
    with pytest.raises(NotFittedError, match="instance is not fitted"):
        selector.score_feature_matrix(T)


def test_residual_features_ndim(test_data):
    """
    Ensure that ValueError is raised if you try and use `score_feature_matrix`
    on data that has different dimensions to that used to fit the convex hull.
    """
    T = test_data["T"]
    X = test_data["X"]
    y = test_data["y"]

    selector = DirectionalConvexHull()
    selector.fit(T, y)
    match = (
        "X has 10 features, but DirectionalConvexHull is expecting 4 features as input."
    )
    with pytest.raises(ValueError, match=match):
        selector.score_feature_matrix(X)


def test_negative_score(test_data):
    """
    Ensure that when a point lies below the convex hull, the distance to the hull
    in the target (y) dimension, obtained using the `score_samples` function,
    returns a negative value.
    """
    T = test_data["T"]
    y = test_data["y"]
    below_hull_point = test_data["below_hull_point"]

    selector = DirectionalConvexHull()
    selector.fit(T, y)
    distance = selector.score_samples(below_hull_point[:, 1:], below_hull_point[:, 0])[
        0
    ]
    print("distance", distance)
    assert distance < 0.0


def test_positive_score():
    """
    Ensure that when we score on the points we fitted that we obtain only >= 0
    distances.

    In an old implementation we observed this bug for the dataset we use in this
    test (see issue #162).
    """
    X = [
        [1.88421449, 0.86675162],
        [1.88652863, 0.86577001],
        [1.89200182, 0.86573224],
        [1.89664107, 0.86937211],
        [1.90181908, 0.85964603],
        [1.90313135, 0.85695238],
        [1.90063025, 0.84948309],
        [1.90929015, 0.87526563],
        [1.90924666, 0.85509754],
        [1.91139146, 0.86115512],
        [1.91199225, 0.8681867],
        [1.90681563, 0.85036791],
        [1.90193881, 0.84168907],
        [1.90544262, 0.84451744],
        [1.91498802, 0.86010812],
        [1.91305204, 0.85333203],
        [1.89779902, 0.83731807],
        [1.91725967, 0.86630218],
        [1.91309514, 0.85046796],
        [1.89822103, 0.83522425],
    ]
    y = [
        -2.69180967,
        -2.72443825,
        -2.77293913,
        -2.797828,
        -2.12097652,
        -2.69428482,
        -2.70275134,
        -2.80617667,
        -2.79199375,
        -2.01707974,
        -2.74203922,
        -2.24217962,
        -2.03472,
        -2.72612763,
        -2.7071123,
        -2.75706683,
        -2.68925596,
        -2.77160335,
        -2.69528665,
        -2.70911598,
    ]
    selector = DirectionalConvexHull(low_dim_idx=[0, 1])
    selector.fit(X, y)
    distances = selector.score_samples(X, y)
    assert np.all(distances >= -selector.tolerance)


def test_score_function_warnings():
    """Ensure that calling `score_samples` with points outside the range causes an
    error.
    """
    selector = DirectionalConvexHull(low_dim_idx=[0])
    # high-dimensional dummy data, not important for the test
    X_high_dimensional = [1.0, 2.0, 3.0]
    # interpolating the range [0, 3]
    X_low_dimensional = [0.0, 2.0, 3.0]
    X = np.vstack((X_low_dimensional, X_high_dimensional)).T
    # dummy y data, not important for the test
    y = [1.0, 2.0, 3.0]
    selector.fit(X, y)

    # check for score_feature_matrix
    with warnings.catch_warnings(record=True) as warning:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning because it is outsite of range [0, 3]
        selector.score_feature_matrix([[4.0, 1.0]])
        # Verify some things
        assert len(warning) == 1
        assert issubclass(warning[0].category, UserWarning)
        assert (
            "There are samples in X with a low-dimensional part that is outside "
            "of the range of the convex surface. Distance will contain nans."
            == str(warning[0].message)
        )
