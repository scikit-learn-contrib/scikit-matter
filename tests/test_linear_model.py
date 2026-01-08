import numpy as np
import pytest
from parameterized import parameterized
from sklearn.datasets import load_iris
from sklearn.utils import check_random_state, extmath

from skmatter.linear_model import OrthogonalRegression, Ridge2FoldCV


@pytest.fixture(scope="module")
def base_test_data():
    features_all = load_iris().data
    features_small = features_all[:, [0, 1]]
    features_large = features_all[:, [0, 1, 0, 1]]
    eps = 1e-9
    random_state = 0
    random_state = check_random_state(random_state)
    random_orthonormal_mat = extmath.randomized_range_finder(
        np.eye(features_small.shape[1]),
        size=features_small.shape[1],
        n_iter=10,
        random_state=random_state,
    )
    features_rotated_small = features_small @ random_orthonormal_mat
    return {
        "features_all": features_all,
        "features_small": features_small,
        "features_large": features_large,
        "features_rotated_small": features_rotated_small,
        "eps": eps,
    }


def test_orthogonal_regression_small_to_rotated_small(base_test_data):
    # tests if OrthogonalRegression can predict rotated small features using small
    # features with use_orthogonal_projector False
    features_small = base_test_data["features_small"]
    features_rotated_small = base_test_data["features_rotated_small"]
    eps = base_test_data["eps"]

    err = np.linalg.norm(
        features_rotated_small
        - OrthogonalRegression(use_orthogonal_projector=False)
        .fit(features_small, features_rotated_small)
        .predict(features_small)
    )
    assert abs(err) < eps, f"error {err} surpasses threshold for zero {eps}"


def test_orthogonal_regression_large_to_small(base_test_data):
    # tests if prediction is padded to larger feature size
    features_small = base_test_data["features_small"]
    features_large = base_test_data["features_large"]

    n_features = (
        OrthogonalRegression(use_orthogonal_projector=False)
        .fit(features_large, features_small)
        .predict(features_large)
        .shape[1]
    )
    assert n_features == features_large.shape[1], (
        f"n_features {n_features} does not match larger feature size "
        f"{features_large.shape[1]}"
    )


def test_orthogonal_regression_use_orthogonal_projector_small_to_rotated_small(
    base_test_data,
):
    # tests if OrthogonalRegression can predict rotated small features using small
    # features with use_orthogonal_projector True
    features_small = base_test_data["features_small"]
    features_rotated_small = base_test_data["features_rotated_small"]
    eps = base_test_data["eps"]

    err = np.linalg.norm(
        features_rotated_small
        - OrthogonalRegression(use_orthogonal_projector=True)
        .fit(features_small, features_rotated_small)
        .predict(features_small)
    )
    assert abs(err) < eps, f"error {err} surpasses threshold for zero {eps}"


def test_orthogonal_regression_use_orthogonal_projector_small_to_large(base_test_data):
    # tests if prediction is projected to prediction feature space
    features_small = base_test_data["features_small"]
    features_large = base_test_data["features_large"]

    n_features = (
        OrthogonalRegression(use_orthogonal_projector=True)
        .fit(features_small, features_large)
        .predict(features_small)
        .shape[1]
    )
    assert n_features == features_large.shape[1], (
        f"n_features {n_features} does not match projection feature size "
        f"{features_large.shape[1]}"
    )


def test_orthogonal_regression_use_orthogonal_projector_large_to_small(base_test_data):
    # tests if prediction is projected to prediction feature space
    features_small = base_test_data["features_small"]
    features_large = base_test_data["features_large"]

    n_features = (
        OrthogonalRegression(use_orthogonal_projector=True)
        .fit(features_large, features_small)
        .predict(features_large)
        .shape[1]
    )
    assert n_features == features_small.shape[1], (
        f"n_features {n_features} does not match projection feature size "
        f"{features_small.shape[1]}"
    )


@pytest.fixture(scope="module")
def ridge_test_data():
    features_all = load_iris().data
    features_small = features_all[:, [0, 1]]
    features_large = features_all[:, [0, 1, 0, 1]]
    eps = 5e-8
    np.random.RandomState(0).seed(0x5F3759DF)
    alphas = [1e-9, 1e-3, 1e-1, 0.5]
    return {
        "features_all": features_all,
        "features_small": features_small,
        "features_large": features_large,
        "eps": eps,
        "alphas": alphas,
    }


def test_ridge_regression_2fold_regularization_method_raise_error(ridge_test_data):
    # tests if wrong regularization_method in Ridge2FoldCV raises error
    features_small = ridge_test_data["features_small"]
    match = "regularization method .* is not known"
    with pytest.raises(ValueError, match=match):
        Ridge2FoldCV(
            regularization_method="dummy",
        ).fit(features_small, features_small)


def test_ridge_regression_2fold_alpha_type_raise_error(ridge_test_data):
    # tests if wrong alpha type in Ridge2FoldCV raises error
    features_small = ridge_test_data["features_small"]
    match = "alpha type.*is not known"
    with pytest.raises(ValueError, match=match):
        Ridge2FoldCV(
            alpha_type="dummy",
        ).fit(features_small, features_small)


def test_ridge_regression_2fold_relative_alpha_type_raise_error(ridge_test_data):
    # tests if an error is raised if alpha not in [0,1)
    features_small = ridge_test_data["features_small"]
    match = "alphas are not within the range"
    with pytest.raises(ValueError, match=match):
        Ridge2FoldCV(alphas=[1], alpha_type="relative").fit(
            features_small, features_small
        )

    with pytest.raises(ValueError, match="alphas are not within the range"):
        Ridge2FoldCV(alphas=[-0.1], alpha_type="relative").fit(
            features_small, features_small
        )


def test_ridge_regression_2fold_iterable_cv(ridge_test_data):
    # tests if we can use iterable as cv parameter
    features_small = ridge_test_data["features_small"]
    cv = [([0, 1, 2, 3], [4, 5, 6])]
    Ridge2FoldCV(alphas=[1], cv=cv).fit(features_small, features_small)


ridge_parameters = [
    ["absolute_tikhonov", "absolute", "tikhonov"],
    ["absolute_cutoff", "absolute", "cutoff"],
    ["relative_tikhonov", "relative", "tikhonov"],
    ["relative_cutoff", "relative", "cutoff"],
]


@pytest.mark.parametrize("name,alpha_type,regularization_method", ridge_parameters)
def test_ridge_regression_2fold_cv_small_to_small(
    ridge_test_data, name, alpha_type, regularization_method
):
    # tests if Ridge2FoldCV can predict small features using small
    # features with use_orthogonal_projector False
    features_small = ridge_test_data["features_small"]
    alphas = ridge_test_data["alphas"]
    eps = ridge_test_data["eps"]

    err = np.linalg.norm(
        features_small
        - Ridge2FoldCV(
            alphas=alphas,
            alpha_type=alpha_type,
            regularization_method=regularization_method,
        )
        .fit(features_small, features_small)
        .predict(features_small)
    )
    assert abs(err) < eps, f"error {err} surpasses threshold for zero {eps}"


@pytest.mark.parametrize("name,alpha_type,regularization_method", ridge_parameters)
def test_ridge_regression_2fold_cv_small_to_large(
    # tests if Ridge2FoldCV can predict large features using small
    # features with use_orthogonal_projector False
    ridge_test_data,
    name,
    alpha_type,
    regularization_method,
):
    features_small = ridge_test_data["features_small"]
    features_large = ridge_test_data["features_large"]
    alphas = ridge_test_data["alphas"]
    eps = ridge_test_data["eps"]

    err = np.linalg.norm(
        features_large
        - Ridge2FoldCV(
            alphas=alphas,
            alpha_type=alpha_type,
            regularization_method=regularization_method,
        )
        .fit(features_small, features_large)
        .predict(features_small)
    )
    assert abs(err) < eps, f"error {err} surpasses threshold for zero {eps}"


@pytest.mark.parametrize("name,alpha_type,regularization_method", ridge_parameters)
def test_ridge_regression_2fold_regularization(
    ridge_test_data, name, alpha_type, regularization_method
):
    # tests if the regularization in the CV split of
    # Ridge2FoldCV does effect the results

    # regularization parameters are chosen to match the singular values o
    # the features, thus each regularization parameter affects the minimized
    # weight matrix and thus the error
    features_all = ridge_test_data["features_all"]
    eps = ridge_test_data["eps"]

    _, singular_values, _ = np.linalg.svd(features_all)
    if alpha_type == "absolute":
        alphas = singular_values[1:][::-1]
    if alpha_type == "relative":
        alphas = singular_values[1:][::-1] / singular_values[0]

    # tests if Ridge2FoldCV does do regularization correct
    ridge = Ridge2FoldCV(
        alphas=alphas,
        alpha_type=alpha_type,
        regularization_method=regularization_method,
        scoring="neg_root_mean_squared_error",
    ).fit(features_all, features_all)
    twofold_rmse = -np.array(ridge.cv_values_)

    # since the data can be perfectly reconstructed,
    # larger regularization parameters (alphas) should result in
    # larger errors
    error_grad = twofold_rmse[1:] - twofold_rmse[:-1]
    assert np.all(error_grad > eps), (
        "error does not strictly increase with larger regularization\n"
        f"\ttwofold RMSE: {twofold_rmse}\n"
        f"\tregularization parameters: {ridge.alphas}"
    )
