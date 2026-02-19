import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import load_digits

from skmatter.decomposition import SketchMap
from skmatter.decomposition._sketchmap import (
    analyze_distance_distribution,
    classical_mds,
    sigmoid_derivative,
    sigmoid_inverse,
    sigmoid_transform,
    suggest_sigmoid_params,
)


@pytest.fixture
def sample_data():
    digits = load_digits(n_class=4)
    X = digits.data[:100]
    return X


class TestSketchMap:
    def test_basic_fit(self, sample_data):
        X = sample_data
        sm = SketchMap(n_components=2, preopt_steps=10, max_iter=10)
        sm.fit(X)

        assert hasattr(sm, "embedding_")
        assert sm.embedding_.shape == (X.shape[0], 2)
        assert hasattr(sm, "stress_")
        assert sm.stress_ >= 0
        assert hasattr(sm, "params_")
        assert "sigma" in sm.params_
        assert "a_high" in sm.params_

    def test_n_components(self, sample_data):
        X = sample_data

        for n_comp in [2, 3, 5]:
            sm = SketchMap(n_components=n_comp, preopt_steps=5, max_iter=5)
            sm.fit(X)

            assert sm.embedding_.shape == (X.shape[0], n_comp)

    def test_sample_weights(self, sample_data):
        X = sample_data
        rng = np.random.default_rng(42)
        weights = rng.random(X.shape[0])

        sm = SketchMap(n_components=2, preopt_steps=10, max_iter=10)
        sm.fit(X, sample_weights=weights)

        assert sm.embedding_.shape == (X.shape[0], 2)
        assert sm.stress_ >= 0

    def test_init_parameter(self, sample_data):
        X = sample_data
        rng = np.random.default_rng(42)
        init = rng.standard_normal((X.shape[0], 2))

        sm = SketchMap(n_components=2, preopt_steps=5, max_iter=5, init=init)
        sm.fit(X)

        assert sm.embedding_.shape == (X.shape[0], 2)

    def test_stress_positive(self, sample_data):
        X = sample_data
        sm = SketchMap(n_components=2, preopt_steps=20, max_iter=20)
        sm.fit(X)

        assert sm.stress_ > 0

    def test_auto_params(self, sample_data):
        X = sample_data
        sm = SketchMap(
            n_components=2,
            preopt_steps=10,
            max_iter=10,
        )
        sm.fit(X)

        assert hasattr(sm, "params_")
        assert sm.params_["sigma"] > 0
        assert sm.params_["a_high"] > 0
        assert sm.params_["b_high"] > 0
        assert sm.params_["a_low"] > 0
        assert sm.params_["b_low"] > 0
        assert hasattr(sm, "suggested_params_")
        assert "sigma" in sm.suggested_params_

    def test_partial_params(self, sample_data):
        X = sample_data

        sm = SketchMap(
            n_components=2,
            params={"sigma": 5.0},
            preopt_steps=10,
            max_iter=10,
        )
        sm.fit(X)

        assert sm.params_["sigma"] == 5.0
        assert sm.params_["a_high"] > 0
        assert sm.params_["b_high"] > 0

    def test_full_params(self, sample_data):
        X = sample_data
        params = {
            "sigma": 7.0,
            "a_high": 4.0,
            "b_high": 2.0,
            "a_low": 2.0,
            "b_low": 2.0,
        }
        sm = SketchMap(
            n_components=2,
            params=params,
            preopt_steps=10,
            max_iter=10,
        )
        sm.fit(X)

        for key, value in params.items():
            assert sm.params_[key] == value

    def test_mixing_ratio(self, sample_data):
        X = sample_data

        for ratio in [0.0, 0.5, 1.0]:
            sm = SketchMap(
                n_components=2,
                mixing_ratio=ratio,
                preopt_steps=5,
                max_iter=5,
            )
            sm.fit(X)

            assert sm.embedding_.shape == (X.shape[0], 2)

    def test_transform(self, sample_data):
        X = sample_data
        sm = SketchMap(n_components=2, preopt_steps=5, max_iter=5)
        sm.fit(X)

        result = sm.transform(X)

        assert result.shape == (X.shape[0], 2)
        np.testing.assert_allclose(result, sm.embedding_)

    def test_reproducibility_with_random_state(self, sample_data):
        X = sample_data

        sm1 = SketchMap(n_components=2, preopt_steps=10, max_iter=10, random_state=42)
        sm1.fit(X)

        sm2 = SketchMap(n_components=2, preopt_steps=10, max_iter=10, random_state=42)
        sm2.fit(X)

        np.testing.assert_allclose(sm1.embedding_, sm2.embedding_, rtol=1e-10)
        np.testing.assert_allclose(sm1.stress_, sm2.stress_, rtol=1e-10)

    def test_center_parameter(self, sample_data):
        X = sample_data

        sm_centered = SketchMap(n_components=2, center=True, preopt_steps=5, max_iter=5)
        sm_centered.fit(X)

        sm_not_centered = SketchMap(
            n_components=2, center=False, preopt_steps=5, max_iter=5
        )
        sm_not_centered.fit(X)

        assert sm_centered.embedding_.shape == (X.shape[0], 2)
        assert sm_not_centered.embedding_.shape == (X.shape[0], 2)

    def test_mds_opt_steps(self, sample_data):
        X = sample_data

        sm_no_preopt = SketchMap(
            n_components=2,
            mds_opt_steps=0,
            preopt_steps=5,
            max_iter=5,
            random_state=42,
        )
        sm_no_preopt.fit(X)

        sm_with_preopt = SketchMap(
            n_components=2,
            mds_opt_steps=10,
            preopt_steps=5,
            max_iter=5,
            random_state=42,
        )
        sm_with_preopt.fit(X)

        assert sm_no_preopt.embedding_.shape == (X.shape[0], 2)
        assert sm_with_preopt.embedding_.shape == (X.shape[0], 2)
        assert sm_no_preopt.stress_ >= 0
        assert sm_with_preopt.stress_ >= 0
        assert not np.allclose(sm_no_preopt.embedding_, sm_with_preopt.embedding_)

    def test_global_opt(self, sample_data):
        X = sample_data

        sm_no_global = SketchMap(
            n_components=2,
            mds_opt_steps=10,
            preopt_steps=10,
            max_iter=10,
            global_opt_steps=None,
            random_state=42,
        )
        sm_no_global.fit(X)

        sm_with_global = SketchMap(
            n_components=2,
            mds_opt_steps=10,
            preopt_steps=10,
            max_iter=10,
            global_opt_steps=3,
            random_state=42,
        )
        sm_with_global.fit(X)

        assert sm_no_global.embedding_.shape == (X.shape[0], 2)
        assert sm_with_global.embedding_.shape == (X.shape[0], 2)
        assert sm_no_global.stress_ >= 0
        assert sm_with_global.stress_ >= 0

    def test_global_opt_invalid_params(self, sample_data):
        X = sample_data

        sm = SketchMap(
            n_components=2,
            preopt_steps=5,
            max_iter=5,
            global_opt_steps=-1,
        )

        match = "global_opt_steps must be a positive int"
        with pytest.raises(ValueError, match=match):
            sm.fit(X)


class TestHelperFunctions:
    def test_sigmoid_transform(self):
        sigma = 7.0
        a = 4.0
        b = 2.0

        x = np.array([[10.0]])
        result = sigmoid_transform(x, sigma, a, b)

        # Expected: 1 - (1 + (2^(a/b) - 1) * (x/sigma)^a)^(-b/a)
        ratio = 10.0 / 7.0
        base = 1 + (2 ** (a / b) - 1) * (ratio**a)
        expected = 1 - base ** (-b / a)

        np.testing.assert_allclose(result[0, 0], expected, rtol=1e-10)

    def test_sigmoid_at_sigma(self):
        for sigma in [1.0, 5.0, 10.0]:
            for a in [2.0, 4.0]:
                for b in [2.0, 4.0]:
                    x = np.array([[sigma]])
                    result = sigmoid_transform(x, sigma, a, b)

                    np.testing.assert_allclose(result[0, 0], 0.5, rtol=1e-10)

    def test_sigmoid_inverse(self):
        sigma, a, b = 7.0, 4.0, 2.0
        distances = np.array([[1.0, 5.0, 10.0, 15.0]])

        transformed = sigmoid_transform(distances, sigma, a, b)
        recovered = sigmoid_inverse(transformed, sigma, a, b)

        np.testing.assert_allclose(recovered, distances, rtol=1e-10)

    def test_sigmoid_derivative(self):
        sigma, a, b = 7.0, 4.0, 2.0
        distances = np.array([[5.0, 7.0, 10.0]])

        deriv = sigmoid_derivative(distances, sigma, a, b)

        assert np.all(deriv > 0)

        eps = 1e-7
        d_plus = sigmoid_transform(distances + eps, sigma, a, b)
        d_minus = sigmoid_transform(distances - eps, sigma, a, b)
        numerical_deriv = (d_plus - d_minus) / (2 * eps)

        np.testing.assert_allclose(deriv, numerical_deriv, rtol=1e-5)

    def test_classical_mds(self):
        rng = np.random.default_rng(42)

        points = rng.standard_normal((20, 3))

        distances = squareform(pdist(points))

        embedding = classical_mds(distances, n_components=2)

        assert embedding.shape == (20, 2)
        assert np.isfinite(embedding).all()

    def test_analyze_distance_distribution(self):
        rng = np.random.default_rng(42)
        distances = rng.random((100, 100)) * 10
        distances = (distances + distances.T) / 2
        np.fill_diagonal(distances, 0)

        analysis = analyze_distance_distribution(distances, n_bins=50)

        assert "bin_centers" in analysis
        assert "prob_density" in analysis
        assert "peak_distance" in analysis
        assert analysis["peak_distance"] > 0

    def test_suggest_sigmoid_params(self):
        rng = np.random.default_rng(42)
        distances = rng.random((50, 50)) * 10
        distances = (distances + distances.T) / 2
        np.fill_diagonal(distances, 0)

        params, _analysis = suggest_sigmoid_params(
            distances, n_components=2, n_features=10
        )

        assert "sigma" in params
        assert "a_high" in params
        assert "b_high" in params
        assert "a_low" in params
        assert "b_low" in params

        for value in params.values():
            assert value > 0
