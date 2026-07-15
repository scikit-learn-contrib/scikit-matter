import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform
from sklearn.base import clone
from sklearn.datasets import load_digits

from skmatter.decomposition import SketchMap
from skmatter.decomposition._sketchmap_utils import (
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


def quick_sketchmap(**kwargs):
    """SketchMap with cheap settings for shape/API tests."""
    kwargs.setdefault("max_iter", 10)
    kwargs.setdefault("global_opt_steps", 0)
    return SketchMap(**kwargs)


class TestSketchMap:
    def test_basic_fit(self, sample_data):
        X = sample_data
        sm = quick_sketchmap(n_components=2)
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
            sm = quick_sketchmap(n_components=n_comp, max_iter=5)
            sm.fit(X)

            assert sm.embedding_.shape == (X.shape[0], n_comp)

    def test_resolve_global_opt(self):
        # auto: anneal geometrically from MDS-like (1.0) down to pure sigmoid (0)
        schedule = SketchMap()._resolve_global_opt()
        assert len(schedule) == 5
        assert schedule[0] == 1.0
        assert schedule[-1] == 0.0
        assert all(x > y for x, y in zip(schedule, schedule[1:]))

        # annealed gradient works in any dimension (no 2D restriction)
        assert SketchMap(n_components=3)._resolve_global_opt() is not None

        # explicit schedule passes through verbatim
        schedule = SketchMap(
            global_opt_steps=2, mixing_schedule=[1.0, 0.0]
        )._resolve_global_opt()
        assert schedule == (1.0, 0.0)

        # disabled
        for off in (0, None):
            assert SketchMap(global_opt_steps=off)._resolve_global_opt() is None

        # no annealing: a single relaxation at the fixed mixing_ratio
        assert SketchMap(mixing_schedule=None)._resolve_global_opt() == (0.0,)

    def test_sample_weight(self, sample_data):
        X = sample_data
        rng = np.random.default_rng(42)
        weights = rng.random(X.shape[0])

        sm = quick_sketchmap()
        sm.fit(X, sample_weight=weights)

        assert sm.embedding_.shape == (X.shape[0], 2)
        assert sm.stress_ >= 0

    def test_unweighted_equals_explicit_ones(self, sample_data):
        # the uniform (weights=None) fast path must match passing all-ones
        # weights explicitly, which builds the full pairwise matrix
        X = sample_data[:40]
        a = quick_sketchmap(max_iter=20)
        a.fit(X)
        b = quick_sketchmap(max_iter=20)
        b.fit(X, sample_weight=np.ones(X.shape[0]))
        np.testing.assert_allclose(a.embedding_, b.embedding_, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(a.stress_, b.stress_, rtol=1e-9)

    def test_float32_input(self, sample_data):
        X = sample_data[:60]
        f64 = quick_sketchmap(max_iter=30)
        f64.fit(X.astype(np.float64))
        f32 = quick_sketchmap(max_iter=30)
        f32.fit(X.astype(np.float32))

        assert np.isfinite(f32.embedding_).all()
        # same ballpark stress (float32 noise / a different basin are possible)
        assert abs(f32.stress_ - f64.stress_) < 0.05 * max(f64.stress_, 1e-6) + 1e-3

    def test_init_parameter(self, sample_data):
        X = sample_data
        rng = np.random.default_rng(42)
        init = rng.standard_normal((X.shape[0], 2))

        sm = quick_sketchmap(max_iter=5, init=init)
        sm.fit(X)

        assert sm.embedding_.shape == (X.shape[0], 2)

    def test_invalid_init_shape(self, sample_data):
        X = sample_data
        sm = quick_sketchmap(init=np.zeros((3, 2)))
        with pytest.raises(ValueError, match="init has shape"):
            sm.fit(X)

    def test_auto_params(self, sample_data):
        X = sample_data
        sm = quick_sketchmap()
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

        sm = quick_sketchmap(sigma=5.0)
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
        sm = quick_sketchmap(**params)
        sm.fit(X)

        for key, value in params.items():
            assert sm.params_[key] == value

    def test_deterministic(self, sample_data):
        X = sample_data

        sm1 = quick_sketchmap()
        sm1.fit(X)

        sm2 = quick_sketchmap()
        sm2.fit(X)

        np.testing.assert_allclose(sm1.embedding_, sm2.embedding_, rtol=1e-10)
        np.testing.assert_allclose(sm1.stress_, sm2.stress_, rtol=1e-10)

    def test_mds_opt_steps(self, sample_data):
        X = sample_data

        sm_no_preopt = quick_sketchmap(mds_opt_steps=0, max_iter=5)
        sm_no_preopt.fit(X)

        sm_with_preopt = quick_sketchmap(mds_opt_steps=10, max_iter=5)
        sm_with_preopt.fit(X)

        assert sm_no_preopt.embedding_.shape == (X.shape[0], 2)
        assert sm_with_preopt.embedding_.shape == (X.shape[0], 2)
        assert sm_no_preopt.stress_ >= 0
        assert sm_with_preopt.stress_ >= 0
        assert not np.allclose(sm_no_preopt.embedding_, sm_with_preopt.embedding_)

    def test_global_opt_improves_stress(self, sample_data):
        X = sample_data

        sm_no_global = SketchMap(
            mds_opt_steps=10,
            max_iter=10,
            global_opt_steps=0,
        )
        sm_no_global.fit(X)

        sm_with_global = SketchMap(
            mds_opt_steps=10,
            max_iter=10,
            global_opt_steps=3,
        )
        sm_with_global.fit(X)

        assert sm_no_global.embedding_.shape == (X.shape[0], 2)
        assert sm_with_global.embedding_.shape == (X.shape[0], 2)
        assert sm_no_global.stress_ >= 0
        # the extra grid cycles must not make the map worse
        assert sm_with_global.stress_ <= sm_no_global.stress_ + 1e-12

    def test_global_opt_invalid_params(self, sample_data):
        X = sample_data

        sm = SketchMap(max_iter=5, global_opt_steps=-1)
        with pytest.raises(ValueError, match="global_opt_steps must be"):
            sm.fit(X)

        sm = SketchMap(mixing_schedule="bogus")
        with pytest.raises(ValueError, match="mixing_schedule must be"):
            sm.fit(X)

    def test_global_opt_works_in_3d(self, sample_data):
        # annealed gradient global optimization is not restricted to 2D
        sm = SketchMap(n_components=3, max_iter=20, global_opt_steps=3)
        sm.fit(sample_data[:40])
        assert sm.embedding_.shape == (40, 3)
        assert sm.stress_ >= 0

    def test_invalid_optimizer(self, sample_data):
        sm = SketchMap(optimizer="Nelder-Mead")
        with pytest.raises(ValueError, match="optimizer must be"):
            sm.fit(sample_data)

    def test_progress_bar(self, sample_data):
        pytest.importorskip("tqdm")
        X = sample_data[:30]
        sm = SketchMap(max_iter=5, global_opt_steps=1, progress_bar=True)
        sm.fit(X)
        assert sm.embedding_.shape == (30, 2)

    def test_clone(self, sample_data):
        # fit() must not modify constructor parameters (sklearn contract)
        sm = quick_sketchmap(max_iter=5)
        params_before = sm.get_params()
        sm.fit(sample_data[:30])
        assert sm.get_params() == params_before
        clone(sm)


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

    def test_sigmoid_derivative_at_zero(self):
        deriv = sigmoid_derivative(np.array([0.0, 1.0]), 7.0, 2.0, 2.0)
        assert deriv[0] == 0.0
        assert np.isfinite(deriv).all()

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

        params, _analysis = suggest_sigmoid_params(distances, n_components=2)

        assert "sigma" in params
        assert "a_high" in params
        assert "b_high" in params
        assert "a_low" in params
        assert "b_low" in params

        for value in params.values():
            assert value > 0
