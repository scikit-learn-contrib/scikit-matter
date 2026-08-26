from os.path import dirname, join

import numpy as np
import pytest
from scipy.spatial import procrustes
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.base import clone
from sklearn.datasets import load_digits

from skmatter.decomposition import SketchMap
from skmatter.decomposition._sketchmap_utils import (
    _analyze_distance_distribution,
    _classical_mds,
    _sigmoid_and_derivative,
    _sigmoid_transform,
    _suggest_sigmoid_params,
)


@pytest.fixture
def sample_data():
    digits = load_digits(n_class=4)
    X = digits.data[:100]
    return X


def quick_sketchmap(**kwargs):
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
        # by default no annealing: every level relaxes at mixing_ratio
        assert SketchMap()._resolve_global_opt() == (0.0,)

        # auto: anneal geometrically from MDS-like (1.0) down to pure sigmoid
        schedule = SketchMap(mixing_schedule="auto")._resolve_global_opt()
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

    def test_unweighted_equals_explicit_ones(self, sample_data):
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
        assert abs(f32.stress_ - f64.stress_) < 0.05 * max(f64.stress_, 1e-6) + 1e-3

    def test_init_parameter(self, sample_data):
        X = sample_data[:40]
        rng = np.random.default_rng(42)
        init = rng.standard_normal((X.shape[0], 2))

        untouched = quick_sketchmap(max_iter=0, init=init).fit(X)
        np.testing.assert_allclose(untouched.embedding_, init)

        first = quick_sketchmap(max_iter=5, init=init).fit(X)
        second = quick_sketchmap(max_iter=5, init=-init).fit(X)
        assert not np.allclose(first.embedding_, second.embedding_)

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

    def test_auto_params_use_sample_weight(self):
        rng = np.random.default_rng(0)
        tight = rng.normal(0.0, 0.05, (30, 2))
        broad = rng.normal(0.0, 2.0, (30, 2))

        X = np.vstack([tight, broad])
        weights = np.array([1e-6] * 30 + [1.0] * 30)
        distances = cdist(X, X)

        unweighted, _ = _suggest_sigmoid_params(distances, 2)
        weighted, _ = _suggest_sigmoid_params(distances, 2, sample_weight=weights)
        assert weighted["sigma"] > 10 * unweighted["sigma"]

        sm = quick_sketchmap()
        sm.fit(X, sample_weight=weights)
        assert np.isclose(sm.params_["sigma"], weighted["sigma"])

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
        # fit() must not modify constructor parameters
        sm = quick_sketchmap(max_iter=5)
        params_before = sm.get_params()
        sm.fit(sample_data[:30])
        assert sm.get_params() == params_before
        clone(sm)

    def test_not_equivalent_to_mds(self):
        X = load_digits().data[:150].astype(np.float64)
        hd_distances = cdist(X, X)
        params = dict(sigma=30.0, a_high=4.0, b_high=2.0, a_low=2.0, b_low=2.0)

        def sigmoid_stress(embedding):
            hd_t = _sigmoid_transform(
                hd_distances, params["sigma"], params["a_high"], params["b_high"]
            )
            ld = cdist(embedding, embedding)
            ld_t = _sigmoid_transform(
                ld, params["sigma"], params["a_low"], params["b_low"]
            )
            upper = np.triu_indices(len(X), k=1)
            return np.mean((hd_t[upper] - ld_t[upper]) ** 2)

        mds_map = _classical_mds(hd_distances, 2)
        sketch_map = SketchMap(n_components=2, **params).fit_transform(X)

        assert sigmoid_stress(sketch_map) < 0.5 * sigmoid_stress(mds_map)

        _, _, disparity = procrustes(mds_map, sketch_map)
        assert disparity > 0.1

    def test_precomputed_matches_euclidean(self, sample_data):
        # feeding the distance matrix directly must reproduce the euclidean fit
        X = sample_data[:40]
        euclidean = quick_sketchmap(max_iter=20).fit(X)
        precomputed = quick_sketchmap(max_iter=20, dissimilarity="precomputed").fit(
            cdist(X, X)
        )
        np.testing.assert_allclose(euclidean.embedding_, precomputed.embedding_)

    def test_precomputed_requires_square(self, sample_data):
        with pytest.raises(ValueError, match="square distance matrix"):
            quick_sketchmap(dissimilarity="precomputed").fit(sample_data[:40])

    def test_dissimilarity_invalid(self, sample_data):
        with pytest.raises(ValueError, match="dissimilarity must be"):
            SketchMap(dissimilarity="cosine").fit(sample_data[:30])

    def test_grid_optimizer_lowers_stress(self, sample_data):
        X = sample_data[:60]
        common = dict(
            n_components=2,
            sigma=30.0,
            a_high=4.0,
            b_high=2.0,
            a_low=2.0,
            b_low=2.0,
            global_opt_steps=2,
        )

        gradient = SketchMap(**common).fit(X)
        grid = SketchMap(**common, global_optimizer="grid").fit(X)

        assert grid.stress_ <= gradient.stress_ + 1e-9
        assert grid.embedding_.shape == (60, 2)

    def test_grid_optimizer_keeps_map_bounded(self, sample_data):
        X = sample_data[:60]
        common = dict(
            n_components=2,
            sigma=10.0,
            a_high=8.0,
            b_high=8.0,
            a_low=2.0,
            b_low=8.0,
            global_opt_steps=3,
        )

        gradient = SketchMap(**common).fit(X)
        grid = SketchMap(**common, global_optimizer="grid").fit(X)

        gradient_radius = np.sqrt((gradient.embedding_**2).sum(axis=1)).max()
        grid_radius = np.sqrt((grid.embedding_**2).sum(axis=1)).max()
        assert grid_radius <= 1.05 * gradient_radius

    def test_grid_optimizer_requires_2d(self, sample_data):
        with pytest.raises(ValueError, match="only implemented for"):
            SketchMap(n_components=3, global_optimizer="grid").fit(sample_data[:30])

    def test_annealed_grid_runs_and_stays_bounded(self, sample_data):
        X = sample_data[:60]
        common = dict(
            n_components=2,
            sigma=30.0,
            a_high=4.0,
            b_high=2.0,
            a_low=2.0,
            b_low=2.0,
            global_opt_steps=2,
        )

        gradient = SketchMap(**common).fit(X)
        annealed = SketchMap(
            **common, global_optimizer="grid", mixing_schedule="auto"
        ).fit(X)

        assert np.isfinite(annealed.stress_)
        assert annealed.embedding_.shape == (60, 2)
        gradient_radius = np.sqrt((gradient.embedding_**2).sum(axis=1)).max()
        annealed_radius = np.sqrt((annealed.embedding_**2).sum(axis=1)).max()
        assert annealed_radius < 5 * gradient_radius

    def test_global_optimizer_invalid(self, sample_data):
        with pytest.raises(ValueError, match="global_optimizer must be"):
            SketchMap(global_optimizer="annealing").fit(sample_data[:30])

    def test_n_components_exceeding_n_samples(self):
        rng = np.random.default_rng(0)

        with pytest.raises(ValueError, match="cannot exceed the number of"):
            quick_sketchmap(n_components=6).fit(rng.standard_normal((4, 6)))

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            (dict(mixing_ratio=2.0), "mixing_ratio must lie between"),
            (dict(mixing_ratio=-1.0), "mixing_ratio must lie between"),
            (dict(sigma=0.0), "sigma must be positive"),
            (dict(sigma=-5.0), "sigma must be positive"),
            (dict(a_high=0.0), "a_high must be positive"),
            (dict(a_high=-4.0), "a_high must be positive"),
            (dict(b_high=0.0), "b_high must be positive"),
            (dict(a_low=-1.0), "a_low must be positive"),
            (dict(b_low=0.0), "b_low must be positive"),
            (dict(n_components=0), "n_components must be at least 1"),
            (dict(mds_opt_steps=-10), "mds_opt_steps must be non-negative"),
            (dict(max_iter=-10), "max_iter must be non-negative"),
        ],
    )
    def test_invalid_parameter_values(self, sample_data, kwargs, message):
        with pytest.raises(ValueError, match=message):
            quick_sketchmap(**kwargs).fit(sample_data[:30])

    def test_grid_rejects_explicit_mixing_schedule(self, sample_data):
        with pytest.raises(ValueError, match="explicit mixing schedules"):
            SketchMap(global_optimizer="grid", mixing_schedule=[1.0, 0.5, 0.0]).fit(
                sample_data[:30]
            )

    def test_annealed_grid_zero_stress(self):
        # two points are embedded exactly, so the stress is 0 from the start
        sm = SketchMap(
            global_optimizer="grid",
            mixing_schedule="auto",
            global_opt_steps=3,
            sigma=1.0,
            a_high=4,
            b_high=2,
            a_low=2,
            b_low=2,
        ).fit([[0, 0], [1, 0]])

        assert np.isfinite(sm.stress_)

    def test_annealed_grid_collapsed_embedding(self):
        # identical samples collapse to a single point, leaving no grid to scan
        sm = SketchMap(
            global_optimizer="grid",
            mixing_schedule="auto",
            global_opt_steps=2,
            sigma=1.0,
            a_high=4,
            b_high=2,
            a_low=2,
            b_low=2,
        ).fit(np.zeros((4, 2)))

        assert np.isfinite(sm.stress_)

    def test_degenerate_sample_weight(self, sample_data):
        # a single nonzero weight leaves every pair weightless
        X = sample_data[:20]
        weights = np.zeros(20)
        weights[3] = 1.0

        with pytest.raises(ValueError, match="at least two positive entries"):
            quick_sketchmap().fit(X, sample_weight=weights)

    def test_annealed_grid_stress_regression(self):
        # test of the same grid annealed pipeline as in the c++
        rng = np.random.default_rng(0)
        X = np.vstack(
            [
                center + rng.normal(0, 0.5, (30, 6))
                for center in rng.normal(0, 4, (4, 6))
            ]
        )

        sm = SketchMap(
            sigma=4.0,
            a_high=4.0,
            b_high=2.0,
            a_low=2.0,
            b_low=2.0,
            global_optimizer="grid",
            mixing_schedule="auto",
        ).fit(X)

        np.testing.assert_allclose(sm.stress_, 2.34413e-4, rtol=1e-3)


class TestHelperFunctions:
    def test_sigmoid_transform(self):
        sigma = 7.0
        a = 4.0
        b = 2.0

        x = np.array([[10.0]])
        result = _sigmoid_transform(x, sigma, a, b)

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
                    result = _sigmoid_transform(x, sigma, a, b)

                    np.testing.assert_allclose(result[0, 0], 0.5, rtol=1e-10)

    def test_sigmoid_and_derivative_matches_transform(self):
        sigma, a, b = 7.0, 4.0, 2.0
        distances = np.array([[1.0, 5.0, 10.0, 15.0]])

        value, _ = _sigmoid_and_derivative(distances, sigma, a, b)

        np.testing.assert_allclose(
            value, _sigmoid_transform(distances, sigma, a, b), rtol=1e-12
        )

    def test_sigmoid_and_derivative_gradient(self):
        sigma, a, b = 7.0, 4.0, 2.0
        distances = np.array([[5.0, 7.0, 10.0]])

        deriv = _sigmoid_and_derivative(distances, sigma, a, b)[1]

        assert np.all(deriv > 0)

        eps = 1e-7
        d_plus = _sigmoid_transform(distances + eps, sigma, a, b)
        d_minus = _sigmoid_transform(distances - eps, sigma, a, b)
        numerical_deriv = (d_plus - d_minus) / (2 * eps)

        np.testing.assert_allclose(deriv, numerical_deriv, rtol=1e-5)

    def test_sigmoid_and_derivative_at_zero(self):
        value, deriv = _sigmoid_and_derivative(np.array([0.0, 1.0]), 7.0, 2.0, 2.0)
        assert value[0] == 0.0
        assert deriv[0] == 0.0
        assert np.isfinite(deriv).all()

    def test_classical_mds(self):
        rng = np.random.default_rng(42)
        points = rng.standard_normal((20, 3))
        distances = squareform(pdist(points))

        embedding = _classical_mds(distances, n_components=3)

        assert embedding.shape == (20, 3)
        np.testing.assert_allclose(squareform(pdist(embedding)), distances, atol=1e-10)

    def test_classical_mds_truncates_to_leading_components(self):
        rng = np.random.default_rng(42)
        points = np.column_stack([rng.standard_normal((20, 2)), np.zeros(20)])
        distances = squareform(pdist(points))

        embedding = _classical_mds(distances, n_components=2)

        assert embedding.shape == (20, 2)
        np.testing.assert_allclose(squareform(pdist(embedding)), distances, atol=1e-10)

    def test_classical_mds_sign_convention(self):
        rng = np.random.default_rng(7)
        distances = squareform(pdist(rng.standard_normal((30, 4))))

        embedding = _classical_mds(distances, n_components=3)

        for column in embedding.T:
            assert column[np.argmax(np.abs(column))] > 0

    def test_analyze_distance_distribution(self):
        # points on a circle of radius 5 all sit 10 apart at most and their
        # distance histogram peaks near the diameter
        angles = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        points = 5.0 * np.column_stack([np.cos(angles), np.sin(angles)])
        distances = squareform(pdist(points))

        analysis = _analyze_distance_distribution(distances, n_bins=50)

        assert 8.0 < analysis["peak_distance"] <= 10.0
        assert analysis["max_distance"] <= 10.0 + 1e-9
        assert len(analysis["bin_centers"]) == 50
        assert len(analysis["prob_density"]) == 50

    def test_suggest_sigmoid_params(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 5))
        distances = squareform(pdist(X))

        params, analysis = _suggest_sigmoid_params(distances, n_components=2)

        # sigma sits just below the peak of the distance histogram
        assert np.isclose(params["sigma"], 0.9 * analysis["peak_distance"])

        # the high-dimensional pair keeps the long tail: b below a, a twice b
        assert 2.0 <= params["b_high"] <= 6.0
        assert np.isclose(params["a_high"], 2.0 * params["b_high"])

        for n_components in (2, 3, 5):
            params, _ = _suggest_sigmoid_params(distances, n_components=n_components)
            assert params["a_low"] == float(n_components)
            assert params["b_low"] == float(n_components)

    def test_suggest_sigmoid_params_scales_with_the_data(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((80, 5))

        small, _ = _suggest_sigmoid_params(squareform(pdist(X)), n_components=2)
        large, _ = _suggest_sigmoid_params(
            squareform(pdist(1000.0 * X)), n_components=2
        )

        assert np.isclose(large["sigma"] / small["sigma"], 1000.0, rtol=1e-6)
        assert np.isclose(large["a_high"], small["a_high"], rtol=0.05)
        assert np.isclose(large["b_high"], small["b_high"], rtol=0.05)


class TestReferenceCpp:
    """Validate against the reference C++ ``dimred`` from sketchmap.org.

    ``tests/data/sketchmap_dimred_reference.dat`` was computed with the C++ code on
    ``load_digits().data[:64]``, see its header for the exact settings
    """

    params = dict(sigma=30.0, a_high=4.0, b_high=2.0, a_low=2.0, b_low=2.0)
    cpp_stress = 0.0129757

    @pytest.fixture(scope="class")
    def fitted(self):
        X = load_digits().data[:64].astype(np.float64)
        cpp_map = np.loadtxt(
            join(dirname(__file__), "data", "sketchmap_dimred_reference.dat")
        )
        sm = SketchMap(n_components=2, **self.params).fit(X)
        return X, cpp_map, sm

    def test_objective_matches_cpp(self, fitted):
        X, cpp_map, sm = fitted
        hd_distances = cdist(X, X)
        hd_transformed = _sigmoid_transform(
            hd_distances,
            self.params["sigma"],
            self.params["a_high"],
            self.params["b_high"],
        )
        n_pairs = X.shape[0] * (X.shape[0] - 1) / 2.0
        problem = (hd_distances, hd_transformed, None, n_pairs)

        stress_at_cpp = sm._stress_and_grad(cpp_map.ravel(), problem, 0.0)[0]

        np.testing.assert_allclose(stress_at_cpp, self.cpp_stress, rtol=1e-3)

    def test_fit_reproduces_cpp(self, fitted):
        _X, cpp_map, sm = fitted

        _, _, disparity = procrustes(cpp_map, sm.embedding_)

        assert disparity < 0.1
        np.testing.assert_allclose(sm.stress_, self.cpp_stress, rtol=0.1)
