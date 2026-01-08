# scikit-matter Development Guide

## Overview
scikit-matter is a scikit-learn-compatible toolbox of methods from computational chemistry and materials science. All estimators follow sklearn API conventions (fit/transform/predict) and inherit from sklearn base classes.

## Architecture

### Core Selection Framework
The codebase centers on a unique **dual-selection architecture** split across `feature_selection/` and `sample_selection/` with shared implementation in `_selection.py`:

- **`_selection.py`**: Contains base classes (`GreedySelector`, `_CUR`, `_FPS`, `_PCovCUR`, `_PCovFPS`) that implement core algorithms independent of axis
- **`feature_selection/_base.py`**: Thin wrappers that instantiate base classes with `selection_type="feature"` and inherit from `SelectorMixin` (enables `transform()`)
- **`sample_selection/_base.py`**: Thin wrappers with `selection_type="sample"` - return indices via `selected_idx_` attribute (no `transform()`)

Example: `FPS` (Farthest Point Sampling) exists as both `feature_selection.FPS` and `sample_selection.FPS`, sharing the same `_FPS` implementation but differing only in which axis they select along.

### Module Organization
- **`decomposition/`**: PCovR (Principal Covariates Regression) and variants - supervised dimensionality reduction combining PCA-like and regression objectives
- **`linear_model/`**: `OrthogonalRegression`, `Ridge2FoldCV` (custom 2-fold CV for efficiency)
- **`metrics/`**: Reconstruction measures (GRE, GRD, LRE) and prediction rigidities (LPR, CPR)
- **`preprocessing/`**: `StandardFlexibleScaler`, `SparseKernelCenterer` with column-wise scaling options
- **`utils/`**: Orthogonalizers, PCovR utilities, progress bar helpers
- **`datasets/`**: Chemistry/materials datasets (CSD-1000r, CH4 manifolds, etc.)

## Development Workflows

### Testing
```bash
# Run all tests with coverage
tox -e tests

# Run specific test file
tox -e tests -- tests/test_feature_simple_cur.py

# Run tests against sklearn dev version
tox -e tests-dev
```

Tests use pytest-style assertions and fixtures. Common patterns:
- Use `@pytest.fixture` for test data setup
- Use `assert` statements instead of `self.assertEqual()`
- Use `pytest.raises()` for exception testing
- Use `pytest.warns()` for warning testing
- Use `pytest.mark.parametrize` for parameterized tests
- Tests often load datasets via `skmatter.datasets.load_*()`

### Linting & Formatting
```bash
# Check only (CI mode)
tox -e lint

# Auto-format code
tox -e format

# More aggressive fixes (review changes carefully)
tox -e format-unsafe
```

Uses `ruff` for both formatting and linting. Configuration in `pyproject.toml` ignores F401 (unused imports in `__init__.py`).

### Building Docs
```bash
tox -e docs  # Builds HTML docs, runs examples via sphinx-gallery
```

Documentation uses Sphinx with `.rst` format. Examples in `examples/` are executed during doc builds.

### Building Package
```bash
tox -e build  # Builds wheel and sdist, runs check-manifest and twine check
```

Uses `setuptools_scm` for versioning from git tags. Version file auto-generated at `src/skmatter/_version.py`.

## Key Conventions

### scikit-learn Compliance
- All estimators inherit from appropriate sklearn mixins (`RegressorMixin`, `TransformerMixin`, `SelectorMixin`)
- Use `validate_data()` (not deprecated `check_X_y()`) for input validation
- Implement `fit()` returning `self`, store fitted attributes with trailing underscore (`selected_idx_`, `n_selected_`)
- Support `warm_start` parameter in selectors to continue from previous fit

### Selection Methods Patterns
```python
# Feature selection (returns transformed X)
from skmatter.feature_selection import CUR, FPS, PCovCUR, PCovFPS
selector = CUR(n_to_select=10, progress_bar=True)
X_reduced = selector.fit(X).transform(X)

# Sample selection (returns indices)
from skmatter.sample_selection import CUR
selector = CUR(n_to_select=10)
selector.fit(X)
X_subset = X[selector.selected_idx_]
```

### PCovR Methods
Always center and scale inputs (`StandardFlexibleScaler`) before using PCovR/PCovC - results change drastically near α→0 or α→1 otherwise. Use `column_wise=True` when features are comparable.

### Progress Bars
Optional `tqdm` progress bar via `progress_bar=True` parameter. Implementation uses utility functions `get_progress_bar()` / `no_progress_bar()` from `utils/`.

## Dependencies & Python Support
- **Python**: 3.11+ (as of v0.3.3)
- **Core**: scikit-learn 1.8.x, scipy ≥1.15
- **Optional**: matplotlib, pandas, tqdm (for examples)
- **Testing**: Requires Python 3.11 and 3.14 on Ubuntu, macOS, Windows

## Pull Request Requirements
- Update tests for new features/bugfixes
- Update documentation for new features  
- Reference issue numbers in PR description
- Reviewer updates CHANGELOG for important changes (not contributor)

## Common Pitfalls
- Don't use deprecated sklearn APIs - check sklearn version in `pyproject.toml`
- Selection methods: feature vs sample selection use same algorithms but different interfaces
- PCovR requires pre-scaled data - document this in examples
- Test files use pytest - use fixtures, `assert`, `pytest.raises()`, not unittest classes
