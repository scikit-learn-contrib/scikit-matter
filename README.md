# scikit-cosmo

[![Test](https://github.com/lab-cosmo/scikit-cosmo/workflows/Test/badge.svg)](https://github.com/lab-cosmo/scikit-cosmo/actions?query=workflow%3ATest)
[![codecov](https://codecov.io/gh/lab-cosmo/scikit-cosmo/branch/main/graph/badge.svg?token=UZJPJG34SM)](https://codecov.io/gh/lab-cosmo/scikit-cosmo/)

A collection of scikit-learn compatible utilities that implement methods
developed in the COSMO laboratory


## Deprecation Warning
__IMPORTANT:__ This package has been renamed to scikit-matter ([skmatter](https://github.com/lab-cosmo/scikit-matter)). This package will no longer be maintained and updated.

Please install the new package, skmatter, using:

```bash
pip install skmatter
```

## Installation

```bash
pip install skcosmo 
```

You can then `import skcosmo` in your code!

## Developing the package

Start by installing the development dependencies:

```bash
pip install tox black flake8
```

Then this package itself

```bash
git clone https://github.com/lab-cosmo/scikit-cosmo
cd scikit-cosmo
pip install -e .
```

This install the package in development mode, making is `import`able globally
and allowing you to edit the code and directly use the updated version.

### Running the tests

```bash
cd <scikit-cosmo PATH>
# run unit tests
tox
# run the code formatter
black --check .
# run the linter
flake8
```

You may want to setup your editor to automatically apply the
[black](https://black.readthedocs.io/en/stable/) code formatter when saving your
files, there are plugins to do this with [all major
editors](https://black.readthedocs.io/en/stable/editor_integration.html).

## License and developers

This project is distributed under the BSD-3-Clauses license. By contributing to
it you agree to distribute your changes under the same license.
