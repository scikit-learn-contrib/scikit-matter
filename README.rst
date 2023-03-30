scikit-matter
=============

|tests| |codecov| |docs| |pypi| |conda| |docs|

A collection of scikit-learn compatible utilities that implement methods born out of the
materials science and chemistry communities.

Installation
------------

You can install *scikit-matter* either via pip using

.. code-block:: bash

    pip install skmatter


or conda

.. code-block:: bash

    conda install -c conda-forge skmatter


You can then `import skmatter` in your code!

Developing the package
----------------------

Start by installing the development dependencies:

.. code-block:: bash

    pip install tox black flake8


Then this package itself

.. code-block:: bash

    git clone https://github.com/lab-cosmo/scikit-matter
    cd scikit-matter
    pip install -e .


This install the package in development mode, making is ``import`` able globally and
allowing you to edit the code and directly use the updated version.

Running the tests
^^^^^^^^^^^^^^^^^

.. code-block:: bash

    cd <scikit-matter PATH>
    # run unit tests
    tox
    # run the code formatter
    black --check .
    # run the linter
    flake8


You may want to setup your editor to automatically apply the `black`_ code formatter
when saving your files, there are plugins to do this with `all major editors`_.

License and developers
----------------------

This project is distributed under the BSD-3-Clauses license. By contributing to it you
agree to distribute your changes under the same license.

.. _`black`: https://black.readthedocs.io/en/stable/
.. _`all major editors`: https://black.readthedocs.io/en/stable/editor_integration.html

.. |tests| image:: https://github.com/lab-cosmo/scikit-matter/workflows/Test/badge.svg
   :alt: Github Actions Tests Job Status
   :target: https://github.com/lab-cosmo/scikit-matter/actions?query=workflow%3ATests

.. |codecov| image:: https://codecov.io/gh/lab-cosmo/scikit-matter/branch/main/graph/badge.svg?token=UZJPJG34SM
   :alt: Code coverage
   :target: https://codecov.io/gh/lab-cosmo/scikit-matter/

.. |pypi| image:: https://img.shields.io/pypi/v/skmatter.svg
   :alt: Latest PYPI version
   :target: https://pypi.org/project/skmatter

.. |conda| image:: https://anaconda.org/conda-forge/skmatter/badges/version.svg
   :alt: Latest conda version
   :target: https://anaconda.org/conda-forge/skmatter

.. |docs| image:: https://img.shields.io/badge/documentation-latest-sucess
   :alt: Documentation
   :target: https://scikit-matter.readthedocs.io
