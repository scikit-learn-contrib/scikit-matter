
Welcome to sklearn-COSMO's documentation!
=========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
##################

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Sklearn-cosmo
#############

A collection of scikit-learn compatible utilities that implement methods
developed in the COSMO laboratory.

**WARNING**: this package is a work in progress, you can
currently find the prototype code in the
`kernel-tutorials <https://github.com/cosmo-epfl/kernel-tutorials>`_ repository.

Installation
############

.. code-block:: bash

  pip install https://github.com/cosmo-epfl/sklearn-cosmo

You can then import skcosmo in your code!

Developing the package
######################

Start by installing the development dependencies:

.. code-block:: bash

  pip install tox black flake8


Then this package itself

.. code-block:: bash

  git clone https://github.com/cosmo-epfl/sklearn-cosmo
  cd sklearn-cosmo
  pip install -e .

This install the package in development mode, making it importable globally
and allowing you to edit the code and directly use the updated version.

Running the tests
#################

.. code-block:: bash

  cd <sklearn-cosmo PATH>
  # run unit tests
  tox
  # run the code formatter
  black --check .
  # run the linter
  flake8

You may want to setup your editor to automatically apply the
`black <https://black.readthedocs.io/en/stable/>`_ code formatter when saving your
files, there are plugins to do this with `all major
editors <https://black.readthedocs.io/en/stable/editor_integration.html>`_.

License and developers
######################

This project is distributed under the BSD-3-Clauses license. By contributing to
it you agree to distribute your changes under the same license.
