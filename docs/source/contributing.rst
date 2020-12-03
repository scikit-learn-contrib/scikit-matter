Contributing
============

Start by installing the development dependencies:

.. code-block:: bash

  pip install tox black flake8


Then this package itself

.. code-block:: bash

  git clone https://github.com/cosmo-epfl/scikit-cosmo
  cd scikit-cosmo
  pip install -e .

This install the package in development mode, making it importable globally
and allowing you to edit the code and directly use the updated version.

Running the tests
#################

.. code-block:: bash

  cd <scikit-cosmo PATH>
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
