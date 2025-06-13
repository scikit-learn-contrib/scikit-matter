.. _contributing:

Contributing
============

.. include:: ../../README.rst
   :start-after: marker-contributing
   :end-before: marker-contributors

Getting started
---------------

To help with developing start by installing the development dependencies:

.. code-block:: bash

  pip install tox


Then this package itself

.. code-block:: bash

  git clone https://github.com/lab-cosmo/scikit-matter
  cd scikit-matter
  pip install -e .

This install the package in development mode, making it importable globally and allowing
you to edit the code and directly use the updated version. To see a list of all
supported tox environments please use

.. code-block:: bash

  tox -av

Running the tests
-----------------

The testsuite is implemented using Python's `unittest`_ framework and should be set-up
and run in an isolated virtual environment with `tox`_. All tests can be run with

.. code-block:: bash

  tox                  # all tests

If you wish to test only specific functionalities, for example:

.. code-block:: bash

  tox -e lint          # code style
  tox -e tests         # unit tests
  tox -e examples      # test the examples


You can also use ``tox -e format`` to use tox to do actual formatting instead of just
testing it. Also, you may want to setup your editor to automatically apply the `black
<https://black.readthedocs.io/en/stable/>`_ code formatter when saving your files, there
are plugins to do this with `all major editors
<https://black.readthedocs.io/en/stable/editor_integration.html>`_.

.. _unittest: https://docs.python.org/3/library/unittest.html
.. _tox: https://tox.readthedocs.io/en/latest

Contributing to the documentation
---------------------------------

The documentation is written in reStructuredText (rst) and uses `sphinx`_ documentation
generator. In order to modify the documentation, first create a local version on your
machine as described above. Then, build the documentation with

.. code-block:: bash

    tox -e docs

You can then visualize the local documentation with your favorite browser using the
following command (or open the :file:`docs/build/html/index.html` file manually).

.. code-block:: bash

    # on linux, depending on what package you have installed:
    xdg-open docs/build/html/index.html
    firefox docs/build/html/index.html

    # on macOS:
    open docs/build/html/index.html

.. _`sphinx` : https://www.sphinx-doc.org

Contributing new examples
-------------------------

The examples and tutorials are written as plain Python files and will be converted and
rendered for the documentation using `Sphinx-Gallery
<https://sphinx-gallery.github.io/stable/index.html>`.

All examples are located in the ``examples`` directory in the root of the repository. To
contribute a new example create a new ``.py`` file in one of the subdirectories. For
writing the example/tutorial you can use another file for inspiration. Details on how to
structure a Python script for Sphinx-Gallery are given in the `Sphinx-Gallery
documentation <https://sphinx-gallery.github.io/stable/syntax.html>`.

We encourage yoy to at least add one plot to your example to provide a nice image for
the gallery on the website.


Contributing Datasets
---------------------

Have an example dataset that would fit into scikit-matter?

Contributing a dataset is easy. First, copy your numpy file into
``src/skmatter/datasets/data/`` with an informative name. Here, we'll call it
``my-dataset.npz``.

Next, create a documentation file in ``src/skmatter/datasets/data/my-dataset.rst``.
This file should look like this:

.. code-block:: rst

  .. _my-dataset:

  My Dataset
  ##########

  This is a summary of my dataset. My dataset was originally published in My Paper.

  Function Call
  -------------

  .. function:: skmatter.datasets.load_my_dataset

  Data Set Characteristics
  ------------------------

  :Number of Instances: ______

  :Number of Features: ______

  The representations were computed using the _____ package using the hyperparameters:


  +------------------------+------------+
  | key                    |   value    |
  +------------------------+------------+
  | hyperparameter 1       |    _____   |
  +------------------------+------------+
  | hyperparameter 2       |    _____   |
  +------------------------+------------+

  Of the ____ resulting features, ____ were selected via _____.

  References
  ----------

  Reference Code
  --------------


Then, show ``scikit-matter`` how to load your data by adding a loader function to
``skmatter/datasets/_base.py``. It should look like this:

.. code-block:: python

    def load_my_dataset():
        """Load and returns my dataset.

        Returns
        -------
        my_data : sklearn.utils.Bunch
            Dictionary-like object, with the following attributes:

            data : `sklearn.utils.Bunch` --
            contains the keys ``X`` and ``y``.
            My input vectors and properties, respectively.

            DESCR: `str` --
            The full description of the dataset.
        """
        module_path = dirname(__file__)
        target_filename = join(module_path, "data", "my-dataset.npz")
        raw_data = np.load(target_filename)
        data = Bunch(
            X=raw_data["X"],
            y=raw_data["y"],
        )
        with open(join(module_path, "descr", "my-dataset.rst")) as rst_file:
            fdescr = rst_file.read()

        return Bunch(data=data, DESCR=fdescr)

Add this function to ``src/skmatter/datasets/__init__.py``.

Finally, add a test to ``tests/test_datasets.py`` to see that your dataset loads
properly. It should look something like this:

.. code-block:: python

    class MyDatasetTests(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.my_data = load_my_data()

        def test_load_my_data(self):
            # test if representations and properties have commensurate shape
            self.assertTrue(
                self.my_data.data.X.shape[0] == self.my_data.data.y.shape[0]
            )

        def test_load_my_data_descr(self):
            self.my_data.DESCR


You're good to go! Time to submit a `pull request.
<https://github.com/lab-cosmo/scikit-matter/pulls>`_

How to Perform a Release
-------------------------

1. **Prepare a Release Pull Request**

   - Based on the main branch create branch ``release-0.4`` and a PR.
   - Ensure that all `CI tests
     <https://github.com/scikit-learn-contrib/scikit-matter/actions>`_ pass.
   - Optionally, run the tests locally to double-check.

2. **Update the Changelog**

   - Edit the changelog located in ``CHANGELOG``:
      - Add a new section for the new version, summarizing the changes based on the
        PRs merged since the last release.
      - Leave a placeholder section titled *Unreleased* for future updates.

3. **Merge the PR and Create a Tag**

   - Merge the release PR.
   - Update the ``main`` branch and check that the latest commit is the release PR with
     ``git log``
   - Create a tag on directly the ``main`` branch.
   - Push the tag to GitHub. For example for a release of version ``0.4``:

     .. code-block:: bash

        git checkout main
        git pull
        git tag -a v0.4 -m "Release v0.4"
        git push --tags

4. **Finalize the GitHub Release**

   - Once the PR is merged, the CI will automatically:
      - Publish the package to PyPI.
      - Create a draft release on GitHub.
   - Update the GitHub release notes by pasting the changelog for the version.

5. **Merge Conda Recipe Changes**

   - May resolve and then merge an automatically created PR on the `conda recipe
     <https://github.com/conda-forge/skmatter-feedstock>`_.
   - Once thus PR is merged and the new version will be published automatically on the
     `conda-forge <https://anaconda.org/conda-forge/skmatter>`_ channel.
