scikit-matter
=============
|tests| |codecov| |pypi| |conda| |docs| |doi|

A collection of ``scikit-learn`` compatible utilities that implement methods born out of
the materials science and chemistry communities.

For details, tutorials, and examples, please have a look at our `documentation`_.

.. _`documentation`: https://scikit-matter.readthedocs.io

.. marker-installation

Installation
------------
You can install *scikit-matter* either via pip using

.. code-block:: bash

    pip install skmatter

or conda

.. code-block:: bash

    conda install -c conda-forge skmatter

You can then ``import skmatter`` and use scikit-matter in your projects!

.. marker-ci-tests

Tests
-----
We are testing our code for Python 3.8 and 3.12 on  `Windows Server 2019`_, `macOS 11`_
and `Ubuntu LTS 22.04`_.

.. _`Windows Server 2019`: https://github.com/actions/runner-images/blob/main/images/win/Windows2019-Readme.md
.. _`macOS 11`: https://github.com/actions/runner-images/blob/main/images/macos/macos-11-Readme.md
.. _`Ubuntu LTS 22.04`: https://github.com/actions/runner-images/blob/main/images/linux/Ubuntu2204-Readme.md

.. marker-issues

Having problems or ideas?
-------------------------
Having a problem with scikit-matter? Please let us know by `submitting an issue
<https://github.com/scikit-learn-contrib/scikit-matter/issues>`_.

Submit new features or bug fixes through a `pull request
<https://github.com/scikit-learn-contrib/scikit-matter/pulls>`_.

.. marker-contributing

Call for Contributions
----------------------
We always welcome new contributors. If you want to help us take a look at our
`contribution guidelines`_ and afterwards you may start with an open issue marked as
`good first issue`_.

Writing code is not the only way to contribute to the project. You can also:

* review `pull requests`_
* help us stay on top of new and old `issues`_
* develop `examples and tutorials`_
* maintain and `improve our documentation`_
* contribute `new datasets`_

.. _`contribution guidelines`: https://scikit-matter.readthedocs.io/en/latest/contributing.html
.. _`good first issue`: https://github.com/scikit-learn-contrib/scikit-matter/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22
.. _`pull requests`: https://github.com/scikit-learn-contrib/scikit-matter/pulls
.. _`issues`: https://github.com/scikit-learn-contrib/scikit-matter/issues
.. _`improve our documentation`: https://scikit-matter.readthedocs.io/en/latest/contributing.html#contributing-to-the-documentation
.. _`examples and tutorials`: https://scikit-matter.readthedocs.io/en/latest/contributing.html#contributing-new-examples
.. _`new datasets`: https://scikit-matter.readthedocs.io/en/latest/contributing.html#contributing-datasets

.. marker-citing

Citing scikit-matter
--------------------
If you use *scikit-matter* for your work, please cite:

Goscinski A, Principe VP, Fraux G et al. scikit-matter :
A Suite of Generalisable Machine Learning Methods Born out of Chemistry
and Materials Science. Open Res Europe 2023, 3:81.
`10.12688/openreseurope.15789.2`_

.. _`10.12688/openreseurope.15789.2`: https://doi.org/10.12688/openreseurope.15789.2

.. marker-contributors

Contributors
------------
Thanks goes to all people that make scikit-matter possible:

.. image:: https://contrib.rocks/image?repo=scikit-learn-contrib/scikit-matter
   :target: https://github.com/scikit-learn-contrib/scikit-matter/graphs/contributors

.. |tests| image:: https://github.com/scikit-learn-contrib/scikit-matter/workflows/Tests/badge.svg
   :alt: Github Actions Tests Job Status
   :target: action_

.. |codecov| image:: https://codecov.io/gh/scikit-learn-contrib/scikit-matter/branch/main/graph/badge.svg?token=UZJPJG34SM
   :alt: Code coverage
   :target: https://codecov.io/gh/scikit-learn-contrib/scikit-matter/

.. |docs| image:: https://img.shields.io/badge/documentation-latest-sucess
   :alt: Python
   :target: documentation_

.. |pypi| image:: https://img.shields.io/pypi/v/skmatter.svg
   :alt: Latest PYPI version
   :target: https://pypi.org/project/skmatter

.. |conda| image:: https://anaconda.org/conda-forge/skmatter/badges/version.svg
   :alt: Latest conda version
   :target: https://anaconda.org/conda-forge/skmatter

.. |doi| image:: https://img.shields.io/badge/DOI-10.12688-blue
   :alt: ORE Paper
   :target: `10.12688/openreseurope.15789.2`_

.. _`action`: https://github.com/scikit-learn-contrib/scikit-matter/actions?query=branch%3Amain
