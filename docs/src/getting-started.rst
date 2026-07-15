Getting started
===============

This guide illustrates the main functionalities that ``scikit-matter`` provides. It
assumes a very basic working knowledge of how ``scikit-learn`` works. Please refer to
our :ref:`installation` instructions for installing ``scikit-matter``.

For a detailed explaination of the functionalities, please look at the
:ref:`selection-api`

.. _getting_started-selection:

Features and Samples Selection
------------------------------

.. automodule:: skmatter._selection
   :noindex:

Notebook Examples
^^^^^^^^^^^^^^^^^

.. include:: examples/selection/index.rst
   :start-line: 4


.. _getting_started-reconstruction:

Metrics
-------

.. automodule:: skmatter.metrics
   :noindex:

Notebook Examples
^^^^^^^^^^^^^^^^^

.. include:: examples/reconstruction/index.rst
   :start-line: 4

.. _getting_started-hybrid:

Hybrid Mapping Techniques
-------------------------

.. automodule:: skmatter.decomposition
   :noindex:

Notebook Examples
^^^^^^^^^^^^^^^^^

.. include:: examples/pcovr/index.rst
   :start-line: 4
.. include:: examples/pcovc/index.rst
   :start-line: 4

.. _getting_started-decomposition:

Non-linear Dimensionality Reduction
-----------------------------------

For visualising high-dimensional data with strong intermediate-range structure,
``scikit-matter`` provides :ref:`Sketch-Map <SketchMap-api>` [Ceriotti2011]_ - a
nonlinear projection method that uses sigmoid-transformed pairwise distances to
preserve the *intermediate* range while compressing both very short distances
(noise / fluctuations) and very long ones (the high-D
"curse-of-dimensionality" regime).

Because the fit scales as :math:`O(N^2)`, large datasets are handled by fitting
a representative subset:

1. select :math:`N` landmarks (e.g. via :class:`skmatter.sample_selection.FPS`),
2. weight each landmark by its Voronoi cell population, as a density proxy
   (:func:`skmatter.sample_selection.voronoi_weights`),
3. fit Sketch-Map on the weighted landmarks.

Notebook Examples
^^^^^^^^^^^^^^^^^

.. include:: examples/decomposition/index.rst
   :start-line: 4
