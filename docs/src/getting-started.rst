Getting started
===============

A small introduction to all methods implemented in scikit-matter.
For a detailed explaination, please look at the :ref:`selection-api`

Features and Samples Selection
------------------------------

  .. include:: selection.rst
     :start-after: marker-selection-introduction-begin
     :end-before: marker-selection-introduction-end


These selectors are available:

* :ref:`CUR-api`: a decomposition: an iterative feature selection method based upon the
  singular value decoposition.
* :ref:`PCov-CUR-api` decomposition extends upon CUR by using augmented right or left
  singular vectors inspired by Principal Covariates Regression.
* :ref:`FPS-api`: a common selection technique intended to exploit the diversity of
  the input space. The selection of the first point is made at random or by a
  separate metric
* :ref:`PCov-FPS-api` extends upon FPS much like PCov-CUR does to CUR.
* :ref:`Voronoi-FPS-api`: conduct FPS selection, taking advantage of Voronoi
  tessellations to accelerate selection.
* :ref:`DCH-api`: selects samples by constructing a directional convex hull and
  determining which samples lie on the bounding surface.

Examples
^^^^^^^^

.. include:: examples/selection/index.rst
   :start-line: 4


Reconstruction Measures
-----------------------

  .. include:: gfrm.rst
     :start-after: marker-reconstruction-introduction-begin
     :end-before: marker-reconstruction-introduction-end


These reconstruction measures are available:

* :ref:`GRE-api` (GRE) computes the amount of linearly-decodable information
  recovered through a global linear reconstruction.
* :ref:`GRD-api` (GRD) computes the amount of distortion contained in a global linear
  reconstruction.
* :ref:`LRE-api` (LRE) computes the amount of decodable information recovered through
  a local linear reconstruction for the k-nearest neighborhood of each sample.

Examples
^^^^^^^^

.. include:: examples/reconstruction/index.rst
   :start-line: 4

Principal Covariates Regression
-------------------------------

  .. include:: pcovr.rst
     :start-after: marker-pcovr-introduction-begin
     :end-before: marker-pcovr-introduction-end

It includes

* :ref:`PCovR-api` the standard Principal Covariates Regression. Utilises a
  combination between a PCA-like and an LR-like loss, and therefore attempts to find
  a low-dimensional projection of the feature vectors that simultaneously minimises
  information loss and error in predicting the target properties using only the
  latent space vectors :math:`\mathbf{T}`.
* :ref:`KPCovR-api` the Kernel Principal Covariates Regression
  a kernel-based variation on the
  original PCovR method, proposed in [Helfrecht2020]_.


Examples
^^^^^^^^

.. include:: examples/pcovr/index.rst
   :start-line: 4
