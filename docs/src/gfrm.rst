.. _gfrm:

Reconstruction Measures
=======================

.. marker-reconstruction-introduction-begin

.. automodule:: skmatter.metrics

These reconstruction measures are available:

* :ref:`GRE-api` (GRE) computes the amount of linearly-decodable information
  recovered through a global linear reconstruction.
* :ref:`GRD-api` (GRD) computes the amount of distortion contained in a global linear
  reconstruction.
* :ref:`LRE-api` (LRE) computes the amount of decodable information recovered through
  a local linear reconstruction for the k-nearest neighborhood of each sample.

.. marker-reconstruction-introduction-end

.. currentmodule:: skmatter.metrics

.. _GRE-api:

Global Reconstruction Error
---------------------------

.. autofunction:: pointwise_global_reconstruction_error
.. autofunction:: global_reconstruction_error

.. _GRD-api:

Global Reconstruction Distortion
--------------------------------

.. autofunction:: pointwise_global_reconstruction_distortion
.. autofunction:: global_reconstruction_distortion

.. _LRE-api:

Local Reconstruction Error
--------------------------

.. autofunction:: pointwise_local_reconstruction_error
.. autofunction:: local_reconstruction_error
