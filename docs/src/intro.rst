What's in scikit-matter?
=======================

``scikit-matter`` is a collection of `scikit-learn <https://scikit.org/>`_
compatible utilities that implement methods born out of the materials science
and chemistry communities.

This package serves two purposes: 1) as a development ground for models and patches that may ultimately be suitable for inclusion
in sklearn, and 2) to coalesce field-specific sklearn-like routines and models in
a well-documented and standardized repository.

Currently, scikit-matter contains models described in [Imbalzano2018]_, [Helfrecht2020]_, [Goscinski2021]_ and [Cersonsky2021]_, as well
as some modifications to sklearn functionalities and minimal datasets that are useful within the field
of computational materials science and chemistry.



- Fingerprint Selection:
   Multiple data sub-selection modules, for selecting the most relevant features and samples out of a large set of candidates [Imbalzano2018]_, [Helfrecht2020]_ and [Cersonsky2021]_.

   * :ref:`CUR-api` decomposition: an iterative feature selection method based upon the singular value decoposition.
   * :ref:`PCov-CUR-api` decomposition extends upon CUR by using augmented right or left singular vectors inspired by Principal Covariates Regression.
   * :ref:`FPS-api`: a common selection technique intended to exploit the diversity of the input space. The selection of the first point is made at random or by a separate metric.
   * :ref:`PCov-FPS-api` extends upon FPS much like PCov-CUR does to CUR.
   * :ref:`Voronoi-FPS-api`: conduct FPS selection, taking advantage of Voronoi tessellations to accelerate selection.
   * :ref:`DCH-api`: selects samples by constructing a directional convex hull and determining which samples lie on the bounding surface.

- Reconstruction Measures:
   A set of easily-interpretable error measures of the relative information capacity of feature space `F` with respect to feature space `F'`.
   The methods returns a value between 0 and 1, where 0 means that `F` and `F'` are completey distinct in terms of linearly-decodable information, and where 1 means that `F'` is contained in `F`.
   All methods are implemented as the root mean-square error for the regression of the feature matrix `X_F'` (or sometimes called `Y` in the doc) from `X_F` (or sometimes called `X` in the doc) for transformations with different constraints (linear, orthogonal, locally-linear).
   By default a custom 2-fold cross-validation :py:class:`skosmo.linear_model.RidgeRegression2FoldCV` is used to ensure the generalization of the transformation and efficiency of the computation, since we deal with a multi-target regression problem.
   Methods were applied to compare different forms of featurizations through different hyperparameters and induced metrics and kernels [Goscinski2021]_ .

   * :ref:`GRE-api` (GRE) computes the amount of linearly-decodable information recovered through a global linear reconstruction.
   * :ref:`GRD-api` (GRD) computes the amount of distortion contained in a global linear reconstruction. 
   * :ref:`LRE-api` (LRE) computes the amount of decodable information recovered through a local linear reconstruction for the k-nearest neighborhood of each sample.

- Principal Covariates Regression

   * PCovR: the standard Principal Covariates Regression [deJong1992]_. Utilises a combination between a PCA-like and an LR-like loss, and therefore attempts to find a low-dimensional projection of the feature vectors that simultaneously minimises information loss and error in predicting the target properties using only the latent space vectors $\mathbf{T}$ :ref:`PCovR-api`.
   * Kernel Principal Covariates Regression (KPCovR) a kernel-based variation on the original PCovR method, proposed in [Helfrecht2020]_ :ref:`KPCovR-api`.
  
If you would like to contribute to scikit-matter, check out our :ref:`contributing` page!
