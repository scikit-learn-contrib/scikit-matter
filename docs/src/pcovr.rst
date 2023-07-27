Principal Covariates Regression (PCovR)
=======================================


.. marker-pcovr-introduction-begin

Often, one wants to construct new ML features from their
current representation in order to compress data or visualise
trends in the dataset. In the archetypal method for this
dimensionality reduction, principal components analysis (PCA),
features are transformed into the latent space which best
preserves the variance of the original data. Principal Covariates
Regression (PCovR), as introduced by [deJong1992]_,
is a modification to PCA that incorporates target information,
such that the resulting embedding could be tuned using a
mixing parameter α to improve performance in regression
tasks (:math:`\alpha = 0` corresponding to linear regression
and :math:`\alpha = 1` corresponding to PCA).
[Helfrecht2020]_ introduced the non-linear
version, Kernel Principal Covariates Regression (KPCovR),
where the mixing parameter α now interpolates between kernel ridge
regression (:math:`\alpha = 0`) and kernel principal components
analysis (KPCA, :math:`\alpha = 1`)

.. marker-pcovr-introduction-end

.. _PCovR-api:

PCovR
#####

.. currentmodule:: skmatter.decomposition

.. autoclass:: PCovR
    :show-inheritance:
    :special-members:

    .. automethod:: fit

        .. automethod:: _fit_feature_space
        .. automethod:: _fit_sample_space

    .. automethod:: transform
    .. automethod:: predict
    .. automethod:: inverse_transform
    .. automethod:: score

.. _KPCovR-api:

Kernel PCovR
############

.. currentmodule:: skmatter.decomposition

.. autoclass:: KernelPCovR
    :show-inheritance:
    :special-members:

    .. automethod:: fit
    .. automethod:: transform
    .. automethod:: predict
    .. automethod:: inverse_transform
    .. automethod:: score
