Principal Covariates Regression (PCovR)
=======================================

.. _PCovR-api:

PCovR
#####

.. currentmodule:: skcosmo.decomposition

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

.. currentmodule:: skcosmo.decomposition

.. autoclass:: KernelPCovR
    :show-inheritance:
    :special-members:

    .. automethod:: fit
    .. automethod:: transform
    .. automethod:: predict
    .. automethod:: inverse_transform
    .. automethod:: score
