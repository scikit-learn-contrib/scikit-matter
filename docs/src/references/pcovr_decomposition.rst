Principal Covariates Regression (PCovR)
================================================================

.. _PCovR-api:

PCovR
-----

.. autoclass:: skmatter.decomposition.PCovR
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
------------

.. autoclass:: skmatter.decomposition.KernelPCovR
    :show-inheritance:
    :special-members:

    .. automethod:: fit
    .. automethod:: transform
    .. automethod:: predict
    .. automethod:: inverse_transform
    .. automethod:: score
