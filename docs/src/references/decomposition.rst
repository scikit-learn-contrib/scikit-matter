Hybrid Mapping Techniques
=========================

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

.. _PCovC-api:

PCovC
-----

.. autoclass:: skmatter.decomposition.PCovC
    :show-inheritance:
    :special-members:

    .. automethod:: fit

        .. automethod:: _fit_feature_space
        .. automethod:: _fit_sample_space

    .. automethod:: transform
    .. automethod:: predict
    .. automethod:: inverse_transform
    .. automethod:: decision_function
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

.. _KPCovC-api:

Kernel PCovC
------------

.. autoclass:: skmatter.decomposition.KernelPCovC
    :show-inheritance:
    :special-members:

    .. automethod:: fit
    .. automethod:: transform
    .. automethod:: predict
    .. automethod:: inverse_transform
    .. automethod:: decision_function
    .. automethod:: score

.. _SketchMap-api:

SketchMap
---------

sketch-map [Ceriotti2011]_ is a nonlinear dimensionality-reduction method for atomistic
simulations, where configurations cluster into basins. Short distances mostly measure
thermal fluctuations inside a basin, long ones say little more than that two
configurations are far apart, so the structure worth preserving lives in between.
sketch-map therefore passes the high- and the low-dimensional distances through a
saturating sigmoid before matching them, and the embedding is driven by the intermediate
range.

The :ref:`example <sphx_glr_examples_decomposition_sketchmap.py>` introduces the method,
walks through the landmark workflow used for large datasets, and shows how the
implementation reproduces a published sketch-map of the reference C++ code.

.. autoclass:: skmatter.decomposition.SketchMap
    :show-inheritance:
    :special-members:

    .. automethod:: fit
    .. automethod:: fit_transform
