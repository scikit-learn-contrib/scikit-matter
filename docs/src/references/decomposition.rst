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

Sketch-Map [Ceriotti2011]_ is a nonlinear dimensionality-reduction
algorithm that selectively preserves intermediate-range pairwise
distances using sigmoid transforms. See
:doc:`/examples/decomposition/sketchmap` for an end-to-end worked
example reproducing the analysis from the MAD paper [Mazitov2025a]_,
including validation against the reference C++ implementation.

.. autoclass:: skmatter.decomposition.SketchMap
    :show-inheritance:
    :special-members:

    .. automethod:: fit
    .. automethod:: transform
    .. automethod:: fit_transform
