"""
Sequential sample selection
"""

from .._selection import (
    _CUR,
    _FPS,
    _PCovCUR,
    _PCovFPS,
)


class FPS(_FPS):
    """Transformer that performs Greedy Sample Selection using Farthest Point Sampling."""

    def __init__(self, **kwargs):

        super().__init__(selection_type="sample", **kwargs)


class PCovFPS(_PCovFPS):
    """Transformer that performs Greedy Sample Selection using PCovR-weighted
    Farthest Point Sampling.
    """

    def __init__(self, **kwargs):

        super().__init__(selection_type="sample", **kwargs)


class CUR(_CUR):
    """Transformer that performs Greedy Sample Selection by choosing samples
    which maximize the magnitude of the left singular vectors, consistent with
    classic CUR matrix decomposition.

    """

    def __init__(self, **kwargs):
        super().__init__(selection_type="sample", **kwargs)


class PCovCUR(_PCovCUR):
    r"""Transformer that performs Greedy Sample Selection by choosing samples
    which maximize the importance score :math:`\pi`, which is the sum over
    the squares of the first :math:`k` components of the PCovR-modified
    left singular vectors.

    """

    def __init__(self, **kwargs):

        super().__init__(selection_type="sample", **kwargs)
