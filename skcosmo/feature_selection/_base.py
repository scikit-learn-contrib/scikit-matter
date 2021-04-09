"""
Sequential feature selection
"""

from .._selection import (
    _CUR,
    _FPS,
    _PCovCUR,
    _PCovFPS,
)


class FPS(_FPS):
    """Transformer that performs Greedy Feature Selection using Farthest Point Sampling."""

    def __init__(self, **kwargs):
        super().__init__(selection_type="feature", **kwargs)


class PCovFPS(_PCovFPS):
    """Transformer that performs Greedy Feature Selection using PCovR-weighted
    Farthest Point Sampling.
    """

    def __init__(self, **kwargs):
        super().__init__(selection_type="feature", **kwargs)


class CUR(_CUR):
    """Transformer that performs Greedy Feature Selection by choosing features
    which maximize the magnitude of the right singular vectors, consistent with
    classic CUR matrix decomposition.
    """

    def __init__(self, **kwargs):
        super().__init__(selection_type="feature", **kwargs)


class PCovCUR(_PCovCUR):
    """Transformer that performs Greedy Feature Selection by choosing features
    which maximize the importance score :math:`\\pi`, which is the sum over
    the squares of the first :math:`k` components of the PCovR-modified
    right singular vectors.

    """

    def __init__(self, **kwargs):

        super().__init__(selection_type="feature", **kwargs)
