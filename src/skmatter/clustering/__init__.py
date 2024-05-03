"""
This module implements the quick shift clustering algorithm, which is used in
probabilistic analysis of molecular motifs (PAMM). See `Gasparotto and Ceriotti
<https://doi.org/10.1063/1.4900655>`_ for more details."""

from ._quick_shift import QuickShift

__all__ = [
    "QuickShift",
]
