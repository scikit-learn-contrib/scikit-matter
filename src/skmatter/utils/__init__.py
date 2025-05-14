"""
The :mod:`skmatter.utils` module includes functions which are
used by multiple packages
"""

from ._orthogonalizers import (
    X_orthogonalizer,
    Y_feature_orthogonalizer,
    Y_sample_orthogonalizer,
)
from ._pcovr_utils import (
    check_krr_fit,
    check_lr_fit,
    pcovr_covariance,
    pcovr_kernel,
)

from ._pcovc_utils import check_cl_fit

from ._progress_bar import (
    get_progress_bar,
    no_progress_bar,
)

from ._sparsekde import (
    effdim,
    oas,
)

__all__ = [
    "get_progress_bar",
    "no_progress_bar",
    "pcovr_covariance",
    "pcovr_kernel",
    "check_krr_fit",
    "check_lr_fit",
    "X_orthogonalizer",
    "Y_sample_orthogonalizer",
    "Y_feature_orthogonalizer",
    "effdim",
    "oas",
]
