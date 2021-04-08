import numpy as np
from scipy.linalg import eig
from scipy.sparse.linalg import eigs as speig

from ..utils import pcovr_covariance
from .simple_cur import CUR


class PCovCUR(CUR):
    """Transformer that performs Greedy Feature Selection by choosing features
    which maximize the importance score :math:`\\pi`, which is the sum over
    the squares of the first :math:`k` components of the right singular vectors

    .. math::

        \\pi_j =
        \\sum_i^k \\left(\\mathbf{U}_\\mathbf{\\tilde{C}}\\right)_{ij}^2.

    where :math:`{\\mathbf{\\tilde{C}} = \\alpha \\mathbf{X}^T\\mathbf{X} +
    (1 - \\alpha)(\\mathbf{X}^T\\mathbf{X})^{-1/2}\\mathbf{X}^T
    \\mathbf{\\hat{Y}\\hat{Y}}^T\\mathbf{X}(\\mathbf{X}^T\\mathbf{X})^{-1/2}}`
    for some mixing parameter :math:`{\\alpha}`. When :math:`{\\alpha = 1}`,
    this defaults to the covariance matrix
    :math:`{\\mathbf{C} = \\mathbf{X}^T\\mathbf{X}}` used in CUR.

    Parameters
    ----------

    mixing: float, default=0.5
            The PCovR mixing parameter, as described in PCovR as
            :math:`{\\alpha}`

    n_features_to_select : int or float, default=None
        The number of features to select. If `None`, half of the features are
        selected. If integer, the parameter is the absolute number of features
        to select. If float between 0 and 1, it is the fraction of features to
        select.

    score_thresh_to_select : float, default=None
        Threshold for the score. If `None` selection will continue until the
        number or fraction given by n_features_to_select is chosen. Otherwise
        will stop when the score falls below the threshold.

    iterative : boolean
                whether to orthogonalize after each selection, defaults to `true`

    k : int
        number of eigenvectors to compute the importance score with, defaults to 1

    tolerance: float
         threshold below which scores will be considered 0, defaults to 1E-12

    progress_bar: boolean, default=False
                  option to use `tqdm <https://tqdm.github.io/>`_
                  progress bar to monitor selections

    Attributes
    ----------

    n_features_to_select : int
        The number of features that were selected.

    X_selected_ : ndarray (n_samples, n_features_to_select)
                  The features selected

    X_current : ndarray (n_samples, n_features)
                  The features, orthogonalized by previously selected features

    y_current : ndarray (n_samples, n_properties)
                The properties, if supplied, orthogonalized by a regression on
                the previously selected features

    n_selected_ : int
        The number of features that have been selected thus far

    report_progress : callable
        A wrapper to report the progress of the selector using a `tqdm` style
        progress bar

    score_threshold : float (optional)
        A score below which to stop selecting points

    selected_idx_ : ndarray of integers
                    indices of the selected features, with respect to the
                    original fitted matrix

    support_ : ndarray of shape (n_features,), dtype=bool
        The mask of selected features.

    """

    def __init__(
        self,
        mixing=0.5,
        n_features_to_select=None,
        score_thresh_to_select=None,
        iterative=True,
        k=1,
        tolerance=1e-12,
        progress_bar=False,
    ):

        self.mixing = mixing

        super().__init__(
            n_features_to_select=n_features_to_select,
            score_thresh_to_select=score_thresh_to_select,
            iterative=iterative,
            k=k,
            tolerance=tolerance,
            progress_bar=progress_bar,
        )

    def _compute_pi(self, X, y=None):
        """
        For feature selection, the importance score :math:`\\pi` is the sum over
        the squares of the first :math:`k` components of the right singular vectors

        .. math::

            \\pi_j =
            \\sum_i^k \\left(\\mathbf{U}_\\mathbf{\\tilde{C}}\\right)_{ij}^2.

        where :math:`{\\mathbf{C} = \\mathbf{X}^T\\mathbf{X}.
        """

        Ct = pcovr_covariance(
            self.mixing,
            X,
            y,
            rcond=1e-12,
            rank=None,
        )

        if self.k < Ct.shape[0] - 1:
            v, U = speig(Ct, k=self.k, tol=1e-12)
        else:
            v, U = eig(Ct)
        U = U[:, np.flip(np.argsort(v))]
        pi = (np.real(U)[:, : self.k] ** 2.0).sum(axis=1)

        return pi
