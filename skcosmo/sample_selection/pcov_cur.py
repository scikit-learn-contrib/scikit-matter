import numpy as np
from scipy.sparse.linalg import eigs as speig
from scipy.linalg import eig

from .simple_cur import CUR
from ..utils import pcovr_kernel


class PCovCUR(CUR):
    r"""Transformer that performs Greedy Feature Selection using by choosing samples
    which maximize the importance score :math:`\pi`, which is the sum over
    the squares of the first :math:`k` components of the right singular vectors

    .. math::

        \pi_j =
        \sum_i^k \left(\mathbf{U}_\mathbf{\tilde{K}}\right)_{ij}^2.

    where :math:`{\mathbf{\tilde{K}} = \alpha \mathbf{XX}^T +
    (1 - \alpha)\mathbf{\hat{Y}\hat{Y}}^T}` for some mixing parameter
    :math:`{\alpha}`. When :math:`{\alpha = 1}`, this defaults to the Gram
    matrix :math:`{\mathbf{K} = \mathbf{X}\mathbf{X}^T}`.

    Parameters
    ----------

    mixing: float, default=0.5
            The PCovR mixing parameter, as described in PCovR as
            :math:`{\alpha}`

    n_samples_to_select : int or float, default=None
        The number of samples to select. If `None`, half of the samples are
        selected. If integer, the parameter is the absolute number of samples
        to select. If float between 0 and 1, it is the fraction of samples to
        select.

    score_thresh_to_select : float, default=None
        Threshold for the score. If `None` selection will continue until the
        number or fraction given by n_samples_to_select is chosen. Otherwise
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

    n_samples_to_select : int
        The number of samples that were selected.

    X_selected_ : ndarray (n_samples_to_select, n_features)
                  The samples selected

    y_selected_ : ndarray (n_samples_to_select, n_properties)
                  The targets selected

    X_current_ : ndarray (n_samples, n_features)
                  The samples, orthogonalized by previously selected samples

    y_current_ : ndarray (n_samples, n_properties)
                The properties, if supplied, orthogonalized by a regression on
                the previously selected samples

    eligible_ : ndarray of shape (n_samples,), dtype=bool
        A mask of samples eligible for selection

    n_selected_ : int
        The number of samples that have been selected thus far

    report_progress : callable
        A wrapper to report the progress of the selector using a `tqdm` style
        progress bar

    score_threshold : float (optional)
        A score below which to stop selecting points

    selected_idx_ : ndarray of integers
                    indices of the selected samples, with respect to the
                    original fitted matrix

    support_ : ndarray of shape (n_samples,), dtype=bool
        The mask of selected samples.

    """

    def __init__(
        self,
        mixing=0.5,
        n_samples_to_select=None,
        score_thresh_to_select=None,
        iterative=True,
        k=1,
        tolerance=1e-12,
        iterated_power="auto",
        random_state=None,
        progress_bar=False,
    ):

        self.mixing = mixing

        super().__init__(
            n_samples_to_select=n_samples_to_select,
            score_thresh_to_select=score_thresh_to_select,
            iterative=iterative,
            k=k,
            tolerance=tolerance,
            iterated_power=iterated_power,
            random_state=random_state,
            progress_bar=progress_bar,
        )

    def _compute_pi(self, X, y=None):
        r"""
        For sample selection, the importance score :math:`\pi` is the sum over
        the squares of the first :math:`k` components of the right singular vectors

        .. math::

            \pi_j =
            \sum_i^k \left(\mathbf{U}_\mathbf{\tilde{K}}\right)_{ij}^2.

        where :math:`{\mathbf{K} = \mathbf{X}\mathbf{X}^T.
        """

        Kt = pcovr_kernel(
            self.mixing,
            X,
            y,
        )

        if self.k < Kt.shape[0] - 1:
            v, U = speig(Kt, k=self.k, tol=1e-12)
        else:
            v, U = eig(Kt)

        U = U[:, np.flip(np.argsort(v))]
        pi = (np.real(U)[:, : self.k] ** 2.0).sum(axis=1)

        return pi
