import numpy as np
from scipy.sparse.linalg import eigs as speig
from scipy.linalg import eig

from .simple_cur import SimpleCUR
from ..utils import pcovr_covariance


class PCovCUR(SimpleCUR):
    """Transformer that performs Greedy Feature Selection using by choosing features
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
    :math:`{\\mathbf{C} = \\mathbf{X}^T\\mathbf{X}}` used in SimpleCUR.

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

    iterated_power : int or 'auto', default='auto'
         Number of iterations for the power method computed by
         svd_solver == 'randomized'.
         Must be of range [0, infinity).

    random_state : int, RandomState instance or None, default=None
         Pass an int for reproducible results across multiple function calls.

    progress_bar: boolean, default=False
                  option to use `tqdm <https://tqdm.github.io/>`_
                  progress bar to monitor selections

    Attributes
    ----------

    n_features_to_select : int
        The number of features that were selected.

    X_selected_ : ndarray (n_samples, n_features_to_select)
                  The features selected

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
        iterated_power="auto",
        random_state=None,
        progress_bar=False,
    ):

        self.mixing = mixing

        super().__init__(
            n_features_to_select=n_features_to_select,
            score_thresh_to_select=score_thresh_to_select,
            iterative=iterative,
            k=k,
            tolerance=tolerance,
            iterated_power=iterated_power,
            random_state=random_state,
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
            random_state=self.random_state,
        )

        if self.k < Ct.shape[0] - 1:
            v, U = speig(Ct, k=self.k, tol=1e-12)
        else:
            v, U = eig(Ct)
        U = U[:, np.flip(np.argsort(v))]
        pi = (np.real(U)[:, : self.k] ** 2.0).sum(axis=1)
        # print(X.shape, np.argsort(-pi))
        return pi
