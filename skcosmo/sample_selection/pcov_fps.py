import numpy as np

from ..utils import pcovr_kernel
from .simple_fps import FPS


class PCovFPS(FPS):
    """Transformer that performs Greedy Feature Selection using PCovR-weighted
    Farthest Point Sampling. Traditional FPS employs a column-wise Euclidean
    distance, which can be expressed using the covariance matrix
    :math:`\\mathbf{C} = \\mathbf{X} ^ T \\mathbf{X}`

    .. math::
        \\operatorname{d}_c(i, j) = C_{ii} - 2 C_{ij} + C_{jj}.

    When mixing < 1, this will use PCov-FPS, where a modified covariance matrix
    is used to express the distances

    .. math::
        \\mathbf{\\tilde{C}} = \\alpha \\mathbf{X}^T\\mathbf{X} +
        (1 - \\alpha)(\\mathbf{X}^T\\mathbf{X})^{-1/2}\\mathbf{X}^T
        \\mathbf{\\hat{Y}\\hat{Y}}^T\\mathbf{X}(\\mathbf{X}^T\\mathbf{X})^{-1/2}

    Please note: the best results

    Parameters
    ----------

    mixing: float, default=0.5
            The PCovR mixing parameter, as described in PCovR as
            :math:`{\\alpha}`

    n_samples_to_select : int or float, default=None
        The number of samples to select. If `None`, half of the samples are
        selected. If integer, the parameter is the absolute number of samples
        to select. If float between 0 and 1, it is the fraction of samples to
        select.

    initialize: int or 'random', default=0
        Index of the first sample to be selected. If 'random', picks a random
        value when fit starts.

    progress_bar: boolean, default=False
                  option to use `tqdm <https://tqdm.github.io/>`_
                  progress bar to monitor selections

    tolerance: float, defaults to 1E-12
                threshold below which distances will be considered 0

    Attributes
    ----------
    haussdorf_ : ndarray of shape (n_samples,)
                 the minimum distance from each sample to the set of selected
                 samples in PCovR space. once a sample is selected, the
                 distance is not updated; the final list will reflect the
                 distances when selected.

    norms_ : ndarray of shape (n_samples,)
        The self-covariances of each of the samples

    n_samples_to_select : int
        The number of samples that were selected.

    X_selected_ : ndarray (n_samples_to_select, n_features)
              The samples selected

    y_selected_ : ndarray (n_samples_to_select, n_properties)
              The corresponding target values for the samples selected

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
        initialize=0,
        tolerance=1e-12,
        progress_bar=False,
    ):

        if mixing == 1.0:
            raise ValueError(
                "Mixing = 1.0 corresponds to traditional FPS."
                "Please use the FPS class."
            )

        self.mixing = mixing

        super().__init__(
            initialize=initialize,
            n_samples_to_select=n_samples_to_select,
            progress_bar=progress_bar,
            tolerance=tolerance,
        )

    def _set_norms(self, X, y):
        """
        Set the norms as the augmented dot product of samples
        """
        self.modified_kernel_ = pcovr_kernel(mixing=self.mixing, X=X, Y=y)
        self.norms_ = np.diag(self.modified_kernel_)

    def _calculate_distances(self, X, last_selected):
        """
        Using the norms saved, calculate the distance between points
        """
        return (
            self.norms_
            + self.norms_[last_selected]
            - 2 * self.modified_kernel_[:, last_selected]
        )

    def _more_tags(self):
        """
        Pass that this method requires a target vector
        """
        return {
            "requires_y": True,
        }
