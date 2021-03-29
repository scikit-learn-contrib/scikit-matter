import numpy as np

from ..utils import pcovr_covariance
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

    initialize: int or 'random', default=0
        Index of the first feature to be selected. If 'random', picks a random
        value when fit starts.

    progress_bar: boolean, default=False
                  option to use `tqdm <https://tqdm.github.io/>`_
                  progress bar to monitor selections

    tolerance: float, defaults to 1E-12
                threshold below which distances will be considered 0

    Attributes
    ----------
    haussdorf_ : ndarray of shape (n_features,)
                 the minimum distance from each feature to the set of selected
                 features in PCovR space. once a feature is selected, the
                 distance is not updated; the final list will reflect the
                 distances when selected.

    norms_ : ndarray of shape (n_features,)
        The self-covariances of each of the features

    n_features_to_select : int
        The number of features that were selected.

    X_selected_ : ndarray (n_samples, n_features_to_select)
                  The features selected

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
            n_features_to_select=n_features_to_select,
            progress_bar=progress_bar,
            tolerance=tolerance,
        )

    def _set_norms(self, X, y):
        """
        Set the norms as the augmented dot product of features
        """
        self.modified_covariance_ = pcovr_covariance(mixing=self.mixing, X=X, Y=y)
        self.norms_ = np.diag(self.modified_covariance_)

    def _calculate_distances(self, X, last_selected):
        """
        Using the norms saved, calculate the distance between points
        """
        return (
            self.norms_
            + self.norms_[last_selected]
            - 2 * self.modified_covariance_[:, last_selected]
        )

    def _more_tags(self):
        """
        Pass that this method requires a target vector
        """
        return {
            "requires_y": True,
        }
