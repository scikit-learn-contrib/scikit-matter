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
    """
    Transformer that performs Greedy Sample Selection using Farthest Point Sampling.

    Parameters
    ----------

    initialize: int, list of int, or 'random', default=0
        Index of the first selection(s). If 'random', picks a random
        value when fit starts. Stored in :py:attr:`self.initialize`.

    n_to_select : int or float, default=None
        The number of selections to make. If `None`, half of the samples are
        selected. If integer, the parameter is the absolute number of selections
        to make. If float between 0 and 1, it is the fraction of the total dataset to
        select. Stored in :py:attr:`self.n_to_select`.

    score_threshold : float, default=None
        Threshold for the score. If `None` selection will continue until the
        n_to_select is chosen. Otherwise will stop when the score falls below the threshold.
        Stored in :py:attr:`self.score_threshold`.

    score_threshold_type : str, default="absolute"
        How to interpret the ``score_threshold``. When "absolute", the score used by
        the selector is compared to the threshold directly. When "relative", at each iteration,
        the score used by the selector is compared proportionally to the score of the first
        selection, i.e. the selector quits when ``current_score / first_score < threshold``.
        Stored in :py:attr:`self.score_threshold_type`.

    progress_bar: bool, default=False
              option to use `tqdm <https://tqdm.github.io/>`_
              progress bar to monitor selections. Stored in :py:attr:`self.report_progress`.

    full : bool, default=False
        In the case that all non-redundant selections are exhausted, choose
        randomly from the remaining samples. Stored in :py:attr:`self.full`.

    random_state: int or RandomState instance, default=0

    Attributes
    ----------

    n_selected_ : int
                  Counter tracking the number of selections that have been made
    X_selected_ : ndarray,
                  Matrix containing the selected samples, for use in fitting
    y_selected_ : ndarray,
                  In sample selection, the matrix containing the selected targets, for use in fitting

    """

    def __init__(
        self,
        initialize=0,
        n_to_select=None,
        score_threshold=None,
        score_threshold_type="absolute",
        progress_bar=False,
        full=False,
        random_state=0,
    ):
        super().__init__(
            selection_type="sample",
            initialize=initialize,
            n_to_select=n_to_select,
            score_threshold=score_threshold,
            score_threshold_type=score_threshold_type,
            progress_bar=progress_bar,
            full=full,
            random_state=random_state,
        )


class PCovFPS(_PCovFPS):
    """Transformer that performs Greedy Sample Selection using PCovR-weighted
    Farthest Point Sampling.

    Parameters
    ----------

    mixing: float, default=0.5
            The PCovR mixing parameter, as described in PCovR as
            :math:`{\\alpha}`

    initialize: int or 'random', default=0
        Index of the first selection. If 'random', picks a random
        value when fit starts.

    n_to_select : int or float, default=None
        The number of selections to make. If `None`, half of the samples are
        selected. If integer, the parameter is the absolute number of selections
        to make. If float between 0 and 1, it is the fraction of the total dataset to
        select. Stored in :py:attr:`self.n_to_select`.

    score_threshold : float, default=None
        Threshold for the score. If `None` selection will continue until the
        n_to_select is chosen. Otherwise will stop when the score falls below the threshold.
        Stored in :py:attr:`self.score_threshold`.

    score_threshold_type : str, default="absolute"
        How to interpret the ``score_threshold``. When "absolute", the score used by
        the selector is compared to the threshold directly. When "relative", at each iteration,
        the score used by the selector is compared proportionally to the score of the first
        selection, i.e. the selector quits when ``current_score / first_score < threshold``.
        Stored in :py:attr:`self.score_threshold_type`.

    progress_bar: bool, default=False
              option to use `tqdm <https://tqdm.github.io/>`_
              progress bar to monitor selections. Stored in :py:attr:`self.report_progress`.

    full : bool, default=False
        In the case that all non-redundant selections are exhausted, choose
        randomly from the remaining samples. Stored in :py:attr:`self.full`.

    random_state: int or RandomState instance, default=0

    Attributes
    ----------

    n_selected_ : int
                  Counter tracking the number of selections that have been made
    X_selected_ : ndarray,
                  Matrix containing the selected samples, for use in fitting
    y_selected_ : ndarray,
                  In sample selection, the matrix containing the selected targets, for use in fitting


    """

    def __init__(
        self,
        mixing=0.5,
        initialize=0,
        n_to_select=None,
        score_threshold=None,
        score_threshold_type="absolute",
        progress_bar=False,
        full=False,
        random_state=0,
    ):
        super().__init__(
            selection_type="sample",
            mixing=mixing,
            initialize=initialize,
            n_to_select=n_to_select,
            score_threshold=score_threshold,
            score_threshold_type=score_threshold_type,
            progress_bar=progress_bar,
            full=full,
            random_state=random_state,
        )


class CUR(_CUR):
    """Transformer that performs Greedy Sample Selection by choosing samples
     which maximize the magnitude of the left singular vectors, consistent with
     classic CUR matrix decomposition.

    Parameters
     ----------
     recompute_every : int
                       number of steps after which to recompute the pi score
                       defaults to 1, if 0 no re-computation is done

     k : int
         number of eigenvectors to compute the importance score with, defaults to 1

     tolerance: float
          threshold below which scores will be considered 0, defaults to 1E-12

     n_to_select : int or float, default=None
         The number of selections to make. If `None`, half of the samples are
         selected. If integer, the parameter is the absolute number of selections
         to make. If float between 0 and 1, it is the fraction of the total dataset to
         select. Stored in :py:attr:`self.n_to_select`.

     score_threshold : float, default=None
         Threshold for the score. If `None` selection will continue until the
         n_to_select is chosen. Otherwise will stop when the score falls below the threshold.
         Stored in :py:attr:`self.score_threshold`.

     score_threshold_type : str, default="absolute"
         How to interpret the ``score_threshold``. When "absolute", the score used by
         the selector is compared to the threshold directly. When "relative", at each iteration,
         the score used by the selector is compared proportionally to the score of the first
         selection, i.e. the selector quits when ``current_score / first_score < threshold``.
         Stored in :py:attr:`self.score_threshold_type`.

     progress_bar: bool, default=False
               option to use `tqdm <https://tqdm.github.io/>`_
               progress bar to monitor selections. Stored in :py:attr:`self.report_progress`.

     full : bool, default=False
         In the case that all non-redundant selections are exhausted, choose
         randomly from the remaining samples. Stored in :py:attr:`self.full`.

     random_state: int or RandomState instance, default=0

     Attributes
     ----------

     X_current_ : ndarray (n_samples, n_features)
                   The original matrix orthogonalized by previous selections

     n_selected_ : int
                   Counter tracking the number of selections that have been made
     X_selected_ : ndarray,
                   Matrix containing the selected samples, for use in fitting
     y_selected_ : ndarray,
                   In sample selection, the matrix containing the selected targets, for use in fitting

    """

    def __init__(
        self,
        recompute_every=1,
        k=1,
        tolerance=1e-12,
        n_to_select=None,
        score_threshold=None,
        score_threshold_type="absolute",
        progress_bar=False,
        full=False,
        random_state=0,
    ):
        super().__init__(
            selection_type="sample",
            recompute_every=recompute_every,
            k=k,
            tolerance=tolerance,
            n_to_select=n_to_select,
            score_threshold=score_threshold,
            score_threshold_type=score_threshold_type,
            progress_bar=progress_bar,
            full=full,
            random_state=random_state,
        )


class PCovCUR(_PCovCUR):
    r"""Transformer that performs Greedy Sample Selection by choosing samples
    which maximize the importance score :math:`\pi`, which is the sum over
    the squares of the first :math:`k` components of the PCovR-modified
    left singular vectors.

    Parameters
    ----------

    mixing: float, default=0.5
            The PCovR mixing parameter, as described in PCovR as
            :math:`{\\alpha}`. Stored in :py:attr:`self.mixing`.

    recompute_every : int
                      number of steps after which to recompute the pi score
                      defaults to 1, if 0 no re-computation is done


    k : int
        number of eigenvectors to compute the importance score with, defaults to 1

    tolerance: float
         threshold below which scores will be considered 0, defaults to 1E-12

    n_to_select : int or float, default=None
        The number of selections to make. If `None`, half of the samples are
        selected. If integer, the parameter is the absolute number of selections
        to make. If float between 0 and 1, it is the fraction of the total dataset to
        select. Stored in :py:attr:`self.n_to_select`.

    score_threshold : float, default=None
        Threshold for the score. If `None` selection will continue until the
        n_to_select is chosen. Otherwise will stop when the score falls below the threshold.
        Stored in :py:attr:`self.score_threshold`.

    score_threshold_type : str, default="absolute"
        How to interpret the ``score_threshold``. When "absolute", the score used by
        the selector is compared to the threshold directly. When "relative", at each iteration,
        the score used by the selector is compared proportionally to the score of the first
        selection, i.e. the selector quits when ``current_score / first_score < threshold``.
        Stored in :py:attr:`self.score_threshold_type`.

    progress_bar: bool, default=False
              option to use `tqdm <https://tqdm.github.io/>`_
              progress bar to monitor selections. Stored in :py:attr:`self.report_progress`.

    full : bool, default=False
        In the case that all non-redundant selections are exhausted, choose
        randomly from the remaining samples. Stored in :py:attr:`self.full`.

    random_state: int or RandomState instance, default=0

    Attributes
    ----------

    X_current_ : ndarray (n_samples, n_features)
                  The original matrix orthogonalized by previous selections

    y_current_ : ndarray (n_samples, n_properties)
                The targets orthogonalized by a regression on
                the previous selections.

    n_selected_ : int
                  Counter tracking the number of selections that have been made
    X_selected_ : ndarray,
                  Matrix containing the selected samples, for use in fitting
    y_selected_ : ndarray,
                  In sample selection, the matrix containing the selected targets, for use in fitting

    """

    def __init__(
        self,
        mixing=0.5,
        recompute_every=1,
        k=1,
        tolerance=1e-12,
        n_to_select=None,
        score_threshold=None,
        score_threshold_type="absolute",
        progress_bar=False,
        full=False,
        random_state=0,
    ):
        super().__init__(
            selection_type="sample",
            mixing=mixing,
            recompute_every=recompute_every,
            k=k,
            tolerance=tolerance,
            n_to_select=n_to_select,
            score_threshold=score_threshold,
            score_threshold_type=score_threshold_type,
            progress_bar=progress_bar,
            full=full,
            random_state=random_state,
        )
