# -*- coding: utf-8 -*-
"""

This module contains Farthest Point Sampling (FPS) classes for sub-selecting
features or samples from given datasets. Each class supports a Principal
Covariates Regression (PCov)-inspired variant, using a mixing parameter and
target values to bias the selections.

Authors: Rose K. Cersonsky
         Michele Ceriotti

"""

"""
    /* ---------------------------------------------------------------------- */
    std::tuple<Eigen::ArrayXi, Eigen::ArrayXd, Eigen::ArrayXd, Eigen::ArrayXi,
               Eigen::ArrayXd>
    select_fps_voronoi(const Eigen::Ref<const RowMatrixXd> & feature_matrix,
                       int n_sparse, int i_first_point) {
      // number of inputs
      int n_inputs = feature_matrix.rows();
      // number of features
      int n_features = feature_matrix.cols();

      // defaults to full sorting of the inputs
      if (n_sparse == 0) {
        n_sparse = n_inputs;
      }

      // TODO(ceriottm) <- use the exception mechanism
      // for librascal whatever it is
      if (n_sparse > n_inputs) {
        throw std::runtime_error("Cannot FPS more inputs than those provided");
      }

      // return arrays
      // FPS indices
      auto sparse_indices = Eigen::ArrayXi(n_sparse);
      // minmax distances^2
      auto sparse_minmax_d2 = Eigen::ArrayXd(n_sparse);
      // size^2 of Voronoi cells
      auto voronoi_r2 = Eigen::ArrayXd(n_sparse);
      // assignment of points to Voronoi cells
      auto voronoi_indices = Eigen::ArrayXi(n_inputs);

      // work arrays
      // index of the maximum-d2 point in each cell
      auto voronoi_i_far = Eigen::ArrayXd(n_sparse);
      // square moduli of inputs
      auto feature_x2 = Eigen::ArrayXd(n_inputs);
      // list of distances^2 to latest FPS point
      auto list_new_d2 = Eigen::ArrayXd(n_inputs);
      // list of minimum distances^2 to each input
      auto list_min_d2 = Eigen::ArrayXd(n_inputs);
      // flags for "active" cells
      auto f_active = Eigen::ArrayXi(n_sparse);
      // list of dist^2/4 to previously selected points
      auto list_sel_d2q = Eigen::ArrayXd(n_sparse);
      // feaures of the latest FPS point
      auto feature_new = Eigen::VectorXd(n_features);
      // matrix of the features for the active point selection
      auto feature_sel = RowMatrixXd(n_sparse, n_features);

      int i_new{};
      double d2max_new{};

      // computes the squared modulus of input points
      feature_x2 = feature_matrix.cwiseAbs2().rowwise().sum();

      // initializes arrays taking the first point provided in input
      sparse_indices(0) = i_first_point;
      //  distance square to the selected point
      list_new_d2 =
          feature_x2 + feature_x2(i_first_point) -
          2 * (feature_matrix * feature_matrix.row(i_first_point).transpose())
                  .array();
      list_min_d2 = list_new_d2;  // we only have this point....

      voronoi_r2 = 0.0;
      voronoi_indices = 0;
      // picks the initial Voronoi radius and the farthest point index
      voronoi_r2(0) = list_min_d2.maxCoeff(&voronoi_i_far(0));

      feature_sel.row(0) = feature_matrix.row(i_first_point);

#ifdef DO_TIMING
      // timing code
      double tmax{0}, tactive{0}, tloop{0};
      int64_t ndist_eval{0}, npoint_skip{0}, ndist_active{0};
      auto gtstart = hrclock::now();
#endif
      for (int i = 1; i < n_sparse; ++i) {
#ifdef DO_TIMING
        auto tstart = hrclock::now();
#endif
        /*
         * find the maximum minimum distance and the corresponding point.  this
         * is our next FPS. The maxmin point must be one of the voronoi
         * radii. So we pick it from this smaller array. Note we only act on the
         * first i items as the array is filled incrementally picks max dist and
         * index of the cell
         */
        d2max_new = voronoi_r2.head(i).maxCoeff(&i_new);
        // the actual index of the fartest point
        i_new = voronoi_i_far(i_new);
#ifdef DO_TIMING
        auto tend = hrclock::now();
        tmax +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(tend - tstart)
                .count();
#endif
        // store properties of the new FPS selection
        sparse_indices(i) = i_new;
        sparse_minmax_d2(i - 1) = d2max_new;
        feature_new = feature_matrix.row(i_new);
        /*
         * we store indices of the selected features because we can then compute
         * some of the distances with contiguous array operations
         */
        feature_sel.row(i) = feature_new;

        /*
         * now we find the "active" Voronoi cells, i.e. those
         * that might change due to the new selection.
         */
        f_active = 0;

#ifdef DO_TIMING
        tstart = hrclock::now();
        ndist_active += i;
#endif
        /*
         * must compute distance of the new point to all the previous FPS.  some
         * of these might have been computed already, but bookkeeping could be
         * worse that recomputing (TODO: verify!)
         */
        list_sel_d2q.head(i) =
            feature_x2(i_new) -
            2 * (feature_sel.topRows(i) * feature_new).array();
        for (ssize_t j = 0; j < i; ++j) {
          list_sel_d2q(j) += feature_x2(sparse_indices(j));
        }
        list_sel_d2q.head(i) *= 0.25;  // triangle inequality: voronoi_r < d/2

        for (ssize_t j = 0; j < i; ++j) {
          /*
           * computes distances to previously selected points and uses triangle
           * inequality to find which voronoi sets might be affected by the
           * newly selected point divide by four so we don't have to do that
           * later to speed up the bound on distance to the new point
           */
          if (list_sel_d2q(j) < voronoi_r2(j)) {
            f_active(j) = 1;
            //! size of active cells will have to be recomputed
            voronoi_r2(j) = 0;
#ifdef DO_TIMING
          } else {
            ++npoint_skip;
#endif
          }
        }

#ifdef DO_TIMING
        tend = hrclock::now();
        tactive +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(tend - tstart)
                .count();

        tstart = hrclock::now();
#endif

        for (ssize_t j = 0; j < n_inputs; ++j) {
          int voronoi_idx_j = voronoi_indices(j);
          // only considers "active" points
          if (f_active(voronoi_idx_j) > 0) {
            /*
             * check if we can skip this check for point j. this is a tighter
             * bound on the distance, since |x_j-x_sel|<rvoronoi_sel
             */
            if (list_sel_d2q(voronoi_idx_j) < list_min_d2(j)) {
              double d2_j = feature_x2(i_new) + feature_x2(j) -
                            2 * feature_new.dot(feature_matrix.row(j));
#ifdef DO_TIMING
              ndist_eval++;
#endif
              /*
               * we have to reassign point j to the new selection. also, the
               * voronoi center is actually that of the new selection
               */
              if (d2_j < list_min_d2(j)) {
                list_min_d2(j) = d2_j;
                voronoi_indices(j) = voronoi_idx_j = i;
              }
            }
            // also must update the voronoi radius
            if (list_min_d2(j) > voronoi_r2(voronoi_idx_j)) {
              voronoi_r2(voronoi_idx_j) = list_min_d2(j);
              // stores the index of the FP of the cell
              voronoi_i_far(voronoi_idx_j) = j;
            }
          }
        }

#ifdef DO_TIMING
        tend = hrclock::now();
        tloop +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(tend - tstart)
                .count();
#endif
      }
      sparse_minmax_d2(n_sparse - 1) = 0;

#ifdef DO_TIMING
      auto gtend = hrclock::now();

      std::cout << "Skipped " << npoint_skip << " FPS centers of "
                << n_sparse * (n_sparse - 1) / 2 << " - "
                << npoint_skip * 100. / (n_sparse * (n_sparse - 1) / 2)
                << "%\n";
      std::cout << "Computed " << ndist_eval << " distances rather than "
                << n_inputs * n_sparse << " - "
                << ndist_eval * 100. / (n_inputs * n_sparse) << " %\n";

      std::cout << "Time total "
                << std::chrono::duration_cast<std::chrono::nanoseconds>(gtend -
                                                                        gtstart)
                           .count() *
                       1e-9
                << "\n";
      std::cout << "Time looking for max " << tmax * 1e-9 << "\n";
      std::cout << "Time looking for active " << tactive * 1e-9 << " with "
                << ndist_active << " distances\n";
      std::cout << "Time general loop " << tloop * 1e-9 << "\n";
#endif

      return std::make_tuple(sparse_indices, sparse_minmax_d2, list_min_d2,
                             voronoi_indices, voronoi_r2);
    }


"""

from abc import abstractmethod
import numpy as np
from skcosmo.pcovr.pcovr_distances import pcovr_covariance, pcovr_kernel
from sklearn import TransformerMixin
from sklearn.utils import check_X_y, check_array
from skcosmo.utils import get_progress_bar

class GreedySelector(TransformerMixin):
    """ Selects features or samples in an iterative way """
    
    
    def __init__(self, n_select = None, support = None, kernel = None):
        self.support_ = None # TODO implement some kind of restart mechanism.
        self.n_select_ = n_select
        self.n_selected_ = 0 
        
        if kernel is None:
            kernel = lambda x: x
        self.kernel_ = kernel # TODO implement support of providing a kernel function, sklearn style
    
    def get_support(self):
        
        return self.support_[:self.n_selected_]
        
    def transform(X):
        return X[self.support_]

class SimpleFPS(GreedySelector):        
    
    def fit(X, initial = 0):
        
        self.norms_ := (X**2).sum(axis=1)
        
        # first point
        self.support_[0] = initial
        self.n_select_ = 1 
        
        self.haussdorf_ 
        
        
        
        
        
        
        
        

def _calc_distances_(K, ref_idx, idxs=None):
    """
    Calculates the distance between points in ref_idx and idx

    Assumes

    .. math::
        d(i, j) = K_{i,i} - 2 * K_{i,j} + K_{j,j}

    : param K : distance matrix, must contain distances for ref_idx and idxs
    : type K : array

    : param ref_idx : index of reference points
    : type ref_idx : int

    : param idxs : indices of points to compute distance to ref_idx
                   defaults to all indices in K
    : type idxs : list of int, None

    """
    if idxs is None:
        idxs = range(K.shape[0])
    return np.array(
        [np.real(K[j][j] - 2 * K[j][ref_idx] + K[ref_idx][ref_idx]) for j in idxs]
    )


"""
   select_fps(const Eigen::Ref<const RowMatrixXd> & feature_matrix,
               int n_sparse, int i_first_point,
               const FPSReturnTupleConst & restart) {
      // number of inputs
      int n_inputs = feature_matrix.rows();

      // n. of sparse points. defaults to full sorting of the inputs
      if (n_sparse == 0) {
        n_sparse = n_inputs;
      }

      // TODO(ceriottm) <- use the exception mechanism
      // for librascal whatever it is
      if (n_sparse > n_inputs) {
        throw std::runtime_error("Cannot FPS more inputs than those provided");
      }

      /* return arrays */
      // FPS indices
      auto sparse_indices = Eigen::ArrayXi(n_sparse);
      // minmax distances (squared)
      auto sparse_minmax_d2 = Eigen::ArrayXd(n_sparse);

      // square moduli of inputs
      auto feature_x2 = Eigen::ArrayXd(n_inputs);
      // list of square distances to latest FPS point
      auto list_new_d2 = Eigen::ArrayXd(n_inputs);
      // list of minimum square distances to each input
      auto list_min_d2 = Eigen::ArrayXd(n_inputs);
      int i_new{};
      double d2max_new{};

      // computes the squared modulus of input points
      feature_x2 = feature_matrix.cwiseAbs2().rowwise().sum();

      // extracts (possibly empty) restart arrays
      auto i_restart = std::get<0>(restart);
      auto d_restart = std::get<1>(restart);
      auto ld_restart = std::get<2>(restart);
      ssize_t i_start_index = 1;
      if (i_restart.size() > 0) {
        if (i_restart.size() >= n_sparse) {
          throw std::runtime_error("Restart arrays larger than target ");
        }

        sparse_indices.head(i_restart.size()) = i_restart;
        i_start_index = i_restart.size();

        if (d_restart.size() > 0) {
          // restart the FPS calculation from previous run.
          // all information is available
          if (d_restart.size() != i_restart.size()) {
            throw std::runtime_error("Restart indices and distances mismatch");
          }

          // sets the part of the data we know already
          sparse_minmax_d2.head(i_restart.size()) = d_restart;
          list_min_d2 = ld_restart;
        } else {
          // distances are not available, so we recompute them.
          // note that this is as expensive as re-running a full
          // FPS, but it allows us to extend an existing FPS set
          list_new_d2 = feature_x2 + feature_x2(i_restart[0]) -
                        2 * (feature_matrix *
                             feature_matrix.row(i_restart[0]).transpose())
                                .array();
          list_min_d2 = list_new_d2;
          // this is basically the standard algorithm below, only that
          // it is run on the first i_start_index points. see below
          // for comments
          for (ssize_t i = 1; i < i_start_index; ++i) {
            // if the feature matrix has been expanded, the data will
            // not be selected in the same order, so we have to
            // override the selection
            i_new = i_restart[i];
            d2max_new = list_min_d2[i_new];
            /*d2max_new = list_min_d2.maxCoeff(&i_new);
             if (i_new != i_restart[i])
             throw std::runtime_error("Reconstructed distances \
              are inconsistent with restart array");*/
            sparse_indices(i) = i_new;
            sparse_minmax_d2(i - 1) = d2max_new;
            list_new_d2 =
                feature_x2 + feature_x2(i_new) -
                2 * (feature_matrix * feature_matrix.row(i_new).transpose())
                        .array();
            list_min_d2 = list_min_d2.min(list_new_d2);
          }
        }
      } else {
        // standard initialization
        // initializes arrays taking the first point provided in input
        sparse_indices(0) = i_first_point;
        //  distance square to the selected point
        list_new_d2 =
            feature_x2 + feature_x2(i_first_point) -
            2 * (feature_matrix * feature_matrix.row(i_first_point).transpose())
                    .array();
        list_min_d2 = list_new_d2;  // we only have this point....
      }

      for (ssize_t i = i_start_index; i < n_sparse; ++i) {
        // picks max dist and its index
        d2max_new = list_min_d2.maxCoeff(&i_new);
        sparse_indices(i) = i_new;
        sparse_minmax_d2(i - 1) = d2max_new;

        // compute distances^2 to the new point
        // TODO(ceriottm): encapsulate the distance calculation
        // into an interface function
        // dispatching to the proper distance/kernel needed
        list_new_d2 =
            feature_x2 + feature_x2(i_new) -
            2 * (feature_matrix * feature_matrix.row(i_new).transpose())
                    .array();

        // this actually returns a list with the element-wise minimum between
        // list_min_d2(i) and list_new_d2(i)
        list_min_d2 = list_min_d2.min(list_new_d2);
      }
      sparse_minmax_d2(n_sparse - 1) = 0;

      return std::make_tuple(sparse_indices, sparse_minmax_d2, list_min_d2);
    }


"""

class SimpleFPS:
    """
    Base Class defined for FPS selection methods

    :param idxs: predetermined indices; if None provided, first index selected
                 is random
    :type idxs: list of int, None

    :param progress_bar: option to use `tqdm <https://tqdm.github.io/>`_
                         progress bar to monitor selections
    :type progress_bar: boolean

    """

    def __init__(self, tol=1e-12, idxs=None, progress_bar=False):

        if not hasattr(self, "tol"):
            self.tol = tol

        if idxs is not None:
            self.idx = idxs
        else:
            self.idx = [np.random.randint(self.product.shape[0])]
        self.distances = np.min([self.calc_distance(i) for i in self.idx], axis=0)
        self.distance_selected = np.nan * np.zeros(self.product.shape[0])

        if progress_bar:
            self.report_progress = get_progress_bar()
        else:
            self.report_progress = lambda x: x

    def select(self, n):
        """Method for FPS select based upon a product of the input matrices

        Parameters
        ----------
        n : number of selections to make, must be > 0

        Returns
        -------
        idx: list of n selections
        """

        if n <= 0:
            raise ValueError("You must call select(n) with n > 0.")

        if len(self.idx) > n:
            return self.idx[:n]

        # Loop over the remaining points...
        for i in self.report_progress(range(len(self.idx) - 1, n - 1)):
            for j in np.where(self.distances > 0)[0]:
                self.distances[j] = min(
                    self.distances[j], self.calc_distance(self.idx[i], [j])
                )

            self.idx.append(np.argmax(self.distances))
            self.distance_selected[i + 1] = self.distances[i + 1]

            if np.abs(self.distances).max() < self.tol:
                return self.idx
        return self.idx

    @abstractmethod
    def calc_distance(self, idx_1, idx_2=None):
        """
            Abstract method to be used for calculating the distances
            between two indexed points. Should be overwritten if default
            functionality is not desired

        : param idx_1 : index of first point to use
        : type idx_1 : int

        : param idx_2 : index of first point to use; if None, calculates the
                        distance between idx_1 and all points
        : type idx_2 : list of int or None
        """
        return _calc_distances_(self.product, idx_1, idx_2)


class SampleFPS(_BaseFPS):
    """

    For sample selection, traditional FPS employs a row-wise Euclidean
    distance, which can be expressed using the Gram matrix
    :math:`\\mathbf{K} = \\mathbf{X} \\mathbf{X}^T`

    .. math::
        \\operatorname{d}_r(i, j) = K_{ii} - 2 K_{ij} + K_{jj}.

    When mixing < 1, this will use PCov-FPS, where a modified Gram matrix is
    used to express the distances

    .. math::
        \\mathbf{\\tilde{K}} = \\alpha \\mathbf{XX}^T +
        (1 - \\alpha)\\mathbf{\\hat{Y}\\hat{Y}}^T

    :param idxs: predetermined indices; if None provided, first index selected
                 is random
    :type idxs: list of int, None

    :param X: Data matrix :math:`\\mathbf{X}` from which to select a
                   subset of the `n` rows
    :type X: array of shape (n x m)

    :param mixing: mixing parameter, as described in PCovR as
                  :math:`{\\alpha}`, defaults to 1
    :type mixing: float

    :param progress_bar: option to use `tqdm <https://tqdm.github.io/>`_
                         progress bar to monitor selections
    :type progress_bar: boolean

    :param tol: threshold below which values will be considered 0,
                      defaults to 1E-12
    :type tol: float

    :param Y: array to include in biased selection when mixing < 1;
              required when mixing < 1, throws AssertionError otherwise
    :type Y: array of shape (n x p), optional when :math:`{\\alpha = 1}`

    """

    def __init__(self, X, mixing=1.0, tol=1e-12, Y=None, **kwargs):

        self.mixing = mixing
        self.tol = tol

        if mixing < 1:
            try:
                self.A, self.Y = check_X_y(X, Y, copy=True, multi_output=True)
            except AssertionError:
                raise Exception(r"For $\alpha < 1$, $Y$ must be supplied.")
        else:
            self.A, self.Y = check_array(X, copy=True), None

        self.product = pcovr_kernel(self.mixing, self.A, self.Y)
        super().__init__(tol=tol, **kwargs)


class FeatureFPS(_BaseFPS):
    """

    For feature selection, traditional FPS employs a column-wise Euclidean
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

    :param idxs: predetermined indices; if None provided, first index selected
                 is random
    :type idxs: list of int, None

    :param X: Data matrix :math:`\\mathbf{X}` from which to select a
                   subset of the `n` columns
    :type X: array of shape (n x m)

    :param mixing: mixing parameter, as described in PCovR as
                   :math:`{\\alpha}`, defaults to 1
    :type mixing: float

    :param progress_bar: option to use `tqdm <https://tqdm.github.io/>`_
                         progress bar to monitor selections
    :type progress_bar: boolean

    :param tol: threshold below which values will be considered 0,
                      defaults to 1E-12
    :type tol: float

    :param Y: array to include in biased selection when mixing < 1;
              required when mixing < 1, throws AssertionError otherwise
    :type Y: array of shape (n x p), optional when :math:`{\\alpha = 1}`


    """

    def __init__(self, X, mixing=1.0, tol=1e-12, Y=None, **kwargs):

        self.mixing = mixing
        self.tol = tol

        if mixing < 1:
            try:
                self.A, self.Y = check_X_y(X, Y, copy=True, multi_output=True)
            except AssertionError:
                raise Exception(r"For $\alpha < 1$, $Y$ must be supplied.")
        else:
            self.A, self.Y = check_array(X, copy=True), None

        self.product = pcovr_covariance(self.mixing, self.A, self.Y, rcond=self.tol)
        super().__init__(tol=tol, **kwargs)
