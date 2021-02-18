import numpy as np

from skcosmo.feature_selection.voronoi_fps import VoronoiFPS
from skcosmo.feature_selection.simple_fps import SimpleFPS

import time


class VoronoiBenchmark(VoronoiFPS):
    def _init_greedy_search(self, X, y, n_to_select):
        self.start_ = time.time()
        self.times_ = np.zeros(n_to_select)
        self.n_dist_calc_each_ = np.zeros((2, n_to_select))
        self.n_dist_calc_each_[1] = np.arange(n_to_select)
        super()._init_greedy_search(X, y, n_to_select)

    def _continue_greedy_search(self, X, y, n_to_select):
        n_pad = n_to_select - self.n_selected_
        self.start_ = time.time()
        self.times_ = np.pad(self.times_, (0, n_pad), "constant", constant_values=0)
        self.n_dist_calc_each_ = np.pad(
            self.n_dist_calc_each_, (0, n_pad), "constant", constant_values=0
        )
        self.n_dist_calc_each_[1] = np.arange(n_to_select)
        super()._continue_greedy_search(X, y, n_to_select)

    def _get_active(self, X, last_selected):
        f_active = super()._get_active(X, last_selected)

        if (
            np.sum(self.number_in_voronoi[f_active]) / X.shape[1]
            > self.voronoi_cutoff_fraction
        ):
            self.n_dist_calc_each_[0][self.n_selected_] = X.shape[-1]
        else:
            self.n_dist_calc_each_[0][self.n_selected_] = np.sum(
                self.number_in_voronoi[f_active]
            )
        return f_active

    def _update_post_selection(self, X, y, last_selected):
        self.times_[self.n_selected_] = time.time() - self.start_
        self.start_ = time.time()
        super()._update_post_selection(X, y, last_selected)

    def _get_benchmarks(self):
        return self.times_, self.n_dist_calc_each_


class SimpleBenchmark(SimpleFPS):
    def _init_greedy_search(self, X, y, n_to_select):
        self.start_ = time.time()
        self.times_ = np.zeros(n_to_select)
        self.n_dist_calc_ = np.zeros(n_to_select)
        super()._init_greedy_search(X, y, n_to_select)

    def _continue_greedy_search(self, X, y, n_to_select):
        n_pad = n_to_select - self.n_selected_
        self.start_ = time.time()
        self.times_ = np.pad(self.times_, (0, n_pad), "constant", constant_values=0)
        self.n_dist_calc_ = np.pad(
            self.n_dist_calc_, (0, n_pad), "constant", constant_values=0
        )
        super()._continue_greedy_search(X, y, n_to_select)

    def _get_dist(self, X, last_selected):
        self.n_dist_calc_[self.n_selected_] = X.shape[-1]
        return super()._get_dist(X, last_selected)

    def _update_post_selection(self, X, y, last_selected):
        self.times_[self.n_selected_] = time.time() - self.start_
        self.start_ = time.time()
        super()._update_post_selection(X, y, last_selected)

    def _get_benchmarks(self):
        return self.times_, self.n_dist_calc_


def run(benchmark, X, **benchmark_args):

    b = benchmark(n_features_to_select=X.shape[-1] - 1, **benchmark_args)
    b.fit(X)

    return b._get_benchmarks()


if __name__ == "__main__":

    X = np.load("./skcosmo/datasets/data/csd-1000r-large.npz")["X"]

    times, calcs = run(SimpleBenchmark, X)
    vtimes, vcalcs = run(VoronoiBenchmark, X, voronoi_cutoff_fraction=0.9)

    from matplotlib import pyplot as plt

    plt.figure()
    plt.title("Times per Iteration")
    plt.loglog(times, label="Simple FPS")
    plt.loglog(vtimes, label="Voronoi FPS")
    plt.xlabel("iteration")
    plt.ylabel("time")
    plt.legend()

    plt.figure()
    plt.title("Total Number of Distances Calculated by each Iteration")
    plt.loglog([np.sum(calcs[:i]) for i in range(X.shape[-1] - 1)], label="Simple FPS")
    plt.loglog(
        [np.sum(np.sum(vcalcs, axis=0)[:i]) for i in range(X.shape[-1] - 1)],
        label="Voronoi FPS",
    )
    plt.xlabel("iteration")
    plt.ylabel("Total number of distances computed")
    plt.legend()

    plt.figure()
    plt.title("Percentage of Distances Calculated at Each Iteration")
    plt.loglog(calcs / X.shape[-1], label="Simple FPS")
    plt.loglog(np.sum(vcalcs, axis=0) / X.shape[-1], label="Voronoi FPS")
    plt.xlabel("iteration")
    plt.ylabel("percentage of distances computed")
    plt.legend()

    plt.figure()
    plt.title("Computations per Step in Voronoi FPS")
    plt.plot(vcalcs[0], label="points inside `active` polyhedra")
    plt.plot(vcalcs[1], label="centers of `active` polyhedra")
    plt.xlabel("iteration")
    plt.ylabel("number of distances computed")
    plt.legend(title="Calculating distance\nbetween previous\nselected and: ")
    plt.show()
