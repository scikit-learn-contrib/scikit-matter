import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt

from skcosmo.feature_selection.voronoi_fps import VoronoiFPS
from skcosmo.feature_selection.simple_fps import FPS

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

    def _calculate_distances(self, X, last_selected, **kwargs):
        super()._calculate_distances(X, last_selected, **kwargs)
        self.n_dist_calc_each_[0][self.n_selected_ - 1] = self.number_calculated_dist
        return self.haussdorf_

    def _update_post_selection(self, X, y, last_selected):
        self.times_[self.n_selected_] = time.time() - self.start_
        self.start_ = time.time()
        super()._update_post_selection(X, y, last_selected)

    def _get_benchmarks(self):
        return self.times_, self.n_dist_calc_each_


class SimpleBenchmark(FPS):
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

    def _update_post_selection(self, X, y, last_selected):
        self.n_dist_calc_[self.n_selected_] = X.shape[-1]
        self.times_[self.n_selected_] = time.time() - self.start_
        self.start_ = time.time()
        super()._update_post_selection(X, y, last_selected)

    def _get_benchmarks(self):
        return self.times_, self.n_dist_calc_


def run(benchmark, X, **benchmark_args):
    number_of_samples  = np.shape(X)[1]
    initialize = np.random.randint(0, number_of_samples)
    b = benchmark(n_features_to_select=X.shape[-1] - 1, initialize=initialize, **benchmark_args)
    b.fit(X)

    return (b._get_benchmarks())

if __name__ == "__main__":

    X = np.load("./skcosmo/datasets/data/csd-1000r-large.npz")["X"]
    
    X = np.random.normal(size=(10000,100))
    X *= np.arange(X.shape[1])
    
    
    simple_times = []
    voronoi_times = []
    for i in range(2):
        simple_times_i, _ = run(SimpleBenchmark, X)
        voronoi_times_i, _ = run(VoronoiBenchmark, X)
        simple_times.append(simple_times_i)
        voronoi_times.append(voronoi_times_i)

    voronoi_times = np.array(voronoi_times)
    simple_times = np.array(simple_times)
    voronoi_mean_time = np.mean(voronoi_times, axis = 0)
    simple_mean_time = np.mean(simple_times, axis = 0)
    voronoi_time_std = np.std(voronoi_times, axis = 0)
    simple_time_std = np.std(simple_times, axis = 0)

    index = []
    for i in range(20):
        index.append(np.power(10, i/10))
    index = np.around(index, decimals = 1)
    index = index.astype(int)
    plt.figure(figsize=(10,8))
    mpl.rcParams['font.size'] = 20
    plt.yscale('log')
    plt.xscale('log')
    plt.title("Time taken per iteration")
    plt.errorbar(index, simple_mean_time[index], simple_time_std[index], capsize=5, color = 'r', ecolor='k', errorevery = 3,  label="Simple FPS")
    plt.errorbar(index, voronoi_mean_time[index], voronoi_time_std[index], capsize=5, color = 'b', ecolor='g', errorevery = 3, label=" Voronoi FPS")
    plt.xlabel("$n_{iteration}$")
    plt.ylabel("time ($s$)")
    plt.legend()
    plt.show()

    _, simple_n_calcs = run(SimpleBenchmark, X)
    _, voronoi_n_calcs = run(VoronoiBenchmark, X)
    plt.figure(figsize=(10,8))
    mpl.rcParams['font.size'] = 20
    plt.title("Total number of distances calculated by each iteration")
    plt.loglog([np.sum(simple_n_calcs[:i]) for i in range(X.shape[-1] - 1)], color = 'r', label="Simple FPS")
    plt.loglog(
        [np.sum(np.sum(voronoi_n_calcs, axis=0)[:i]) for i in range(X.shape[-1] - 1)], color = 'b',
        label="Voronoi FPS",
    )
    plt.xlabel("$n_{iteration}$")
    plt.ylabel("Total number of computed distances")
    plt.legend()
