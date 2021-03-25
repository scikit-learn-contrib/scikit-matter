import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from sklearn.datasets import make_low_rank_matrix

from skcosmo.feature_selection.voronoi_fps import VoronoiFPS
from skcosmo.feature_selection.simple_fps import FPS

class VoronoiBenchmark(VoronoiFPS):
    def _init_greedy_search(self, X, y, n_to_select):
        self.n_dist_calc_each_ = np.zeros((2, n_to_select))
        self.n_dist_calc_each_[1] = np.arange(n_to_select)
        super()._init_greedy_search(X, y, n_to_select)

    def _continue_greedy_search(self, X, y, n_to_select):
        n_pad = n_to_select - self.n_selected_
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
        super()._update_post_selection(X, y, last_selected)

    def _get_benchmarks(self):
        return self.n_dist_calc_each_


class SimpleBenchmark(FPS):
    def _init_greedy_search(self, X, y, n_to_select):
        self.n_dist_calc_ = np.zeros(n_to_select)
        super()._init_greedy_search(X, y, n_to_select)

    def _continue_greedy_search(self, X, y, n_to_select):
        n_pad = n_to_select - self.n_selected_
        self.n_dist_calc_ = np.pad(
            self.n_dist_calc_, (0, n_pad), "constant", constant_values=0
        )
        super()._continue_greedy_search(X, y, n_to_select)

    def _update_post_selection(self, X, y, last_selected):
        self.n_dist_calc_[self.n_selected_] = X.shape[-1]
        super()._update_post_selection(X, y, last_selected)

    def _get_benchmarks(self):
        return self.n_dist_calc_


def run(benchmark, X, **benchmark_args):
    number_of_samples = np.shape(X)[1]
    initialize = np.random.randint(0, number_of_samples)
    b = benchmark(initialize=initialize, **benchmark_args)
    b.fit(X)

    return b._get_benchmarks()

def data_generation(n_samples, n_features):
        X = np.random.normal(size=(n_samples, n_features))
        X += np.random.poisson(size=(n_samples, n_features)) * 0.1
        X *= 1 / (1.0 + np.arange(X.shape[1]) ** 2)
        X -= np.random.normal(size=(n_samples, n_features)) * 0.01
        X = X.T
        return X

if __name__ == "__main__":
    print("Enter number of samples")
    n_samples = int(input())
    ntfs = int(np.round(0.08*n_samples))
    n_features = [2**i for i in range(5, -1, -1)]
    n_iter = 20
    aver_par = max(10, int(np.round(0.001*10000)))
    statistics = []
    for feature in n_features:
        print('=======================================')
        print(f"n_samples = {n_samples}", f"n_features = {feature}")
        stats = np.zeros((2,ntfs))
        for i in range(n_iter):
            X = data_generation(n_samples, feature)
            voronoi_n_calcs = run(VoronoiBenchmark, X.copy(), n_features_to_select=ntfs)
            stats += [voronoi_n_calcs[1]/n_samples, voronoi_n_calcs[0]/n_samples]
        averaged = [np.sum((stats[1] - stats[0])[i:i+aver_par]/n_iter)/aver_par for i in range(0, ntfs-10)]
        index = np.arange(ntfs-10)
        statistics.append([index/n_samples, averaged])

    plt.figure(figsize=(10, 8))
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams["font.size"] = 20
    print('=======================================')
    print("FPS,", f"n_samples = {n_samples}")
    X = data_generation(n_samples, 1)
    simple_n_calcs = run(SimpleBenchmark, X, n_features_to_select=ntfs)
    iterations = np.arange(ntfs)/n_samples
    plt.xscale('log')
    plt.plot(iterations, simple_n_calcs/n_samples - iterations, "--k")
    plt.text(x = 0.9, y = 0.8, s = 'FPS', transform = plt.gca().transAxes, ha = 'right')
    for i in range(len(n_features)):
        line = r"$n_\mathrm{features}$ = " + r"{}".format(n_features[i])
        plt.plot(statistics[i][0],statistics[i][1], label = line)
    plt.ylabel("$n_\mathrm{checked}/n_\mathrm{samples}$", fontsize = 25)
    plt.xlabel(r"$m/n_\mathrm{samples}$", fontsize = 25)
    line = "$n_\mathrm{samples}$ = " + "{}".format(n_samples)
    plt.title(line)
    plt.legend( prop={'size': 14}, title = 'Voronoi FPS')
    plt.savefig("fps_benchmark.pdf")
