import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from sklearn.datasets import make_low_rank_matrix

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
    number_of_samples = np.shape(X)[1]
    initialize = np.random.randint(0, number_of_samples)
    b = benchmark(initialize=initialize, **benchmark_args)
    b.fit(X)

    return b._get_benchmarks()

def building_figure(n_samples, n_features, n_features_to_select, scaling = True):
        print('=======================================')
        print(f"n_samples = {n_samples}", f"n_features = {n_features}")
        X = make_low_rank_matrix(n_samples = n_samples, n_features=n_features, effective_rank = 10)
        if scaling:
            X *= 1 / (1.0 + np.arange(X.shape[1]) ** 2)
        simple_times = []
        voronoi_times = []
        st = []
        vt = []
        sd = []
        vd = []
        for nfts in n_features_to_select:
            sb_time = -time.time()
            simple_times_i, simple_n_calcs = run(SimpleBenchmark, X, n_features_to_select=nfts, tolerance = 0)
            sb_time += time.time()
            vr_time = -time.time()
            voronoi_times_i, voronoi_n_calcs = run(VoronoiBenchmark, X, n_features_to_select=nfts, tolerance = 0)
            vr_time += time.time()
            st.append(sb_time)
            vt.append(vr_time)
            sd.append(np.sum(np.sum(voronoi_n_calcs, axis=0)[:-1]))
            vd.append(np.sum(simple_n_calcs[:-1]))

        simple_times.append(simple_times_i)
        voronoi_times.append(voronoi_times_i)

        voronoi_times = np.array(voronoi_times)
        simple_times = np.array(simple_times)
        voronoi_mean_time = np.mean(voronoi_times, axis=0)
        simple_mean_time = np.mean(simple_times, axis=0)
        voronoi_time_std = np.std(voronoi_times, axis=0)
        simple_time_std = np.std(simple_times, axis=0)
        plt.figure(figsize=(10, 8))
        mpl.rcParams["font.size"] = 20
        plt.yscale("log")
        plt.xscale("log")
        plt.title("Time taken per iteration")
        index = [i for i in range(n_features_to_select[-1])]
        plt.errorbar(
            index,
            simple_mean_time[index],
            simple_time_std[index],
            capsize=5,
            color="r",
            ecolor="k",
            errorevery=3,
            label="Simple FPS",
        )
        plt.errorbar(
            index,
            voronoi_mean_time[index],
            voronoi_time_std[index],
            capsize=5,
            color="b",
            ecolor="g",
            errorevery=3,
            label=" Voronoi FPS",
        )
        plt.xlabel("$n_{iteration}$")
        plt.ylabel("time ($s$)")
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 8))
        mpl.rcParams["font.size"] = 20
        plt.title("Total number of distances calculated by each iteration")
        plt.loglog(
            [np.sum(simple_n_calcs[:i]) for i in range(n_features_to_select[-1])],
            color="r",
            label="Simple FPS",
        )
        plt.loglog(
            [np.sum(np.sum(voronoi_n_calcs, axis=0)[:i]) for i in range(n_features_to_select[-1])],
            color="b",
            label="Voronoi FPS",
        )
        plt.xlabel("$n_{iteration}$")
        plt.ylabel("Total number of computed distances")
        plt.legend()
        plt.show()
        return [n_features_to_select/n_features, st, vt, sd, vd]

if __name__ == "__main__":

    samples = [10**i for i in range(0, 4)]
    n_features_to_select = np.array([1, 10, 100, 500])
    n_features = 1000
    stats_1 = {}
    for sample in samples:
        stats_1[sample] =  building_figure(n_features = n_features, n_samples = sample, n_features_to_select=n_features_to_select)

    samples = [10**i for i in range(3, 6)]
    n_features_to_select = np.array([10, 100, 500, 999])
    n_features = 1000
    stats_2 = {}
    for sample in samples:
        stats_2[sample] =  building_figure(n_features = n_features, n_samples = sample, n_features_to_select=n_features_to_select)

    mpl.rcParams["font.size"] = 15
    fig, axs = plt.subplots(2, 2,figsize=(20,16))
    axs[0, 0].plot(stats_1[1][0], stats_1[1][1], label = 'SimpleFPS')
    axs[0, 0].plot(stats_1[1][0], stats_1[1][2], label = 'VoronoiFPS')
    axs[0, 1].plot(stats_1[10][0], stats_1[10][1], label = 'SimpleFPS')
    axs[0, 1].plot(stats_1[10][0], stats_1[10][2], label = 'VoronoiFPS')
    axs[1, 0].plot(stats_1[100][0], stats_1[100][1], label = 'SimpleFPS')
    axs[1, 0].plot(stats_1[100][0], stats_1[100][2], label = 'VoronoiFPS')
    axs[1, 1].plot(stats_1[1000][0], stats_1[1000][1], label = 'SimpleFPS')
    axs[1, 1].plot(stats_1[1000][0], stats_1[1000][2], label = 'VoronoiFPS')
    axs[0, 0].set_title('1000 n_samples = n_features')
    axs[0, 0].legend()
    axs[0, 1].set_title('100 n_samples =  n_features')
    axs[0, 1].legend()
    axs[1, 0].set_title('10 n_samples = n_features')
    axs[1, 0].legend()
    axs[1, 1].set_title('n_samples = n_features')
    axs[1, 1].legend()
    for ax in axs.flat:
        ax.set(xlabel='$n_{features\_to\_select}/n_{features}$', ylabel='time')
    plt.show()

    mpl.rcParams["font.size"] = 15
    fig, axs = plt.subplots(2, 2,figsize=(20,16))
    axs[0, 0].plot(stats_2[1000][0], stats_2[1000][1], label = 'SimpleFPS')
    axs[0, 0].plot(stats_2[1000][0], stats_2[1000][2], label = 'VoronoiFPS')
    axs[0, 1].plot(stats_2[10000][0], stats_2[10000][1], label = 'SimpleFPS')
    axs[0, 1].plot(stats_2[10000][0], stats_2[10000][2], label = 'VoronoiFPS')
    axs[1, 0].plot(stats_2[100000][0], stats_2[100000][1], label = 'SimpleFPS')
    axs[1, 0].plot(stats_2[100000][0], stats_2[100000][2], label = 'VoronoiFPS')
    axs[0, 0].set_title('n_samples = n_features')
    axs[0, 0].legend()
    axs[0, 1].set_title('n_samples = 10 n_features')
    axs[0, 1].legend()
    axs[1, 0].set_title('n_samples = 100 n_features')
    axs[1, 0].legend()
    for ax in axs.flat:
        ax.set(xlabel='$n_{features\_to\_select}/n_{features}$', ylabel='time')
    plt.show()

    fig, axs = plt.subplots(2, 2,figsize=(20,16))
    axs[0, 0].plot(stats_1[1][0], stats_1[1][3], label = 'SimpleFPS')
    axs[0, 0].plot(stats_1[1][0], stats_1[1][4], label = 'VoronoiFPS')
    axs[0, 1].plot(stats_1[10][0], stats_1[10][3], label = 'SimpleFPS')
    axs[0, 1].plot(stats_1[10][0], stats_1[10][4], label = 'VoronoiFPS')
    axs[1, 0].plot(stats_1[100][0], stats_1[100][3], label = 'SimpleFPS')
    axs[1, 0].plot(stats_1[100][0], stats_1[100][4], label = 'VoronoiFPS')
    axs[1, 1].plot(stats_1[1000][0], stats_1[1000][3], label = 'SimpleFPS')
    axs[1, 1].plot(stats_1[1000][0], stats_1[1000][4], label = 'VoronoiFPS')
    axs[0, 0].set_title('1000 n_samples = n_features')
    axs[0, 0].legend()
    axs[0, 1].set_title('100 n_samples =  n_features')
    axs[0, 1].legend()
    axs[1, 0].set_title('10 n_samples = n_features')
    axs[1, 0].legend()
    axs[1, 1].set_title('n_samples = n_features')
    axs[1, 1].legend()
    for ax in axs.flat:
        ax.set(xlabel='$n_{features\_to\_select}/n_{features}$', ylabel='$n_{calc\_dist}$')
    plt.show()
    fig, axs = plt.subplots(2, 2,figsize=(20,16))
    axs[0, 0].plot(stats_2[1000][0], stats_2[1000][3], label = 'SimpleFPS')
    axs[0, 0].plot(stats_2[1000][0], stats_2[1000][4], label = 'VoronoiFPS')
    axs[0, 1].plot(stats_2[10000][0], stats_2[10000][3], label = 'SimpleFPS')
    axs[0, 1].plot(stats_2[10000][0], stats_2[10000][4], label = 'VoronoiFPS')
    axs[1, 0].plot(stats_2[100000][0], stats_2[100000][3], label = 'SimpleFPS')
    axs[1, 0].plot(stats_2[100000][0], stats_2[100000][4], label = 'VoronoiFPS')
    axs[0, 0].set_title('n_samples = n_features')
    axs[0, 0].legend()
    axs[0, 1].set_title('n_samples = 10 n_features')
    axs[0, 1].legend()
    axs[1, 0].set_title('n_samples = 100 n_features')
    axs[1, 0].legend()
    for ax in axs.flat:
        ax.set(xlabel='$n_{features\_to\_select}/n_{features}$', ylabel='$n_{calc\_dist}$')
    plt.show()
