import numpy as np

from skcosmo.feature_selection.voronoi_fps import VoronoiFPS
from skcosmo.feature_selection.simple_fps import FPS

import time

idx = [
    0,
    113,
    194,
    177,
    30,
    18,
    28,
    51,
    74,
    140,
    92,
    131,
    172,
    160,
    12,
    40,
    116,
    4,
    97,
    55,
    68,
    27,
    106,
    162,
    90,
    70,
    184,
    80,
    25,
    100,
    22,
    47,
    120,
    150,
    2,
    61,
    3,
    82,
    126,
    29,
    165,
    65,
    142,
    57,
    110,
    132,
    145,
    85,
    108,
    38,
    115,
    84,
    53,
    8,
    191,
    91,
    5,
    180,
    111,
    42,
    67,
    36,
    88,
    24,
    176,
    93,
    186,
    46,
    16,
    63,
    170,
    155,
    188,
    34,
    52,
    107,
    69,
    139,
    179,
    169,
    118,
    168,
    17,
    157,
    161,
    6,
    117,
    60,
    133,
    196,
    33,
    125,
    81,
    56,
    71,
    175,
    109,
    41,
    127,
    114,
    149,
    11,
    190,
    99,
    39,
    159,
    105,
    163,
    152,
    102,
    64,
    128,
    1,
    124,
    164,
    185,
    13,
    148,
    44,
    76,
    83,
    173,
    94,
    141,
    112,
    89,
    62,
    7,
    153,
    43,
    134,
    26,
    66,
    14,
    178,
    73,
    146,
    167,
    48,
    144,
    50,
    130,
    31,
    78,
    121,
    21,
    174,
    104,
    183,
    187,
    119,
    49,
    137,
    151,
    10,
    198,
    101,
    45,
    147,
    195,
    95,
    136,
    15,
    19,
    54,
    182,
    75,
    98,
    79,
    72,
    181,
    123,
    9,
    197,
]


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
        active_points = super()._get_active(X, last_selected)

        if len(active_points) / X.shape[1] > (1.0 / 6.0):
            self.n_dist_calc_each_[0][self.n_selected_ - 1] = X.shape[-1]
        else:
            self.n_dist_calc_each_[0][self.n_selected_ - 1] = len(active_points)
        return active_points

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

    b = benchmark(**benchmark_args)
    b.fit(X.T)

    return (*b._get_benchmarks(), b.selected_idx_)


if __name__ == "__main__":

    #X = np.load("./skcosmo/datasets/data/csd-1000r-large.npz")["X"]
    X = np.random.normal(size=(100000,1000))
    X*=1/(1.0+np.arange(X.shape[1])**2)
    
    times, calcs, idx = run(SimpleBenchmark, X, n_features_to_select=100)
    vtimes, vcalcs, vidx = run(VoronoiBenchmark, X, n_features_to_select= 100)

    n = min(len(idx), len(vidx))

    assert np.allclose(vidx, idx)

    from matplotlib import pyplot as plt

    plt.figure()
    plt.title("Times per Iteration")
    plt.semilogy(times, 'b.', label="Simple FPS")
    plt.semilogy(vtimes, 'r.', label="Voronoi FPS")
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
    plt.loglog(calcs / X.shape[-1], 'b.', label="Simple FPS")
    plt.loglog(np.sum(vcalcs, axis=0) / X.shape[-1], 'r.', label="Voronoi FPS")
    plt.xlabel("iteration")
    plt.ylabel("percentage of distances computed")
    plt.legend()

    plt.figure()
    plt.title("Computations per Step in Voronoi FPS")
    plt.plot(vcalcs[0], 'r.', label="points inside `active` polyhedra")
    plt.plot(vcalcs[1], 'b-', label="centers of `active` polyhedra")
    plt.xlabel("iteration")
    plt.ylabel("number of distances computed")
    plt.legend(title="Calculating distance\nbetween previous\nselected and: ")
    
    plt.figure()
    plt.title("Computations vs time in Voronoi FPS")
    plt.plot((vcalcs[0]+vcalcs[1]), vtimes,  'r.')
    plt.xlabel("number of distances computed")
    plt.ylabel("time")
    plt.legend(title="Calculating distance\nbetween previous\nselected and: ")
    plt.show()
