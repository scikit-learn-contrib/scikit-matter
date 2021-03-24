import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from sklearn.datasets import make_low_rank_matrix

from skcosmo.feature_selection.voronoi_fps import VoronoiFPS
from skcosmo.feature_selection.simple_fps import FPS
"""
The number of calculated distances during the VoronoiFPS algorithm can be represented as:
$n_{selected} +{n_updated}$,
where
$n_{selected}$ is the number of vertices already selected, and $n_{updated}$ is the number of points
that can get into the resulting Voronoi polyhedron.
According to this it is possible to define 3 modes of algorithm performance:
1. Initial - small number of selected points (relative to the size of dataset). There are few polyhedrons
in the system so far, a large number of points at each step can change its polyhedron, so the number of
calculations is large. Here the main contribution is made by the second summand
2. Intermediate - the number of polyhedrons is already large enough that a significant part of points does not
change its polyhedron. At the same time, the number of selected points relative to the dataset size is also
small. This is the target mode of the algorithm, at which it gives the best speedup. In this mode, both
summands are much smaller than the size of the dataset, the contribution of the first summand becomes
predominant.
3. Final, the number of selected points is comparable to the size of the dataset. At this stage, the
first summand is already much larger than the second summand, it can be neglected in this limit.
At this stage the Voronoi algorithm is already undesirable, because at each iteration the distance from the
new vertex to all the previously selected ones is calculated. Simple FPS at this stage already calculates
distances only from the selected vertex to the remaining points, the number of which is much smaller.
Consequently, an important task is to determine the intermediate mode conditions for different parameters
$n_{select}$, $N$ and $M$. This can be done using the presented algorithm.
The number of samples in the dataset is input. Then a random matrix of size $NxM$ is generated - where M
takes the value $2^k$, k lies from 0 to 7. Next, all the features are scaled, and a small noise is added to
them. The result of the algorithm is a plot of the dependence $(n_{d}-n_{selected})/N$ on $n_{selected}/N$
where $n_{d}$ is the number of distances calculated at each step. The sooner the plot goes to zero,
the sooner the algorithm begins to work in the intermediate mode. The above graphs are an example of
the "curse of dimensionality$ - as the number of features increases, the time to reach the intermediate mode
also increases significantly.
"""
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
    plt.figure(figsize=(10, 8))
    mpl.rcParams["font.size"] = 20
    print('=======================================')
    print("SimpleFPS,", f"n_samples = {n_samples}")
    X = data_generation(n_samples, 1)
    simple_n_calcs = run(SimpleBenchmark, X, n_features_to_select=ntfs)
    iterations = np.arange(ntfs)/n_samples
    plt.plot(iterations, (simple_n_calcs - iterations)/n_samples, label = "SimpleFPS calculation")
    aver_par = max(10, int(np.round(0.001*10000)))
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
        plt.plot(index/n_samples, averaged, label = f"n_features = {feature}")
    plt.ylabel("$n_{dist}/N$", fontsize = 25)
    plt.xlabel("$n_{selected}/N$", fontsize = 25)
    plt.title(f"n_samples = {n_samples}")
    plt.legend()
    plt.show()
