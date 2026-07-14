#!/usr/bin/env python
# coding: utf-8
"""
Sketch-Map: nonlinear dimensionality reduction
==============================================

Sketch-Map [Ceriotti2011]_ is a nonlinear dimensionality reduction method
designed for **diverse, high-dimensional data**. Unlike PCA, which preserves
the directions of largest variance, Sketch-Map preserves *intermediate-range*
pairwise distances by mapping them through sigmoid functions: very short
distances are compressed (treated as "the same"), very long distances are
saturated (treated as "far apart"), and the optimisation focuses on the range
of distances that actually distinguishes points.

This example introduces the standard landmark workflow on a synthetic dataset:

1. select representative landmarks via Farthest Point Sampling,
2. weight each landmark by the size of its Voronoi cell,
3. fit Sketch-Map on the weighted landmarks,
4. project the rest of the dataset with :meth:`SketchMap.transform`.

A final, optional section reproduces the analysis of the Massive Atomic
Diversity (MAD) dataset [Mazitov2025a]_ and validates it against the reference
C++ implementation from `sketchmap.org <https://sketchmap.org>`_; it runs only
when the (large) MAD landmark file is present locally.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA

from skmatter.decomposition import SketchMap
from skmatter.sample_selection import FPS, voronoi_weights

# %%
# Quick start
# -----------
#
# In the simplest case ``SketchMap()`` works with no arguments: the sigmoid
# parameters are estimated from the distance statistics of the data, and the
# default optimisation settings (including annealed pointwise global
# optimisation for 2D maps) are tuned to give a publication-quality map out
# of the box. The parameter values actually used are stored in ``params_``.

X_blobs, blob_labels = make_blobs(
    n_samples=400,
    n_features=5,
    centers=5,
    cluster_std=[0.5, 0.8, 0.3, 1.2, 0.6],
    random_state=0,
)

sm_quick = SketchMap().fit(X_blobs)
print("Auto-estimated sigmoid parameters:")
for key, value in sm_quick.params_.items():
    print(f"  {key} = {value:.3f}")

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(
    sm_quick.embedding_[:, 0],
    sm_quick.embedding_[:, 1],
    c=blob_labels,
    cmap="tab10",
    s=20,
    edgecolor="k",
    lw=0.3,
)
ax.set_aspect("equal")
ax.set_xlabel("Sketch-Map 1")
ax.set_ylabel("Sketch-Map 2")
ax.set_title("Quick start on synthetic blobs")
plt.tight_layout()
plt.show()


# %%
# Understanding the sigmoid parameters
# ------------------------------------
#
# Every pair of points enters the optimisation only through the sigmoid
#
# .. math::
#
#     s(r) = 1 - \left(1 + (2^{a/b}-1)\,(r/\sigma)^a\right)^{-b/a},
#
# so the three numbers :math:`\sigma, a, b` fully determine *which* distances
# the map preserves:
#
# - :math:`\sigma` is the switching distance (:math:`s(\sigma) = 0.5`):
#   distances below it count as "close", above it as "far". Place it just
#   below the peak of the pairwise distance histogram, or at the size of the
#   structures you want to resolve.
# - :math:`a` controls how sharply distances *below* :math:`\sigma` are
#   compressed toward 0 (larger = more aggressive squashing of short-range
#   noise).
# - :math:`b` controls how quickly distances *above* :math:`\sigma` saturate
#   to 1 (smaller = gentler saturation).
#
# Separate sigmoids are used for the high-dimensional distances
# (``a_high``, ``b_high``) and the low-dimensional ones (``a_low``,
# ``b_low``); the low-D exponents are usually smaller to compensate for the
# volume difference between the spaces. The figure of the high- and low-D
# sigmoids overlaid on the distance distribution in the original publication
# [Ceriotti2011]_ and the `sketchmap.org analysis tutorial
# <https://sketchmap.org/index.html?page=tuts&psub=analysis>`_ illustrates the
# effect: only distances near :math:`\sigma`, where the sigmoid is steep,
# meaningfully drive the embedding.


# %%
# The landmark workflow
# ---------------------
#
# Sketch-Map jointly optimises all pairwise relations, so
# :meth:`SketchMap.fit` scales as :math:`O(N^2)` and is meant for up to a few
# thousand points. For larger datasets the standard recipe is to **fit** on a
# small set of representative *landmarks* and **transform** everything else
# onto the fixed map afterwards.
#
# Here we draw 2000 points from a 5D blob mixture, pick 100 landmarks with
# Farthest Point Sampling, weight each by the number of points in its Voronoi
# cell (:func:`~skmatter.sample_selection.voronoi_weights`, a density proxy),
# and fit Sketch-Map on the weighted landmarks.

X_full, full_labels = make_blobs(
    n_samples=2000,
    n_features=5,
    centers=5,
    cluster_std=[0.5, 0.8, 0.3, 1.2, 0.6],
    random_state=42,
)
fps = FPS(n_to_select=100, random_state=42).fit(X_full)
X_landmarks = X_full[fps.selected_idx_]
landmark_weights = voronoi_weights(X_full, X_landmarks)

sm = SketchMap().fit(X_landmarks, sample_weight=landmark_weights)

# %%
# :meth:`~SketchMap.transform` then places every point of the full dataset
# onto the fixed map. Internally this is an *out-of-sample projection*: each
# new point's 2D position is found by minimising its single-point stress
# against the fixed landmarks, at a cost independent of how the map was fit.
# Points that coincide with a landmark simply get the landmark's coordinates.

embedding_full = sm.transform(X_full)

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(
    embedding_full[:, 0],
    embedding_full[:, 1],
    c=full_labels,
    cmap="tab10",
    s=8,
    alpha=0.5,
)
ax.scatter(
    sm.embedding_[:, 0],
    sm.embedding_[:, 1],
    c="black",
    s=30,
    marker="x",
    linewidths=1.2,
    label="Landmarks (fit)",
)
ax.set_aspect("equal")
ax.set_xlabel("Sketch-Map 1")
ax.set_ylabel("Sketch-Map 2")
ax.set_title("FPS landmarks + voronoi_weights + SketchMap.transform")
ax.legend()
plt.tight_layout()
plt.show()


# %%
# Sketch-Map vs PCA
# -----------------
#
# Linear PCA captures the directions of largest variance, which on a
# heterogeneous dataset are dominated by the most spread-out clusters. This
# compresses the rest of the data into a small region. Sketch-Map's sigmoid
# deliberately ignores very large distances, so the 2D map devotes its real
# estate to resolving the bulk of the data instead.

pca_emb = PCA(n_components=2).fit_transform(X_full)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, emb, title in [
    (axes[0], pca_emb, "PCA"),
    (axes[1], embedding_full, "Sketch-Map"),
]:
    ax.scatter(emb[:, 0], emb[:, 1], c=full_labels, cmap="tab10", s=8, alpha=0.5)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
plt.tight_layout()
plt.show()


# %%
# Validating against the reference C++ implementation
# ---------------------------------------------------
#
# The section below reproduces the MAD analysis [Mazitov2025a]_ and compares
# the embedding against the C++ ``dimred`` reference. It needs the 1000-point
# MAD landmark file (too large to ship with the examples), so it runs only
# when ``highd-landmarks`` is present in the working directory.
#
# Because the stress is non-convex and the layout is only defined up to
# rotations, reflections and local minima, the right comparison is not
# point-by-point coordinates but the *stress* itself: the value of the
# objective both implementations minimise, evaluated on each embedding with
# the same weights.

mad_landmarks_file = "highd-landmarks"

if os.path.exists(mad_landmarks_file):
    data = np.loadtxt(mad_landmarks_file)
    mad_X = data[:, :-1]
    mad_weights = data[:, -1]

    sm_mad = SketchMap(sigma=7.0, a_high=4, b_high=2, a_low=2, b_low=2)
    mad_embedding = sm_mad.fit_transform(mad_X, sample_weight=mad_weights)
    reference = np.loadtxt("low_landmarks.dat", comments="#")[:, :2]

    def sketchmap_sigmoid(r, sigma, a, b):
        """Sketch-Map sigmoid, s(sigma) = 0.5."""
        amplitude = 2 ** (a / b) - 1
        return 1 - (1 + amplitude * (r / sigma) ** a) ** (-b / a)

    # pairwise quantities in scipy's condensed (upper-triangle) ordering
    fhd_pairs = sketchmap_sigmoid(pdist(mad_X - mad_X.mean(0)), 7.0, 4.0, 2.0)
    triu = np.triu_indices(len(mad_X), k=1)
    pair_weights = np.outer(mad_weights, mad_weights)[triu]

    def sketchmap_stress(emb):
        """Weighted sigmoid stress of an embedding of the MAD landmarks."""
        fld = sketchmap_sigmoid(pdist(emb), 7.0, 2.0, 2.0)
        return np.sum(pair_weights * (fhd_pairs - fld) ** 2) / np.sum(pair_weights)

    stress_py = sketchmap_stress(mad_embedding)
    stress_cpp = sketchmap_stress(reference)
    print(f"stress of the Python map:        {stress_py:.6f}")
    print(f"stress of the C++ reference map: {stress_cpp:.6f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, emb, title in [
        (axes[0], mad_embedding, f"Python (stress = {stress_py:.4f})"),
        (axes[1], reference, f"C++ reference (stress = {stress_cpp:.4f})"),
    ]:
        sc = ax.scatter(
            emb[:, 0], emb[:, 1], c=mad_weights, cmap="viridis", s=20, lw=0.3
        )
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("Sketch-Map 1")
        ax.set_ylabel("Sketch-Map 2")
    fig.colorbar(sc, ax=axes, label="Voronoi weight", fraction=0.04, pad=0.02)
    plt.show()

    # distance-preservation diagnostic: transformed high-D vs low-D distance
    # for every pair. A perfect map would lie on the diagonal; the sigmoid
    # makes the saturated corners (0,0) and (1,1) cheap, so the diagonal is
    # followed most closely at the intermediate values that carry structure.
    fld_pairs = sketchmap_sigmoid(pdist(mad_embedding), 7.0, 2.0, 2.0)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    hb = ax.hexbin(fhd_pairs, fld_pairs, gridsize=60, bins="log", cmap="Blues")
    ax.plot([0, 1], [0, 1], "r--", lw=1)
    ax.set_xlabel("high-D sigmoid distance $s_{hd}(D_{ij})$")
    ax.set_ylabel("low-D sigmoid distance $s_{ld}(d_{ij})$")
    ax.set_title("Pairwise distance preservation")
    fig.colorbar(hb, ax=ax, label="pairs (log scale)")
    plt.tight_layout()
    plt.show()
else:
    print(
        f"{mad_landmarks_file!r} not found; skipping the MAD / C++ validation "
        "section. See the example header for how to obtain the data."
    )


# %%
# References
# ----------
#
# Citations [Ceriotti2011]_, [Mazitov2025a]_, [Mazitov2025b]_ are listed in
# the main :ref:`bibliography <bibliography>`.
