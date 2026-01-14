#!/usr/bin/env python
# coding: utf-8

"""
Sketch-map example using pre-selected high-dimensional landmarks
==================================================================

This example demonstrates a minimal, sphinx-gallery friendly usage of the
`SketchMap` estimator. It loads the provided landmark file, fits the
estimator on the landmark set (using per-sample `sample_weights`) and
plots the resulting 2D embedding.

Notes
-----
- The lightweight Python implementation of Sketch-map lives in
  `skmatter.decomposition._sketchmap`.
- The example fits only the landmark set (callers that want a subset
  should pre-select and pass that array to `fit`).
"""

# %%
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

from skmatter.decomposition import SketchMap


# %%
# Load example landmark data. The file included with the examples contains
# high-dimensional descriptors and the last column holds a per-sample weight.
data_file = "highd-landmarks"
data = np.loadtxt(data_file)
X_land = data[:, :-1]
weights = data[:, -1]


# %%
# Fit SketchMap on the landmark set and provide `sample_weights`.
# We fit on the landmarks only here (no FPS or internal selection).
sm = SketchMap(n_components=2, auto_histogram=True, preopt_steps=50)
sm.fit(X_land, sample_weights=weights, mixing_ratio=0.2, n_components=2)
T = sm.embedding_


# %%
# Plot the resulting embedding colored by the per-sample weight.
cmap = cm.get_cmap("viridis")
fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter(T[:, 0], T[:, 1], c=weights, s=40, cmap=cmap, edgecolor="k")
ax.set_title("Sketch-map embedding (example)")
ax.set_xlabel("Sketchmap 1")
ax.set_ylabel("Sketchmap 2")
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("landmark weight")
fig.tight_layout()
plt.show()
