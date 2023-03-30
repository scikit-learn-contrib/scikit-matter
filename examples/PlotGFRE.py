#!/usr/bin/env python
# coding: utf-8

"""
Global Feature Reconstruction Error (GFRE) and Distortion (GFRD)
================================================================

Example for the usage of the :class:`skmatter.metrics.global_reconstruction_error`
as global feature reconstruction error (GFRE) and
:class:`skmatter.metrics.global_reconstruction_distortion` global feature reconstruction
distortion (GFRD). We apply the global reconstruction measures on the degenerate
CH4 manifold dataset. This dataset was specifically constructed to be
representable by a 4-body features (bispectrum) but not by a 3-body features
(power spectrum). In other words the dataset contains environments which are
different, but have the same 3-body features. For more details about the dataset
please refer to `Pozdnyakov 2020 <https://doi.org/10.1103/PhysRevLett.125.166001>`_.

The skmatter dataset already contains the 3 and 4-body features computed with
`librascal <https://github.com/lab-cosmo/librascal>`_ so we can load it and
compare it with the GFRE/GFRD.
"""
# %%
#

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from skmatter.datasets import load_degenerate_CH4_manifold
from skmatter.metrics import (
    global_reconstruction_distortion,
    global_reconstruction_error,
)


mpl.rc("font", size=20)

# %%
#
# load features

degenerate_manifold = load_degenerate_CH4_manifold()
power_spectrum_features = degenerate_manifold.data.SOAP_power_spectrum
bispectrum_features = degenerate_manifold.data.SOAP_bispectrum

# %%
#

gfre_matrix = np.zeros((2, 2))
print("Computing GFRE...")

# %%
#
# reconstruction error of power spectrum features using power spectrum features

gfre_matrix[0, 0] = global_reconstruction_error(
    power_spectrum_features, power_spectrum_features
)

# %%
#
# reconstruction error of bispectrum features using power spectrum features

gfre_matrix[0, 1] = global_reconstruction_error(
    power_spectrum_features, bispectrum_features
)

# %%
#
# reconstruction error of power spectrum features using bispectrum features

gfre_matrix[1, 0] = global_reconstruction_error(
    bispectrum_features, power_spectrum_features
)

# %%
#
# reconstruction error of bispectrum features using bispectrum features

gfre_matrix[1, 1] = global_reconstruction_error(
    bispectrum_features, bispectrum_features
)

print("Computing GFRE finished.")


# %%
#


gfrd_matrix = np.zeros((2, 2))
print("Computing GFRD...")

# %%
#
# reconstruction distortion of power spectrum features using power spectrum features

gfrd_matrix[0, 0] = global_reconstruction_distortion(
    power_spectrum_features, power_spectrum_features
)

# %%
#
# reconstruction distortion of power spectrum features using bispectrum features

gfrd_matrix[0, 1] = global_reconstruction_distortion(
    power_spectrum_features, bispectrum_features
)

# %%
#
# reconstruction distortion of bispectrum features using power spectrum features

gfrd_matrix[1, 0] = global_reconstruction_distortion(
    bispectrum_features, power_spectrum_features
)

# %%
#
# reconstruction distortion of bipsectrum features using bispectrum features

gfrd_matrix[1, 1] = global_reconstruction_distortion(
    bispectrum_features, bispectrum_features
)

print("Computing GFRD finished.")


# %%
#


fig, (axGFRE, axGFRD, cbar_ax) = plt.subplots(
    1,
    3,
    figsize=(10, 4),
    gridspec_kw=dict(width_ratios=(1, 1, 0.2)),
)


pcm1 = axGFRE.imshow(gfre_matrix, vmin=0, vmax=0.25)
axGFRE.set_ylabel("F")
axGFRE.set_xlabel("F'")
axGFRE.set_title("GFRE(F, F')")

axGFRE.set_xticks([0, 1])
axGFRE.set_xticklabels(["3-body", "4-body"])
axGFRE.set_yticks([0, 1])
axGFRE.set_yticklabels(["3-body", "4-body"])

pcm2 = axGFRD.imshow(gfrd_matrix, vmin=0, vmax=0.25)
axGFRD.set_xlabel("F'")
axGFRD.set_title("GFRD(F, F')")

axGFRD.set_xticks([0, 1])
axGFRD.set_xticklabels(["3-body", "4-body"])
axGFRD.set_yticks([0, 1])
axGFRD.set_yticklabels(["", ""])

cbar = fig.colorbar(pcm2, cax=cbar_ax, label="GFRE or GFRD")
plt.show()

# %%
#
# It can be seen that the reconstruction error of 4-body features with 3-body
# features in the left plot in the upper right corner is large, expressing that the
# dataset contains 4-body information that cannot be well linearly reconstructed using
# 3-body information. This is expected, since the dataset was specifically designed for
# this purpose (for more information please read
# `Pozdnyakov 2020 <https://doi.org/10.1103/PhysRevLett.125.166001>`_). On the other
# hand the 3-body features can be perfectly reconstructed from the 4-body features
# as seen in the left plot in the lower left corner. However, this reconstruction distorts
# the 4-body features significantly as seen in the right plot in the lower left corner
# which is a typical behaviour of higher order features and can be also observed for
# polynomial kernel features.
