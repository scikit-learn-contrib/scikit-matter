#!/usr/bin/env python
# coding: utf-8
"""
Pointwise Local Reconstruction Error
====================================

Example for the usage of the :class:`skmatter.metrics.pointwise_local_reconstruction_error`
as pointwise local reconstruction error (LFRE) on the degenerate CH4 manifold. We apply
the local reconstruction measure on the degenerate CH4 manifold dataset. This dataset
was specifically constructed to be representable by a 4-body features (bispectrum) but
not by a 3-body features (power spectrum). In other words the dataset contains
environments which are different, but have the same 3-body features. For more details
about the dataset please refer to
`Pozdnyakov 2020 <https://doi.org/10.1103/PhysRevLett.125.166001>`_.

The skmatter dataset already contains the 3 and 4-body features computed with
`librascal <https://github.com/lab-cosmo/librascal>`_ so we can load it and compare
it with the LFRE.
"""
# %%
#


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from skmatter.datasets import load_degenerate_CH4_manifold
from skmatter.metrics import pointwise_local_reconstruction_error

mpl.rc("font", size=20)


# load features
degenerate_manifold = load_degenerate_CH4_manifold()
power_spectrum_features = degenerate_manifold.data.SOAP_power_spectrum
bispectrum_features = degenerate_manifold.data.SOAP_bispectrum

print(degenerate_manifold.DESCR)


# %%
#


n_local_points = 20

print("Computing pointwise LFRE...")

# %%

# local reconstruction error of power spectrum features using bispectrum features
power_spectrum_to_bispectrum_pointwise_lfre = pointwise_local_reconstruction_error(
    power_spectrum_features,
    bispectrum_features,
    n_local_points,
    train_idx=np.arange(0, len(power_spectrum_features), 2),
    test_idx=np.arange(0, len(power_spectrum_features)),
    estimator=None,
    n_jobs=4,
)

# local reconstruction error of bispectrum features using power spectrum features
bispectrum_to_power_spectrum_pointwise_lfre = pointwise_local_reconstruction_error(
    bispectrum_features,
    power_spectrum_features,
    n_local_points,
    train_idx=np.arange(0, len(power_spectrum_features), 2),
    test_idx=np.arange(0, len(power_spectrum_features)),
    estimator=None,
    n_jobs=4,
)

print("Computing pointwise LFRE finished.")

print(
    "LFRE(3-body, 4-body) = ",
    np.linalg.norm(power_spectrum_to_bispectrum_pointwise_lfre)
    / np.sqrt(len(power_spectrum_to_bispectrum_pointwise_lfre)),
)

print(
    "LFRE(4-body, 3-body) = ",
    np.linalg.norm(bispectrum_to_power_spectrum_pointwise_lfre)
    / np.sqrt(len(power_spectrum_to_bispectrum_pointwise_lfre)),
)


# %%
#


fig, (ax34, ax43) = plt.subplots(
    1, 2, constrained_layout=True, figsize=(16, 7.5), sharey="row", sharex=True
)

vmax = 0.5

X, Y = np.meshgrid(np.linspace(0.7, 0.9, 9), np.linspace(-0.1, 0.1, 9))
pcm = ax34.contourf(
    X,
    Y,
    power_spectrum_to_bispectrum_pointwise_lfre[81:].reshape(9, 9).T,
    vmin=0,
    vmax=vmax,
)

ax43.contourf(
    X,
    Y,
    bispectrum_to_power_spectrum_pointwise_lfre[81:].reshape(9, 9).T,
    vmin=0,
    vmax=vmax,
)

ax34.axhline(y=0, color="red", linewidth=5)
ax43.axhline(y=0, color="red", linewidth=5)
ax34.set_ylabel(r"v/$\pi$")
ax34.set_xlabel(r"u/$\pi$")
ax43.set_xlabel(r"u/$\pi$")

ax34.set_title(r"$X^-$ LFRE(3-body, 4-body)")
ax43.set_title(r"$X^-$ LFRE(4-body, 3-body)")

cbar = fig.colorbar(pcm, ax=[ax34, ax43], label="LFRE", location="bottom")

plt.show()

# %%
#
# The environments span a manifold which is described by the coordinates :math:`v/\pi` and
# :math:`u/\pi` (please refer to `Pozdnyakov 2020 <https://doi.org/10.1103/PhysRevLett.125.166001>`_
# for a concrete understanding of the manifold). The LFRE is presented for each environment
# in the manifold in the two contour plots. It can be seen that the reconstruction error
# of 4-body features using 3-body features (the left plot) is most significant along the
# degenerate line (the horizontal red line). This agrees with the fact that the 3-body
# features remain the same on the degenerate line and can therefore not reconstruct the
# 4-body features. On the other hand the 4-body features can perfectly reconstruct the
# 3-body features as seen in the right plot.
