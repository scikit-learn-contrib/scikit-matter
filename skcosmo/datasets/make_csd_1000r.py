import numpy as np
from ase.io import read
from rascal.representations import SphericalInvariants as SOAP

from skcosmo.feature_selection import CUR
from skcosmo.preprocessing import StandardFlexibleScaler
from skcosmo.sample_selection import FPS

# read all of the frames and book-keep the centers and species
filename = "/path/to/CSD-1000R.xyz"
frames = np.asarray(
    read(filename, ":"),
    dtype=object,
)

n_centers = np.array([len(frame) for frame in frames])
center_idx = np.array([i for i, f in enumerate(frames) for p in f])
n_env_accum = np.zeros(len(frames) + 1, dtype=int)
n_env_accum[1:] = np.cumsum(n_centers)

numbers = np.concatenate([frame.numbers for frame in frames])

# compute radial soap vectors as first pass
hypers = dict(
    soap_type="PowerSpectrum",
    interaction_cutoff=2.5,
    max_radial=6,
    max_angular=0,
    gaussian_sigma_type="Constant",
    gaussian_sigma_constant=0.4,
    cutoff_smooth_width=0.5,
    normalize=False,
    global_species=[1, 6, 7, 8],
    expansion_by_species_method="user defined",
)
soap = SOAP(**hypers)

X_raw = StandardFlexibleScaler(column_wise=False).fit_transform(
    soap.transform(frames).get_features(soap)
)

# rank the environments in terms of diversity
n_samples = 500
i_selected = FPS(n_to_select=n_samples, initialize=0).fit(X_raw).selected_idx_

# book-keep which frames these samples belong in
f_selected = center_idx[i_selected]
reduced_f_selected = list(sorted(set(f_selected)))
frames_selected = frames[f_selected].copy()
ci_selected = i_selected - n_env_accum[f_selected]

properties_select = [
    frames[fi].arrays["CS_local"][ci] for fi, ci in zip(f_selected, ci_selected)
]

# mask other environments in the frames so that SOAP vectors
# will not be computed for other environments on next pass
for frame, ci, fi in zip(frames_selected, ci_selected, f_selected):
    frame.arrays["center_atoms_mask"] = np.zeros(len(frame), dtype=bool)
    frame.arrays["center_atoms_mask"][ci] = True

# compute a larger power spectrum for these frames
hypers["max_angular"] = 6
soap_select = SOAP(**hypers)
X_sample_select = StandardFlexibleScaler(column_wise=False).fit_transform(
    soap_select.transform(frames_selected).get_features(soap_select)
)
X_sample_select.shape

# select 100 / 2520 soap features
n_select = 100
X_select = CUR(n_to_select=n_select).fit_transform(X_sample_select)
Y_select = np.array(properties_select).reshape(-1, 1)

data = dict(
    X=X_select,
    Y=Y_select,
    original_mapping=[(fi, ci) for fi, ci in zip(f_selected, ci_selected)],
)
np.savez("./skcosmo/datasets/data/csd-1000r.npz", **data, size=(n_samples, n_select))
