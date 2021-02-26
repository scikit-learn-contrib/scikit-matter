import numpy as np
from ase.io import read

from skcosmo.feature_selection import FPS
from rascal.representations import SphericalInvariants as SOAP


# read all of the frames and book-keep the centers and species
# filename = "/path/to/CSD-1000r.xyz"
filename = "/home/michele/lavoro/projects/KernelPCovR/datasets/CSD-1000R.xyz"
n_features = 200
n_samples = 1000
frames = np.asarray(
    read(filename, ":"),
    dtype=object,
)

n_centers = [len(frame) for frame in frames]
n_env_accum = [sum(n_centers[: i + 1]) for i in range(len(n_centers))]
n_env = sum(n_centers)

numbers = np.concatenate([frame.numbers for frame in frames])
number_loc = np.array([np.where(numbers == i)[0] for i in [1, 6, 7, 8]], dtype=object)


# compute radial soap vectors as first pass
hypers = dict(
    soap_type="PowerSpectrum",
    interaction_cutoff=3.5,
    max_radial=6,
    max_angular=0,
    gaussian_sigma_type="Constant",
    gaussian_sigma_constant=0.4,
    cutoff_smooth_width=0.5,
)
soap = SOAP(**hypers)
X_raw = soap.transform(frames).get_features(soap)


# select 100 diverse samples
i_selected = FPS(n_features_to_select=n_samples).fit(X_raw.T).selected_idx_

# book-keep which frames these samples belong in
frames_select = [np.where(n_env_accum > i)[0][0] for i in i_selected]
reduced_frames_select = list(sorted(set(frames_select)))

properties_select = [
    frame.arrays["CS_local"] for frame in frames[reduced_frames_select]
]

n_centers_select = [len(frame) for frame in frames[reduced_frames_select]]
n_env_accum_select = [
    sum(n_centers_select[: i + 1]) for i in range(len(n_centers_select))
]
n_env_select = sum(n_centers_select)


# compute a larger power spectrum for these frames
hypers["max_angular"] = 6
soap_select = SOAP(**hypers)
X_raw_select = soap_select.transform(frames[reduced_frames_select]).get_features(
    soap_select
)


# pull the soap vectors only pertaining to the selected environments
i_select_reduced = []
properties_select_reduced = np.zeros(len(i_selected), dtype=float)
for i in range(len(i_selected)):
    my_orig_frame = frames_select[i]
    my_frame = reduced_frames_select.index(my_orig_frame)
    if my_orig_frame != 0:
        orig_loc = i_selected[i] - n_env_accum[my_orig_frame - 1]
        new_loc = orig_loc + n_env_accum_select[my_frame - 1]
    else:
        orig_loc = i_selected[i]
        new_loc = i_selected[i]
    i_select_reduced.append(new_loc)
    properties_select_reduced[i] = frames[my_orig_frame].arrays["CS_local"][orig_loc]

X_sample_select = X_raw_select[i_select_reduced]


# select 100 / 2520 soap features
X_select = FPS(n_features_to_select=n_features).fit_transform(X_sample_select)
Y_select = properties_select_reduced.reshape(-1, 1)

data = dict(X=X_select, Y=Y_select)
np.savez("data/csd-1000r.npz", **data)
