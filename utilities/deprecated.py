from ase.io import read, write

# Librascal
from rascal.representations import SphericalInvariants as SOAP


def load_xyz(N=10, input_file="../datasets/CSD-1000R.xyz", property = "CS_local"):
    """
        Loads the data from the xyz file
    """

    # Read the first N frames of CSD-500
    frames = read(input_file, index=":")
    if(N<len(frames)):
        idx = np.random.choice(range(len(frames)), size=(N,))
        print(f"Randomly choosing {N}/{len(frames)} frames")
    else:
        idx = range(len(frames))
    frames = [frames[i] for i in idx]
    N_env = sum([len(frame) for frame in frames])

    # Extract chemical shifts
    for frame in frames:
        frame.wrap()

    if(property in frames[0].arrays):
        Y = np.concatenate([frame.arrays[property] for frame in frames])
    else:
        Y = np.array([frame.info[property] for frame in frames])

    print("Within the {} frames we have {} environments.".format(min(N, len(frames)), N_env))
    return frames, Y

def compute_soap(frames, n_FPS=200, soap_hypers={}):
    """
        Computes the soap vectors and does FPS, if desired
    """
    soap_default = dict(soap_type="PowerSpectrum",
               interaction_cutoff=3.5,
               max_radial=6,
               max_angular=6,
               gaussian_sigma_type="Constant",
               gaussian_sigma_constant=0.4,
               cutoff_smooth_width=0.5)

    for h in soap_default:
        if h not in soap_hypers:
            soap_hypers[h] = soap_default[h]
    # Compute SOAPs (from librascal tutorial)
    soap = SOAP(**soap_hypers)

    for frame in frames:
        frame.wrap()

    soap_rep = soap.transform(frames)
    X_raw = soap_rep.get_features(soap)

    num_features = X_raw.shape[1]

    if(n_FPS is not None):
        print(f"Each SOAP vector contains {num_features} components.\
               \nWe will use furthest point sampling to generate a subsample containing {n_FPS} components of our SOAP vectors.")

        # FPS the components
        col_idxs, col_dist = FPS(X_raw.T, n_FPS)
        X = X_raw[:, col_idxs]
    else:
        X = X_raw

    return X#, X_split
