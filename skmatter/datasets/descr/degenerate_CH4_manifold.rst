.. _degenerate_manifold:

Degenerate CH4 manifold
#######################

The dataset contains two representations (SOAP power spectrum and bispectrum) of the two manifolds spanned by the carbon atoms of two times 81 methane structures.
The SOAP power spectrum representation the two manifolds intersect creating a degenerate manifold/line for which the representation remains the same.
In contrast for higher body order representations as the (SOAP) bispectrum the carbon atoms can be uniquely represented and do not create a degenerate manifold.
Following the naming convention of [Pozdnyakov2020]_ for each representation the first 81 samples correspond to the X minus manifold and the second 81 samples contain the X plus manifold

Function Call
-------------

.. function:: skmatter.datasets.load_degenerate_CH4_manifold

Data Set Characteristics
------------------------

    :Number of Instances: Each representation 162

    :Number of Features: Each  representation 12

    The representations were computed with [D1]_ using the hyperparameters:

    :rascal hyperparameters:

    +---------------------------+------------+
    | key                       |   value    |
    +===========================+============+
    | radial_basis:             |    "GTO"   |
    +---------------------------+------------+
    | interaction_cutoff:       |      4     |
    +---------------------------+------------+
    | max_radial:               |      2     |
    +---------------------------+------------+
    | max_angular:              |      2     |
    +---------------------------+------------+
    | gaussian_sigma_constant": |     0.5    |
    +---------------------------+------------+
    | gaussian_sigma_type:      |  "Constant"|
    +---------------------------+------------+
    | cutoff_smooth_width:      |     0.5    |
    +---------------------------+------------+
    | normalize:                |    False   |
    +---------------------------+------------+

The SOAP bispectrum features were in addition reduced to 12 features with principal component analysis (PCA) [D2]_.

References
----------

   .. [D1] https://github.com/lab-cosmo/librascal commit 8d9ad7a
   .. [D2] https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
