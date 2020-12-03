Principal Covariates Regression
===============================

.. _PCovR-api:

PCovR
#####

Principal Covariates Regression, as described in `[S. de Jong and
H. A. L. Kiers, 1992] <https://doi.org/10.1016/0169-7439(92)80100-I>`_
determines a latent-space projection :math:`\mathbf{T}` which
minimizes a combined loss in supervised and unsupervised tasks.

This projection is determined by the eigendecomposition of a modified gram
matrix :math:`\mathbf{\tilde{K}}`

.. math::

  \mathbf{\tilde{K}} = \alpha \mathbf{X} \mathbf{X}^T +
        (1 - \alpha) \mathbf{\hat{Y}}\mathbf{\hat{Y}}^T

where :math:`\alpha` is a mixing parameter and
:math:`\mathbf{X}` and :math:`\mathbf{\hat{Y}}` are matrices of shapes
:math:`(n_{samples}, n_{features})` and :math:`(n_{samples}, n_{properties})`,
respectively, which contain the input and approximate targets. For
:math:`(n_{samples} < n_{features})`, this can be more efficiently computed
using the eigendecomposition of a modified covariance matrix
:math:`\mathbf{\tilde{C}}`

.. math::

  \mathbf{\tilde{C}} = \alpha \mathbf{X}^T \mathbf{X} +
        (1 - \alpha) \left(\left(\mathbf{X}^T
        \mathbf{X}\right)^{-\frac{1}{2}} \mathbf{X}^T
        \mathbf{\hat{Y}}\mathbf{\hat{Y}}^T \mathbf{X} \left(\mathbf{X}^T
        \mathbf{X}\right)^{-\frac{1}{2}}\right)

.. currentmodule:: skcosmo.pcovr.pcovr

.. autoclass:: PCovR
    :show-inheritance:
    :special-members:

    .. automethod:: fit

        .. automethod:: _fit_feature_space
        .. automethod:: _fit_structure_space

    .. automethod:: transform
    .. automethod:: predict
    .. automethod:: inverse_transform

.. _PCovR_dist-api:

.. currentmodule:: skcosmo.pcovr.pcovr_distances

Modified Gram Matrix :math:`\mathbf{\tilde{K}}`
###############################################

.. autofunction:: pcovr_kernel


Modified Covariance Matrix :math:`\mathbf{\tilde{C}}`
#####################################################

.. autofunction:: pcovr_covariance
