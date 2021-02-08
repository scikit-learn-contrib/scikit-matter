Principal Covariates Regression (PCovR)
=======================================

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

For all PCovR methods, it is strongly suggested that :math:`\mathbf{X}` and
:math:`\mathbf{Y}` are centered and scaled to unit variance, otherwise the
results will change drastically near :math:`alpha \to 0` and :math:`alpha \to 0`.
This can be done with the companion preprocessing classes, where

>>> from skcosmo.preprocessing import StandardFlexibleScaler as SFS
>>>
>>> # Set column_wise to True when the columns are relative to one another,
>>> # False otherwise.
>>> scaler = SFS(column_wise=True)
>>>
>>> scaler.fit(A) # replace with your matrix
>>> A = scaler.transform(A)

.. currentmodule:: skcosmo.decomposition

.. autoclass:: PCovR
    :show-inheritance:
    :special-members:

    .. automethod:: fit

        .. automethod:: _fit_feature_space
        .. automethod:: _fit_sample_space

    .. automethod:: transform
    .. automethod:: predict
    .. automethod:: inverse_transform
    .. automethod:: score

.. _KPCovR-api:

Kernel PCovR
############

Kernel Principal Covariates Regression, as described in `[Helfrecht, et al., 2020]
<https://iopscience.iop.org/article/10.1088/2632-2153/aba9ef>`_
determines a latent-space projection :math:`\mathbf{T}` which
minimizes a combined loss in supervised and unsupervised tasks in the
reproducing kernel Hilbert space (RKHS).

This projection is determined by the eigendecomposition of a modified gram
matrix :math:`\mathbf{\tilde{K}}`

.. math::

  \mathbf{\tilde{K}} = \alpha \mathbf{K} +
        (1 - \alpha) \mathbf{\hat{Y}}\mathbf{\hat{Y}}^T

where :math:`\alpha` is a mixing parameter,
:math:`\mathbf{K}` is the input kernel of shape :math:`(n_{samples}, n_{samples})`
and :math:`\mathbf{\hat{Y}}` is the target matrix of shape
:math:`(n_{samples}, n_{properties})`.

.. currentmodule:: skcosmo.decomposition

.. autoclass:: KPCovR
    :show-inheritance:
    :special-members:

    .. automethod:: fit
    .. automethod:: transform
    .. automethod:: predict
    .. automethod:: inverse_transform
    .. automethod:: score
