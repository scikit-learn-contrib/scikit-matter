#!/usr/bin/env python
# coding: utf-8

"""
PCovC with the Breast Cancer Dataset
====================================
"""
# %%
#

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

from skmatter.decomposition import PCovC


plt.rcParams["image.cmap"] = "tab10"
plt.rcParams["scatter.edgecolors"] = "k"

random_state = 0

# %%
#
# For this, we will use the :func:`sklearn.datasets.load_breast_cancer` dataset from
# ``sklearn``.

X, y = load_breast_cancer(return_X_y=True)
print(load_breast_cancer().DESCR)

# %%
#
# Scale Feature Data
# ------------------
#
# Below, we transform the Breast Cancer feature data to have a mean of zero
# and standard deviation of one, while preserving relative relationships
# between feature values.

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %%
#
# PCA
# ---
#
# We use Principal Component Analysis to reduce the Breast Cancer feature
# data to two features that retain as much information as possible
# about the original dataset.

pca = PCA(n_components=2)

pca.fit(X_scaled, y)
T_pca = pca.transform(X_scaled)

fig, axis = plt.subplots()
scatter = axis.scatter(T_pca[:, 0], T_pca[:, 1], c=y)
axis.set(xlabel="PC$_1$", ylabel="PC$_2$")
axis.legend(
    scatter.legend_elements()[0][::-1],
    load_breast_cancer().target_names[::-1],
    loc="upper right",
    title="Classes",
)

# %%
#
# LDA
# ---
#
# Here, we use Linear Discriminant Analysis to find a projection
# of the feature data that maximizes separability between
# the benign/malignant classes.

lda = LinearDiscriminantAnalysis(n_components=1)
lda.fit(X_scaled, y)

T_lda = lda.transform(X_scaled)

fig, axis = plt.subplots()
axis.scatter(-T_lda[:], np.zeros(len(T_lda[:])), c=y)

# %%
#
# PCA, PCovC, and LDA
# -------------------
#
# Below, we see a side-by-side comparison of PCA, PCovC (Logistic
# Regression classifier, :math:`\alpha` = 0.5), and LDA maps of the data.

mixing = 0.5
n_models = 3
fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

models = {
    PCA(n_components=2): "PCA",
    PCovC(
        mixing=mixing,
        n_components=2,
        random_state=random_state,
        classifier=LogisticRegressionCV(),
    ): "PCovC",
    LinearDiscriminantAnalysis(n_components=1): "LDA",
}

for id in range(0, n_models):
    model = list(models)[id]

    model.fit(X_scaled, y)
    T = model.transform(X_scaled)

    if isinstance(model, LinearDiscriminantAnalysis):
        axes[id].scatter(-T_lda[:], np.zeros(len(T_lda[:])), c=y)
    else:
        axes[id].scatter(T[:, 0], T[:, 1], c=y)

    axes[id].set_title(models[model])
