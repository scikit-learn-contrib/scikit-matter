#!/usr/bin/env python
# coding: utf-8
"""
The Benefits of Kernel PCovR for the WHO Dataset
================================================
"""
# %%
#

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.decomposition import PCA, KernelPCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from skmatter.datasets import load_who_dataset
from skmatter.decomposition import KernelPCovR, PCovR
from skmatter.preprocessing import StandardFlexibleScaler

# %%
#
# Load the Dataset
# ----------------


df = load_who_dataset()["data"]
print(df)

# %%
#

columns = [
    "SP.POP.TOTL",
    "SH.TBS.INCD",
    "SH.IMM.MEAS",
    "SE.XPD.TOTL.GD.ZS",
    "SH.DYN.AIDS.ZS",
    "SH.IMM.IDPT",
    "SH.XPD.CHEX.GD.ZS",
    "SN.ITK.DEFC.ZS",
    "NY.GDP.PCAP.CD",
]

X_raw = np.array(df[columns])

# %%
#
# We are taking the logarithm of the population  and GDP to avoid extreme distributions

log_scaled = ["SP.POP.TOTL", "NY.GDP.PCAP.CD"]
for ls in log_scaled:
    print(X_raw[:, columns.index(ls)].min(), X_raw[:, columns.index(ls)].max())
    if ls in columns:
        X_raw[:, columns.index(ls)] = np.log10(X_raw[:, columns.index(ls)])
y_raw = np.array(df["SP.DYN.LE00.IN"])
y_raw = y_raw.reshape(-1, 1)
X_raw.shape

# %%
#
# Scale and Center the Features and Targets
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

x_scaler = StandardFlexibleScaler(column_wise=True)
X = x_scaler.fit_transform(X_raw)

y_scaler = StandardFlexibleScaler(column_wise=True)
y = y_scaler.fit_transform(y_raw)

n_components = 2

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=True, random_state=0
)

# %%
#
# Train the Different Linear DR Techniques
# ----------------------------------------
#
# Best Error for Linear Regression

RidgeCV(cv=5, alphas=np.logspace(-8, 2, 20), fit_intercept=False).fit(
    X_train, y_train
).score(X_test, y_test)

# %%
#
# PCovR
# ^^^^^

pcovr = PCovR(
    n_components=n_components,
    regressor=Ridge(alpha=1e-4, fit_intercept=False),
    mixing=0.5,
    random_state=0,
).fit(X_train, y_train)
T_train_pcovr = pcovr.transform(X_train)
T_test_pcovr = pcovr.transform(X_test)
T_pcovr = pcovr.transform(X)

r_pcovr = Ridge(alpha=1e-4, fit_intercept=False, random_state=0).fit(
    T_train_pcovr, y_train
)
yp_pcovr = r_pcovr.predict(T_test_pcovr)

plt.scatter(y_scaler.inverse_transform(y_test), y_scaler.inverse_transform(yp_pcovr))
r_pcovr.score(T_test_pcovr, y_test)

# %%
#
# PCA
# ^^^

pca = PCA(
    n_components=n_components,
    random_state=0,
).fit(X_train, y_train)
T_train_pca = pca.transform(X_train)
T_test_pca = pca.transform(X_test)
T_pca = pca.transform(X)

r_pca = Ridge(alpha=1e-4, fit_intercept=False, random_state=0).fit(T_train_pca, y_train)
yp_pca = r_pca.predict(T_test_pca)

plt.scatter(y_scaler.inverse_transform(y_test), y_scaler.inverse_transform(yp_pca))
r_pca.score(T_test_pca, y_test)


# %%
#


for c, x in zip(columns, X.T):
    print(c, pearsonr(x, T_pca[:, 0])[0], pearsonr(x, T_pca[:, 1])[0])

# %%
#
# Train the Different Kernel DR Techniques
# ----------------------------------------
#
# Select Kernel Hyperparameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

param_grid = {"gamma": np.logspace(-8, 3, 20), "alpha": np.logspace(-8, 3, 20)}
clf = KernelRidge(kernel="rbf")

gs = GridSearchCV(estimator=clf, param_grid=param_grid)
gs.fit(X_train, y_train)
gs.best_estimator_


# %%
#
# Best Error for Kernel Regression

gs.best_score_

# %%
#

kernel_params = {"kernel": "rbf", "gamma": gs.best_estimator_.gamma}


# %%
#
# KPCovR
# ^^^^^^


kpcovr = KernelPCovR(
    n_components=n_components,
    regressor=KernelRidge(alpha=gs.best_estimator_.alpha, **kernel_params),
    mixing=0.5,
    **kernel_params,
).fit(X_train, y_train)

T_train_kpcovr = kpcovr.transform(X_train)
T_test_kpcovr = kpcovr.transform(X_test)
T_kpcovr = kpcovr.transform(X)

r_kpcovr = KernelRidge(**kernel_params).fit(T_train_kpcovr, y_train)
yp_kpcovr = r_kpcovr.predict(T_test_kpcovr)

plt.scatter(y_scaler.inverse_transform(y_test), y_scaler.inverse_transform(yp_kpcovr))
r_kpcovr.score(T_test_kpcovr, y_test)

# %%
#
# KPCA
# ^^^^

kpca = KernelPCA(n_components=n_components, **kernel_params, random_state=0).fit(
    X_train, y_train
)

T_train_kpca = kpca.transform(X_train)
T_test_kpca = kpca.transform(X_test)
T_kpca = kpca.transform(X)

r_kpca = KernelRidge(**kernel_params).fit(T_train_kpca, y_train)
yp_kpca = r_kpca.predict(T_test_kpca)

plt.scatter(y_scaler.inverse_transform(y_test), y_scaler.inverse_transform(yp_kpca))
r_kpca.score(T_test_kpca, y_test)

# %%
#
# Correlation of the different variables with the KPCovR axes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

for c, x in zip(columns, X.T):
    print(c, pearsonr(x, T_kpcovr[:, 0])[0], pearsonr(x, T_kpcovr[:, 1])[0])

# %%
#
# Plot Our Results
# ----------------

fig, axes = plt.subplot_mosaic(
    """
                                AFF.B
                                A.GGB
                                .....
                                CHH.D
                                C.IID
                                .....
                                EEEEE
                                """,
    figsize=(7.5, 7.5),
    gridspec_kw=dict(
        height_ratios=(0.5, 0.5, 0.1, 0.5, 0.5, 0.1, 0.1),
        width_ratios=(1, 0.1, 0.2, 0.1, 1),
    ),
)
axPCA, axPCovR, axKPCA, axKPCovR = axes["A"], axes["B"], axes["C"], axes["D"]
axPCAy, axPCovRy, axKPCAy, axKPCovRy = axes["F"], axes["G"], axes["H"], axes["I"]


def add_subplot(ax, axy, T, yp, let=""):
    """Adding a subplot to a given axis."""
    p = ax.scatter(-T[:, 0], T[:, 1], c=y_raw, s=4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.annotate(
        xy=(0.025, 0.95), xycoords="axes fraction", text=f"({let})", va="top", ha="left"
    )
    axy.scatter(
        y_scaler.inverse_transform(y_test),
        y_scaler.inverse_transform(yp),
        c="k",
        s=1,
    )
    axy.plot([y_raw.min(), y_raw.max()], [y_raw.min(), y_raw.max()], "r--")
    axy.annotate(
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        text=r"R$^2$=%0.2f" % round(r2_score(y_test, yp), 3),
        va="top",
        ha="left",
        fontsize=8,
    )
    axy.set_xticks([])
    axy.set_yticks([])
    return p


p = add_subplot(axPCA, axPCAy, T_pca, yp_pca, "a")
axPCA.set_xlabel("PC$_1$")
axPCA.set_ylabel("PC$_2$")

add_subplot(axPCovR, axPCovRy, T_pcovr @ np.diag([-1, 1]), yp_pcovr, "b")
axPCovR.yaxis.set_label_position("right")
axPCovR.set_xlabel("PCov$_1$")
axPCovR.set_ylabel("PCov$_2$", rotation=-90, va="bottom")

add_subplot(axKPCA, axKPCAy, T_kpca @ np.diag([-1, 1]), yp_kpca, "c")
axKPCA.set_xlabel("Kernel PC$_1$", fontsize=10)
axKPCA.set_ylabel("Kernel PC$_2$", fontsize=10)

add_subplot(axKPCovR, axKPCovRy, T_kpcovr, yp_kpcovr, "d")
axKPCovR.yaxis.set_label_position("right")
axKPCovR.set_xlabel("Kernel PCov$_1$", fontsize=10)
axKPCovR.set_ylabel("Kernel PCov$_2$", rotation=-90, va="bottom", fontsize=10)

plt.colorbar(
    p, cax=axes["E"], label="Life Expectancy [years]", orientation="horizontal"
)
fig.subplots_adjust(wspace=0, hspace=0.4)
fig.suptitle(
    "Linear and Kernel PCovR for Predicting Life Expectancy", y=0.925, fontsize=10
)
plt.show()
