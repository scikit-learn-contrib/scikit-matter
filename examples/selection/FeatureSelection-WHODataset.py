#!/usr/bin/env python
# coding: utf-8

"""
Feature Selection on the WHO Dataset
====================================
"""
# %%
#

import numpy as np
import scipy
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from skmatter.datasets import load_who_dataset
from skmatter.feature_selection import CUR, FPS, PCovCUR, PCovFPS
from skmatter.preprocessing import StandardFlexibleScaler


# %%
#
# Load the Dataset
# ----------------

df = load_who_dataset()["data"]
print(df)

# %%
#

columns = np.array(
    [
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
)

column_names = np.array(
    [
        "Population",
        "Tuberculosis",
        "Immunization, measles",
        "Educ. Expenditure",
        "HIV",
        "Immunization, DPT",
        "Health Expenditure",
        "Undernourishment",
        "GDP per capita",
    ]
)

columns = columns[[8, 4, 2, 6, 1, 7, 0, 5, 3]].tolist()
column_names = column_names[[8, 4, 2, 6, 1, 7, 0, 5, 3]].tolist()

# %%
#

X_raw = np.array(df[columns])

# %%
#
# We are taking the logarithm of the population and GDP to avoid extreme distributions

log_scaled = ["SP.POP.TOTL", "NY.GDP.PCAP.CD"]
for ls in log_scaled:
    print(X_raw[:, columns.index(ls)].min(), X_raw[:, columns.index(ls)].max())
    if ls in columns:
        X_raw[:, columns.index(ls)] = np.log10(X_raw[:, columns.index(ls)])
y_raw = np.array(df["SP.DYN.LE00.IN"])  # [np.where(df['Year']==2000)[0]])
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
# Provide an estimated target for the feature selector
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


kernel_params = {"kernel": "rbf", "gamma": 0.08858667904100832}
lr = LinearRegression(fit_intercept=False)


yp_train = lr.fit(X_train, y_train).predict(X_train)

# %%
#
# Compute the Selections for Each Selector Type
# ---------------------------------------------

n_select = X.shape[1]

# %%
# PCov-CUR
# ^^^^^^^^


pcur = PCovCUR(n_to_select=n_select, progress_bar=True, mixing=0.0)
pcur.fit(X_train, yp_train)

# %%
#
# PCov-FPS
# ^^^^^^^^

pfps = PCovFPS(
    n_to_select=n_select,
    progress_bar=True,
    mixing=0.0,
    initialize=pcur.selected_idx_[0],
)
pfps.fit(X_train, yp_train)

# %%
#
# CUR
# ^^^

cur = CUR(n_to_select=n_select, progress_bar=True)
cur.fit(X_train, y_train)


# %%
#
# FPS
# ^^^

fps = FPS(n_to_select=n_select, progress_bar=True, initialize=cur.selected_idx_[0])
fps.fit(X_train, y_train)

# %%
#
# (For Comparison) Recursive Feature Addition
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


class RecursiveFeatureAddition:
    """Class for recursive feature addition"""

    def __init__(self, n_to_select):
        """Init."""
        self.n_to_select = n_to_select
        self.selected_idx_ = np.zeros(n_to_select, dtype=int)

    def fit(self, X, y):
        """Perform the fit."""
        remaining = np.arange(X.shape[1])
        for n in range(self.n_to_select):
            errors = np.zeros(len(remaining))
            for i, pp in enumerate(remaining):
                lr.fit(X[:, [*self.selected_idx_[:n], pp]], y)
                errors[i] = lr.score(X[:, [*self.selected_idx_[:n], pp]], y)
            self.selected_idx_[n] = remaining[np.argmax(errors)]
            remaining = np.array(np.delete(remaining, np.argmax(errors)), dtype=int)
        return self


rfa = RecursiveFeatureAddition(n_select).fit(X_train, y_train)

# %%
#
# Plot our Results
# ----------------


fig, axes = plt.subplots(
    2,
    1,
    figsize=(3.75, 5),
    gridspec_kw=dict(height_ratios=(1, 1.5)),
    sharex=True,
    dpi=150,
)
ns = np.arange(1, n_select, dtype=int)

all_errors = {}
for selector, color, linestyle, label in zip(
    [cur, fps, pcur, pfps, rfa],
    ["red", "lightcoral", "blue", "dodgerblue", "black"],
    ["solid", "solid", "solid", "solid", "dashed"],
    [
        "CUR",
        "FPS",
        "PCov-CUR\n" + r"($\alpha=0.0$)",
        "PCov-FPS\n" + r"($\alpha=0.0$)",
        "Recursive\nFeature\nSelection",
    ],
):
    if label not in all_errors:
        errors = np.zeros(len(ns))
        for i, n in enumerate(ns):
            lr.fit(X_train[:, selector.selected_idx_[:n]], y_train)
            errors[i] = lr.score(X_test[:, selector.selected_idx_[:n]], y_test)
        all_errors[label] = errors
    axes[0].plot(ns, all_errors[label], c=color, label=label, linestyle=linestyle)
    axes[1].plot(
        ns, selector.selected_idx_[: max(ns)], c=color, marker=".", linestyle=linestyle
    )

axes[1].set_xlabel(r"$n_{select}$")
axes[1].set_xticks(range(1, n_select))
axes[0].set_ylabel(r"R$^2$")
axes[1].set_yticks(np.arange(X.shape[1]))
axes[1].set_yticklabels(column_names, rotation=30, fontsize=10)
axes[0].legend(ncol=2, fontsize=8, bbox_to_anchor=(0.5, 1.0), loc="lower center")
axes[1].invert_yaxis()
axes[1].grid(axis="y", alpha=0.5)
plt.tight_layout()
plt.show()


# %%
#
# Plot correlation between selectors
# ----------------------------------


selected_idx = np.array(
    [selector.selected_idx_ for selector in [cur, fps, pcur, pfps, rfa]]
).T

similarity = np.zeros((len(selected_idx.T), len(selected_idx.T)))
for i in range(len(selected_idx.T)):
    for j in range(len(selected_idx.T)):
        similarity[i, j] = scipy.stats.weightedtau(
            selected_idx[:, i], selected_idx[:, j], rank=False
        )[0]

labels = ["CUR", "FPS", "PCovCUR", "PCovFPS,", "RFA"]

plt.imshow(similarity, cmap="Greens")
plt.xticks(np.arange(len(labels)), labels=labels)
plt.yticks(np.arange(len(labels)), labels=labels)

plt.title("Feature selection similarity")
for i in range(len(labels)):
    for j in range(len(labels)):
        value = np.round(similarity[i, j], 2)
        color = "white" if value > 0.5 else "black"
        text = plt.gca().text(j, i, value, ha="center", va="center", color=color)

plt.colorbar()
plt.show()
