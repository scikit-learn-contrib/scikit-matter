 
import numpy as np
from sklearn.base import check_is_fitted
from sklearn.calibration import LinearSVC
from sklearn.discriminant_analysis import StandardScaler
from sklearn.exceptions import NotFittedError
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.datasets import load_breast_cancer as get_dataset
from sklearn.datasets import load_iris as get_dataset2
from sklearn.datasets import load_diabetes as get_dataset3
from sklearn.metrics import accuracy_score
from _kernel_pcovr import KernelPCovR

from skmatter.decomposition import KernelPCovC, KernelPCovR, PCovR, PCovC
X, Y = get_dataset(return_X_y=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

classifier = LogisticRegression()
classifier.fit(X_scaled, Y)

Yhat = classifier.predict(X_scaled)
W = classifier.coef_.reshape(X_scaled.shape[1], -1)
pcovc1 = PCovC(mixing=0.5, classifier="precomputed", n_components=1)
pcovc1.fit(X_scaled, Yhat, W)
t1 = pcovc1.transform(X_scaled)
print(pcovc1.score(X_scaled, Y))

pcovc2 = PCovC(mixing=0.5, classifier=classifier, n_components=1)
pcovc2.fit(X_scaled, Y)
t2 = pcovc2.transform(X_scaled)
print(pcovc2.score(X_scaled, Y))

print(np.linalg.norm(t1-t2))

# classifier = LinearRegression()
# classifier.fit(X_scaled, Y)

# Yhat = classifier.predict(X_scaled)
# W = classifier.coef_.reshape(X_scaled.shape[1], -1)
# pcovc1 = PCovR(mixing=0.5, regressor="precomputed", n_components=1)
# pcovc1.fit(X_scaled, Yhat, W)
# t1 = pcovc1.transform(X_scaled)
# print(pcovc1.score(X_scaled, Y))

# pcovc2 = PCovR(mixing=0.5, regressor=classifier, n_components=1)
# pcovc2.fit(X_scaled, Y)
# t2 = pcovc2.transform(X_scaled)
# print(pcovc2.score(X_scaled, Y))

