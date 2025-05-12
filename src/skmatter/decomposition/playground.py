 
import numpy as np
from sklearn.base import check_is_fitted
from sklearn.calibration import LinearSVC
from sklearn.discriminant_analysis import StandardScaler
from sklearn.exceptions import NotFittedError
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeClassifier
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

ke = KernelPCovC()
ke.fit(X_scaled, Y)
print(ke.score(X_scaled, Y))