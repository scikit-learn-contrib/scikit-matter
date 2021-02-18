# This example shows how to use an FPS, then CUR seleuutor
# to subselect features before training a RidgeCV

import numpy as np
from matplotlib import pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from skcosmo.feature_selection import SimpleFPS, SimpleCUR

scaler = StandardScaler()
fps = SimpleFPS(n_features_to_select=10)
cur = SimpleCUR(n_features_to_select=4)
ridge = RidgeCV(cv=2, alphas=np.logspace(-8, 2, 10))

X, y = load_boston(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

pipe = Pipeline(
    [("scaler", scaler), ("selector1", fps), ("selector2", cur), ("ridge", ridge)]
)
pipe.fit(X_train.copy(), y_train.copy())

plt.scatter(y_test, pipe.predict(X_test))
plt.gca().set_aspect("equal")
plt.plot(plt.xlim(), plt.xlim(), "r--")
plt.show()
