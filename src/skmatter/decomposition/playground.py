 
import numpy as np
from sklearn.base import check_is_fitted
from sklearn.calibration import LinearSVC
from sklearn.discriminant_analysis import StandardScaler
from sklearn.exceptions import NotFittedError
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from kernel_pcovc_new import KernelPCovC
from _kernel_pcovr import KernelPCovR
from pcovc_new import PCovC
from sklearn.datasets import load_breast_cancer as get_dataset
from sklearn.datasets import load_iris as get_dataset2
from sklearn.datasets import load_diabetes as get_dataset3
from sklearn.metrics import accuracy_score
from _kernel_pcovr import KernelPCovR

X, Y = get_dataset(return_X_y=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ke = KernelPCovC(mixing=0.5,classifier=SVC(), n_components=2)
# ke.fit(X, Y)
# y_pred = ke.predict(X)
# print(ke.decision_function(X))

model = KernelPCovC(mixing=0.5, kernel="rbf", classifier=LogisticRegression(), n_components=2)
model.fit(X_scaled, Y)
print(model.n_features_in_)
T = model.transform(X_scaled)

Z = model.decision_function(X_scaled)
X = model.inverse_transform(T)
print(T.shape)
y_pred = model.predict(X_scaled)
print(model.score(X_scaled, Y)) # we should have KPCovC match PCovC decision function shape 

model2 = PCovC(mixing=0.5, classifier=LogisticRegression(), n_components=2)
model2.fit(X_scaled, Y)
T_2 = model2.transform(X_scaled)
y_pred_2 = model2.predict(X_scaled)
print(model2.score(X_scaled, Y))

# ke = KernelPCovC(mixing=1.0, classifier=SVC(verbose=1), svd_solver="full",n_components=2)
# ke.fit(X, Y)

# for svd_solver in ["auto", "full"]:
#     # this one should pass
# ke = KernelPCovC(n_components=2, svd_solver="full")
# ke.fit(X, Y)

            # this one should pass
# ke = KernelPCovC(classifier=SVC(verbose=1), n_components="mle", svd_solver="auto")
# ke.fit(X, Y)
# y_pred = ke.predict(X)
# print(accuracy_score(Y, y_pred))

# ke.fit(X, Y)
# print(ke.predict(X))
# y_pred = ke.predict(X)
# print(accuracy_score(Y, y_pred))
# X, Y = get_dataset2(return_X_y=True)
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# kr = KernelPCovR(mixing=1.0, regressor=KernelRidge(), n_components=2)
# kr.fit(X, Y)














# X_or = X
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# classifier = LogisticRegression()
# classifier.fit(X, Y)
# Yhat = classifier.decision_function(X)
# W = classifier.coef_.reshape(X.shape[1], -1)
# pcovc1 = PCovC(mixing=0.5, classifier="precomputed", n_components=1)
# pcovc1.fit(X, Yhat, W)
# t1 = pcovc1.transform(X)

# pcovc2 = PCovC(mixing=0.5, classifier=classifier, n_components=1)
# pcovc2.fit(X, Y)
# t2 = pcovc2.transform(X)

# print(np.linalg.norm(t1 - t2))




# pcovc = PCovC(mixing=0.0, classifier=LogisticRegression(), n_components=2)
# pcovc.fit(X,Y)
# T = pcovc.transform(X)

# pcovc2 = PCovC(mixing=0.0, classifier=LogisticRegression(), n_components=2)
# pcovc2.classifier.fit(X, Y)
# print(pcovc2.classifier.coef_.shape)
# pcovc2.classifier.fit(T, Y)
# print(pcovc2.classifier.coef_.shape)





# model = PCovR(mixing=0.5, regressor=LinearRegression())
# model.fit(X,Y)
# print(isinstance(model, PCovR))

# import numpy as np

# X = np.array([[-1, 0, -2, 3], [3, -2, 0, 1], [-3, 0, -1, -1], [1, 3, 0, -2]])
# Y = np.array([[0], [1], [2], [0]])

# print("AA23")       
# print(Y.shape)
# pcovc = PCovC(mixing=0.1, n_components=2)
# pcovc.fit(X, Y)
# T= pcovc.transform(X)
# print(T)
# array([[ 3.2630561 ,  0.06663787],
#            [-2.69395511, -0.41582771],
#            [ 3.48683147, -0.83164387],
#            [-4.05593245,  1.18083371]])
# Y = pcovc.predict(X)
# print(Y.shape)
# array([[ 0.01371776, -5.00945512],
#            [-1.02805338,  1.06736871],
#            [ 0.98166504, -4.98307078],
#            [-2.9963189 ,  1.98238856]])



# classifier = LogisticRegression()
# classifier.fit(X, Y)
# pcovc = PCovC(mixing=0.5, classifier=classifier, n_components=2)
# pcovc.fit(X,Y)


# X, Y = get_dataset2(return_X_y=True)
# print(X.shape)
# pcovr = PCovR(mixing = 0.5, regressor=LinearRegression())
# pcovr.fit(X,Y)




# classifier = LogisticRegression()
# classifier.fit(X, Y)

# print(classifier.coef_.ndim)

# pcovc = PCovC(mixing=0.5, classifier=LogisticRegression())
# print(pcovc.classifier.coef_.ndim)

# pcovc.fit(X, Y)
# X = [[1, 2, 3, 4, 5],
#      [2, 3, 4, 5, 6]]
# Y = [[0, 1, 0, 1, 0],
#      [0, 1, 0, 1, 0]]

# classifier = LogisticRegression()
# classifier.fit(X, Y)
# model = PCovC(classifier=classifier)

#model2 = PCovC(classifier=LogisticRegression())
#model2.fit(X, Y)

#problem is that coef_.shape (1, n_features=30) is not the same as 
# print(model.classifier.coef_.shape)
# #print(model2.classifier.coef_.ndim)

# model.fit(X, Y)
# y_pred = model.predict(X)
# print(accuracy_score(y_pred, Y))

# X_new, Y_new = get_dataset2(return_X_y=True)
# print(X_new.shape)
# print(Y_new.shape)


'''
Problem is this: check_lr_fit and check_cl_fit do different things because the coefficients for Logistic/Linear regression are different.
So we need to change check_cl_fit

scaler = StandardScaler()
X_new = scaler.fit_transform(X_new)
regressor = LinearRegression()

regressor.fit(X_new, Y_new)
model2 = PCovR(regressor = regressor)
print(model2.regressor.coef_)'''
