 
from sklearn.discriminant_analysis import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from _kernel_pcovc import KernelPCovC
from _kernel_pcovr import KernelPCovR
from _pcovc import PCovC
from sklearn.datasets import load_breast_cancer as get_dataset
from sklearn.datasets import load_diabetes as get_dataset2
from sklearn.metrics import accuracy_score
from pcovr_new import PCovR

X, Y = get_dataset2(return_X_y=True)

scaler = StandardScaler()
X = scaler.fit_transform(X)

model = PCovR(mixing=0.5, regressor=LinearRegression())
model.fit(X,Y)
print(isinstance(model, PCovR))





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
