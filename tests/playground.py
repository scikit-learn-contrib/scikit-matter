 
from sklearn.discriminant_analysis import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from kernel_pcovc import KernelPCovC
from kernel_pcovr import KernelPCovR
from pcovc import PCovC
from sklearn.datasets import load_breast_cancer as get_dataset
from sklearn.metrics import accuracy_score

X, Y = get_dataset(return_X_y=True)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# classifier = LogisticRegression()
# classifier.fit(X, Y)

# print(classifier.coef_.ndim)

# pcovc = PCovC(mixing=0.5, classifier=LogisticRegression())
# print(pcovc.classifier.coef_.ndim)

# pcovc.fit(X, Y)

model = PCovC(classifier=LogisticRegression())
model.fit(X, Y)
y_pred = model.predict(X)
print(accuracy_score(y_pred, Y))

# model = KernelPCovC(
#             mixing=0.5,
#             classifier=SVC(),
#             n_components=4
# )

# model2 = KernelPCovR(
#             mixing=0.5,
#             regressor=KernelRidge(gamma="scale"),
#             n_components=4
# )
# model3 = SVC()
# model3.fit(X, Y)
# print(model3.dual_coef_.shape)
# # print(model2.gamma, model2.regressor.gamma)
# # model2.fit(X, Y)