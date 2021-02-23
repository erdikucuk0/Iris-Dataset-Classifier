# import libraries
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score

# load the iris data
iris = load_iris()
x = iris.data[:, :4]
y = iris.target

# split the data
from sklearn.model_selection import train_test_split
X_train, x_test, Y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
x_test = scaler.transform(x_test)

# logistic regression
from sklearn.linear_model import LogisticRegression
log = LogisticRegression(random_state=0)
log.fit(X_train, Y_train)
log_pred = log.predict(x_test)
cm = confusion_matrix(y_test, log_pred)
print("Logistic Regression")
print(cm)
print("Adjusted R2 Score:", end="")
print(r2_score(y_test, log.predict(x_test)))

# k-nearest neighbors
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3, metric='minkowski')
knn.fit(X_train, Y_train)
knn_pred = knn.predict(x_test)
cm = confusion_matrix(y_test, knn_pred)
print('K-Nearest Neighbors')
print(cm)
print("Adjusted R2 Score:", end="")
print(r2_score(y_test, knn.predict(x_test)))

# support vector machine
from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(X_train, Y_train)
svc_pred = svc.predict(x_test)
cm = confusion_matrix(y_test, svc_pred)
print('Support Vector Classifier')
print(cm)
print("Adjusted R2 Score:", end="")
print(r2_score(y_test, svc.predict(x_test)))

# decision tree
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train, Y_train)
dtc_pred = dtc.predict(x_test)
cm = confusion_matrix(y_test, dtc_pred)
print("Decision Tree Classifier")
print(cm)
print("Adjusted R2 Score:", end="")
print(r2_score(y_test, dtc.predict(x_test)))

# random tree
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, criterion='gini')
rfc.fit(X_train, Y_train)
rfc_pred = rfc.predict(x_test)
cm = confusion_matrix(y_test, rfc_pred)
print("Random Tree Classifier")
print(cm)
print("Adjusted R2 Score:", end="")
print(r2_score(y_test, rfc.predict(x_test)))

# gaussian naive bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, Y_train)
gnb_pred = gnb.predict(x_test)
cm = confusion_matrix(y_test, gnb_pred)
print("Gaussian NB")
print(cm)
print("Adjusted R2 Score:", end="")
print(r2_score(y_test, gnb.predict(x_test)))