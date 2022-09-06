from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

import pandas as pd

## 분류 (classification) ##

iris = load_iris()

iris_data = iris.data
iris_label = iris.target
# print(iris_label, iris.target_names)

iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)
iris_df['label'] = iris.target
# print(iris_df)

X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_label, test_size=0.2, random_state=11)

dt_clf = DecisionTreeClassifier(random_state=11)

dt_clf.fit(X_train,y_train)

pred = dt_clf.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test ,pred))

print(type(iris))
print(iris.keys())
print(iris.target_names)


### K Fold ###

from sklearn.model_selection import KFold
import numpy as np

iris = load_iris()
features = iris.data
label = iris.target
dt_clf = DecisionTreeClassifier(random_state=156)

kfold = KFold(n_splits=5)
cv_accuracy = []

print(features.shape[0])

n_iter = 0


for train_index, test_index in kfold.split(features):
    X_train, X_test = features[train_index], features[test_index]
    Y_train, y_test = label[train_index], label[test_index]

    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)
    n_iter += 1

    accuracy = np.round(accuracy_score(y_test, pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]

    print('\n#{0} 교차 검증 정확도 : {1}, 학습 데이터 크기 : {2}, 검증 데이터 크기 : {3}'.format(n_iter, accuracy, train_size, test_size))
    print('\n#{0} 검증 세트 인덱스 : {1}'.format(n_iter, test_index))

    cv_accuracy.append(accuracy)

print('\n## 평균 검증 정확도:', np.mean(cv_accuracy))
print(cv_accuracy)


