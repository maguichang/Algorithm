# -*- coding: utf-8 -*-
# author:maguichang time:2018/11/21

# 交叉验证
import numpy as np
from sklearn import datasets

digits = datasets.load_digits()
X = digits.data
y = digits.target

"""
测试train_test_split
"""

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=666)

from sklearn.neighbors import KNeighborsClassifier

# best_k,best_p,best_score = 0,0,0
# for k in range(2,11):
#     for p in range(1,6):
#         knn_clf = KNeighborsClassifier(weights="distance",n_neighbors=k,p = p)
#         knn_clf.fit(X_train,y_train)
#         score = knn_clf.score(X_test,y_test)
#         if score>best_score:
#             best_k,best_p,best_score = k,p,score
#     print("Best K = ",best_k)
#     print("Best P = ",best_p)
#     print("Best Score = ",best_score)


"""
使用交叉验证
"""

from sklearn.model_selection import cross_val_score
knn_clf = KNeighborsClassifier()
cross_val_score(knn_clf,X_train,y_train)

best_k, best_p, best_score = 0, 0, 0
for k in range(2, 11):
    for p in range(1, 6):
        knn_clf = KNeighborsClassifier(weights="distance", n_neighbors=k, p=p)
        scores = cross_val_score(knn_clf, X_train, y_train)
        score = np.mean(scores)
        if score > best_score:
            best_k, best_p, best_score = k, p, score

print("Best K =", best_k)
print("Best P =", best_p)
print("Best Score =", best_score)

best_knn_clf = KNeighborsClassifier(weights="distance", n_neighbors=2, p=2)
best_knn_clf.fit(X_train, y_train)
best_knn_clf.score(X_test, y_test)

