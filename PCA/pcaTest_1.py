# -*- coding: utf-8 -*-
# author:maguichang time:2018/11/13

# pca demo
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
# 加载样本数据集,
digits = datasets.load_digits()
X = digits.data
y = digits.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=666)

from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train,y_train)

knn_clf.score(X_test,y_test)


"""
PCA处理
"""
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(X_train)
X_train_reduction = pca.transform(X_train)
X_test_reduction = pca.transform(X_test)
# 由于手写数字的像素范围均为0-255，故此处不需要demean处理
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_reduction,y_train)
knn_clf.score(X_test_reduction,y_test)


"""
主成分所解释的方差
"""
pca.explained_variance_ratio_
pca.explained_variance_

# 求解每一个主成分所解释的方差并绘图
pca = PCA(n_components=X_train.shape[1])
pca.fit(X_train)
pca.explained_variance_ratio_

plt.plot([i for i in range(X_train.shape[1])],\
         [np.sum(pca.explained_variance_ratio_[:i+1]) for i in range(X_train.shape[1])])
plt.show()



