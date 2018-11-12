# -*- coding: utf-8 -*-
# author:maguichang time:2018/11/12

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# 数据标准化
from sklearn.preprocessing import StandardScaler

# 测试数据
iris = datasets.load_iris()
X = iris.data
y = iris.target
# train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=666)

standardScalar = StandardScaler()
standardScalar.fit(X_train)
# standardScalar.mean_
# standardScalar.scale_
X_train = standardScalar.transform(X_train)
X_test_standard = standardScalar.transform(X_test)

# 使用归一化后的数据进行knn分类
knn_clf = KNeighborsClassifier(n_neighbors=3)
# 训练拟合
knn_clf.fit(X_train,y_train)
# 测试得分(此时不能传入没有归一化的测试数据)
print(knn_clf.score(X_test_standard,y_test))

