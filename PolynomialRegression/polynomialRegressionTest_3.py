# -*- coding: utf-8 -*-
# author:maguichang time:2018/11/20

# 多项式回归
"""
scikit-learn中的多项式回归和Pipeline
"""

import numpy as np
import matplotlib.pyplot as plt

"""
关于PolynomialFeatures
"""
XX = np.arange(1,11).reshape(-1,2)
XX
poly = PolynomialFeatures(degree=2)
poly.fit(XX)
XX2 = poly.transform(XX)
"""
第一列0次幂，2,3列为原始值x1、x2，第四列为x1**2，
第五列x1*x2,第六列x2**2
array([[   1.,    1.,    2.,    1.,    2.,    4.],
       [   1.,    3.,    4.,    9.,   12.,   16.],
       [   1.,    5.,    6.,   25.,   30.,   36.],
       [   1.,    7.,    8.,   49.,   56.,   64.],
       [   1.,    9.,   10.,   81.,   90.,  100.]])
"""
XX2.shape

# Pipline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

poly_reg = Pipeline([
    ("poly",PolynomialFeatures(degree=2)),
    ("std_scaler",StandardScaler()),
    ("lin_reg",LinearRegression())
])

poly_reg.fit(X,y)
y_predict = poly_reg.predict(X)

plt.scatter(x,y)
"""
argsort函数返回的是数组值从小到大的索引值
"""
plt.plot(np.sort(x),y_predict2[np.argsort(x)],color='r')
plt.show()