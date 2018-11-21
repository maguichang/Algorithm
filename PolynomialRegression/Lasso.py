# -*- coding: utf-8 -*-
# author:maguichang time:2018/11/21

"""
正则化，岭回归与LASSO，旨在解决过拟合问题
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
x = np.random.uniform(-3.0, 3.0, size=100)
X = x.reshape(-1, 1)
y = 0.5 * x + 3 + np.random.normal(0, 1, size=100)

from sklearn.model_selection import train_test_split

np.random.seed(666)
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def PolynomialRegression(degree):
    return Pipeline([
        ("Poly",PolynomialFeatures(degree=degree)),
        ("std_scaler",StandardScaler()),
        ("lin_reg",LinearRegression())
    ])

from sklearn.metrics import mean_squared_error

poly_reg = PolynomialRegression(degree=20)
poly_reg.fit(X_train,y_train)

y_predict = poly_reg.predict(X_test)
mean_squared_error(y_test,y_predict)

"""
Lasso,
如果是岭回归，则导入
from sklearn.linear_model import Ridge
将Lasso的地方替换为Ridge
"""
from sklearn.linear_model import Lasso

def LassoRegression(degree,alpha):
    return Pipeline([
        ("Poly",PolynomialFeatures(degree=degree)),
        ("std_scaler",StandardScaler()),
        ("lasso_reg",LinearRegression(alpha = alpha))
    ])

lasso_reg = LassoRegression(20,0.01)
lasso_reg.fit(X_train,y_train)

y1_predict = lasso_reg.predict(X_test)
mean_squared_error(y_test,y_predict)
