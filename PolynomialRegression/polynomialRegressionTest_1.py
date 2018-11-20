# -*- coding: utf-8 -*-
# author:maguichang time:2018/11/20

# 多项式回归基础
"""
多项式回归本质上就是将一个特征的高次幂也作为特征来处理，
是对线性回归的改进
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.random.uniform(-3,3,size=100)
""" 
这个-1代表的意思就是，我不知道可以分成多少行，
但是我的需要是分成1列，多少行我不关心，不得不感叹，
果然是人生苦短，我用python
"""
X = x.reshape(-1,1)
y = 0.5*x**2+x+2+np.random.normal(0,1,100)

# 线性回归？
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X,y)
y_predict = lin_reg.predict(X)

# plt.scatter(x,y)
# plt.plot(x,y_predict,color='r')
# plt.show()

# 解决方案，添加一个特征
X2=np.hstack([X,X**2])
X2.shape

lin_reg2 = LinearRegression()
lin_reg2.fit(X2,y)
y_predict2 = lin_reg2.predict(X2)

plt.scatter(x,y)
"""
argsort函数返回的是数组值从小到大的索引值
"""
plt.plot(np.sort(x),y_predict2[np.argsort(x)],color='r')
plt.show()

# 打印相关系数与截距
lin_reg2.coef_
lin_reg2.intercept_