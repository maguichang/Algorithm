# -*- coding: utf-8 -*-
# author:maguichang time:2018/12/10

# tf 求解线性回归
import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1+0.3

# 随机初始化权重
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

# 估计的y值
y = Weights*x_data+biases

# 估计的y 和真实的y,计算cost
loss = tf.reduce_mean(tf.square(y-y_data))

# 梯度下降优化
optimizer = tf.train.GradientDescentOptimizer(0.5) # 0.5 学习率
train = optimizer.minimize(loss)

"""
到目前为止，我们只是建立了神经网络的结构，还没有使用这个结构。
在使用这个结构之前，我们必须初始化所有之前定义的Variable，这一步很重要
"""
init = tf.global_variables_initializer()

# 创建会话
sess = tf.Session()
sess.run(init) # 用Session来run每一次training的数据
print(x_data)
print(y_data)

for step in range(201):
    sess.run(train)
    if step%20 ==0:
        print(step,sess.run(Weights),sess.run(biases))