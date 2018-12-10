# -*- coding: utf-8 -*-
# author:maguichang time:2018/12/10

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 定义添加神经网络的函数def add_layer()
# 我们设定的默认的激励函数是None
# from __future__ import print_function
# 在每一层做计算时，要搞清楚矩阵的维度
"""
这里的表示是：input*weights
例如，输入层的节点个数是2，隐层是3
input=[n*2],weights=[2*3],bias=[1,3]
input*weigths = [n,3]+[1,3]这样矩阵相加的时候矩阵会执行它的广播机制
所以，这一层的输出维度为[n,3]
"""

def add_layer(inputs,in_size,out_size,activation_function=None):
    """

    :param inputs: 输入值
    :param in_size: 输入大小
    :param out_size: 输出大小
    :param activation_function: 激励函数
    :return:
    """
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b = tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# 构造一个数据集
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data)-0.5+noise

# placeholder 占个位
xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

# add hidden layer
l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
# add output layer
# 上一层的输出是这一层的输入
prediction = add_layer(l1,10,1,activation_function=None)

# the error between prediction and real data
# loss 函数和使用梯度下降的方式俩求解
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# plot the real data
plt.scatter(x_data,y_data)

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50 ==0:
        prediction_value = sess.run(prediction,feed_dict={xs:x_data})
        # 在带有placeholder的变量里面，每一次sess.run都需要给一个feed_dict,这个不能省略啊
        # print("loss:",sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        plt.plot(x_data,prediction_value)
plt.show()