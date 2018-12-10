# -*- coding: utf-8 -*-
# author:maguichang time:2018/12/10

# tensorflow 中的常量与变量

import tensorflow as tf
# 定义一个变量
var = tf.Variable(0,name = 'myvar')
# 定义一个常量
con = tf.constant(1)
# 定义一个加法
var2 = tf.add(var,con)
# 开始计算
# 初始化，在初始化之前变量是没有值的
init = tf.global_variables_initializer()
# 这里变量没激活，需要在sess里，sess.run(init)
sess = tf.Session()
# 计算
sess.run(init)
# 输出
print('var:',sess.run(var))
print('con:',sess.run(con))
print('var2:',sess.run(var2))
# 关闭会话
sess.close()
"""
with tf.Session() as sess:
    sess.run(init)
    print('var:',sess.run(var))
    print('con:',sess.run(con))
    print('var2:',sess.run(var2))
"""