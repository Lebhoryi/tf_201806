#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : 2017-11-30 14:57:27
# @Author  : Lebhoryi@gmail.com
# @Link    : https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/2-2-example2/
# @Version : Tensorflow，简单的线性拟合,一维输入和输出，不包隐藏层

import tensorflow as tf
import numpy as np

# creat data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3
# 真实值y,线性函数生成

### creat tensorflow strucure start ###
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
# 产生随机一维变量，范围[-1.0,1.0]，为需要学习的权值，实际0.1
biases = tf.Variable(tf.zeros([1]))
# 一维变量，初值为0，为需要学习的偏置，实际训练接近0.3
y = Weights*x_data + biases
# 线性模拟真实值，得到预测值，目标就是提升y的准确度

loss = tf.reduce_mean(tf.square(y-y_data)) # 计算预测值y与真实值y_data的差值,最小平方差均值
optimizer = tf.train.GradientDescentOptimizer(0.5) # 神经网络的优化器，学习效率0.5
train = optimizer.minimize(loss) # 优化器使误差最小化

init =  tf.initialize_all_variables()
# init = tf.global_variables_initializer()
### creat tensorflow strucure end ###

sess = tf.Session()
sess.run(init)    #Important

for step in range(201):
	sess.run(train)
	if step % 10 == 0:
		print(step,sess.run(Weights),sess.run(biases))
