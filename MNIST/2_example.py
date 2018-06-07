# !usr/bin/env python
# coding = utf-8
# Tensorflow的第一个实例复现,2018/04/11

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 构造线性模型
x_data = np.random.rand(150)
y_data = x_data * 1.1 + 2.2

weights = tf.Variable(0.)
baises = tf.Variable(0.)
y = weights * x_data + baises

# 二次代价函数
loss = tf.reduce_mean(tf.square(y_data - y))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for step in range(201):
		sess.run(train)
		if step % 20 == 0:
			print(step,sess.run([weights,baises]))
			
		y_value = sess.run(y)
		plt.figure()
		plt.scatter(x_data,y_data)
		plt.plot(x_data,y_value,'r-',lw = 5)
		plt.show()


