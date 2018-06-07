#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-12-09 10:02:51
# @Author  : Lebhoryi@gmail.com
# @Link1   : http://blog.csdn.net/xuan_zizizi/article/details/77815986
# @Link2   : https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/3-1-add-layer/
# @Version : tensorflow添加函数

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs,in_size,out_size,activation_function = None):
# 定义一个添加层，输入值，输入尺寸，输出尺寸，以及激励函数，此处None默认为线性的激励函数
	Weights = tf.Variable(tf.random_normal([in_size,out_size]))
	# 定义权值矩阵，in_size行，out_size列，随机初始权值
	biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
	# 定义一个列表，一行，out_size列，值全部为0.1
	Wx_plus_b = tf.add(tf.matmul(inputs,Weights),biases)
	# inputs*Weights + biases,预测值，未激活
	if activation_function is None:
		outputs = Wx_plus_b
		# 结果为线性输出，激活结果
	else:
		outputs = activation_function(Wx_plus_b)
		# 激励函数处理
	return outputs

# 定义数据集
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape) # 噪声均值为0,方差为0.05,与x_data格式相同
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

# add hidden layer
l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
# 隐含层输入层input（1个神经元）：输入1个神经元，隐藏层10个神经元

# add output layer
prediction = add_layer(l1,10,1,activation_function=None)
# 输出层也是1个神经元：隐藏层10个神经元，输出1个神经元

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
					reduction_indices=[1]))
# tf.reduce_mean(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)
# 先求误差平方和的和求平均，reduce_sum表示对矩阵求和，reduction_indices=[1]方向

optimizer = tf.train.GradientDescentOptimizer(0.1)
train_step = optimizer.minimize(loss)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# 可视化结果
fig = plt.figure() # 生成图片框
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()

for i in range(1000):
	# trianing
	sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
	if i%50 == 0:
		# print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
		try:
			ax.lines.remove(lines[0])
		except Exception:
			pass
		prediction_value = sess.run(prediction,feed_dict = {xs:x_data})
		# 画出预测
		lines = ax.plot(x_data,prediction_value,'r-',lw=5)
		plt.pause(0.1)

