#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-04-19 20:46:43
# @Author  : Lebhoryi@gmail.com
# @Link    : http://example.org
# @Version : 深度学习旅程日志总结及规划

import os

'''
	1.深度学习算法学习:(未习)
	链接: https://pan.baidu.com/s/1dhnCb-RJYWAz6kNGKz1J_w 密码: j5mp
	链接: https://pan.baidu.com/s/1Iybu8EL2EQ6rkTEGpX7Dkw 密码: bd48

	1.1快速过一遍机器学习算法学习:(未习)
	链接: https://pan.baidu.com/s/1L6AfZs-wsau_sE1jvb0PVg 密码: pc7j

	1.2快速浏览神经网络框架,cnn,rnn,梯度下降与反向推导:(未习)
	链接: https://pan.baidu.com/s/1L6AfZs-wsau_sE1jvb0PVg 密码: pc7j

	2.深度学习代码学习:(在学)
	链接: https://pan.baidu.com/s/1zZXMtx2kD9YtQt4tAflLQw 密码: kpab
	
	还有一个视频链接,莫烦老师的,他有numpy/tesnorflow/keras等等一系列的教学视频
	https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/

	####
		第一个链接是我现在正在撸的代码,有讲解,有完整的源码,正在撸,代码中不懂的解决方法:
			1.找网易云课堂的微专业,稍后会贴出链接,吴恩达主讲
			2.上CSDN看别人的详解

		撸完入门级的三个程序:
			1.第一个线性Tensorflow实例
			2.非线性回归,一层隐含层,一层输出层
			3.手写MNIST数据简单分类(真正入门级代码)


		打算撸完第一个的教程之后,快速浏览莫烦老师的视频,2017年已经刷过一遍,好多不懂.
	
	3.学习大纲是按照这个链接来的:(在学)
		机器学习/深度学习 问题总结及解答 
		https://www.nowcoder.com/discuss/71482?type=2&order=3&pos=24&page=2
	
	####
		心中一直坚信,逆向学习比正向学习有意思

	4.吴恩达网易云课堂深度学习微专业(在学)
	http://mooc.study.163.com/smartSpec/detail/1001319001.htm
		
	####
		结合别人的读书笔记:(重要)
		链接: https://pan.baidu.com/s/1uwSH5jw5MGVJASISLLy0Mg 密码: vuku

	5.新找到了一个大牛的网站,大牛说:(未习)
		我们坚信，学习深度学习的最好方式就是动手实现深度学习模型。
	https://zh.gluon.ai/index.html

	#########
	近期打算:(时间不够,快速撸了几个机器学习的算法,没有推导,后期还要补的,还有数学基础)
		1.撸完2.1的视频,1-1.5周;
		2.撸完AlexNet网络,1周.

	初涉深度学习:
		先看一遍莫烦的视频,看到看不懂的地方停止,
		根据牛客网总结的知识点,(这可是面试的知识点哦)去看网易云课堂吴恩达的视频.
		最建议就是:不废话,直接莽代码,正面刚2的代码,不懂就去查.
		要学会总结,看别人的总结和详解很重要...
		看到微信公众号上有人总结:
			数据获取是基础;
			机器学习算法是核心;
			数学基础是根基;
			linux系统是灵魂;
			编程语言是发动机;

		突突突,折腾才会有进步!



'''

############# 
#1.第一个线性Tensorflow实例
#############
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



############# 
#2.非线性回归,一层隐含层,一层输出层
#############

# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-01-18 16:30:58
# @Author  : Lebhoryi@gmail.com
# @Link    : http://example.org
# @Version : 3-1 非线性回归

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 生成200随机点
x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data) + noise

# 两个初始变量值
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])


# 神经网络隐藏层
weights_L1 = tf.Variable(tf.random_normal([1,10]))
biases_L1 = tf.Variable(tf.zeros([1,10]))
wx_plus_b_L1 = tf.matmul(x,weights_L1) + biases_L1
L1 = tf.nn.tanh(wx_plus_b_L1)

# 神经网络输出层
weights_L2 = tf.Variable(tf.random_normal([10,1]))
biases_L2 = tf.Variable(tf.zeros([1,1]))
wx_plus_b_L2 = tf.matmul(L1,weights_L2) + biases_L2
prediction = tf.nn.tanh(wx_plus_b_L2)

# 二次代价函数
loss = tf.reduce_mean(tf.square(y-prediction))
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(2000):
        sess.run(train,feed_dict={x:x_data,y:y_data})
    
    # 获得预测值
    prediction_value = sess.run(prediction,feed_dict={x:x_data})
    # 画图
    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction_value,'r-',lw=5)
    plt.show()


############# 
#3.手写MNIST数据简单分类(真正入门级代码)
#############

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

# 每个批次的大小
batch_size = 100
# 一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

# 定义两个变量
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

# 创建一个简单的神经网络
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x,W)+b)

# 二次代价函数
loss = tf.reduce_mean(tf.square(y-prediction))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# 结果存放
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
# 准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# 初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
#     for _ in range(200):
#         print(sess.run([w,b]))
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter" + str(epoch) + ",Testing Accuracy " + str(acc)) 