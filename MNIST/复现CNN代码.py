#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-04-26 17:37:09
# @Author  : Lebhoryi@gmail.com
# @Link    : http://example.org
# @Version : CNN实现Mnist识别,准确率提高98%以上

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/',one_hot = True)

# 批次
batch_size = 100
# 多少个批次
n_batch = mnist.train.num_examples // batch_size

# weight,bias,conv2d,max_pooling
def Weight_variable(shape):
	initial = tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1,shape=shape)
	return tf.Variable(initial)

def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):	
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# placeholder
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])
lr = tf.Variable(0.001,dtype=tf.float32)
keep_prob = tf.placeholder(tf.float32)

# 输入的x
x_image = tf.reshape(x,[-1,28,28,1])

# conv1
W_conv1 = Weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_max_pool1 = max_pool_2x2(h_conv1)

# conv2
W_conv2 = Weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_max_pool1,W_conv2) + b_conv2)
h_max_pool2 = max_pool_2x2(h_conv2)

# 经过两层卷积和池化,变成了一张7*7*64的feature map

h_pool2_flat = tf.reshape(h_max_pool2,[-1,7*7*64])

# fc1
W_fc1 = Weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

# fc2 ----> prediction
W_fc2 = Weight_variable([1024,10])
b_fc2 = bias_variable([10])

prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

# 交叉熵损失函数以及优化器
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# accuracy
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float32"))

# initial
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	for epoch in range(11):
		sess.run(tf.assign(lr,0.001 * (0.95 ** epoch)))
		for batch in range(n_batch):
			batch = mnist.train.next_batch(batch_size)
			sess.run(train_step,feed_dict={x:batch[0],y:batch[1],keep_prob:0.7})

		learning_rate = sess.run(lr)
		test_acc = sess.run(accuracy,feed_dict={x:batch[0],y:batch[1],keep_prob:1.0})
		print('Iter: ' + str(epoch) + ',Test_accuracy: ' + str(test_acc) + ',Learning_rate = ' + str(learning_rate))



# Iter: 0,Test_accuracy: 0.99,Learning_rate = 0.001
# Iter: 1,Test_accuracy: 0.99,Learning_rate = 0.00095
# Iter: 2,Test_accuracy: 0.98,Learning_rate = 0.0009025
# Iter: 3,Test_accuracy: 1.0,Learning_rate = 0.000857375
# Iter: 4,Test_accuracy: 0.98,Learning_rate = 0.00081450626
# Iter: 5,Test_accuracy: 0.99,Learning_rate = 0.0007737809
# Iter: 6,Test_accuracy: 0.98,Learning_rate = 0.0007350919
# Iter: 7,Test_accuracy: 0.99,Learning_rate = 0.0006983373
# Iter: 8,Test_accuracy: 1.0,Learning_rate = 0.0006634204
# Iter: 9,Test_accuracy: 0.99,Learning_rate = 0.0006302494
# Iter: 10,Test_accuracy: 1.0,Learning_rate = 0.0005987369
# [Finished in 1555.7s]


















