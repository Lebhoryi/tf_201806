#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-01-14 13:33:23
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$

import tensorflow as tf
import numpy as np

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))
y = Weights*x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# init = tf.global_variables_initializer()
init =  tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)

	for step in range(201):
		sess.run(train)
		if step % 10 == 0:
			print(step,sess.run(Weights),sess.run(biases))
