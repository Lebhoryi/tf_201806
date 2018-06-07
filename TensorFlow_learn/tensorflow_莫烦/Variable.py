#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-11-30 19:51:50
# @Author  : Lebhoryi@gmail.com
# @Link    : https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/2-4-variable/
# @Version : Variable 变量

import tensorflow as tf

state = tf.Variable(0,name='counter')
print(state.name)
one = tf.constant(1)

new_value = tf.add(state,one)    # add
update = tf.assign(state,new_value)    # 赋值，new_value赋值state

init = tf.global_variables_initializer()   #must have if define variable
# init = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)
	for _ in range(3):
		sess.run(update)
		print(sess.run(state))

