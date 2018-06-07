#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-11-30 20:22:37
# @Author  : Lebhoryi@gmail.com
# @Link    : https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/2-5-placeholde/
# @Version : Placeholder 传入值，中文意思为占位符，运行时必须传入值。

import tensorflow as tf
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1,input2)
# 定义乘法运算

with tf.Session() as sess:
	
	print(sess.run(output,feed_dict={input1:[4.],input2:[2.]}))
	# 执行时要传入placeholder的值