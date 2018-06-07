#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-11-30 19:27:06
# @Author  : Lebhoryi@gmail.com
# @Link    : https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/2-3-session/
# @Version : Session 会话控制

import tensorflow as tf

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],
					   [2]])
product = tf.matmul(matrix1,matrix2)    # matrix multiply,np.dot(m1,m2)


# method 1
# sess = tf.Session()
# result = sess.run(product)
# print(result)
# sess.close()

# method2 推荐用后面一种
with tf.Session() as sess:
	result2 = sess.run(product)
	print(result2)
