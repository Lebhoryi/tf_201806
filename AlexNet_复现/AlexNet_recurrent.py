#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-04-28 13:18:37
# @Author  : Lebhoryi@gmail.com
# @Link1   : https://blog.csdn.net/cyh_24/article/details/51440344
# @Link2   : http://www.sohu.com/a/134347664_642762
# @Version : 从LeNet到AlexNet

'''
AlexNet 之所以能够成功，深度学习之所以能够重回历史舞台，原因在于：
	1.非线性激活函数：ReLU
	2.防止过拟合的方法：Dropout，Data augmentation
	3.大数据训练：百万级ImageNet图像数据
	4.其他：GPU实现，LRN归一化层的使用
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)

# 每个批次的大小
batch_size = 100
# 总共多少批次
n_batch = mnist.train.num_examples // batch_size

# 初始化权值，偏置，卷积层，池化层,lrn层
def weight_variable(shape):
    initial = tf.truncated_normal(shape,dtype=tf.float32,stddev=1e-1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0,shape=shape,dtype=tf.float32)
    return tf.Variable(initial)

# 原strides=[1,4,4,1]
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

# 原ksize=[1,3,3,1],strides=[1,2,2,1]
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

# 这儿的参数是抄的，别问我，我也不知道怎么推算出来的，基本上的AlexNet都在用同一个数据
def lrn(x):
    return tf.nn.lrn(x,4,bias=0.001/9.0,beta=0.75)

# 定义placeholder,占位符
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])
lr = tf.Variable(0.001,dtype=tf.float32)
keep_prob = tf.placeholder(tf.float32)

# x --> 4维向量
x_image = tf.reshape(x,[-1,28,28,1])

# 输入是28*28*1，conv1，图片比较小，正宗用的是224*224，我用的是28*28
W_conv1 = weight_variable([5,5,1,32]) 
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
h_lrn1 = lrn(h_pool1)
# 只有CPU，没有GPU加速，所以丢了个dropout在每一层
h_drop1 = tf.nn.dropout(h_lrn1,keep_prob)


# 上一层输出是14*14*32，conv2
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_drop1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
h_lrn2 = lrn(h_pool2)
h_drop2 = tf.nn.dropout(h_lrn2,keep_prob)

# 上一层输出是7*7*64,conv3
W_conv3 = weight_variable([5,5,64,128])
b_conv3 = bias_variable([128])

h_conv3 = tf.nn.relu(conv2d(h_drop2,W_conv3) + b_conv3)
h_drop3 = tf.nn.dropout(h_conv3,keep_prob)

# 上一层的输出是7*7*128，conv4
W_conv4 = weight_variable([5,5,128,128])
b_conv4 = bias_variable([128])

h_conv4 = tf.nn.relu(conv2d(h_drop3,W_conv4) + b_conv4)
h_drop4 = tf.nn.dropout(h_conv4,keep_prob)

# 上一层的输出是7*7*128，conv5
W_conv5 = weight_variable([5,5,128,64])
b_conv5 = bias_variable([64])

h_conv5 = tf.nn.relu(conv2d(h_drop4,W_conv5) + b_conv5)
h_pool5 = tf.nn.max_pool(h_conv5,ksize=[1,1,1,1],strides=[1,1,1,1],padding='VALID')
h_drop5 = tf.nn.dropout(h_pool5,keep_prob)

# 上一层输出为7*7*64，矩阵变为一维
h_drop5_flat = tf.reshape(h_drop5,[-1,7*7*64])

# 上一层的输出是(7*7*64)，FC1
W_fc1 = weight_variable([7*7*64,2048])
b_fc1 = bias_variable([2048])

h_fc1 = tf.nn.relu(tf.matmul(h_drop5_flat,W_fc1) + b_fc1)
h_drop6 = tf.nn.dropout(h_fc1,keep_prob)

# 上一层的输出是（2048）,FC2
W_fc2 = weight_variable([2048,1000])
b_fc2 = bias_variable([1000])

h_fc2 = tf.nn.relu(tf.matmul(h_drop6,W_fc2) + b_fc2)
h_drop7 = tf.nn.dropout(h_fc2,keep_prob)

# 上一层的输出是1000，softmax
W_fc3 = weight_variable([1000,10])
b_fc3 = bias_variable([10])

prediction = tf.nn.softmax(tf.matmul(h_drop7,W_fc3) + b_fc3)

# 交叉熵
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = prediction,labels = y))
# train_step
train_step =tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# 结果存放
correct_pred = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
# 准确率
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

# 初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(11):
        sess.run(tf.assign(lr,0.001 * (0.95 * epoch)))
        for _ in range(n_batch):
            batch = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch[0],y:batch[1],keep_prob:0.7})
        
        learning_rate = sess.run(lr)
        train_acc = sess.run(accuracy,feed_dict={x:mnist.train.images,y:mnist.train.labels,keep_prob:1.0})
        test_acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
        print("Iter" + str(epoch) + "Test accuracy: " + str(test_acc) + "Train accuracy: " + str(train_acc) + "Learning_rate: " + str(learning_rate))




