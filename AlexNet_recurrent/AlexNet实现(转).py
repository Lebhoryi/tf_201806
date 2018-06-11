#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-04-24 11:51:18
# @Author  : Lebhoryi@gmail.com
# @Link1   : https://www.cnblogs.com/georgeli/p/8476307.html
# @Link2   : https://blog.csdn.net/taoyanqi8932/article/details/71081390
# @Version : 这里AlexNet的实现将不涉及实际数据的训练，但会创建一个完整的AlexNet卷积神经网络，然后对它每个batch的前馈计算（forward）和反馈计算（backward）的速度进行测试。下面使用随机图片数据来计算每轮前馈、反馈的平均耗时。

from datetime import datetime
import math
import time
import tensorflow as tf
 
# 这里总共测试100个batch的数据。
batch_size=32
num_batches=100

# 定义一个用来显示网络每一层结构的函数print_activations，展示每一个卷积层或池化层输出的tensor尺寸。
# 这个函数接受一个tensor作为输入，并显示其名称（t.op.name）和tensor尺寸（t.get_shape.as_list()）
def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())
 
# 设计AlexNet网络结构。
# 设定inference函数，用于接受images作为输入，返回最后一层pool5（第5个池化层）及parameters（AlexnNet中所有需要训练的模型参数）
# 该函数包括多个卷积层和池化层。
def inference(images):
    parameters = []
    
    # 定义第一个卷积层，并在其后添加LRN层和最大池化层
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        print_activations(conv1)
        parameters += [kernel, biases]
 
  # 添加LRN层和最大池化层
    lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='lrn1')
    pool1 = tf.nn.max_pool(lrn1,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool1')
    print_activations(pool1)
 
  # 设计第2个卷积层，大部分步骤和第1个卷积层相同，只有几个参数不同。
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
    print_activations(conv2)
 
  # 对第2个卷积层的输出进行处理，同样也是先做LRN处理再做最大化池处理。
    lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='lrn2')
    pool2 = tf.nn.max_pool(lrn2,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool2')
    print_activations(pool2)
 
  # 设计第3个卷积层
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv3)
 
  # 设计第4个卷积层
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv4)
 
  # 设计第5个卷积层
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv5)
 
  # 最大池化层
    pool5 = tf.nn.max_pool(conv5,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool5')
    print_activations(pool5)
 
    return pool5, parameters

# 接下来实现一个评估AlexNet每轮计算时间的函数time_tensorflow_run.第一个输入是Tensorflow的Session，
# 第二个变量是需要测评的运算算子，第三个变量是测试的名称。先定义预热轮数num_steps_burn_in=10,它的作用是
# 给程序预热
def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print ('%s: step %d, duration = %.3f' %
                       (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print ('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
           (datetime.now(), info_string, num_batches, mn, sd))
 
#主函数
def run_benchmark():
    with tf.Graph().as_default():
        # image_size = 224
        image_size = 227
        images = tf.Variable(tf.random_normal([batch_size,
                                           image_size,
                                           image_size, 3],
                                          dtype=tf.float32,
                                          stddev=1e-1))
 
        
        pool5, parameters = inference(images)
 
        init = tf.global_variables_initializer()
 
        # config = tf.ConfigProto()
        # config.gpu_options.allocator_type = 'BFC'
        # sess = tf.Session(config=config)
        sess = tf.Session()
        sess.run(init)
        # print(sess.run(images))
     
        time_tensorflow_run(sess, pool5, "Forward")
      
        objective = tf.nn.l2_loss(pool5)
    
        grad = tf.gradients(objective, parameters)
	    # Run the backward benchmark.
        time_tensorflow_run(sess, grad, "Forward-backward")
 
#执行主函数
run_benchmark()
