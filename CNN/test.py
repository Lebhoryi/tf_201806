import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

# 每个批次的大小
batch_size = 100
# 共多少个批次
n_batch = mnist.train.num_examples // batch_size

# 初始化权值
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

# 初始化偏置
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

# 卷积层
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

# 池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# 定义两个placeholder
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)

# x --> 4维向量
x_image = tf.reshape(x,[-1,28,28,1])

# 第一层卷积层
# 初始化第一个卷积层的权值和偏置
W_conv1 = weight_variable([5,5,1,32]) # 5*5的卷积核,1chanel,32个卷积核
b_conv1 = bias_variable([32]) # 每一个卷积核一个偏置值,自动广播

# 卷积 + 最大池化
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积层
# 权值和偏置
W_conv2 = weight_variable([5,5,32,64]) #5*5的卷积核,32chanel,64个卷积核
b_conv2 = bias_variable([64])

# 卷积 + 最大池化
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 经过两层卷积和两层最大池化层
# 28*28的图片,第一次卷积之后28*28,第一次池化后为14*14
# 第二次卷积后14*14,第二次池化7*7
# 最终结果是64张7*7的平面

# 初始化第一个全连接层的权值和偏置
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])

# 将最终的卷积结果输出为1维
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
# 第一个全连接层的输出
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
# dropout
h_drop = tf.nn.dropout(h_fc1,keep_prob)

# 初始化第二个全连接层
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

# 计算预测值
prediction = tf.nn.softmax(tf.matmul(h_drop,W_fc2) + b_fc2)

# 二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
# 交叉熵
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 结果存放
correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
# 准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# 初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
#         sess.run(tf.assign(lr,0.001 * (0.95 ** epoch)))
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.7})
        
#         learning_rate = sess.run(lr)
        test_acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
#         print("Iter" + str(epoch) + ",Testing Accuracy : " + str(train_acc) + ",Learning_rate : " + str(learning_rate))
#         train_acc = sess.run(accuracy,feed_dict={x:mnist.train.images,y:mnist.trian.labeks,keep_prob:1.0})
        print("Iter" + str(epoch) + ",Testing Accuracy : " + str(test_acc))


