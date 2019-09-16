'''
author: lilinxiao
data: 2019-09-16
aim: realize the simple netural network
'''

import tensorflow as tf
import numpy as np

# x = tf.constant([[0.7,0.9]])
# y = tf.constant([[1.0]])

x = np.arange(10, dtype=np.float32).reshape(5,2) # 使用np来创造两个样本
y = np.array([1,0,1,0,0], dtype=np.float32).reshape(5,1) # 使用np来创造两个label

w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

b1 = tf.Variable(tf.zeros([3]))
b2 = tf.Variable(tf.zeros([1]))

a = tf.nn.relu(tf.matmul(x,w1)+b1)
y_ = tf.matmul(a,w2)+b2

cost = tf.nn.sigmoid_cross_entropy_with_logits(logits = y_,labels = y,name = None)

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    for i in range(100):
        sess.run(train_op)
    print(sess.run(w1))
    print(sess.run(w2))
    print(sess.run(b1))
    print(sess.run(b2))
    print(sess.run(y_))

'''
测试
'''
x1 = np.arange(0,4,dtype = np.float32).reshape(2,2)
a1 = tf.nn.relu(tf.matmul(x1,w1)+b1)
y1 = tf.nn.sigmoid(tf.matmul(a1,w2)+b2)
# y1 = tf.matmul(a1,w2)+b2

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(y1))