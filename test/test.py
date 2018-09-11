# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

"""
Created on Tue Jul 17 10:03:21 2018
@author: C.H.
"""
tf.reset_default_graph()  # 这一句话非常重要，如果没有这句话，就会出现重复定义变量的错误
x = tf.placeholder(tf.float32, shape=(1, 500, 500, 3))
# 分别设置3*3,5*5,7*7三种大小的卷积核
weights1 = tf.get_variable('weights1', shape=[8, 8, 3, 16], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
weights2 = tf.get_variable('weights2', shape=[5, 5, 3, 16], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
weights3 = tf.get_variable('weights3', shape=[7, 7, 3, 16], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
# 第一组实验采用步长为1，填充采用SAME，然后采用三种不同大小的卷积核来进行实验，讨论卷积核对卷积后图像大小的影响。第一组实验为其他实验的对照组
conv1 = tf.nn.conv2d(x, weights1, strides=[1, 1, 1, 1], padding='SAME')
conv2 = tf.nn.conv2d(x, weights2, strides=[1, 1, 1, 1], padding='SAME')
conv3 = tf.nn.conv2d(x, weights3, strides=[1, 1, 1, 1], padding='SAME')

# 第二组实验，控制卷积核的大小为3*3，分别采用1,2,3三种步长，padding方式采用SAME，讨论步长对卷积后图像大小的影响。
conv4 = tf.nn.conv2d(x, weights1, strides=[1, 1, 1, 1], padding='SAME')
conv5 = tf.nn.conv2d(x, weights1, strides=[1, 2, 2, 1], padding='SAME')
conv6 = tf.nn.conv2d(x, weights1, strides=[1, 3, 3, 1], padding='SAME')


# 第三组实验，与第一组实验对照，选择和第一组实验相同的卷积核大小和步长，采用padding的填充方式进行测试。讨论不同padding方式对卷积后图像的影响
conv7 = tf.nn.conv2d(x, weights1, strides=[1, 1, 1, 1], padding='VALID')
conv8 = tf.nn.conv2d(x, weights2, strides=[1, 1, 1, 1], padding='VALID')
conv9 = tf.nn.conv2d(x, weights3, strides=[1, 1, 1, 1], padding='VALID')


# 池化过程的'VALID'，'SAME'参数的对照。讨论不同参数设置对最大池化过程后图像大小的影响
pool1 = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 3, 3, 1], padding='VALID')

pool2 = tf.nn.max_pool(x, [1, 4, 4, 1], [1, 4, 4, 1], padding='SAME')

with tf.Session() as sess:
    a = np.full((1, 500, 500, 3), 2)
    show = np.full((3, 4, 2), 2)

    sess.run(tf.global_variables_initializer())



    conv1 = sess.run(conv1, feed_dict={x: a})
    conv2 = sess.run(conv2, feed_dict={x: a})
    conv3 = sess.run(conv3, feed_dict={x: a})
    conv4 = sess.run(conv4, feed_dict={x: a})
    conv5 = sess.run(conv5, feed_dict={x: a})
    conv6 = sess.run(conv6, feed_dict={x: a})
    conv7 = sess.run(conv7, feed_dict={x: a})
    conv8 = sess.run(conv8, feed_dict={x: a})
    conv9 = sess.run(conv9, feed_dict={x: a})
    pool1 = sess.run(pool1, feed_dict={x: a})
    pool2 = sess.run(pool2, feed_dict={x: a})
    print(conv1.shape)
    print(conv2.shape)
    print(conv3.shape)
    print()
    print(conv4.shape)
    print(conv5.shape)
    print(conv6.shape)
    print()
    print(conv7.shape)
    print(conv8.shape)
    print(conv9.shape)
    print()
    print(pool1.shape)
    print(pool2.shape)

    show[1][1][1] = 100
    print(show)
