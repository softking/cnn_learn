#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import tensorflow as tf
# # 定义输入数据为维度5*3的矩阵
# input_data = tf.constant([[1, 2, 3],[1, 2, 3],[1, 2, 3],[1, 2, 3],[1, 2, 3]],shape=[5,3],name="input_data")
# # 定义权重为维度3*10的矩阵
# weight = tf.constant([[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6]],shape=[3,6],name="weight")
#
#
# y = tf.constant([1.0,7.0,1.0])
#
#
# cross = tf.nn.softmax(y)
#
#
# # 计算矩阵乘法结果
# result = tf.matmul(input_data,weight,name="result")
# with tf.Session() as session:
#     session.run(tf.global_variables_initializer())
#     print("result : \n" , session.run(result))


tag_x = []
tag_y = []
for i in range(1000000):
    x = random.randint(1,100000)
    tag_x.append([x*1.0])
    tag_y.append([x*2.0+1.0])

x = tf.placeholder(tf.float32, shape=(None, 1))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

w = tf.Variable([[0.1]], dtype=tf.float32)
b = tf.Variable([0.1], dtype=tf.float32)

y = tf.nn.relu(tf.matmul(x, w) + b)
loss = tf.square(y - y_)

step = tf.train.AdadeltaOptimizer(0.01).minimize(loss)

saver = tf.train.Saver()

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    saver.restore(session, "Model/model.ckpt")

    for j in range(1000000):
        i = random.randint(1,9000)
        session.run(step, feed_dict={x: tag_x[i:i+1000], y_: tag_y[i:i+1000]})

        if i % 5000 == 0:
            input = random.randint(1, 10000)
            print "w:  ", w.eval(), "  b:  ", b.eval(), "check: ", input,"   result: ",  session.run(y, feed_dict={x:[[input]]})
            saver.save(session, "Model/model.ckpt")



