#!/usr/bin/python
#coding=utf-8

import os
import logging as log
import numpy as np
import tensorflow as tf

SIZE = 64
FULL_LAYER = 512

x_data = tf.placeholder(tf.float32, [None, SIZE, SIZE, 3])
y_data = tf.placeholder(tf.float32, [None, None])

keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)

def weightVariable(shape):
    init = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(init)

def biasVariable(shape):
    init = tf.random_normal(shape)
    return tf.Variable(init)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxPool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def dropout(x, keep):
    return tf.nn.dropout(x, keep)

def cnnLayer(classnum):
    # 第一层
    W1 = weightVariable([3, 3, 3, 64]) # 卷积核大小(3,3)， 输入通道(3)， 输出通道(32)
    b1 = biasVariable([64])
    conv1 = tf.nn.relu(conv2d(x_data, W1) + b1)
    pool1 = maxPool(conv1)
    # 减少过拟合，随机让某些权重不更新
    drop1 = dropout(pool1, keep_prob_5) # 32 * 32 * 32 多个输入channel 被filter内积掉了

    # 第二层
    W2 = weightVariable([3, 3, 64, 64])
    b2 = biasVariable([64])
    conv2 = tf.nn.relu(conv2d(drop1, W2) + b2)
    pool2 = maxPool(conv2)
    drop2 = dropout(pool2, keep_prob_5) # 64 * 16 * 16

    # 第三层
    W3 = weightVariable([3, 3, 64, 64])
    b3 = biasVariable([64])
    conv3 = tf.nn.relu(conv2d(drop2, W3) + b3)
    pool3 = maxPool(conv3)
    drop3 = dropout(pool3, keep_prob_5) # 64 * 8 * 8

    pool_shape = pool3.get_shape().as_list()     # 计算全连接层空间大小
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]     # 计算全连接层空间大小

    # 全连接层
    Wf = weightVariable([nodes, FULL_LAYER])
    bf = biasVariable([FULL_LAYER])
    drop3_flat = tf.reshape(drop3, [-1, nodes])
    dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)
    dropf = dropout(dense, keep_prob_75)

    # 输出层
    Wout = weightVariable([FULL_LAYER, classnum])
    bout = weightVariable([classnum])
    #out = tf.matmul(dropf, Wout) + bout
    out = tf.add(tf.matmul(dropf, Wout), bout)
    return out

def train(train_x, train_y, model_path):
    log.debug('train')
    out = cnnLayer(train_y.shape[1])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=out, labels=y_data))
    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y_data, 1)), tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if os.path.exists(model_path + '.meta'):
            saver.restore(sess, model_path)

        batch_size = 50
        num_batch = len(train_x) // 50
        for n in range(50):
            r = np.random.permutation(len(train_x))
            train_x = train_x[r, :]
            train_y = train_y[r, :]

            for i in range(num_batch):
                batch_x = train_x[i*batch_size : (i+1)*batch_size]
                batch_y = train_y[i*batch_size : (i+1)*batch_size]
                _, loss = sess.run([train_step, cross_entropy],\
                                   feed_dict={x_data:batch_x, y_data:batch_y,
                                              keep_prob_5:0.75, keep_prob_75:0.75})

                print(n*num_batch+i, loss)

        # 获取测试数据的准确率
        acc = accuracy.eval({x_data:train_x, y_data:train_y, keep_prob_5:1.0, keep_prob_75:1.0})
        print('after 10 times run: accuracy is ', acc)
        saver.save(sess, model_path)

def validate(test_x, tfsavepath):
    output = cnnLayer(2)
    #predict = tf.equal(tf.argmax(output, 1), tf.argmax(y_data, 1))
    predict = output

    saver = tf.train.Saver()
    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        saver.restore(sess, tfsavepath)
        res = sess.run([predict, tf.argmax(output, 1)],
                       feed_dict={x_data: test_x,
                                  keep_prob_5:1.0, keep_prob_75: 1.0})
        return res
