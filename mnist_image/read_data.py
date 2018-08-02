#!/usr/bin/python3.5
# -*- coding: utf-8 -*-

import tensorflow as tf


def read_data():

    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer(["mnist_record/number.tfrecords"])
    _,serialized_example = reader.read(filename_queue)

    # 解析读取的样例。
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw':tf.FixedLenFeature([],tf.string),
            'label':tf.FixedLenFeature([],tf.int64)
        })

    images = tf.decode_raw(features['image_raw'],tf.uint8)
    images = tf.reshape(images, [28, 28, 1])
    labels = tf.cast(features['label'],tf.int32)
    labels = tf.one_hot(labels, 10, 1, 0)

    return tf.train.shuffle_batch([images, labels], batch_size=100, capacity=10000, min_after_dequeue=5000)


# image_batch, label_batch = read_data()
#
# sess = tf.Session()
# # 启动多线程处理输入数据。
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(sess=sess,coord=coord)
#
#
#
# image, label = sess.run([image_batch, label_batch])
# print image,label
