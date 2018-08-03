#!/usr/bin/python3.5
# -*- coding: utf-8 -*-

import os
import tensorflow as tf

cwd = 'images/all/'
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
writer = tf.python_io.TFRecordWriter("mnist_record/number.tfrecords")  # 要生成的文件

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

session = tf.Session()
for index, name in enumerate(classes):
    class_path = cwd + name + '/'
    for img_name in os.listdir(class_path):
        img_path = class_path + img_name  # 每一个图片的地址
        print img_path
        image_raw_data = tf.gfile.FastGFile(img_path, 'r').read()
        img_data = tf.image.decode_bmp(image_raw_data)
        a = session.run(img_data)
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(index),
            'image_raw': _bytes_feature(a.tostring())
        }))
        writer.write(example.SerializeToString())  # 序列化为字符串

writer.close()


