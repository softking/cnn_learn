#!/usr/bin/python
# coding=utf-8

import logging as log
import numpy as np
import common
import face_conv as myconv

if __name__ == '__main__':

    pathlabelpair, indextoname = common.get_data_lablel('./images')
    train_x, train_y = common.read_image(pathlabelpair)
    train_x = train_x.astype(np.float32) / 255.0
    log.debug('len of train_x : %s', train_x.shape)
    myconv.train(train_x, train_y, './checkpoint/face.ckpt')
    log.debug('training is over, please run again')
