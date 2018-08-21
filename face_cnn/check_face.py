#!/usr/bin/python
#coding=utf-8

import numpy as np
import tensorflow as tf
import cv2
import face_conv as myconv
import common

IMGSIZE = 512

def check_face(chkpoint):
    camera = cv2.VideoCapture(0)
    haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    pathlabelpair, indextoname = common.get_data_lablel('./images')
    output = myconv.cnnLayer(len(pathlabelpair))
    #predict = tf.equal(tf.argmax(output, 1), tf.argmax(y_data, 1))
    predict = output

    saver = tf.train.Saver()
    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        saver.restore(sess, chkpoint)

        n = 1
        while 1:
            if (n <= 20000):
                print('It`s processing %s image.' % n)
                # 读帧
                success, img = camera.read()

                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = haar.detectMultiScale(gray_img, 1.3, 5)
                for f_x, f_y, f_w, f_h in faces:
                    face = img[f_y:f_y+f_h, f_x:f_x+f_w]
                    face = cv2.resize(face, (IMGSIZE, IMGSIZE))
                    #could deal with face to train
                    test_x = np.array([face])
                    test_x = test_x.astype(np.float32) / 255.0
                    
                    res = sess.run([predict, tf.argmax(output, 1)],\
                                   feed_dict={myconv.x_data: test_x,\
                                   myconv.keep_prob_5:1.0, myconv.keep_prob_75: 1.0})
                    print(res)

                    cv2.putText(img, indextoname[res[1][0]], (f_x, f_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)  #显示名字
                    img = cv2.rectangle(img, (f_x, f_y), (f_x + f_w, f_y + f_h), (255, 0, 0), 2)
                    n+=1
                cv2.imshow('img', img)
                key = cv2.waitKey(30) & 0xff
                if key == 27:
                    break
            else:
                break
    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    check_face('./checkpoint/face.ckpt')


