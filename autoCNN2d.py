# -*- coding: utf-8 -*-

import tensorflow as tf
import pickle
import random
import numpy as np
import math

dataset_normal = pickle.load(open('no-em-normal-sound.pkl', 'rb'))

batch_size = 100
training_epochs = 10000

sess = tf.Session()
x = tf.placeholder("float", shape=[None, 10000])

# RMSの計算
def calRms(data):
    square = np.power(data,2)
    rms = math.sqrt(sum(square)/len(data))
    return rms

def make_batch(batch_size):
    normal_num = len(dataset_normal)/200000
    batch = np.zeros((batch_size, 10000))
    batch = np.array(batch, dtype=np.float32)
    for i in range(batch_size):
        rand_num = random.randint(0, normal_num - 2)
        rand_index = random.randint(0, 190000)
        image = dataset_normal[(rand_num*200000 + rand_index) : (rand_num*200000 + rand_index + 10000)] / calRms(dataset_normal[(rand_num*200000 + rand_index) : (rand_num*200000 + rand_index + 10000)])
        batch[i, :] = image
    return batch

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
 
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
 
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
 
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
 
def deconv2d(x, w, b, k, output_shape):
    return tf.nn.bias_add(tf.nn.conv2d_transpose(x,w,strides=[1,k,k,1],output_shape=output_shape), b)

channel1 = 1
channel2 = 5
channel3 = 10
 
# Store layers weight & bias
weights = {
    # filter_h, filter_w, in_ch, out_ch
    'wconv1': weight_variable([5, 5, channel1, channel2]),
    'wconv2': weight_variable([5, 5, channel2, channel3]),
    # filter_h, filter_w, out_ch, in_ch
    'wup1': weight_variable([5, 5, channel2, channel3]),
    'wup2': weight_variable([5, 5, channel1, channel2])
    }
biases = {
    'bconv1': bias_variable([channel2]),
    'bconv2': bias_variable([channel3]),
    'bup1': bias_variable([channel2]),
    'bup2': bias_variable([channel1])
}

def CAE_model(x):
    x_image = tf.reshape(x, [-1,100,100,channel1])
 
    h_conv1 = tf.nn.relu(conv2d(x_image, weights["wconv1"]) + biases["bconv1"])
    h_pool1 = max_pool_2x2(h_conv1)
 
    h_conv2 = tf.nn.relu(conv2d(h_pool1, weights["wconv2"]) + biases["bconv2"])
    encoded = max_pool_2x2(h_conv2)
 
    batch_size = tf.shape(encoded)[0]
    output_shape = tf.stack([batch_size, 50, 50, channel2])
    up_pool1 = deconv2d(encoded, weights['wup1'], biases['bup1'], 2, output_shape)
    up_pool1 = tf.nn.relu(up_pool1)
 
    output_shape2 = tf.stack([batch_size, 100, 100, channel1])
    up_pool2 = deconv2d(up_pool1, weights['wup2'], biases['bup2'], 2, output_shape2)
    up_pool2 = tf.sigmoid(up_pool2)
 
    decoded = tf.reshape(up_pool2, [-1, 10000])
 
    return decoded

decoded = CAE_model(x)
loss = tf.reduce_mean(tf.square(x - decoded))
# cross_entropy = -1. *x *tf.log(decoded) - (1. - x) *tf.log(1. - decoded)
# loss_unlabel = tf.reduce_mean(cross_entropy)
optimizer_unlabel = tf.train.AdamOptimizer(1e-4).minimize(loss)

# 変数初期化
sess.run(tf.initialize_all_variables())

for step in range(training_epochs):
    batch = make_batch(batch_size)
    sess.run(optimizer_unlabel, feed_dict={x: batch})
    if step % 100 == 0:
        print "iteration: {}".format(step)
        print "unlabel_loss: {}".format(sess.run(loss,  feed_dict={x: batch}))
        
saver = tf.train.Saver()
saver.save(sess, 'model.ckpt')
