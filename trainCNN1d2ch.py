# -*- coding: utf-8 -*-

import tensorflow as tf
import pickle
import random
import numpy as np
import math

dataset_normal_sound = pickle.load(open('no-dr-normal-sound.pkl', 'rb'), encoding='latin1')
dataset_inner_sound = pickle.load(open('no-dr-inner-sound.pkl', 'rb'), encoding='latin1')
dataset_normal_vib = pickle.load(open('no-dr-normal-vib.pkl', 'rb'), encoding='latin1')
dataset_inner_vib = pickle.load(open('no-dr-inner-vib.pkl', 'rb'), encoding='latin1')
batch_size = 100
training_epochs = 10000
accMatrix = np.zeros((training_epochs))

sess = tf.Session()
x = tf.placeholder("float", shape=[None, 1000, 2])
y_ = tf.placeholder("float", shape=[None, 2])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d_pad(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_4(x):
    return tf.nn.max_pool(x, ksize=[1, 4, 1, 1],
                          strides=[1, 4, 1, 1], padding='SAME')
def max_pool_5(x):
    return tf.nn.max_pool(x, ksize=[1, 5, 1, 1],
                          strides=[1, 5, 1, 1], padding='SAME')

# RMSの計算
def calRms(data):
    square = np.power(data,2)
    rms = math.sqrt(sum(square)/len(data))
    return rms
    
def make_batch(batch_size):
    normal_num = len(dataset_normal_sound)/200000
    inner_num = len(dataset_inner_sound)/200000
    batch = np.zeros((batch_size, 1000, 2))
    batch = np.array(batch, dtype=np.float32)
    output = np.zeros((batch_size, 2))
    output = np.array(output, dtype=np.int32)
    for i in range(batch_size):
        if i%2 == 0:
            rand_num = random.randint(0, normal_num - 1)
            rand_index = random.randint(0, 199000)
            batch[i, :, 0] = dataset_normal_sound[(rand_num*200000 + rand_index) : (rand_num*200000 + rand_index + 1000)] / calRms(dataset_normal_sound[(rand_num*200000 + rand_index) : (rand_num*200000 + rand_index + 1000)])
            batch[i, :, 1] = dataset_normal_vib[(rand_num*200000 + rand_index) : (rand_num*200000 + rand_index + 1000)] / calRms(dataset_normal_vib[(rand_num*200000 + rand_index) : (rand_num*200000 + rand_index + 1000)])
            output[i, 0] = 1
        else:
            rand_num = random.randint(0, inner_num - 1)
            rand_index = random.randint(0, 199000)
            batch[i, :, 0] = dataset_inner_sound[(rand_num*200000 + rand_index) : (rand_num*200000 + rand_index + 1000)] / calRms(dataset_inner_sound[(rand_num*200000 + rand_index) : (rand_num*200000 + rand_index + 1000)])
            batch[i, :, 1] = dataset_inner_vib[(rand_num*200000 + rand_index) : (rand_num*200000 + rand_index + 1000)] / calRms(dataset_inner_vib[(rand_num*200000 + rand_index) : (rand_num*200000 + rand_index + 1000)])
            output[i, 1] = 1
    return batch, output
    

W_conv1 = weight_variable([10, 1, 2, 16])
b_conv1 = bias_variable([16])
x_image = tf.reshape(x, [-1, 1000, 1, 2])
h_conv1 = tf.nn.relu(conv2d_pad(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_4(h_conv1)

W_conv2 = weight_variable([10, 1, 16, 32])
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv2d_pad(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_5(h_conv2)

W_conv3 = weight_variable([10, 1, 32, 64])
b_conv3 = bias_variable([64])
h_conv3 = tf.nn.relu(conv2d_pad(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_5(h_conv3)

W_conv4 = weight_variable([10, 1, 64, 128])
b_conv4 = bias_variable([128])
h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)

W_fc1 = weight_variable([128, 32])
b_fc1 = bias_variable([32])
h_flat = tf.reshape(h_conv4, [-1, 128])
h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([32, 2])
b_fc2 = bias_variable([2])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())

for i in range(training_epochs):
    batch, output = make_batch(batch_size)
    train_accuracy = accuracy.eval(session=sess, feed_dict={x: batch, y_: output, keep_prob: 1.0})
    accMatrix[i] = train_accuracy
    if i % 100 == 0:
        loss = cross_entropy.eval(session=sess, feed_dict={x: batch, y_: output, keep_prob: 1.0})
        print(i)
        print(train_accuracy)
        print(loss)
    train_step.run(session=sess, feed_dict={x: batch, y_: output, keep_prob: 0.5})
