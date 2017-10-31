# -*- coding: utf-8 -*-

import tensorflow as tf
import pickle
import random
import numpy as np
import math

dataset_normal = pickle.load(open('A5052P-em10-normal-sound.pkl', 'rb'))
dataset_inner = pickle.load(open('A5052P-em10-inner-sound.pkl', 'rb'))

dataset_normal = np.r_[dataset_normal, pickle.load(open('no-em-normal-sound.pkl', 'rb'))]
dataset_inner = np.r_[dataset_inner, pickle.load(open('no-em-inner-sound.pkl', 'rb'))]
    
test_normal = pickle.load(open('A5052P-em06-normal-sound-all.pkl', 'rb'))
test_inner = pickle.load(open('A5052P-em06-inner-sound-all.pkl', 'rb'))

batch_size = 100
training_epochs = 10000
accMatrix = np.zeros((training_epochs))
sumsMatrix = np.zeros((training_epochs))

sess = tf.Session()
x = tf.placeholder("float", shape=[None, 10000])
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
    normal_num = len(dataset_normal)/200000
    inner_num = len(dataset_inner)/200000
    batch = np.zeros((batch_size, 10000))
    batch = np.array(batch, dtype=np.float32)
    output = np.zeros((batch_size, 2))
    output = np.array(output, dtype=np.int32)
    for i in range(batch_size):
        if i%2 == 0:
            rand_num = random.randint(0, normal_num - 1)
            rand_index = random.randint(0, 190000)
            batch[i, :] = dataset_normal[(rand_num*200000 + rand_index) : (rand_num*200000 + rand_index + 10000)] / calRms(dataset_normal[(rand_num*200000 + rand_index) : (rand_num*200000 + rand_index + 10000)])
            output[i, 0] = 1
        else:
            rand_num = random.randint(0, inner_num - 1)
            rand_index = random.randint(0, 190000)
            batch[i, :] = dataset_inner[(rand_num*200000 + rand_index) : (rand_num*200000 + rand_index + 10000)] / calRms(dataset_inner[(rand_num*200000 + rand_index) : (rand_num*200000 + rand_index + 10000)])
            output[i, 1] = 1
    return batch, output

def make_test_batch(batch_size):
    normal_num = len(test_normal)/200000
    inner_num = len(test_inner)/200000
    batch = np.zeros((batch_size, 10000))
    batch = np.array(batch, dtype=np.float32)
    output = np.zeros((batch_size, 2))
    output = np.array(output, dtype=np.int32)
    for i in range(batch_size):
        if i%2 == 0:
            rand_num = random.randint(0, normal_num - 1)
            rand_index = random.randint(0, 190000)
            batch[i, :] = dataset_normal[(rand_num*200000 + rand_index) : (rand_num*200000 + rand_index + 10000)] / calRms(dataset_normal[(rand_num*200000 + rand_index) : (rand_num*200000 + rand_index + 10000)])
            output[i, 0] = 1
        else:
            rand_num = random.randint(0, inner_num - 1)
            rand_index = random.randint(0, 190000)
            batch[i, :] = dataset_inner[(rand_num*200000 + rand_index) : (rand_num*200000 + rand_index + 10000)] / calRms(dataset_inner[(rand_num*200000 + rand_index) : (rand_num*200000 + rand_index + 10000)])
            output[i, 1] = 1
    return batch, output
    

W_conv1 = weight_variable([10, 1, 1, 16])
b_conv1 = bias_variable([16])
x_image = tf.reshape(x, [-1, 10000, 1, 1])
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
h_conv4 = tf.nn.relu(conv2d_pad(h_pool3, W_conv4) + b_conv4)
h_pool4 = max_pool_5(h_conv4)

W_conv5 = weight_variable([5, 1, 128, 256])
b_conv5 = bias_variable([256])
h_conv5 = tf.nn.relu(conv2d_pad(h_pool4, W_conv5) + b_conv5)
h_pool5 = max_pool_4(h_conv5)

W_conv6 = weight_variable([5, 1, 256, 512])
b_conv6 = bias_variable([512])
h_conv6 = tf.nn.relu(conv2d(h_pool5, W_conv6) + b_conv6)

W_fc1 = weight_variable([512, 32])
b_fc1 = bias_variable([32])
h_flat = tf.reshape(h_conv6, [-1, 512])
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
    batch_test, output_test = make_test_batch(batch_size)
    test_accuracy = accuracy.eval(session=sess, feed_dict={x: batch_test, y_: output_test, keep_prob: 1.0})
    sumsMatrix[i] = test_accuracy
    if i%100 ==0:
        loss = cross_entropy.eval(session=sess, feed_dict={x: batch, y_: output, keep_prob: 1.0})
        print(i)
        print(train_accuracy)
        print(test_accuracy)
        print(loss)
    train_step.run(session=sess, feed_dict={x: batch, y_: output, keep_prob: 0.5})
             
    
pickle.dump(accMatrix, open('accMatrix.pkl', 'wb'))
pickle.dump(sumsMatrix, open('sumsMatrix.pkl', 'wb'))
