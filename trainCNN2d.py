# -*- coding: utf-8 -*-

import numpy as np
import pickle
import math
import random
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers, cuda

gpu_flag = 0
TRAINING_EPOCHS = 10000
MINI_BATCH_SIZE = 100
dropout_ratio = 0.1

# 学習データの読み込み
dataset_normal = pickle.load(open('A5052P-em10-normal-sound.pkl', 'rb'))
dataset_inner = pickle.load(open('A5052P-em10-inner-sound.pkl', 'rb'))

dataset_normal = np.r_[dataset_normal, pickle.load(open('no-em-normal-sound.pkl', 'rb'))]
dataset_inner = np.r_[dataset_inner, pickle.load(open('no-em-inner-sound.pkl', 'rb'))]
    
test_normal = pickle.load(open('A5052P-em06-normal-sound-all.pkl', 'rb'))
test_inner = pickle.load(open('A5052P-em06-inner-sound-all.pkl', 'rb'))

if gpu_flag >= 0:
    cuda.get_device(gpu_flag).use()
    xp = cuda.cupy
else:
    xp = np

# RMSの計算
def calRms(data):
    square = np.power(data,2)
    rms = math.sqrt(sum(square)/len(data))
    return rms

def make_batch(batch_size):
    normal_num = len(dataset_normal)/200000
    inner_num = len(dataset_inner)/200000
    batch = np.zeros((batch_size, 1, 100, 100))
    batch = np.array(batch, dtype=xp.float32)
    output = np.zeros((batch_size))
    output = np.array(output, dtype=xp.int32)
    for i in range(batch_size):
        if i%2 == 0:
            rand_num = random.randint(0, normal_num - 1)
            rand_index = random.randint(0, 190000)
            image = dataset_normal[(rand_num*200000 + rand_index) : (rand_num*200000 + rand_index + 10000)] / calRms(dataset_normal[(rand_num*200000 + rand_index) : (rand_num*200000 + rand_index + 10000)])
            batch[i, 0, :, :] = image.reshape(100, 100)
            output[i] = 0
        else:
            rand_num = random.randint(0, inner_num - 1)
            rand_index = random.randint(0, 190000)
            image = dataset_inner[(rand_num*200000 + rand_index) : (rand_num*200000 + rand_index + 10000)] / calRms(dataset_inner[(rand_num*200000 + rand_index) : (rand_num*200000 + rand_index + 10000)])
            batch[i, 0, :, :] = image.reshape(100, 100)
            output[i] = 1
    return batch, output


def make_test_batch(batch_size):
    normal_num = len(test_normal)/200000
    inner_num = len(test_inner)/200000
    batch = np.zeros((batch_size, 1, 100, 100))
    batch = np.array(batch, dtype=xp.float32)
    output = np.zeros((batch_size))
    output = np.array(output, dtype=xp.int32)
    for i in range(batch_size):
        if i%2 == 0:
            rand_num = random.randint(0, normal_num - 1)
            rand_index = random.randint(0, 190000)
            image = dataset_normal[(rand_num*200000 + rand_index) : (rand_num*200000 + rand_index + 10000)] / calRms(dataset_normal[(rand_num*200000 + rand_index) : (rand_num*200000 + rand_index + 10000)])
            batch[i, 0, :, :] = image.reshape(100, 100)
            output[i] = 0
        else:
            rand_num = random.randint(0, inner_num - 1)
            rand_index = random.randint(0, 190000)
            image = dataset_inner[(rand_num*200000 + rand_index) : (rand_num*200000 + rand_index + 10000)] / calRms(dataset_inner[(rand_num*200000 + rand_index) : (rand_num*200000 + rand_index + 10000)])
            batch[i, 0, :, :] = image.reshape(100, 100)
            output[i] = 1
    return batch, output
 
    
    
"""
model = chainer.FunctionSet(conv1=F.Convolution2D(1, 50, 10, stride=2),
                            conv2=F.Convolution2D(50, 100, 8),
                            conv3=F.Convolution2D(100, 200, 5),
                            conv4=F.Convolution2D(200, 500, 2),
                            l1=F.Linear(500, 128),
                            l2=F.Linear(128, 32),
                            l3=F.Linear(32, 2))

"""
model = chainer.FunctionSet(conv1=F.Convolution2D(1, 16, 11),
                            conv2=F.Convolution2D(16, 32, 10),
                            conv3=F.Convolution2D(32, 64, 9),
                            conv4=F.Convolution2D(64, 128, 5),
                            l1=F.Linear(128, 32),
                            l2=F.Linear(32, 2))


if gpu_flag >= 0:
    model.to_gpu(gpu_flag)

for param in model.params():
    data = param.data
    data[:] = xp.random.uniform(-0.1, 0.1, data.shape)
    

optimizer = optimizers.RMSpropGraves()
optimizer.setup(model)
          
accMatrix = xp.zeros((TRAINING_EPOCHS))
sumsMatrix = xp.zeros((TRAINING_EPOCHS))
"""
for epoch in range(TRAINING_EPOCHS):
    imageBatch, output = make_batch(MINI_BATCH_SIZE)
    if gpu_flag >= 0:
        imageBatch = cuda.to_gpu(imageBatch)
        output = cuda.to_gpu(output)
    optimizer.zero_grads()
    x = chainer.Variable(imageBatch)
    t = chainer.Variable(output)
    x = F.relu(model.conv1(x))
    x = F.max_pooling_2d(x, ksize=2, cover_all=True)
    x = F.relu(model.conv2(x))
    x = F.max_pooling_2d(x, ksize=2, cover_all=True)
    x = F.relu(model.conv3(x))
    x = F.max_pooling_2d(x, ksize=2, cover_all=True)
    x = F.relu(model.conv4(x))
    x = F.relu(model.l1(x))
    x = model.l2(F.dropout(x, ratio=dropout_ratio, train=True))
    y = model.l3(F.dropout(x, ratio=dropout_ratio, train=True))
    loss = F.softmax_cross_entropy(y, t)
    loss.backward()
    optimizer.update()
    accMatrix[epoch] = F.accuracy(y, t).data
    if (epoch%100 == 0):
        print(
            "Fin[{j}]training loss:\t{i} \t acc = {k}".format(
                j=epoch, 
                i=loss.data/(imageBatch.shape[0] - 1),
                k=F.accuracy(y, t).data
            )
        )
        
    if (epoch%5000 == 0):
        fn = 'latestmodel.pkl'
        pickle.dump(model, open(fn, 'wb'))
            
"""
for epoch in range(TRAINING_EPOCHS):
    imageBatch, output = make_batch(MINI_BATCH_SIZE)
    if gpu_flag >= 0:
        imageBatch = cuda.to_gpu(imageBatch)
        output = cuda.to_gpu(output)
    optimizer.zero_grads()
    x = chainer.Variable(imageBatch)
    t = chainer.Variable(output)
    x = F.relu(model.conv1(x))
    x = F.max_pooling_2d(x, ksize=2)
    x = F.relu(model.conv2(x))
    x = F.max_pooling_2d(x, ksize=2)
    x = F.relu(model.conv3(x))
    x = F.max_pooling_2d(x, ksize=2)
    x = F.relu(model.conv4(x))
    x = F.relu(model.l1(x))
    y = model.l2(F.dropout(x, ratio=dropout_ratio, train=True))
    loss = F.softmax_cross_entropy(y, t)
    loss.backward()
    optimizer.update()
    accMatrix[epoch] = F.accuracy(y, t).data
    
    if (epoch%100 == 0):
        print(
            "Fin[{j}]training loss:\t{i} \t acc = {k}".format(
                j=epoch, 
                i=loss.data/(imageBatch.shape[0] - 1),
                k=F.accuracy(y, t).data
            )
        )
    imageBatch, output = make_test_batch(MINI_BATCH_SIZE)
    if gpu_flag >= 0:
        imageBatch = cuda.to_gpu(imageBatch)
        output = cuda.to_gpu(output)
    x = chainer.Variable(imageBatch)
    t = chainer.Variable(output)
    x = F.relu(model.conv1(x))
    x = F.max_pooling_2d(x, ksize=2)
    x = F.relu(model.conv2(x))
    x = F.max_pooling_2d(x, ksize=2)
    x = F.relu(model.conv3(x))
    x = F.max_pooling_2d(x, ksize=2)
    x = F.relu(model.conv4(x))
    x = F.relu(model.l1(x))
    y = model.l2(F.dropout(x, ratio=dropout_ratio, train=False))
    sumsMatrix[epoch] = F.accuracy(y, t).data
    if (epoch%100 == 0):
        print(F.accuracy(y, t).data)
    

accMatrix = chainer.cuda.to_cpu(accMatrix)
sumsMatrix = chainer.cuda.to_cpu(sumsMatrix)
pickle.dump(model, open('latestmodel.pkl', 'wb'))
pickle.dump(accMatrix, open('accMatrix.pkl', 'wb'))
pickle.dump(sumsMatrix, open('sumsMatrix.pkl', 'wb'))

