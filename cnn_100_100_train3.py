# -*- coding: utf-8 -*-

import numpy as np
import pickle
import math
from scipy.fftpack import fft
import random
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers

xp = np
PRE_TRAINING_EPOCHS = 1000
TRAINING_EPOCHS = 20000
MINI_BATCH_SIZE = 216
dropout_ratio = 0.1

# 学習データの読み込み
with open('train_dataset_0905.pkl', 'rb') as f:
    train_dataset = pickle.load(f)
with open('test_dataset_0905.pkl', 'rb') as f:
    test_dataset = pickle.load(f)
    
# バッチ作成
def make_batch(dataset, batchSize):
    batch = xp.zeros((batchSize, 1, 100, 100))
    batch = xp.array(batch, dtype=xp.float32)
    output = xp.zeros((batchSize))
    output = xp.array(output, dtype=xp.int32)
    for i in range(batchSize):
        index = random.randint(0, 9999)
        sample = dataset[index:index+10000, i]
        sample = np.array(sample, dtype=xp.float32)
        sample = sample.reshape(100, 100)
        batch[i, 0, :, :] = sample
        output[i] = i/72
    return batch, output

# バッチ作成(test用)
def make_batch_test(dataset, batchSize=24):
    batch = xp.zeros((batchSize, 1, 100, 100))
    batch = xp.array(batch, dtype=xp.float32)
    output = xp.zeros((batchSize))
    output = xp.array(output, dtype=xp.int32)
    for i in range(batchSize):
        index = random.randint(0, 9999)
        sample = dataset[index:index+10000, i]
        sample = np.array(sample, dtype=xp.float32)
        sample = sample.reshape(100, 100)
        batch[i, 0, :, :] = sample
        output[i] = i/8
    return batch, output

model = chainer.FunctionSet(conv1=F.Convolution2D(1, 50, 10, stride=2),
                            conv2=F.Convolution2D(50, 100, 8),
                            conv3=F.Convolution2D(100, 200, 5),
                            conv4=F.Convolution2D(200, 500, 2),
                            l1=F.Linear(500, 128),
                            l2=F.Linear(128, 32),
                            l3=F.Linear(32, 3))

for param in model.params():
    data = param.data
    data[:] = np.random.uniform(-0.1, 0.1, data.shape)
    

# model = pickle.load(open('model/cnn_100_100_model_3kind_all_drop0.05_0822_latest.pkl', 'rb'))
optimizer = optimizers.Adam()
optimizer.setup(model)

batch_test = []
for i in range(10):
    batch, output_test = make_batch_test(test_dataset)
    batch_test.append(batch)
           
sumsMatrix = xp.zeros((TRAINING_EPOCHS))
accMatrix = xp.zeros((TRAINING_EPOCHS))
# sumsMatrix = pickle.load(open('sumsMatrix_latest.pkl', 'rb'))
#ファインチューニング
for epoch in range(TRAINING_EPOCHS):
    imageBatch, output = make_batch(train_dataset, MINI_BATCH_SIZE)
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
    if (epoch%1 == 0):
        print(
            "Fin[{j}]training loss:\t{i} \t acc = {k}".format(
                j=epoch, 
                i=loss.data/(imageBatch.shape[0] - 1),
                k=F.accuracy(y, t).data
            )
        )
    if (epoch%100 == 0):
        pickle.dump(sumsMatrix, open('sumsMatrix_latest.pkl', 'wb'))
        
    if (epoch%1000 == 0):
        fn = 'model/latest.pkl'
        pickle.dump(model, open(fn, 'wb'))
    sums = 0
    for i in range(10):
        
        x = chainer.Variable(batch_test[i])
        t = chainer.Variable(output_test)
        x = F.relu(model.conv1(x))
        x = F.max_pooling_2d(x, ksize=2, cover_all=True)
        x = F.relu(model.conv2(x))
        x = F.max_pooling_2d(x, ksize=2, cover_all=True)
        x = F.relu(model.conv3(x))
        x = F.max_pooling_2d(x, ksize=2, cover_all=True)
        x = F.relu(model.conv4(x))
        x = F.relu(model.l1(x))
        x = model.l2(F.dropout(x, ratio=dropout_ratio, train=False))
        y = model.l3(F.dropout(x, ratio=dropout_ratio, train=False))
        acc = F.accuracy(y, t).data
        y = F.softmax(y).data
        sums += acc
    sumsMatrix[epoch] = sums * 10
    print(sums * 10)
    if (epoch%5000 == 0):
        fname_model = str(epoch) + 'epochs_cnn_100_100_model_3kind_smallchange_only_drop01.pkl'
        fname_sums = str(epoch) + 'epochs_sumsMatrix_smallchange_only_cnn_drop01.pkl'
        fname_acc = str(epoch) + 'epochs_accMatrix_smallchange_only_cnn_drop01.pkl'
        pickle.dump(model, open(fname_model, 'wb'))
        pickle.dump(sumsMatrix, open(fname_sums, 'wb'))
        pickle.dump(accMatrix, open(fname_acc, 'wb'))
            
pickle.dump(model, open('cnn_100_100_model_3kind_smallchange_only_drop01.pkl', 'wb'))
pickle.dump(sumsMatrix, open('sumsMatrix_smallchange_only_cnn_drop01.pkl', 'wb'))
pickle.dump(accMatrix, open('accMatrix_smallchange_only_cnn_drop01.pkl', 'wb'))


# テストデータの読み込み
"""
with open('test_dataset_0713.pkl', 'rb') as f:
    test_dataset = pickle.load(f)
    
batch, output = make_batch_test(test_dataset)
x = chainer.Variable(batch)
t = chainer.Variable(output)
x = F.relu(model.conv1(x))
x = F.max_pooling_2d(x, ksize=2, cover_all=False)
x = F.relu(model.conv2(x))
x = F.max_pooling_2d(x, ksize=2, cover_all=False)
x = F.relu(model.conv3(x))
x = F.relu(model.l1(x))
y = model.l2(F.dropout(x, ratio=dropout_ratio, train=True))
acc = F.accuracy(y, t).data
y = F.softmax(y).data
    
"""

