# -*- coding: utf-8 -*-

import pickle
import numpy as np

dataset = []

for i in range(6):
    fname = 'no-dr-inner-vib' + str(i+1) + '.pkl'
    dataset_add = pickle.load(open(fname, 'rb'))
    if len(dataset) == 0:
        dataset = dataset_add[200000:len(dataset_add)-400000]
    else:
        dataset = np.r_[dataset, dataset_add[200000:len(dataset_add)-400000]]
        
pickle.dump(dataset, open('no-dr-inner-vib.pkl', 'wb'))