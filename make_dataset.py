# -*- coding: utf-8 -*-

import csv
import numpy as np
import pickle

dataset_sound = []
dataset_vib = []

for i in range(1,44):
    if i < 10:
        filename = 'kousaku\jikuuke\psf2017-08-24' + '\\A5052P-dr10-S2000-F025-Z05_0000000' + str(i) + '.csv'
    else:
        filename = 'kousaku\jikuuke\psf2017-08-24' + '\\A5052P-dr10-S2000-F025-Z05_000000' + str(i) + '.csv'
    file = open(filename, 'r')
    dataReader = csv.reader(file, delimiter='\t')
    sound = []
    vib = []
    for j in range(23):
        header = next(dataReader)
    for row in dataReader:
        if len(row) == 0:
            break
        sound.append(float(row[1]))
        vib.append(float(row[2]))
    if len(dataset_sound) == 0:
        dataset_sound = sound
        dataset_vib = vib
    else:
        dataset_sound = np.r_[dataset_sound,sound]
        dataset_vib = np.r_[dataset_vib,vib]
   
"""
pickle.dump(dataset_sound_new, open('A5052P-dr10-inner-sound6.pkl', 'wb'))
pickle.dump(dataset_vib_new, open('A5052P-dr10-inner-vib6.pkl', 'wb'))
"""