#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: chengdicao
"""

import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing import sequence


def load_feature(data_dir, song_id, version_id, feature_len):
    feature = np.load(os.path.join(data_dir, '%d_%d.npy' %(song_id, version_id)))
    feature = np.transpose(feature)
    feature = np.concatenate((feature, feature[0:11,:]),axis=0)
    feature = sequence.pad_sequences(feature, maxlen=feature_len, padding='post',truncating='post', dtype='float')
    
    return feature[np.newaxis,:,:,np.newaxis]


def load_list(data_dir, ls, feature_len):
    feature = []
    label = []
    
    print('loading data list %s...' %ls)
    with open(ls, 'r')as f:
        lines = f.readlines()
    for line in tqdm(lines):
        line = line.strip().split()        
        feature.append(load_feature(data_dir, int(line[0]), int(line[1]), feature_len))
        label.append(int(line[0]))
    print('loading data list %s... done!')
    
    return feature, label


def cosine_decay_lr(epoch, lr, max_epoch):
    return lr * 0.5 * (1 + np.cos(np.pi * epoch / max_epoch))


def get_label_matrix(label):
    num = len(label)
    
    label_matrix = np.zeros((num, num))
    a = 0
    b = 1
    for i in range(num):
        if i == num-1 or label[i] != label[i+1]:
            label_matrix[a:a+b, a:a+b] += 1
            for j in range(a, a+b):
                label_matrix[j][j] = 0
            a += b
            b = 1
        else:
            b += 1

    return label_matrix


def get_cosine_distance_matrix(embedding):
    num = embedding.shape[0]
    distance_matrix = 1 - embedding.dot(embedding.T)
    for i in range(num):
        distance_matrix[i,i] = 1
    return distance_matrix


def get_rank_matrix(distance_matrix):        
    return distance_matrix.argsort(axis=-1)