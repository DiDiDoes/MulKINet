#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: chengdicao
"""

import tensorflow.keras as keras
import numpy as np
import os
from tqdm import tqdm
from tensorflow.keras.preprocessing import sequence


class BasicGenerator(keras.utils.Sequence):
    def __init__(self, data_dir, ls, feature_len, batch_size=64, shuffle=True):
        self.data_dir = data_dir
        self.ls = ls
        self.feature_len = feature_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.prepare()
        self.on_epoch_end()

    def prepare(self):
        print('preparing data list %s...' %self.ls)
        self.song = []
        self.version = []

        with open(self.ls, 'r')as f:
            lines = f.readlines()
        for line in tqdm(lines):
            line = line.strip().split()
            self.song.append(int(line[0]))
            self.version.append(int(line[1]))
        self.label = keras.utils.to_categorical(self.song)
        print('preparing data in %s... done!' %self.ls)
        
    def get_class_num(self):
        return len(np.unique(self.song))

    def load_feature(self, song_id, version_id):
        feature = np.load(os.path.join(self.data_dir, '%d_%d.npy' %(song_id, version_id)))
        feature = np.transpose(feature)

        actual_len = feature.shape[1]
        if actual_len <= self.feature_len:
            feature = sequence.pad_sequences(feature, maxlen=self.feature_len, padding='post',truncating='post', dtype='float')
        else:
            begin = np.random.randint(0, actual_len - self.feature_len)
            feature = feature[:, begin:begin+self.feature_len]
        
        feature = np.concatenate((feature, feature[0:11,:]),axis=0)
        
        return feature[:,:,np.newaxis]

    def __len__(self):
        return int(np.floor(len(self.song) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_song = self.song[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_version = self.version[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_label = self.label[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_feature = [self.load_feature(batch_song[i], batch_version[i])
                         for i in range(self.batch_size)]

        return np.stack(batch_feature), np.stack(batch_label)
   
    def on_epoch_end(self):
        if self.shuffle:
            indices = np.arange(len(self.song))
            np.random.shuffle(indices)
            self.song = [self.song[i] for i in indices]
            self.version = [self.version[i] for i in indices]
            self.label = [self.label[i] for i in indices]
            # print('dataset shuffled.')