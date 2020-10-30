#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: chengdicao
"""

import os
import tensorflow as tf
import numpy as np
import argparse
from tqdm import tqdm

from lib.utils import load_list, get_label_matrix, get_cosine_distance_matrix, get_rank_matrix
from lib.metrics import mean_average_precison, mean_precision_top_n, mean_rank_n


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation codes for cover song identification using Keras')
    parser.add_argument('--data-dir', type=str, default='/data/youtube_hpcp_npy')
    parser.add_argument('--test-ls', type=str, default='meta/SHS100K-TEST')
    parser.add_argument('--model-file', type=str, default='models/test/modelXX.h5')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--feature-len', type=int, default=400)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

    model_all = tf.keras.models.load_model(args.model_file)
    embedding = model_all.get_layer('Embedding_LeakyReLU').output
    embedding_norm = tf.keras.layers.Lambda(lambda x:tf.keras.backend.l2_normalize(x, axis=1), name='Embedding_L2Norm')(embedding)
    model_rep = tf.keras.models.Model(inputs=model_all.input, outputs=embedding_norm)

    test_feature, test_label= load_list(args.data_dir, args.test_ls, args.feature_len)
    test_label_matrix = get_label_matrix(test_label)
    
    test_embedding = []
    for feature in tqdm(test_feature):
        test_embedding.append(model_rep.predict(feature, verbose=0))
    test_embedding = np.vstack(test_embedding)
    test_distance_matrix = get_cosine_distance_matrix(test_embedding)
    test_rank_matrix = get_rank_matrix(test_distance_matrix)
    this_mAP = mean_average_precison(test_label_matrix, test_rank_matrix)
    this_TOP10 = mean_precision_top_n(test_label_matrix, test_rank_matrix, n=10)
    this_MR1 = mean_rank_n(test_label_matrix, test_rank_matrix, n=1)
    
    print('mAP:%.4f\tTOP10:%.4f\tMR1:%.2f' %(this_mAP, this_TOP10, this_MR1))
    
