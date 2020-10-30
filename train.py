#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: chengdicao
"""

import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm

import argparse
from lib.utils import load_list, cosine_decay_lr, get_label_matrix, get_cosine_distance_matrix, get_rank_matrix
from lib.metrics import mean_average_precison, mean_precision_top_n, mean_rank_n
from lib.models import _set_class_num, _set_regularizer, get_model
from lib.generators import BasicGenerator


if __name__ == '__main__':
    # Step 0: Configure arguments
    parser = argparse.ArgumentParser(description='Training codes for cover song identification using Keras')

    parser.add_argument('--tag', type=str, default='test',
                        help='tag for this experiment')

    parser.add_argument('--block', default='wider',
                        help='building block: simple / bottleneck / wider')

    # Argument for dataset
    parser.add_argument('--data-dir', type=str, default='/data/youtube_hpcp_npy',
                        help='directory of dataset')

    parser.add_argument('--train-ls', type=str, default='meta/SHS100K-TRAIN',
                        help='list of training set')

    parser.add_argument('--val-ls', type=str, default='meta/SHS100K-VAL',
                        help='list of validation set')

    parser.add_argument('--feature-len', type=int, default=400,
                        help='length of input feature')

    # Argument for model
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='checkpoint directory')

    parser.add_argument('--regularize', type=float, default=0.0001,
                        help='value of l2-regularization')

    parser.add_argument('--time-field', type=int, default=48,
                        help='temporal reception field of KINet')

    parser.add_argument('--ki-block-num', type=int, default=4,
                        help='number of key-invariant blocks')

    parser.add_argument('--ki-out-channel', type=int, default=256,
                        help='output channel of key-invariant blocks')

    parser.add_argument('--bn-ratio', type=int, default=4,
                        help='squeeze ratio of bottleneck blocks')

    parser.add_argument('--no-chnlatt', action='store_true',
                        help='disable channel attention')

    parser.add_argument('--no-tempatt', action='store_true',
                        help='disable temporal attention')

    parser.add_argument('--attention-ratio', type=int, default=4,
                        help='squeeze ratio of attention modules')

    parser.add_argument('--embedding-len', type=int, default=128,
                        help='length of final music embedding')

    # Argument for training

    parser.add_argument('--batchsize', type=int, default=32,
                        help='batch size for training')

    parser.add_argument('--max-epoch', type=int, default=100,
                        help='max number of training epochs')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')

    parser.add_argument('--gpu', type=str, default='0',
                        help='id of GPU to use')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

    # Step 1: prepare data
    train_generator = BasicGenerator(args.data_dir, args.train_ls, args.feature_len, args.batchsize)
    val_feature, val_label= load_list(args.data_dir, args.val_ls, args.feature_len)
    val_label_matrix = get_label_matrix(val_label)
    _set_class_num(train_generator.get_class_num())

    # Step 2: define model
    if args.checkpoint:
        model_all = tf.keras.models.load_model(args.checkpoint)
        embedding = model_all.get_layer('Embedding_LReLU').output
        embedding = tf.keras.layers.Lambda(lambda x:tf.keras.backend.l2_normalize(x, axis=1),
                                           name='Embedding_L2Norm')(embedding)
        model_embedding = tf.keras.models.Model(inputs=model_all.input, outputs=embedding)
        model_embedding.compile(optimizer='adam', loss=tf_batch_all_loss)  
    else:
        _set_regularizer(args.regularize)
        input_tensor = tf.keras.Input(shape=(23, args.feature_len, 1), name='Feature')
        model_embedding, model_all = get_model(input_tensor, args)
    
    model_embedding.summary()
    
    # Step 3: define file writer
    model_dir = 'models/%s' %args.tag
    log_dir = 'log/%s' %args.tag
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    writer = tf.summary.FileWriter(log_dir, sess.graph)
    train_lr = tf.placeholder(tf.float32, [])
    train_loss = tf.placeholder(tf.float32, [])
    train_acc = tf.placeholder(tf.float32, [])
    val_mAP = tf.placeholder(tf.float32, [])
    val_TOP10 = tf.placeholder(tf.float32, [])
    val_MR1 = tf.placeholder(tf.float32, [])
    tf.summary.scalar('train_loss', train_loss)
    tf.summary.scalar('train_acc', train_acc)
    tf.summary.scalar('train_lr', train_lr)
    tf.summary.scalar('val_mAP', val_mAP)
    tf.summary.scalar('val_TOP10', val_TOP10)
    tf.summary.scalar('val_MR1', val_MR1)

    merged=tf.summary.merge_all()

    # Step 4: train model
    best_loss = float('inf')
    best_acc = 0
    best_mAP = 0
    best_TOP10 = 0
    best_MR1 = float('inf')

    for epoch in range(args.max_epoch):
        this_lr = cosine_decay_lr(epoch, args.lr, args.max_epoch)
        tf.keras.backend.set_value(model_all.optimizer.lr, this_lr)
        
        history = model_all.fit_generator(train_generator, epochs=epoch+1, initial_epoch=epoch, verbose=1)
        this_loss = history.history['loss'][-1]
        this_acc = history.history['acc'][-1]
        
        val_embedding = []
        for feature in tqdm(val_feature):
            val_embedding.append(model_embedding.predict(feature, verbose=0))
        val_embedding = np.vstack(val_embedding)
        val_distance_matrix = get_cosine_distance_matrix(val_embedding)
        val_rank_matrix = get_rank_matrix(val_distance_matrix)
        this_mAP = mean_average_precison(val_label_matrix, val_rank_matrix)
        this_TOP10 = mean_precision_top_n(val_label_matrix, val_rank_matrix, n=10)
        this_MR1 = mean_rank_n(val_label_matrix, val_rank_matrix, n=1)

        improve = False
        if this_loss < best_loss:
            best_loss = this_loss
            improve = True
        if this_acc < best_acc:
            best_acc = this_acc
            improve = True
        if this_mAP < best_mAP:
            best_mAP = this_mAP
            improve = True
        if this_TOP10 < best_TOP10:
            best_TOP10 = this_TOP10
            improve = True
        if this_MR1 < best_MR1:
            best_MR1 = this_MR1
            improve = True

        print('loss:%.4f\tacc:%.4f\tmAP:%.4f\tTOP10:%.4f\tMR1:%.2f' %(this_loss, this_acc, this_mAP, this_TOP10, this_MR1))
        summary = sess.run(merged,
                           feed_dict={train_lr:this_lr,
                                      train_loss:this_loss,
                                      train_acc:this_acc,
                                      val_mAP:this_mAP,
                                      val_TOP10:this_TOP10,
                                      val_MR1:this_MR1
                                      })
        writer.add_summary(summary, epoch)
        if improve:
            model_all.save(os.path.join(model_dir, 'model%d-%.4f-%.4f-%.4f-%.4f-%.2f.h5' %(epoch+1, this_loss, this_acc, this_mAP, this_TOP10, this_MR1)))
