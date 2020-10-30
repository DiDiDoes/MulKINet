#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: chengdicao
"""

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K

# global hyperparameters
_regularizer = None
_class_num = None


def _set_class_num(class_num):
    global _class_num
    _class_num = class_num


def _set_regularizer(value):
    #pass
    global _regularizer
    _regularizer = tf.keras.regularizers.l2(value)


def basic_attention(x, attention_ratio, name):
    # A modified version of attention module
    # Reference: CBAM: Convolutional Block Attention Module

    input_channel = K.int_shape(x)[-1]

    # squeeze the input
    avgpool = layers.GlobalAveragePooling2D(name='%s_AvgPool' %name)(x)
    maxpool = layers.GlobalMaxPooling2D(name='%s_MaxPool' %name)(x)
    
    # shared layer
    attention_layer1 = layers.Dense(input_channel//attention_ratio, kernel_regularizer=_regularizer,
                                    use_bias=False, name='%s_Layer1' %name)
    attention_leakyrelu = layers.LeakyReLU(name='%s_LeakyReLU' %name)
    attention_layer2 = layers.Dense(input_channel, kernel_regularizer=_regularizer, use_bias=False,
                                    name='%s_Layer2' %name)
    
    # process the squeezed input
    avgpool = attention_layer1(avgpool)
    avgpool = attention_leakyrelu(avgpool)
    avgpool = attention_layer2(avgpool)

    maxpool = attention_layer1(maxpool)
    maxpool = attention_leakyrelu(maxpool)
    maxpool = attention_layer2(maxpool)

    # calculate the attention weights
    weight = layers.Add(name='%s_Add' %name)([avgpool, maxpool])
    weight = layers.Activation('sigmoid', name='%s_Sigmoid' %name)(weight)
    weight = layers.Reshape((1, 1, input_channel), name='%s_Reshape' %name)(weight)

    y = layers.Multiply(name='%s_Out' %name)([x, weight])

    return y


def channel_attention(x, attention_ratio, name):
    return basic_attention(x, attention_ratio, name='%s_ChnlAtt' %name)


def temporal_attention(x, attention_ratio, name):
    # different from channel attention,
    # two permutation will be needed
    x = layers.Permute((1, 3, 2), name='%s_TempAtt_InPermute' %name)(x)
    x = basic_attention(x, attention_ratio, name='%s_TempAtt' %name)
    x = layers.Permute((1, 3, 2), name='%s_TempAtt_OutPermute' %name)(x)

    return x


def build_conv_layer(x, channel, kernel_size, name,
                     dilation_rate=(1, 1),
                     tempatt=False, chnlatt=False,
                     attention_ratio=None):
    # Conv + BatchNorm + LeakyReLU

    x = layers.Conv2D(channel, kernel_size, strides=1, padding='valid', use_bias=False,
                      kernel_regularizer=_regularizer, name='%s_Conv' %name, dilation_rate=dilation_rate)(x)

    x = layers.BatchNormalization(name='%s_BN' %name)(x)

    # attention module
    if chnlatt:
        x = channel_attention(x, attention_ratio, name)
    if tempatt:
        x = temporal_attention(x, attention_ratio, name)

    x = layers.LeakyReLU(name='%s_LeakyReLU' %name)(x)

    return x


def build_bottleneck_layer(input_tensor,
                           out_channel, bottleneck_channel,
                           kernel_size,
                           name,
                           chnlatt=False, tempatt=False,
                           attention_ratio=None):

    x = build_conv_layer(input_tensor, bottleneck_channel, (1,1), name='%s_0' %name)
    x = build_conv_layer(x, bottleneck_channel, kernel_size, name='%s_1' %name,
                         chnlatt=chnlatt, tempatt=tempatt, attention_ratio=attention_ratio)
    x = build_conv_layer(x, out_channel, (1,1), name='%s_2' %name)

    return x


def build_wider_bottleneck_layer(input_tensor,
                                 out_channel, bottleneck_channel,
                                 kernel_size,
                                 name,
                                 chnlatt=False, tempatt=False,
                                 attention_ratio=None):

    x = build_conv_layer(input_tensor, out_channel, kernel_size, name='%s_0' %name,
                         chnlatt=chnlatt, tempatt=tempatt, attention_ratio=attention_ratio)
    x = build_conv_layer(x, bottleneck_channel, (1,1), name='%s_1' %name)

    return x


def build_key_invariant_layer(x, args):
    if args.ki_block_num not in [1, 2, 3, 4, 6, 12]:
        raise Exception('ki_block_num %d is not a divisor of 12.' %args.ki_block_num)
    
    freq_kernel_size = 12 // args.ki_block_num
    time_kernel_size = args.time_field // args.ki_block_num

    for block_id in range(args.ki_block_num):
        if block_id == 1:
            freq_kernel_size += 1
            time_kernel_size += 1

        x = build_conv_layer(x,
                             channel=args.ki_out_channel,
                             kernel_size=(freq_kernel_size, time_kernel_size),
                             name='KI_%d' %block_id,
                             chnlatt=False if args.no_chnlatt else True,
                             tempatt=False if args.no_tempatt else True,
                             attention_ratio=args.attention_ratio)

    x = layers.MaxPooling2D((12, 1), strides=1, name='KI_MaxPool')(x)
    return x


def build_key_invariant_bottleneck_layer(x, args):
    if args.ki_block_num not in [1, 2, 3, 4, 6, 12]:
        raise Exception('ki_block_num %d is not a divisor of 12.' %args.ki_block_num)
    
    freq_kernel_size = 12 // args.ki_block_num
    time_kernel_size = args.time_field // args.ki_block_num

    for block_id in range(args.ki_block_num):
        if block_id == 1:
            freq_kernel_size += 1
            time_kernel_size += 1

        if block_id == 0:
            x = build_conv_layer(x,
                                 channel=args.ki_out_channel,
                                 kernel_size=(freq_kernel_size, time_kernel_size),
                                 name='KI_%d' %block_id,
                                 chnlatt=False if args.no_chnlatt else True,
                                 tempatt=False if args.no_tempatt else True,
                                 attention_ratio=args.attention_ratio)
        else:
            x = build_bottleneck_layer(x,
                                       out_channel = args.ki_out_channel,
                                       bottleneck_channel = args.ki_out_channel // args.bn_ratio,
                                       kernel_size = (freq_kernel_size, time_kernel_size),
                                       name = 'KI_Bottleneck%s' %block_id,
                                       chnlatt=False if args.no_chnlatt else True,
                                       tempatt=False if args.no_tempatt else True,
                                       attention_ratio=args.attention_ratio)
        
    y = layers.MaxPooling2D((12, 1), strides=1, name='KICNN_MaxPool')(x)
    return y


def build_key_invariant_wider_bottleneck_layer(x, args):
    if args.ki_block_num not in [1, 2, 3, 4, 6, 12]:
        raise Exception('ki_block_num %d is not a divisor of 12.' %args.ki_block_num)
    
    freq_kernel_size = 12 // args.ki_block_num
    time_kernel_size = args.time_field // args.ki_block_num

    for block_id in range(args.ki_block_num):
        if block_id == 1:
            freq_kernel_size += 1
            time_kernel_size += 1

        if block_id == args.ki_block_num-1:
            x = build_conv_layer(x,
                                 channel=args.ki_out_channel,
                                 kernel_size=(freq_kernel_size, time_kernel_size),
                                 name='KI_%d' %block_id,
                                 chnlatt=False if args.no_chnlatt else True,
                                 tempatt=False if args.no_tempatt else True,
                                 attention_ratio=args.attention_ratio)
        else:
            x = build_wider_bottleneck_layer(x,
                                             out_channel = args.ki_out_channel,
                                             bottleneck_channel = args.ki_out_channel // args.bn_ratio,
                                             kernel_size = (freq_kernel_size, time_kernel_size),
                                             name = 'KI_Wider_Bottleneck%s' %block_id,
                                             chnlatt=False if args.no_chnlatt else True,
                                             tempatt=False if args.no_tempatt else True,
                                             attention_ratio=args.attention_ratio)
        
    y = layers.MaxPooling2D((12, 1), strides=1, name='KICNN_MaxPool')(x)
    return y


def build_classification_layer(x):
    x = layers.Dropout(0.5, name='Dropout')(x)
    return layers.Dense(_class_num, use_bias=False, kernel_regularizer=_regularizer,
                        activation='softmax', name='Classification')(x)


def build_embedding_model(input_tensor, args):

    # key-invariant CNN
    if args.block == 'simple':
        x = build_key_invariant_layer(input_tensor, args)
    elif args.block == 'bottleneck':
        x = build_key_invariant_bottleneck_layer(input_tensor, args)
    elif args.block == 'wider':
        x = build_key_invariant_wider_bottleneck_layer(input_tensor, args)
    else:
        raise Exception('Building block type not recognized.')
    
    # summarize the temporal axis
    x = build_conv_layer(x, args.ki_out_channel, (1,10), name='Temporal_0', dilation_rate=(1, 3),
                         chnlatt=False if args.no_chnlatt else True,
                         tempatt=False if args.no_tempatt else True,
                         attention_ratio=args.attention_ratio)
    x = layers.MaxPooling2D((1, 3), strides=(1, 2), name='Tempora_MaxPool_0')(x)
    
    x = build_conv_layer(x, args.ki_out_channel, (1,10), name='Temporal_1', dilation_rate=(1, 3),
                         chnlatt=False if args.no_chnlatt else True,
                         tempatt=False if args.no_tempatt else True,
                         attention_ratio=args.attention_ratio)
    x = layers.MaxPooling2D((1, 3), strides=(1, 2), name='Tempora_MaxPool_1')(x)
    
    x = layers.GlobalAveragePooling2D(name='Temporal_Summary')(x)

    # generate embedding
    embedding = layers.Dense(args.embedding_len, use_bias=False, kernel_regularizer=_regularizer, name='Embedding')(x)
    embedding = layers.BatchNormalization(name='Embedding_BN')(embedding)
    embedding = layers.LeakyReLU(name='Embedding_LeakyReLU')(embedding)

    return embedding


def get_model(input_tensor, args):
    
    embedding = build_embedding_model(input_tensor, args)

    embedding_norm = layers.Lambda(lambda x:K.l2_normalize(x, axis=1),
                                   name='Embedding_L2Norm')(embedding)
    model_embedding = tf.keras.models.Model(input_tensor, embedding_norm)
    
    # Classification layer
    y = build_classification_layer(embedding)
    
    model_all = tf.keras.models.Model(input_tensor, y)
    model_all.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    return model_embedding, model_all
