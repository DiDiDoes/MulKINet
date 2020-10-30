#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: chengdicao
"""

import numpy as np


def mean_average_precison(label_matrix, rank_matrix):
    num = label_matrix.shape[0]
    average_precisions=[]

    for i in range(num):
        true_positive=0
        true_positive_all = label_matrix[i].sum()
        precisions=[]
        for j in range(num):
            if label_matrix[i][rank_matrix[i, j]] == 1:
                true_positive += 1
                precisions.append(true_positive / (j+1))
                if true_positive == true_positive_all:
                    break
        average_precisions.append(np.mean(precisions))

    return np.mean(average_precisions)


def mean_precision_top_n(label_matrix, rank_matrix, n=10):
    num = label_matrix.shape[0]
    top_ns = []

    for i in range(num):
        top_n = label_matrix[i][rank_matrix[i, :n]].sum()
        top_ns.append(top_n)
    
    return np.mean(top_ns) / n


def mean_rank_n(label_matrix, rank_matrix, n=1):
    num = label_matrix.shape[0]
    rank_ns = []

    for i in range(num):
        true_positive = 0
        true_positive_all = label_matrix[i].sum()

        if true_positive_all < n:
            continue

        for j in range(num):
            if label_matrix[i][rank_matrix[i][j]] == 1:
                true_positive += 1
                if true_positive == n:
                    rank_ns.append(j+1)
                    break

    return np.mean(rank_ns)
