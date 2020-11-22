#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   utils.py
@Contact :   liangyacongyi@163.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/3/15 09:10 AM   liangcong    1.0
"""
import tensorflow as tf
import numpy as np
from numpy import *


def sim_dis_tensorflow(x, y):
    """
    calculate the distance of two matrices along the second dimension using tensorflow
    :param x: matrix x
    :param y: matrix y
    :return: matrix D
    """
    y_G = tf.matmul(y, tf.transpose(y, [1, 0]))
    y_H = tf.tile(tf.expand_dims(tf.diag_part(y_G), 0), (tf.shape(x)[0], 1))
    x_G = tf.matmul(x, tf.transpose(x, [1, 0]))
    x_H = tf.tile(tf.expand_dims(tf.diag_part(x_G), 0), (tf.shape(y)[0], 1))
    R = tf.matmul(x, tf.transpose(y, [1, 0]))
    D = y_H + tf.transpose(x_H, [1, 0]) - 2*R
    return 0.5*D


def _zca(X, U_matrix=None, S_matrix=None, mu=None, flag='train', alpha=1e-5):
    """
    preprocess image dataset by zca
    :param X: image dataset, format: n*d, n: total samples, d: dimentions
    :return:
    """
    if flag == 'train':
        # mu = np.mean(X, axis=0)
        # X = X - mu
        cov = np.cov(X.T)
        U, S, V = np.linalg.svd(cov)
    else:
        # X = X - mu
        U = U_matrix
        S = S_matrix
    x_rot = np.dot(X, U)
    pca_whiten = x_rot / np.sqrt(S + alpha)
    zca_whiten = np.dot(pca_whiten, U.T)
    return zca_whiten, U, S, mu


def _gcn(X, flag=0, scale=55.):
    """
    preprocess image dataset by gcn
    :param X: image dataset, format: n*d, n: total samples, d: dimentions
    :return: the dataset after preprocessing by gcn
    """
    if flag == 0:
        print("1")
        mean = np.mean(X, axis=1)
        X = X - mean[:, np.newaxis]
        contrast = np.sqrt(10. + (X**2).sum(axis=1)) / scale
        contrast[contrast < 1e-8] = 1.
        X = X / contrast[:, np.newaxis]
    else:
        print("3")
        X = X.reshape([-1, 32, 32, 3])
        mu = np.mean(X, axis=(1, 2)).reshape([-1, 1, 1, 3])
        std = np.std(X, axis=(1, 2)).reshape([-1, 1, 1, 3])
        X = X - mu
        X = X / std
        X = X.reshape([-1, 3072])
    return X


def cal_orthogonal_basis_np(x1, x2):
    """
    calculate orthogonal bases matrix given 2 pair matrix
    :param x1: first vector matrix
    :param x2: second vector matrix
    :return: orthogonal bases matrix
    """
    v1 = x1
    v2 = x2 - np.multiply(np.tile(np.sum(np.multiply(x2, v1), axis=1, keepdims=True) /
                                  np.sum(np.square(v1[0])), [1, x1.shape[0]]), v1)
    return v1, v2


def cal_orthogonal_basis_tf(x1, x2):
    """
    calculate orthogonal bases matrix given 2 pair matrices with tensorflow
    :param x1: first vector matrix
    :param x2: second vector matrix
    :return: orthogonal bases matrix
    """
    _v1 = tf.tile(tf.expand_dims(x1, 0), multiples=[tf.shape(x2)[0], 1])
    _v2 = x2 - tf.multiply(
        tf.tile(tf.reduce_sum(tf.multiply(x2, _v1), axis=1, keep_dims=True) /
                tf.reduce_sum(tf.square(_v1[0])), multiples=[1, tf.shape(x2)[1]]), _v1)
    return _v1, _v2


def secondary_optimal_feature_plane_loss(output, _label, _vec, num, flag=True):
    """
    calculate the loss between output in feature space and its corresponding optimal feature plane
    :param output: output of training data in feature space
    :param _label: ground truth
    :param _vec: fixed vector [1., 1., ..., 1.]
    :param num: top num feature point to calculate s-ofp loss
    :param flag: ce loss or with s-ofp loss, True: ce loss, False: with s-ofp loss
    :return: secondary optimal feature plane(ofp) loss
    """
    v1, v2 = cal_orthogonal_basis_tf(_vec, _label)
    p1 = tf.multiply(tf.tile(tf.reduce_sum(tf.multiply(output, v1), axis=1, keep_dims=True) /
                             tf.reduce_sum(tf.square(v1[0])), multiples=[1, tf.shape(v1)[1]]), v1)
    p2 = tf.multiply(tf.tile(tf.reduce_sum(tf.multiply(output, v2), axis=1, keep_dims=True) /
                             tf.reduce_sum(tf.square(v2), axis=1, keep_dims=True), multiples=[1, tf.shape(v2)[1]]), v2)
    pro = p1 + p2
    if flag:
        _ofp_loss = tf.reduce_mean(tf.reduce_sum(tf.square(output - pro), axis=1)) / 2
        _loss = _ofp_loss
    else:
        _ofp_loss = tf.reduce_sum(tf.square(output - pro), axis=1)
        temp_loss_m, _key = tf.nn.top_k(_ofp_loss, k=num)
        _loss = tf.reduce_mean(temp_loss_m) / 2
    return _loss




