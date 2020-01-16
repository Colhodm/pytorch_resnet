import torch
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from Clustering import *

import random

import os

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def group_norm(input, group, running_mean, running_var, weight=None, bias=None,
               use_input_stats=True, momentum=0.1, eps=1e-5):
    r"""Applies Group Normalization for channels in the same group in each data sample in a
    batch.
    See :class:`~torch.nn.GroupNorm1d`, :class:`~torch.nn.GroupNorm2d`,
    :class:`~torch.nn.GroupNorm3d` for details.
    """
    if not use_input_stats and (running_mean is None or running_var is None):
        raise ValueError('Expected running_mean and running_var to be not None when use_input_stats=False')

    b, c = input.size(0), input.size(1)
    if weight is not None:
        weight = weight.repeat(b)
    if bias is not None:
        bias = bias.repeat(b)

    def _instance_norm(input, group, running_mean=None, running_var=None, weight=None,
                       bias=None, use_input_stats=None, momentum=None, eps=None):
        # Repeat stored stats and affine transform params if necessary
        if running_mean is not None:
            running_mean_orig = running_mean
            running_mean = running_mean_orig.repeat(b)
        if running_var is not None:
            running_var_orig = running_var
            running_var = running_var_orig.repeat(b)
        eps = 1e-5
        G = group
        out_final = input
        x_np = input
        # NO PYTORCH EQUIVALENT YET WE CAN USEhttps://discuss.pytorch.org/t/equivalent-to-numpys-nan-to-num/52448
        #x_np = np.nan_to_num(x_np)
        # At this point we have just stored our input on CPU with Numpy
        Nabs, C, H, W = x_np.size()
        N = Nabs
        # N refers to the __? What the batch size?
        # res_x is the original shape of our input
        res_x = torch.zeros((N, C, H, W))
        # this is a looser form where we are just by channel dimensions
        x_np_new = torch.zeros((C,2))
        # we loop through the channels
        #TODO DOUBLE CHECK THIS IS RIGHT!!!
        temp = torch.reshape(x_np, (C, N * H * W))
        img = torch.zeros((C, 2))
        img[:, 0], img[:, 1]= torch.std_mean(temp, dim=1)
        # TODO we could use that other function to replace
        #x_np_new = np.nan_to_num(x_np_new,posinf=0,neginf=0)
        Data = data_preparation(n_cluster=G, data=img[:, :])
        # Up to here we are all correct, as long as we are clustering in the right basis
        count = 0
        Common_mul = 1
        Common_add = 0
        return input
    return _instance_norm(input, group, running_mean=running_mean,
                          running_var=running_var, weight=weight, bias=bias,
                          use_input_stats=use_input_stats, momentum=momentum,
                          eps=eps)


class HoogiNorm(_BatchNorm):
    def __init__(self, num_features, num_groups=5, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=False):
        self.num_groups = num_groups
        self.track_running_stats = track_running_stats
        super(HoogiNorm, self).__init__(int(num_features / num_groups), eps,
                                        momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        return NotImplemented

    def forward(self, input):
        self._check_input_dim(input)

        return group_norm(
            input, self.num_groups, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats, self.momentum, self.eps)

# Deprecated Method of Normalization
def Stat(IN):
    tmp = np.zeros((IN.shape[0]))
    tmp2 = 0
    eps = 1e-5
    sigma = np.zeros((IN.shape[0], 1))
    out = IN
    # print('IN is:')
    # print(IN.shape)
    '''for i in range(IN.shape[1]):
        tmp += np.sum(IN[:,i,:,:])'''
    for i in range(IN.shape[0]):
        # for j in range(IN.shape[1]):
        tmp[i] += np.sum(IN[i, :, :, :])

    # tmp = np.sum(IN, where=[False, True, True, True])
    mu = (1 / (IN.shape[1] * IN.shape[2] * IN.shape[3])) * tmp
    for i in range(IN.shape[0]):
        sigma[i] = np.sqrt(
            (1 / (IN.shape[1] * IN.shape[2] * IN.shape[3])) * (np.sum((IN[i, :, :, :] - mu[i]) ** 2) + eps))
    # sigma = np.sqrt((1 / (IN.shape[0] * IN.shape[2] * IN.shape[3])) * tmp2)
    for i in range(IN.shape[0]):
        for j in range(IN.shape[1]):
            out[i, j, :, :] = (1 / sigma[i]) * (IN[i, j, :, :] - mu[i])
    print(out)
    return out

# Current Method of Normalization
def Stat_torch(IN):
    tmp = torch.zeros((IN.shape[0]))
    tmp2 = 0
    eps = 1e-5
    sigma = torch.zeros((IN.shape[0], 1))
    out = IN
    res = torch.zeros((IN.shape[0], 2, IN.shape[2], IN.shape[3]))
    res[:, 0, :, :], res[:, 1, :, :] = torch.std_mean(IN + eps, dim=1)
    for i in range(IN.shape[0]):
        for j in range(IN.shape[1]):
            out[i, j, :, :] = (1 / res[i, 0]) * (IN[i, j, :, :] - res[i, 1])
    return out

