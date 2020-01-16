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
        input_mod = input
        out_final = input
        input_cpu = input.to('cpu')
        x = input_cpu.to('cpu')
        x_np = x.detach().numpy()
        x_np = np.nan_to_num(x_np)
        # At this point we have just stored our input on CPU with Numpy
        Nabs, C, H, W = x_np.shape
        N = Nabs
        # N refers to the __? What the batch size?
        # res_x is the original shape of our input
        res_x = np.zeros((N, C, H, W))
        # this is a looser form where we are just by channel dimensions
        x_np_new = np.zeros((C,2))
        # we loop through the channels
        for i in range(C):
            # we are converting the raw input data such that there are channel rows of large N*H*W blocks
            # why
           temp = np.reshape(x_np[:, i, :, :], (1, N * H * W))
           x_np_new[i, :] = [temp.mean(),temp.std()]        # 60 by 10000 trying to cluster in this basis
        # we create a copy of our reformatted data
        x_np_new = np.nan_to_num(x_np_new,posinf=0,neginf=0)
        image_vector = np.asarray(x_np_new)
        Data = data_preparation(n_cluster=G, data=image_vector[:, :])
        # Up to here we are all correct, as long as we are clustering in the right basis
        count = 0
        Common_mul = 1
        Common_add = 0
        tmp1 = torch.zeros(())
        for val in range(G):
            # this creates all a list of all the indices where our label is valid 
            inx = np.argwhere(Data.labels_ == val)
            if len(inx) > 0:
                tmp = torch.zeros((input.shape[0], len(inx), input.shape[2], input.shape[3]))
            for idx, idxx in enumerate(inx):
                if len(inx) == 0:
                    pass
                else:
                    # This is line of code is effectively saying fill into our temp channel equal to
                    # the original index from input for all the matching indices. so we have temp has 
                    # number of channels assigned to that cluster
                    tmp[:, idx, :, :] = input[:, idxx[0], :, :]
                    # now we have the correct grouping so we choose to normalize
            # STAT is the function which I think has a bug...
            out_final_tmp = (Stat_torch(tmp)).float().to('cuda')
            for idx, idxx in enumerate(inx):
                if (len(inx) == 0):
                    pass
                else:
                    # here we assign that the the original index of out_final is equal to the temp
                    # index it was given in the normalized group
                    out_final[:, idxx[0], :, :] = out_final_tmp[:, idx, :, :]
        # we reshape it to the original size
        # this is probably fine since code doesnt error...
        # although double check this last line
        return out_final.view(b, c, *input.size()[2:])

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

