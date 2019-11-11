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

        #norm_shape = [1, b * c / group, group]
        #print(norm_shape)
        # Apply instance norm
        #input_reshaped_1 = input.contiguous().view(1, int(b * c/group), group, *input.size()[2:])
        #input_reshaped = input_reshaped_1
        G = group
        input_mod = input
        input_cpu = input.to('cpu')
        x = input_cpu.to('cpu')
        x_np = x.detach().numpy()
        # x_np = x
        Nabs, C, H, W = x_np.shape
        N = Nabs
        # x_np_tmp = x.detach().numpy()
        res_x = np.zeros((N, C, H, W))
        x_np_new = np.zeros((C, N * H * W))
        # tmpp = np.zeros((N, H*W))
        for i in range(C):
            x_np_new[i, :] = np.reshape(x_np[:, i, :, :], (1, N * H * W))
        # x_np_new = np.reshape(x_np, (C, N*H*W))
        # x_np = x_np.transpose()
        image_vector = np.asarray(x_np_new)
        Data = data_preparation(n_cluster=G, data=image_vector[:, :])
        count = 0
        Common = 1
        for val in range(G):
            if (len(np.argwhere(Data.labels_ == val)) == 0):
                pass
            else:
                Common *= len(np.argwhere(Data.labels_ == val))
                #print(Common)
        for val in range(G):
            inx = np.argwhere(Data.labels_ == val)
            for idx, idxx in enumerate(inx):
                if (len(np.argwhere(Data.labels_ == val)) == 0):
                    pass
                else:
                    for d in range(int(Common / len(np.argwhere(Data.labels_ == val)))):
                        input_mod[:, d * len(np.argwhere(Data.labels_ == val)) + count + idx, :, :] = input[:, idxx[0], :, :]
            count = len(inx)
        input_reshaped = input_mod.contiguous().view(1, int(b * Common/group), group, *input.size()[2:])
        out = F.batch_norm(
            input_reshaped, running_mean, running_var, weight=weight, bias=bias,
            training=use_input_stats, momentum=momentum, eps=eps)

        # Reshape back
        if running_mean is not None:
            running_mean_orig.copy_(running_mean.view(b, int(c/group)).mean(0, keepdim=False))
        if running_var is not None:
            running_var_orig.copy_(running_var.view(b, int(c/group)).mean(0, keepdim=False))

        return out.view(b, Common, *input.size()[2:])
    return _instance_norm(input, group, running_mean=running_mean,
                          running_var=running_var, weight=weight, bias=bias,
                          use_input_stats=use_input_stats, momentum=momentum,
                          eps=eps)


class HoogiNorm(_BatchNorm):
    def __init__(self, num_features, num_groups=5, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=False):
        self.num_groups = num_groups
        self.track_running_stats = track_running_stats
        super(HoogiNorm, self).__init__(int(num_features/num_groups), eps,
                                         momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        return NotImplemented

    def forward(self, input):
        self._check_input_dim(input)

        return group_norm(
            input, self.num_groups, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats, self.momentum, self.eps)