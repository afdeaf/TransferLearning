# Copyright (c) 2022 Raven Stock. email:cquptriven@qq.com

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from .torch_utils import grl_hook, entropy_func
import numpy as np


class WeightBCE(nn.Module):
    """
    func: Binary Corss-Entropy loss
    """
    def __init__(self, epsilon: float = 1e-8) -> None:
        super(WeightBCE, self).__init__()
        self.epsilon = epsilon

    def forward(self, x: Tensor, label: Tensor, weight: Tensor) -> Tensor:
        """
        :param x: [N, 1]
        :param label: [N, 1]
        :param weight: [N, 1]
        """
        label = label.float()

        # 逻辑回归的损失函数：-y·log(z + epsilon) - (1-y)·log(1-z + epsilon)。epsilon是防止log(0)发生
        cross_entropy = - label * torch.log(x + self.epsilon) - (1 - label) * torch.log(1 - x + self.epsilon)
        
        return torch.sum(cross_entropy * weight.float()) / 2.


def d_align_uda(softmax_output: Tensor, features: Tensor = None, d_net=None,
                coeff: float = None, ent: bool = False, ret_feature=False):
    """Domain adversarial loss

    Args:
        softmax_output (Tensor): softmax output of the backbone.
        features (Tensor, optional): features of the output of the backbone. Defaults to None.
        d_net (nn.Module, optional): Domain adversarial network. Defaults to None.
        coeff (float, optional): The coeff of the network. Defaults to None.
        ent (bool, optional): Weather use the entropy. Defaults to False.
        ret_feature (bool, optional): Weather return feature. Defaults to False.

    Returns:
        Domain adversarial loss.
    """
    loss_func = WeightBCE()
    d_input = softmax_output if features is None else features
    d_output = d_net(d_input, coeff=coeff)
    d_output = torch.sigmoid(d_output)
    batch_size = softmax_output.size(0) // 2
    labels = torch.tensor([[1]] * batch_size + [[0]] * batch_size).long().cuda()  # 2N x 1
    if ent:
        x = softmax_output
        entropy = entropy_func(x)
        entropy.register_hook(grl_hook(coeff))
        entropy = torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[batch_size:] = 0
        source_weight = entropy * source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[:batch_size] = 0
        target_weight = entropy * target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
    else:
        weight = torch.ones_like(labels).float() / batch_size
    loss_alg = loss_func.forward(d_output, labels, weight.view(-1, 1))
    if ret_feature:
        return loss_alg, d_output[:batch_size, :]
    return loss_alg


class MMDLoss(object):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super().__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def __call__(self, source, target) -> Tensor:
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss


def sift(features: Tensor = None, labels: Tensor = None):
    loss_func = WeightBCE()
    d_output = torch.sigmoid(features)
    batch_size = features.size(0) // 2
    labels = torch.tensor([[1]] * batch_size + [[0]] * batch_size).long().cuda()  # 2N x 1

    weight = torch.ones_like(labels).float() / batch_size
    loss_sift = loss_func.forward(d_output, labels, weight.view(-1, 1))
    return loss_sift


def __compute_covariance(input_data):
    """
    Compute Covariance matrix of the input data
    """
    n = input_data.size(0)  # batch_size

    # Check if using gpu or cpu
    if input_data.is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    id_row = torch.ones(n).resize(1, n).to(device=device)
    sum_column = torch.mm(id_row, input_data)
    mean_column = torch.div(sum_column, n)
    term_mul_2 = torch.mm(mean_column.t(), mean_column)
    d_t_d = torch.mm(input_data.t(), input_data)
    c = torch.add(d_t_d, (-1 * term_mul_2)) * 1 / (n - 1)

    return c


def coral(source: torch.Tensor, target: torch.Tensor) -> Tensor:
    d = source.size(1)  # dim vector
    source_c = __compute_covariance(source)
    target_c = __compute_covariance(target)

    loss = torch.sum(torch.mul((source_c - target_c), (source_c - target_c)))

    loss = loss / (4 * d * d)
    return loss
