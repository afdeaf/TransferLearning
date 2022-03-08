# Copyright (c) 2022 Raven Stock. email:cquptriven@qq.com

import torch
from torch import Tensor
import torch.nn as nn

from utils.torch_utils import grl_hook, entropy_func


class WeightBCE(nn.Module):
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
                coeff: float = None, ent: bool = False):
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

    return loss_alg