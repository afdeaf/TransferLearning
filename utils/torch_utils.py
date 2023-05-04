# Copyright (c) 2022 Raven Stock. email:cquptriven@qq.com

import torch
from torch import Tensor


def grl_hook(coefficient):
    '''
    Gradient reverse layer
    '''
    def func_(grad):
        return -coefficient * grad.clone()
    
    return func_

def entropy_func(x: Tensor) -> Tensor:
    """
    x: [N, C]
    return: entropy: [N,]
    """
    epsilon = 1e-5
    entropy = -x * torch.log(x + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy