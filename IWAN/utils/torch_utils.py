import torch
from torch import Tensor
from torch.autograd import Function

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


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None