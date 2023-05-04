# Copyright (c) 2022 Raven Stock. email:cquptriven@qq.com

from torch import nn
from utils.torch_utils import grl_hook
import torch
import torch.nn.functional as F

__all__ = ['Discriminator', 'Discriminator_no_grl']


class ReLU(nn.Module):
    def __init__(self, min_val=0, max_val=4) -> None:
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.alpha = nn.Parameter(torch.tensor(0.1, requires_grad=True))

    def forward(self, x):
        x = F.hardtanh(x, self.min_val, self.max_val)
        return x*self.alpha

class Discriminator(nn.Module):
    def __init__(self, in_feature: int, hidden_size: int, out_feature: int = 1):
        super().__init__()
        self.layer1 = nn.Linear(in_feature,  hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, out_feature)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)
        self._init_params()

    def _init_params(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.weight)
            if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm1d):
                nn.init.normal_(layer.weight, 1., 0.02)
                nn.init.zeros_(layer.bias)
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def get_parameters(self):
        return [{'params': self.parameters(), 'lr_mult': 5}]

    def forward(self, x, coeff: float):
        x.register_hook(grl_hook(coeff))  # GRL
        # print(self.relu1.alpha)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        x = self.layer2(x)
        x = self.relu2(x)
        x = self.drop2(x)

        y = self.layer3(x)

        return y

class Discriminator_no_grl(nn.Module):
    def __init__(self, in_feature: int, hidden_size: int, out_feature: int = 1):
        super().__init__()
        self.layer1 = nn.Linear(in_feature,  hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, out_feature)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self._init_params()

    def _init_params(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.weight)
            if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm1d):
                nn.init.normal_(layer.weight, 1., 0.02)
                nn.init.zeros_(layer.bias)
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def get_parameters(self):
        return [{'params': self.parameters(), 'lr_mult':5}]

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)

        x = self.layer2(x)
        x = self.relu2(x)

        y = self.layer3(x)
        return y