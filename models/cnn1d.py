# Copyright (c) 2022 Raven Stock. email:cquptriven@qq.com

from models.base_model import BaseModel
from torch import nn
import torch.nn.functional as F
import torch

__all__ = ['cnn1d']


class Conv1x3(nn.Module):
    def __init__(self, in_channel, out_channel, 
                 kernel_size=3, stride=1, 
                 padding=0) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, 
                              kernel_size, stride=stride, 
                              padding=padding)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(4, 4)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.pool(out)
        return out


class CNN(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = Conv1x3(1, 16)
        self.conv2 = Conv1x3(16, 32)
        self.conv3 = Conv1x3(32, 64)
        self._cnn_fidm = 15*64
        self.build_head()
        self.feature_layers = [self.conv1, self.conv2, self.conv3]

        self._init_params()

    def _init_params(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight.data)
                nn.init.zeros_(layer.bias.data)
            if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm1d):
                nn.init.normal_(layer.weight, mean=1.0, std=0.02)
                nn.init.zeros_(layer.bias)

    def get_backbone_parameters(self) -> list:
        feature_layers_params = []
        for m in self.feature_layers:
            feature_layers_params += list(m.parameters())
        parameter_list = [{'params': feature_layers_params, 'lr_mult':3}]
        return parameter_list

    def forward_backbone(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        f = self.conv3(x)
        return f


def cnn1d(**kwargs):
    model = CNN(**kwargs)
    return model.to(kwargs['device'])
