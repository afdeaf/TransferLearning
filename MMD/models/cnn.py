# Copyright (c) 2022 Raven Stock. email:cquptriven@qq.com

from models.base_model import BaseModel
from torch import nn
import torch

__all__ = ['cnn']

class CNN(BaseModel):
    def __init__(self, num_classes: int = 4, **kwargs):
        super().__init__(num_classes, **kwargs)

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, 7),    # 输出为58*58
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     # 输出为29*29
            nn.Dropout2d(0.3),

            nn.Conv2d(16, 32, 5),   # 输出为25*25
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     # 输出为12*12
            nn.Dropout2d(0.3),

            nn.Conv2d(32, 64, 3),   # 输出为10*10
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),     # 输出为5*5, 5*5*64 = 1600, Linear的输入，_fdim
            nn.Dropout2d(0.2),
        )
        self._init_params()
        self._fdim = 1600
        self.build_head()

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
            
    def get_backbone_parameters(self) -> list:
        feature_layers_params = []
        for m in self.feature_extractor:
            feature_layers_params += list(m.parameters())
        parameter_list = [{'params': feature_layers_params, 'lr_mult':1}]

        return parameter_list

    def forward_backbone(self, x):
        feature = self.feature_extractor(x)
        feature = torch.flatten(feature, 1)
        return feature

def cnn(num_classes: int=4, **kwargs):
    model = CNN(num_classes=num_classes, **kwargs)
    return model.cuda()