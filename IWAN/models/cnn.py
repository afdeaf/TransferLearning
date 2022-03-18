import torch
from models.base_model import BaseModel
from torch import nn
from .discriminator import *

__all__ = ['cnn']

class CNN(BaseModel):
    def __init__(self, num_classes: int = 4, **kwargs):
        super().__init__(num_classes, **kwargs)

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 8, 7),     # 输出长度为2554
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),     # 输出长度为 1277
            nn.Dropout(0.3),

            nn.Conv1d(8, 16, 5),    # 输出长度为 1273
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),     # 输出长度为 636
            nn.Dropout(0.3),

            nn.Conv1d(16, 32, 5),   # 输出长度为 632
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),     # 输出长度为 316
            nn.Dropout(0.3),

            nn.Conv1d(32, 64, 5),   # 输出长度为 312
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),     # 输出长度为 156, 156*64 = 9984, Linear的输入，_fdim
            nn.Dropout(0.2),

            nn.GRU(156, 20, 15),     # (输入维度，输出维度，使用隐藏层数量) 128 * 64 = 8192
            # nn.GRU(128, 64, 128)   # 64*64=4096
        ) 
        self._init_params()
        self._fdim = 20 * 64
        # self.build_head()


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
        # only cnn param
        feature_layers_params = []
        for m in self.feature_extractor:
            feature_layers_params += list(m.parameters())
        parameter_list = [{'params': feature_layers_params, 'lr_mult':1}]

        return parameter_list

    def forward_backbone(self, x):
        # assert domain == 'source' or domain == 'target', f'domain \'{domain}\' not found!'
        feature = self.feature_extractor(x)
        feature = torch.flatten(feature[0], 1)
        return feature


def cnn(num_classes: int=4, **kwargs):
    model = CNN(num_classes=num_classes, **kwargs)
    return model.cuda()