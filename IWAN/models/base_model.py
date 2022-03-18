from torch import nn
from abc import abstractmethod


__all__ = ['BaseModel']


class BaseModel(nn.Module):
    def __init__(self, num_classes: int=4, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        
        self._fdim = None

    def build_head(self):
        '''
        Build classification head
        '''
        self.fc =  nn.Linear(self.fdim, self.num_classes)
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)


    @property
    def fdim(self) -> int:
        return self._fdim

    @abstractmethod
    def get_backbone_parameters(self) -> list:
        return []

    def get_parameters(self):
        parameter_list = self.get_backbone_parameters()
        # parameter_list.append({'lr_mult':10})
        return parameter_list
    

    @abstractmethod
    def forward_backbone(self, x):
        return x

    def forward(self, x):
        feature = self.forward_backbone(x)
        return feature
    

    def _init_head(self):
        for layer in self.fc.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
