from torch import nn
import torch.nn.functional as F
import torch


__all__ = ['BaseModel']
    

class BaseModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        if 'num_classes' not in kwargs:
            raise ValueError("num_classes not found!")
        self.num_classes = kwargs['num_classes']
        self._fdim = 128
        self._cnn_fidm = 1600
        self.fc_hidden_size = 512
        self.device = kwargs['device']

    def build_head(self):
        self.fc1 = nn.Linear(self.cdim, self.fc_hidden_size).to(self.device)
        self.fc2 = nn.Linear(self.fc_hidden_size, self.fdim).to(self.device)


        self.fc = nn.Linear(self.fdim, self.num_classes).to(self.device)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    @property
    def fdim(self) -> int:
        return self._fdim
    
    @property
    def cdim(self) -> int:
        return self._cnn_fidm

    def get_backbone_parameters(self) -> list:
        raise NotImplementedError

    def get_parameters(self):
        parameter_list = self.get_backbone_parameters()
        parameter_list.append({'params': self.fc.parameters(), 'lr_mult':1})
        parameter_list.append({'params': self.fc1.parameters(), 'lr_mult':1})
        parameter_list.append({'params': self.fc2.parameters(), 'lr_mult':1})
        return parameter_list

    def forward_backbone(self):
        raise NotImplementedError

    def forward(self, x) -> tuple:
        '''
        return: tuple like (feature, y_pred)
        '''
        c_f = self.forward_backbone(x)
 
        f = torch.flatten(c_f, 1)
        f = self.fc1(f)
        f = F.relu(f)
        f = self.fc2(f)
        f = F.relu(f)

        y = self.fc(f)
        return f, y
    