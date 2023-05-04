from torch import nn
from typing import Optional, Type, Union, List
from models.base_model import BaseModel


__all__ = ['resnet18']


def conv3x3(in_planes: int, out_planes: int, stride: int=1, groups: int=1, dilation: int=1):
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=3, stride=stride,
                     groups=groups, bias=False,
                     padding=dilation, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int=1):
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int=1
    def __init__(self, inplanes: int, planes: int, 
                 stride=1, downsample: Optional[nn.Module]=None) -> None:
        super().__init__()
        # conv -> bn -> relu
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)   # out=(in-k+2*p)/s + 1 = (in-3+2*1)/1 + 1 = in
        out = self.bn1(out)
        out = self.relu1(out) 

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu2(out)
        return out
    

class Bottleneck(nn.Module):
    expansion: int=4
    def __init__(self, inplanes: int, planes: int, 
                 stride=1, downsample: Optional[nn.Module]=None) -> None:
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = conv3x3(planes, planes*self.expansion)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)

        self.downsample = downsample
        self.stride = stride
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu3(out)

        return out
    

class ResNet(BaseModel):
    # ResNet(BasicBlock, layers=[2, 2, 2, 2], **kwargs)
    def __init__(self,
                 block: Type[Union[BasicBlock, Bottleneck]],
                 layers: List[int], 
                 **kwargs):
        super().__init__(**kwargs)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, 
                               kernel_size=7, stride=2, 
                               padding=3, bias=False)  # size = (in-7+2*3)/2+1 = floor(in/2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, padding=1)  # size = floor(in/2)
        self.layer1 = self.__make_layer(block, 64, layers[0])
        self.layer2 = self.__make_layer(block, 128, layers[1], stride=2) # downsample
        self.layer3 = self.__make_layer(block, 256, layers[2], stride=2) # downsample
        self.layer4 = self.__make_layer(block, 512, layers[3], stride=2) # downsample
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.feature_layers = [self.conv1, self.bn1, 
                               self.layer1, self.layer2,
                               self.layer3, self.layer4]
        self._fdim = 512 * block.expansion

        self.__init_params()
        # self._init_params()
        self.build_head()

    def __make_layer(self, block: Type[Union[BasicBlock, Bottleneck]],
                     planes: int, blocks: int, stride: int=1):
        # __make_layer(block=BasicBlock, planes=64, blocks=2), block.expansion=1
        """
        When stride != 1 or self.inplanes != planes * block.expansion, downsample.
        """
        downsample = None
        if stride !=1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    
    def __init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 0, 2)  # mean=0, var=0.02
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_backbone_parameters(self) -> list:
        params = []
        for m in self.feature_layers:
            params += list(m.parameters())
        params_list = [{'params': params, 'lr_mult': 1}]
        return params_list
    
    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
    def forward_backbone(self, x):
        x = self._forward_impl(x)
        f = self.global_avgpool(x)
        return f
    
def resnet18(**kwargs):
    model = ResNet(BasicBlock, layers=[2, 2, 2, 2], **kwargs)
    return model.to(kwargs['device'])