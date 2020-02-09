# from .common_head import *
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import math

__all__  = ['se_res2net50_26w_8s']
class MyAdaptiveMaxPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
    def forward(self, x): 
        inp_size = x.size()
        return nn.functional.max_pool2d(input=x,
                  kernel_size= (inp_size[2], inp_size[3]))

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal', use_se=False):
        """ Constructor
                Args:
                    inplanes: input channel dimensionality
                    planes: output channel dimensionality
                    stride: conv stride. Replaces pooling layer.
                    downsample: None when stride = 1
                    baseWidth: basic width of conv3x3
                    scale: number of scale.
                    type: 'normal': normal set. 'stage': first block of a new stage.
                    use_se: use SE module or not.
                """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))  # 1x1卷积通道变化数目
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))  # K2 K3 K4
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)  # 1*1
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width
        self.se = SELayer(planes * self.expansion) if use_se else None


    def forward(self, x): #这个实现是官方提供的，但是和论文结构有出入
        residual = x

        out = self.conv1(x)  # 1x1卷积
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)  # 把特征通道进行分组，每一份大小为width
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.se is not None:
            out = self.se(out)  # SE-Res2Net

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Res2Net(nn.Module):
    def __init__(self, block, layers,  network_type, baseWidth=26, scale=4, num_classes=1000, att_type=None):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.network_type = network_type
        # different model config between ImageNet and CIFAR
        if network_type == "ImageNet":
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64,  layers[0], att_type=att_type)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, att_type=att_type)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, att_type=att_type)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, att_type=att_type)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.avgpool = MyAdaptiveMaxPool2d()
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        init.kaiming_normal_(self.fc.weight)
        for key in self.state_dict():
            if key.split('.')[-1]=="weight":
                if "conv" in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1]=='bias':
                self.state_dict()[key][...] = 0

    def _make_layer(self, block, planes, blocks, stride=1, att_type=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # layers.append(block(self.inplanes, planes, stride, downsample, use_se=att_type=='SE'))
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            baseWidth=self.baseWidth, scale=self.scale, stype='stage', use_se=att_type=='SE'))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale, use_se=att_type=='SE'))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.network_type == "ImageNet":
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.network_type == "ImageNet":
            x = self.avgpool(x)
        else:
            x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        self.features = x
        x = self.fc(x)
        return x

def Residual2Net(network_type, depth, baseWidth, scale, num_classes, att_type):
    assert network_type in ["ImageNet", "CIFAR10", "CIFAR100"], "network type should be ImageNet or CIFAR10 / CIFAR100"
    assert depth in [50, 101, 152], 'network depth should be 18, 34, 50 or 101, 152'

    if depth == 50:
        model = Res2Net(Bottle2neck, [3, 4, 6, 3], network_type, scale=scale, baseWidth=baseWidth,
                        num_classes=num_classes, att_type=att_type)
    return model


def se_res2net50_26w_8s(pretrained=False, num_classes=1000, **kwargs):
    model = Residual2Net('ImageNet', 50, 26, 8, num_classes, 'SE')
    if pretrained:
        from .res2net import res2net50_26w_8s
        pre_model_dict = res2net50_26w_8s(pretrained=pretrained).state_dict()
        model_dict = model.state_dict()
        pre_model_dict = {k: v for k, v in pre_model_dict.items() if k in model_dict}
        model_dict.update(pre_model_dict)
        model.load_state_dict(model_dict)
    return model


if __name__ == '__main__':
    import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    from collections import OrderedDict
    
    images = torch.rand(1, 3, 224, 224)
    model = se_res2net50_26w_8s(pretrained=False, num_classes=54)
    model_path = 'model_best.pth.tar'
    import torch.optim as optim
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    device = torch.device("cpu")
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = OrderedDict()
    for key, value in checkpoint['state_dict'].items():
        tmp = key[7:]
        state_dict[tmp] = value
    for key, value in checkpoint['optimizer'].items():
        print(key)
    model.load_state_dict(state_dict)
    optimizer.load_state_dict(checkpoint['optimizer'])
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
        }
    # torch.save(state, 'avg2.pth.tar')
    r = model(images)
    print(model(images).size())