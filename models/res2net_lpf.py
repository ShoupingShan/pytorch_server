import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
import numpy as np

__all__ = ['Res2Net', 'lpf_res2net50', 'lpf_res2net50_26w_8s']

model_urls = {
    'res2net50_26w_4s': 'http://data.kaizhao.net/projects/res2net/pretrained/res2net50_26w_4s-06e79181.pth',
    'res2net50_48w_2s': 'http://data.kaizhao.net/projects/res2net/pretrained/res2net50_48w_2s-afed724a.pth',
    'res2net50_14w_8s': 'http://data.kaizhao.net/projects/res2net/pretrained/res2net50_14w_8s-6527dddc.pth',
    'res2net50_26w_6s': 'http://data.kaizhao.net/projects/res2net/pretrained/res2net50_26w_6s-19041792.pth',
    'res2net50_26w_8s': 'http://data.kaizhao.net/projects/res2net/pretrained/res2net50_26w_8s-2c7c9f12.pth',
    'res2net101_26w_4s': 'http://data.kaizhao.net/projects/res2net/pretrained/res2net101_26w_4s-02a759a1.pth',
}


class Downsample(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        # print('Filter size [%i]'%filt_size)
        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))

        self.pad = self.get_pad_layer(pad_type)(self.pad_sizes)

    def get_pad_layer(self, pad_type):
        if(pad_type in ['refl','reflect']):
            PadLayer = nn.ReflectionPad2d
        elif(pad_type in ['repl','replicate']):
            PadLayer = nn.ReplicationPad2d
        elif(pad_type=='zero'):
            PadLayer = nn.ZeroPad2d
        else:
            print('Pad type [%s] not recognized'%pad_type)
        return PadLayer

    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

class Downsample1D(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample1D, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        # print('Filter size [%i]' % filt_size)
        if(self.filt_size == 1):
            a = np.array([1., ])
        elif(self.filt_size == 2):
            a = np.array([1., 1.])
        elif(self.filt_size == 3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size == 4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size == 5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size == 6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size == 7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a)
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :].repeat((self.channels, 1, 1)))

        self.pad = self.get_pad_layer_1d(pad_type)(self.pad_sizes)
    def get_pad_layer_1d(self, pad_type):
        if(pad_type in ['refl', 'reflect']):
            PadLayer = nn.ReflectionPad1d
        elif(pad_type in ['repl', 'replicate']):
            PadLayer = nn.ReplicationPad1d
        elif(pad_type == 'zero'):
            PadLayer = nn.ZeroPad1d
        else:
            print('Pad type [%s] not recognized' % pad_type)
        return PadLayer
    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride]
        else:
            return F.conv1d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()
        # if planes % scale != 0:
        #     raise ValueError('Planes must be divisible by scales')
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
        self._downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
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

        if self._downsample is not None:
            residual = self._downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Module):

    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000, filter_size=1):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.Sequential(*[nn.MaxPool2d(kernel_size=2, stride=1),
                                       Downsample(filt_size=filter_size, stride=2, channels=64)])
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, filter_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = [Downsample(filt_size=filter_size, stride=stride, channels=self.inplanes), ] if (stride != 1) else []
            downsample += [nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                           nn.BatchNorm2d(planes * block.expansion)]
            # print(downsample)
            downsample = nn.Sequential(*downsample)
            # downsample = nn.Sequential(
            #     nn.Conv2d(self.inplanes, planes * block.expansion,
            #               kernel_size=1, stride=stride, bias=False),
            #     nn.BatchNorm2d(planes * block.expansion),
            # )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        self.features = x
        x = self.fc(x)

        return x


def lpf_res2net50(pretrained=False, filter_size=5, **kwargs):
    """Constructs a Res2Net-50 model.
    Res2Net-50 refers to the Res2Net-50_26w_4s.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, filter_size=filter_size, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_26w_4s']))
    return model


def lpf_res2net50_26w_4s(pretrained=False, filter_size=5, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, filter_size=filter_size, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_26w_4s']))
    return model


def lpf_res2net101_26w_4s(pretrained=False, filter_size=5, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4, filter_size=filter_size, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net101_26w_4s']))
    return model


def lpf_res2net50_26w_6s(pretrained=False, filter_size=5, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=6, filter_size=filter_size, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_26w_6s']))
    return model


def lpf_res2net50_26w_8s(pretrained=False, filter_size=5, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=8, filter_size=filter_size, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['res2net50_26w_8s']))
    # return model

    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=8, filter_size=filter_size, **kwargs)
    if pretrained:
        from .res2net import res2net50_26w_8s
        pre_model_dict = res2net50_26w_8s(pretrained=pretrained).state_dict()
        model_dict = model.state_dict()
        pre_model_dict = {k: v for k, v in pre_model_dict.items() if k in model_dict}
        model_dict.update(pre_model_dict)
        model.load_state_dict(model_dict)
    return model

def lpf_res2net50_48w_2s(pretrained=False, filter_size=5, **kwargs):
    """Constructs a Res2Net-50_48w_2s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=48, scale=2, filter_size=filter_size, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_48w_2s']))
    return model


def lpf_res2net50_14w_8s(pretrained=False, filter_size=5, **kwargs):
    """Constructs a Res2Net-50_14w_8s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=14, scale=8, filter_size=filter_size, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_14w_8s']))
    return model


if __name__ == '__main__':
    import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    from collections import OrderedDict
    images = torch.rand(1, 3, 320, 320)
    model = lpf_res2net50_26w_8s(pretrained=False, num_classes=54)
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
