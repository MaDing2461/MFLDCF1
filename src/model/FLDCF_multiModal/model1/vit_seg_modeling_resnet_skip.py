import math

from os.path import join as pjoin
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
affine_par = True

########################################################################
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation = dilation)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

########################################################################


class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel,
                 reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y
    
class SqueezeAndExciteFusionAdd(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExciteFusionAdd, self).__init__()

        self.se_rgb = SqueezeAndExcitation(channels_in,
                                           activation=activation)
        self.se_depth = SqueezeAndExcitation(channels_in,
                                             activation=activation)

    def forward(self, rgb, depth):
        rgb = self.se_rgb(rgb)
        depth = self.se_depth(depth)
        out = rgb + depth
        return out, out
    
def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        # print(x.type)
        # print(w.type)
        x = x.cuda()
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])

        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])

        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))

class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(4, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i+1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]

class FuseResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        self.inplanes = 64
        super().__init__()
        width = int(64 * width_factor)
        self.width = width
        self.activation=nn.ReLU(inplace=True)

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))
        
        self.rootd = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(1, width, kernel_size=7, stride=2, bias=False, padding=3)),
            # ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.se_layer0 = SqueezeAndExciteFusionAdd(
            64, activation=self.activation)
        self.se_layer1 = SqueezeAndExciteFusionAdd(
            256,
            activation=self.activation)
        self.se_layer2 = SqueezeAndExciteFusionAdd(
            512,
            activation=self.activation)
        self.se_layer3 = SqueezeAndExciteFusionAdd(
            1024,
            activation=self.activation)
        
        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))
    
        self.bodyd = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))

        block=Bottleneck
        layers=[1, 2, 2, 1]
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
		
		
        self.res5_con1x1 = nn.Sequential(
            nn.Conv2d(1024+2048, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
			nn.ReLU(True)
            )
        
        
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        affine_par = True
    # self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x, y):
        SE = True
        # SE = False
        features = []
        # b, c, in_size, _ = x.size()
        b, c, in_size, _ = y.size()
        x = self.root(x) #64*128
        # print(y.shape)
        y = self.rootd(y) #64*128
        if SE:
            # print()
            # x1 = x.clone().detach()
            x, y = self.se_layer0(x, y)
            # x = self.se_layer0(x, y)
            # x1 = x.clone().detach() 
            # x = self.layer1(x) + self.layer1(y)
            # y = self.layer1(x1) + self.layer1(y)
            
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        y = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(y)
        for i in range(len(self.body)-1):
            x = self.body[i](x) #256*63, 512*32
            y = self.bodyd[i](y) #256*63, 512*32
            if SE:
                if i == 0:
                    x, y = self.se_layer1(x, y)
                    # x = self.se_layer1(x, y)
                if i == 1:
                    x, y = self.se_layer2(x, y)
                    # x = self.se_layer2(x, y)
                
            right_size = int(in_size / 4 / (i+1))
            # if x.size()[2] != right_size:
            #     pad = right_size - x.size()[2]
            #     assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
            #     feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
            #     feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            # else:
            #     feat = x

            if y.size()[2] != right_size:
                pad = right_size - y.size()[2]
                assert pad < 3 and pad > 0, "y {} should {}".format(y.size(), right_size)
                feat = torch.zeros((b, y.size()[1], right_size, right_size), device=y.device)
                feat[:, :, 0:y.size()[2], 0:y.size()[3]] = y[:]
            else:
                feat = y

            features.append(feat)
        x = self.body[-1](x) # 1024*16
        y = self.bodyd[-1](y) # 1024*16
        if SE:
            x, y = self.se_layer3(x, y)
            # x = self.se_layer3(x, y)
        return x, y, features[::-1]
        # x, y, features = self.hybrid_model(x, y)

    def forward_1(self, x, y):
        SE = True
        # SE = False
        features = []
        # b, c, in_size, _ = x.size()
        b, c, in_size, _ = y.size()
        x = self.root(x) #64*128
        # print(y.shape)
        y = self.rootd(y) #64*128
        # if SE:
        #     # print()
        #     # x1 = x.clone().detach()
        #     x, y = self.se_layer0(x, y)
        #     # x = self.se_layer0(x, y)
        #     # x1 = x.clone().detach() 
        #     # x = self.layer1(x) + self.layer1(y)
        #     # y = self.layer1(x1) + self.layer1(y)
            
        # features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        y = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(y)
        x, y = self.se_layer0(x, y)
        features.append(x)
        for i in range(len(self.body)-1):
            x = self.body[i](x) #256*63, 512*32
            y = self.bodyd[i](y) #256*63, 512*32
            if SE:
                if i == 0:
                    x, y = self.se_layer1(x, y)
                    # x = self.se_layer1(x, y)
                if i == 1:
                    x, y = self.se_layer2(x, y)
                    # x = self.se_layer2(x, y)
                
            right_size = int(in_size / 4 / (i+1))
            # if x.size()[2] != right_size:
            #     pad = right_size - x.size()[2]
            #     assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
            #     feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
            #     feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            # else:
            #     feat = x

            if y.size()[2] != right_size:
                pad = right_size - y.size()[2]
                assert pad < 3 and pad > 0, "y {} should {}".format(y.size(), right_size)
                feat = torch.zeros((b, y.size()[1], right_size, right_size), device=y.device)
                feat[:, :, 0:y.size()[2], 0:y.size()[3]] = y[:]
            else:
                feat = y

            features.append(feat)
        x = self.body[-1](x) # 1024*16
        y = self.bodyd[-1](y) # 1024*16
        if SE:
            x, y = self.se_layer3(x, y)
            # x = self.se_layer3(x, y)
        return x, y, features[::-1]
        # x, y, features = self.hybrid_model(x, y)
