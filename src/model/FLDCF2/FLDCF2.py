"""Cloud detection Network"""

"""
This is the implementation of CDnetV1 without multi-scale inputs. This implementation uses ResNet by default.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data, model_zoo
from torch.autograd import Variable
import math
import numpy as np
affine_par = True
from torch.autograd import Function
import torchvision.models as models
from utils.tools import draw_features
import os

import os
import sys
from functools import partial

import torch
from torch import nn

from torch.utils import model_zoo
from .densenet import densenet121, densenet169, densenet161

from . import resnet
from .dpn import dpn92
from .senet import se_resnext50_32x4d, se_resnext101_32x4d, SCSEModule, senet154





def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine = affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


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

class Classifier_Module(nn.Module):

    def __init__(self, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(2048, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
            return out



class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, norm_layer=nn.BatchNorm2d):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
		
		
class _ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rate, norm_layer):
        super(_ASPPConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate, dilation=atrous_rate, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class _AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(_AsppPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out
		
		
class _ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, norm_layer):
        super(_ASPP, self).__init__()
        out_channels = 512 # changed from 256
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = _ASPPConv(in_channels, out_channels, rate1, norm_layer)
        self.b2 = _ASPPConv(in_channels, out_channels, rate2, norm_layer)
        self.b3 = _ASPPConv(in_channels, out_channels, rate3, norm_layer)			
        self.b4 = _AsppPooling(in_channels, out_channels, norm_layer=norm_layer)

        # self.project = nn.Sequential(
            # nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            # norm_layer(out_channels),
            # nn.ReLU(True),
            # nn.Dropout(0.5))
        self.dropout2d = nn.Dropout2d(0.3)
		
    def forward(self, x):
        feat1 = self.dropout2d(self.b0(x))
        feat2 = self.dropout2d(self.b1(x))
        feat3 = self.dropout2d(self.b2(x))
        feat4 = self.dropout2d(self.b3(x))
        feat5 = self.dropout2d(self.b4(x))	
        x = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        # x = self.project(x)
        return x
		

class _FPM(nn.Module):
    def __init__(self, in_channels, num_classes, norm_layer=nn.BatchNorm2d):
        super(_FPM, self).__init__()
        self.aspp = _ASPP(in_channels, [ 6, 12, 18], norm_layer=norm_layer )
        #self.dropout2d = nn.Dropout2d(0.5)
    def forward(self, x):

        x = torch.cat((x, self.aspp(x)), dim=1)
        #x = self.dropout2d(x) # added
        return x

		


class BR(nn.Module):
    def __init__(self, num_classes, stride=1, downsample=None):
        super(BR, self).__init__()
        self.conv1 = conv3x3(num_classes, num_classes*16, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(num_classes*16, num_classes)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out += residual


        return out

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers
        
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)
        
        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x

################################################################
encoder_params = {
    'dpn92':
        {
            'filters': [64, 336, 704, 1552, 2688],
            'decoder_filters': [64, 128, 256, 256],
            'last_upsample': 64,
            'init_op': dpn92,
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/dpn92_extra-b040e4a9b.pth',
        },
    'resnet18':
        {
            'filters': [64, 64, 128, 256, 512],
            'decoder_filters': [64, 128, 256, 256],
            'last_upsample': 64,
            'init_op': partial(resnet.resnet18, in_channels=3),
            'url': resnet.model_urls['resnet18'],
        },
    'resnet34':
        {
            'filters': [64, 64, 128, 256, 512],
            'decoder_filters': [64, 128, 256, 256],
            'last_upsample': 64,
            'init_op': partial(resnet.resnet34, in_channels=3),
            'url': resnet.model_urls['resnet34'],
        },
    'resnet101':
        {
            'filters': [64, 256, 512, 1024, 2048],
            'decoder_filters': [64, 128, 256, 256],
            'last_upsample': 64,
            'init_op': partial(resnet.resnet101, in_channels=3),
            'url': resnet.model_urls['resnet101'],
        },
    'resnet50':
        {
            'filters': [64, 256, 512, 1024, 2048],
            'decoder_filters': [64, 128, 256, 256],
            'last_upsample': 64,
            'init_op': partial(resnet.resnet50, in_channels=3),
            'url': resnet.model_urls['resnet50'],
        },
    'densenet121':
        {
            'filters': [64, 256, 512, 1024, 1024],
            'decoder_filters': [64, 128, 256, 256],
            'last_upsample': 64,
            'url': None,
            'init_op': densenet121,
        },
    'densenet169':
        {
            'filters': [64, 256, 512, 1280, 1664],
            'decoder_filters': [64, 128, 256, 256],
            'last_upsample': 64,
            'url': None,
            'init_op': densenet169,
        },
    'densenet161':
        {
            'filters': [96, 384, 768, 2112, 2208],
            'decoder_filters': [64, 128, 256, 256],
            'last_upsample': 64,
            'url': None,
            'init_op': densenet161,
        },
    'densenet161_fatter':
        {
            'filters': [96, 384, 768, 2112, 2208],
            'decoder_filters': [128, 128, 256, 256],
            'last_upsample': 128,
            'url': None,
            'init_op': densenet161,
        },
    'seresnext50':
        {
            'filters': [64, 256, 512, 1024, 2048],
            'decoder_filters': [64, 128, 256, 384],
            'init_op': se_resnext50_32x4d,
            # 'url': './src/model/scunet/pretrain/se_resnext50_32x4d-a260b3a4.pth',
            'url': '/media/lscsc/nas/mading/fakedetect/src/model/FLDCF2/pretrain/se_resnext50_32x4d-a260b3a4.pth',
        },
    'senet154':
        {
            'filters': [128, 256, 512, 1024, 2048],
            'decoder_filters': [64, 128, 256, 384],
            'init_op': senet154,
            'url': './models/senet154-c7b49a05.pth',
            # 'url':None
        },
    'seresnext101':
        {
            'filters': [64, 256, 512, 1024, 2048],
            'decoder_filters': [64, 128, 256, 384],
            'last_upsample': 64,
            'init_op': se_resnext101_32x4d,
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth',
        }
}


class AbstractModel(nn.Module):
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def initialize_encoder(self, model, model_url, num_channels_changed=False):
        if os.path.isfile(model_url):
            pretrained_dict = torch.load(model_url)
        else:
            pretrained_dict = model_zoo.load_url(model_url)
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
            pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        if num_channels_changed:
            model.state_dict()[self.first_layer_params_name +
                               '.weight'][:, :3, ...] = pretrained_dict[self.first_layer_params_name + '.weight'].data
            skip_layers = [
                self.first_layer_params_name,
                self.first_layer_params_name + '.weight',
            ]
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if not any(k.startswith(s) for s in skip_layers)
            }
        model.load_state_dict(pretrained_dict, strict=False)

    @property
    def first_layer_params_name(self):
        return 'conv1'


class EncoderDecoder(AbstractModel):
    def __init__(self, num_classes, num_channels=3, encoder_name='resnet34',return_middle_map = False):
        self.return_middle_map = return_middle_map
        if not hasattr(self, 'first_layer_stride_two'):
            self.first_layer_stride_two = False
        if not hasattr(self, 'decoder_block'):
            self.decoder_block = UnetDecoderBlock
        if not hasattr(self, 'bottleneck_type'):
            self.bottleneck_type = ConvBottleneck

        self.filters = encoder_params[encoder_name]['filters']
        self.decoder_filters = encoder_params[encoder_name].get('decoder_filters', self.filters[:-1])
        self.last_upsample_filters = encoder_params[encoder_name].get('last_upsample', self.decoder_filters[0] // 2)

        super().__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes

        self.bottlenecks = nn.ModuleList(
            [
                self.bottleneck_type(self.filters[-i - 2] + f, f)
                for i, f in enumerate(reversed(self.decoder_filters[:]))
            ]
        )

        self.decoder_stages = nn.ModuleList([self.get_decoder(idx) for idx in range(0, len(self.decoder_filters))])

        if self.first_layer_stride_two:
            self.last_upsample = self.decoder_block(
                self.decoder_filters[0],
                self.last_upsample_filters,
                self.last_upsample_filters,
            )

        self.final = self.make_final_classifier(
            self.last_upsample_filters if self.first_layer_stride_two else self.decoder_filters[0],
            num_classes,
        )

        self._initialize_weights()

        encoder = encoder_params[encoder_name]['init_op'](pretrained=False)
        self.encoder_stages = nn.ModuleList([self.get_encoder(encoder, idx) for idx in range(len(self.filters))])
        if encoder_params[encoder_name]['url'] is not None:
            self.initialize_encoder(encoder, encoder_params[encoder_name]['url'], num_channels != 3)

    # noinspection PyCallingNonCallable
    def forward(self, x):
        b,c,w,h = x.shape
        enc_results = []
        return_middle_map = []
        # print(len(self.encoder_stages))
        for stage in self.encoder_stages:
            x = stage(x)
            return_middle_map.append(x)
            enc_results.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())

        last_dec_out = enc_results[-1]
        # size = last_dec_out.size(2)
        # last_dec_out = torch.cat([last_dec_out, F.upsample(angles, size=(size, size), mode="nearest")], dim=1)
        x = last_dec_out
        for idx, bottleneck in enumerate(self.bottlenecks):
            rev_idx = -(idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx - 1])

        if self.first_layer_stride_two:
            x = self.last_upsample(x)
            return_middle_map.append(x)
        f = self.final(x)
        return f

    def get_decoder(self, layer):
        in_channels = (
            self.filters[layer + 1] if layer + 1 == len(self.decoder_filters) else self.decoder_filters[layer + 1]
        )
        return self.decoder_block(
            in_channels,
            self.decoder_filters[layer],
            self.decoder_filters[max(layer, 0)],
        )

    def make_final_classifier(self, in_filters, num_classes):
        return nn.Sequential(nn.Conv2d(in_filters, num_classes, 1, padding=0), nn.Sigmoid())

    def get_encoder(self, encoder, layer):
        raise NotImplementedError

    @property
    def first_layer_params(self):
        return _get_layers_params([self.encoder_stages[0]])

    @property
    def layers_except_first_params(self):
        layers = get_slice(self.encoder_stages, 1, -1) + [
            self.bottlenecks,
            self.decoder_stages,
            self.final,
        ]
        return _get_layers_params(layers)

class EncoderDecoder_withEdgeDecoder(AbstractModel):
    def __init__(self, num_classes, num_channels=3, encoder_name='resnet34'):
        if not hasattr(self, 'first_layer_stride_two'):
            self.first_layer_stride_two = False
        if not hasattr(self, 'decoder_block'):
            self.decoder_block = UnetDecoderBlock
        if not hasattr(self, 'bottleneck_type'):
            self.bottleneck_type = ConvBottleneck

        self.filters = encoder_params[encoder_name]['filters']
        self.decoder_filters = encoder_params[encoder_name].get('decoder_filters', self.filters[:-1])
        self.last_upsample_filters = encoder_params[encoder_name].get('last_upsample', self.decoder_filters[0] // 2)

        super().__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes

        self.bottlenecks = nn.ModuleList(
            [
                self.bottleneck_type(self.filters[-i - 2] + f, f)
                for i, f in enumerate(reversed(self.decoder_filters[:]))
            ]
        )
        self.bottlenecks_edge = nn.ModuleList(
            [
                self.bottleneck_type(self.filters[-i - 2] + f, f)
                for i, f in enumerate(reversed(self.decoder_filters[:]))
            ]
        )
        self.decoder_stages = nn.ModuleList([self.get_decoder(idx) for idx in range(0, len(self.decoder_filters))])
        self.decoder_stages_edge = nn.ModuleList([self.get_decoder(idx) for idx in range(0, len(self.decoder_filters))])

        if self.first_layer_stride_two:
            self.last_upsample = self.decoder_block(
                self.decoder_filters[0],
                self.last_upsample_filters,
                self.last_upsample_filters,
            )
            self.last_upsample_edge = self.decoder_block(
                self.decoder_filters[0],
                self.last_upsample_filters,
                self.last_upsample_filters,
            )
        self.final = self.make_final_classifier(
            self.last_upsample_filters if self.first_layer_stride_two else self.decoder_filters[0],
            num_classes,
        )
        self.final_edge = self.make_final_classifier(
            self.last_upsample_filters if self.first_layer_stride_two else self.decoder_filters[0],
            num_classes,
        )
        self._initialize_weights()

        encoder = encoder_params[encoder_name]['init_op'](pretrained=False)
        self.encoder_stages = nn.ModuleList([self.get_encoder(encoder, idx) for idx in range(len(self.filters))])
        if encoder_params[encoder_name]['url'] is not None:
            self.initialize_encoder(encoder, encoder_params[encoder_name]['url'], num_channels != 3)

    # noinspection PyCallingNonCallable
    def forward(self, x):
        enc_results = []
        for stage in self.encoder_stages:
            x = stage(x)
            enc_results.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())

        last_dec_out = enc_results[-1]
        # size = last_dec_out.size(2)
        # last_dec_out = torch.cat([last_dec_out, F.upsample(angles, size=(size, size), mode="nearest")], dim=1)
        x = last_dec_out
        for idx, bottleneck in enumerate(self.bottlenecks):
            rev_idx = -(idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx - 1])

        if self.first_layer_stride_two:
            x = self.last_upsample(x)

        f = self.final(x)

        '''Decoder Edge'''
        x = last_dec_out
        for idx, bottleneck in enumerate(self.bottlenecks_edge):
            rev_idx = -(idx + 1)
            x = self.decoder_stages_edge[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx - 1])

        if self.first_layer_stride_two:
            x = self.last_upsample_edge(x)

        f_edge = self.final_edge(x)


        return f,f_edge

    def get_decoder(self, layer):
        in_channels = (
            self.filters[layer + 1] if layer + 1 == len(self.decoder_filters) else self.decoder_filters[layer + 1]
        )
        return self.decoder_block(
            in_channels,
            self.decoder_filters[layer],
            self.decoder_filters[max(layer, 0)],
        )

    def make_final_classifier(self, in_filters, num_classes):
        return nn.Sequential(nn.Conv2d(in_filters, num_classes, 1, padding=0), nn.Sigmoid())

    def get_encoder(self, encoder, layer):
        raise NotImplementedError

    @property
    def first_layer_params(self):
        return _get_layers_params([self.encoder_stages[0]])

    @property
    def layers_except_first_params(self):
        layers = get_slice(self.encoder_stages, 1, -1) + [
            self.bottlenecks,
            self.decoder_stages,
            self.final,
        ]
        return _get_layers_params(layers)

def _get_layers_params(layers):
    return sum((list(l.parameters()) for l in layers), [])


def get_slice(features, start, end):
    if end == -1:
        end = len(features)
    return [features[i] for i in range(start, end)]


class ConvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True))

    def forward(self, dec, enc):
        x = torch.cat([dec, enc], dim=1)
        return self.seq(x)

class ConvBottleneck_withBN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True))

    def forward(self, dec, enc):
        x = torch.cat([dec, enc], dim=1)
        return self.seq(x)

class UnetDecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)

class UnetDecoderBlock_BN(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)

class UnetConvTransposeDecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class Resnet(EncoderDecoder):
    def __init__(self, seg_classes, backbone_arch):
        self.first_layer_stride_two = True
        super().__init__(seg_classes, 4, backbone_arch)

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        elif layer == 1:
            return nn.Sequential(encoder.maxpool, encoder.layer1)
        elif layer == 2:
            return encoder.layer2
        elif layer == 3:
            return encoder.layer3
        elif layer == 4:
            return encoder.layer4


class ConvTransposeResnetUnet(EncoderDecoder):
    def __init__(self, seg_classes, backbone_arch):
        self.first_layer_stride_two = True
        self.decoder_block = UnetConvTransposeDecoderBlock
        super().__init__(seg_classes, 3, backbone_arch)

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        elif layer == 1:
            return nn.Sequential(encoder.maxpool, encoder.layer1)
        elif layer == 2:
            return encoder.layer2
        elif layer == 3:
            return encoder.layer3
        elif layer == 4:
            return encoder.layer4


class DPNUnet(EncoderDecoder):
    def __init__(self, seg_classes=1, backbone_arch='dpn92'):
        self.first_layer_stride_two = True
        super().__init__(seg_classes, 4, backbone_arch)

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return nn.Sequential(
                encoder.blocks['conv1_1'].conv,  # conv
                encoder.blocks['conv1_1'].bn,  # bn
                encoder.blocks['conv1_1'].act,  # relu
            )
        elif layer == 1:
            return nn.Sequential(
                encoder.blocks['conv1_1'].pool,  # maxpool
                *[b for k, b in encoder.blocks.items() if k.startswith('conv2_')]
            )
        elif layer == 2:
            return nn.Sequential(*[b for k, b in encoder.blocks.items() if k.startswith('conv3_')])
        elif layer == 3:
            return nn.Sequential(*[b for k, b in encoder.blocks.items() if k.startswith('conv4_')])
        elif layer == 4:
            return nn.Sequential(*[b for k, b in encoder.blocks.items() if k.startswith('conv5_')])

    @property
    def first_layer_params_name(self):
        return 'features.conv1_1.conv'


class DensenetUnet(EncoderDecoder):
    def __init__(self, seg_classes=1, backbone_arch='densenet121'):
        self.first_layer_stride_two = True
        super().__init__(seg_classes, 3, backbone_arch)

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return nn.Sequential(
                encoder.features.conv0,
                encoder.features.norm0,
                encoder.features.relu0,  # conv  # bn  # relu
            )
        elif layer == 1:
            return nn.Sequential(encoder.features.pool0, encoder.features.denseblock1)
        elif layer == 2:
            return nn.Sequential(encoder.features.transition1, encoder.features.denseblock2)
        elif layer == 3:
            return nn.Sequential(encoder.features.transition2, encoder.features.denseblock3)
        elif layer == 4:
            return nn.Sequential(
                encoder.features.transition3,
                encoder.features.denseblock4,
                encoder.features.norm5,
                nn.ReLU(),
            )


class SEUnet(EncoderDecoder):
    def __init__(self, seg_classes=1, backbone_arch='senet154',return_middle_map = False):
        self.first_layer_stride_two = True
        self.return_middle_map = return_middle_map
        super().__init__(seg_classes, num_channels=3, encoder_name=backbone_arch,return_middle_map = self.return_middle_map)

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return encoder.layer0
        elif layer == 1:
            return nn.Sequential(encoder.pool, encoder.layer1)
        elif layer == 2:
            return encoder.layer2
        elif layer == 3:
            return encoder.layer3
        elif layer == 4:
            return encoder.layer4

    @property
    def first_layer_params_name(self):
        return 'layer0.conv1'

class ConvSCSEBottleneckNoBn(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=2):
        # print('bottleneck ', in_channels, out_channels)
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            SCSEModule(out_channels, reduction=reduction, mode='maxout'),
        )

    def forward(self, dec, enc):
        # print(dec.shape,enc.shape)
        # enc = torch.nn.functional.interpolate(enc,
        #                                 size=(dec.shape[2], dec.shape[3]))
        x = torch.cat([dec, enc], dim=1)
        return self.seq(x)

################################################################





# class FLDCF2(nn.Module):
class FLDCF2(SEUnet):
    # def __init__(self,args, block=Bottleneck, layers=[1, 2, 2, 1], num_classes=2, aux=True, \
    def __init__(self,args, block=Bottleneck, num_classes=2, aux=True, \
                 seg_classes=1, backbone_arch='seresnext50',return_middle_map = False):
        # self.inplanes = 64
        # self.aux = aux	

        self.bottleneck_type = ConvSCSEBottleneckNoBn
        self.return_middle_map = return_middle_map
        # super(FLDCF2, self).__init__()
        super().__init__(seg_classes, backbone_arch=backbone_arch,return_middle_map = self.return_middle_map)

        # self.learned = 	Restoretest(args)
        # self.learned.load_state_dict(
        #         # torch.load(
        #         #     # os.path.join('./model', 'model_vi.pt'),
        #         #     os.path.join('/media/lscsc/nas/mading/fakedetect/model', 'model_vi_gray.pt'),  
        #         # ),
        #         torch.load(
        #         # os.path.join('./model', 'model_vi.pt'),
        #         os.path.join('/media/lscsc/nas/mading/fakedetect/model', 'model_vi_md_restore_dsm2.pt'),  
        #         ),
        #         strict=True
        #     )
        
        # self.handlelern1 = nn.Conv2d(32 *4,64, kernel_size=3, stride=2, padding=1, bias=False)
        # self.handlelern2 = nn.Conv2d(32 *4,64, kernel_size=3, stride=2, padding=1, bias=False)

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(64, affine = affine_par)
        # self.conv2 = nn.Conv2d(64*2, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(64, affine = affine_par)		
        # self.conv3 = nn.Conv2d(64*2, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(64, affine = affine_par)
		
		
        # for i in self.bn1.parameters():
        #     i.requires_grad = False
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        # # self.layer1 = self._make_layer(block, 64, layers[0])
        # # self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # # self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        # # self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
		
		
        # self.res5_con1x1 = nn.Sequential(
        #     nn.Conv2d(1024+2048, 512, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(512),
		# 	nn.ReLU(True)
        #     )
					
        # self.fpm1 = _FPM(512, num_classes)
        # #self.fpm2 = _FPM(512, num_classes)
        # #self.fpm3 = _FPM(256, num_classes)	
		
        # self.br1 = BR(num_classes)	
        # #self.br2 = BR(num_classes)	
        # #self.br3 = BR(num_classes)			
        # #self.br4 = BR(num_classes)	
        # self.br5 = BR(num_classes)	
        # self.br6 = BR(num_classes)	
        # self.br7 = BR(num_classes)			
		

        # self.predict1 = self._predict_layer(512*6, num_classes)	
        # #self.predict2 = self._predict_layer(512*6,num_classes)			
        # #self.predict3 = self._predict_layer(512*5+256,num_classes)

        dropout = 0.9

        self.res = models.resnet18(pretrained=True)
        in_features = self.res.fc.in_features

        self.res.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.BatchNorm1d(in_features // 2),
            nn.Dropout(dropout),
            nn.Linear(in_features // 2, 2)
        )

        self.GFF = nn.Sequential(*[
            # nn.Conv2d(5, 3, 1, padding=0, stride=1),
            nn.Conv2d(4, 3, 1, padding=0, stride=1),
            nn.Conv2d(3, 3, 3, padding=(3-1)//2, stride=1)
        ])

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, 0.01)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # #        for i in m.parameters():
        # #            i.requires_grad = False
        # super().__init__(seg_classes, backbone_arch=backbone_arch,return_middle_map = self.return_middle_map)
		
    def _predict_layer(self, in_channels, num_classes):
        return nn.Sequential(nn.Conv2d(in_channels, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
			nn.ReLU(True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=3, stride=1, padding=1, bias=True))

    # def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
    #     downsample = None
    #     if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
    #         downsample = nn.Sequential(
    #             nn.Conv2d(self.inplanes, planes * block.expansion,
    #                       kernel_size=1, stride=stride, bias=False),
    #             nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
    #     for i in downsample._modules['1'].parameters():
    #         i.requires_grad = False
    #     layers = []
    #     layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
    #     self.inplanes = planes * block.expansion
    #     for i in range(1, blocks):
    #         layers.append(block(self.inplanes, planes, dilation=dilation))

    #     return nn.Sequential(*layers)


    
    # def base_forward(self, x,RDNout):
    #     # draw_features(64, RDNout[4][0],"{}/0.png".format('./image'))
    #     # draw_features(64, RDNout[5][0],"{}/1.png".format('./image'))
    #     # draw_features(64, RDNout[6][0],"{}/2.png".format('./image'))
    #     # draw_features(64, RDNout[7][0],"{}/3.png".format('./image'))
    #     rdn1 = self.handlelern1(torch.cat(RDNout[0:4], 1))
    #     rdn2 = self.handlelern2(torch.cat(RDNout[4:8], 1))
    #     x = self.relu(self.bn1(self.conv1(x)))
    #     size_conv1 = x.size()[2:]
    #     x = self.relu(self.bn2(self.conv2(torch.cat([x, rdn1],dim=1))))
    #     # draw_features(64, rdn2[0],"{}/rdn.png".format('./image'))
    #     # draw_features(64, x[0],"{}/x.png".format('./image'))
    #     x = self.relu(self.bn3(self.conv3(torch.cat([x, rdn2],dim=1))))
    #     x = self.maxpool(x)
    #     x = self.layer1(x)
    #     res2 = x
    #     x = self.layer2(x)
    #     res3 = x
    #     x = self.layer3(x)
    #     res4 = x
    #     x = self.layer4(x)
    #     x = self.res5_con1x1(torch.cat([x, res4], dim=1))

		
        # return x, res3, res2, size_conv1
			
    def forward(self, x):
        # result, RDNout = self.learned(x)
        # b,c,w,h = x.size()
        # size = x.size()[2:]
        # score1, score2, score3,  size_conv1 = self.base_forward(x,RDNout)
        # score1 = self.fpm1(score1)
        # score1 = self.predict1(score1) 	# 1/8	
        # score1 = self.br1(score1)
        # score2 = score1
		
		
        # second fusion	
        # size_score3 = score3.size()[2:]
        # score3 = F.interpolate(score2, size_score3, mode='bilinear', align_corners=True)			
        # score3 = self.br5(score3)
        #draw_features(64, score3[:,0],"{}/decoder1.png".format('./image'))
		
        # upsampling + BR	
        # score3 = F.interpolate(score3, size_conv1, mode='bilinear', align_corners=True) 		
        # score3 = self.br6(score3)
        # #draw_features(64, score3[:,0],"{}/decoder2.png".format('./image'))
        # score3 = F.interpolate(score3, size, mode='bilinear', align_corners=True)
        # score3 = self.br7(score3)	

        x1 = x.clone().detach()
        # print(x1.shape)
        ################################################################
        # Forgery Localization Part
        b,c,w,h = x.shape
        enc_results = []
        return_middle_map = []
        # print(len(self.encoder_stages))
        for stage in self.encoder_stages:
            x = stage(x)
            return_middle_map.append(x)
            enc_results.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())

        last_dec_out = enc_results[-1]
        # size = last_dec_out.size(2)
        # last_dec_out = torch.cat([last_dec_out, F.upsample(angles, size=(size, size), mode="nearest")], dim=1)
        x = last_dec_out
        for idx, bottleneck in enumerate(self.bottlenecks):
            rev_idx = -(idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx - 1])

        if self.first_layer_stride_two:
            x = self.last_upsample(x)
            return_middle_map.append(x)
        f = self.final(x)
        # return f
        score3 = f
        ################################################################
        
        # Forgery Detection Part
        score32 = score3.clone().detach()
        # print(x1.shape) 
        # print(x1.shape)
        # print(score32.shape)
        y = torch.cat([x1,score32],dim =1)
        # print(y.shape)
        y = self.GFF(y)
        out = self.res(y)

        return score3, out
        # return score3, None

setattr(sys.modules[__name__], 'resnet_unet', partial(Resnet))
setattr(sys.modules[__name__], 'convt_resnet_unet', partial(ConvTransposeResnetUnet))

setattr(sys.modules[__name__], 'dpn_unet', partial(DPNUnet))
setattr(sys.modules[__name__], 'densenet_unet', partial(DensenetUnet))
setattr(sys.modules[__name__], 'se_unet', partial(SEUnet))
setattr(sys.modules[__name__], 'scse_unet', partial(FLDCF2))
