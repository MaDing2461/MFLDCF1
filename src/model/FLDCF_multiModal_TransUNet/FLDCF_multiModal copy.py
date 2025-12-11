# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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



import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .model1 import vit_seg_configs as configs
# import .FLDCF_multiModal.vit_seg_configs as configs
from .model1.vit_seg_modeling_resnet_skip import FuseResNetV2


logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"



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
        # out_channels = 512 # changed from 256
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
		
# pyramid feature extraction module
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
    
class BR2(nn.Module):
    def __init__(self, num_classes, stride=1, downsample=None):
        super(BR2, self).__init__()
        self.conv1 = conv3x3(num_classes*2, num_classes*16, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(num_classes*16, num_classes)
        self.stride = stride
        
        self.conv3 = conv3x3(num_classes*2, num_classes, stride)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        ################################
        residual = self.conv3(residual)
        
        ################################
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


class Restoretrain(nn.Module):
    def __init__(self, args):
        super(Restoretrain, self).__init__()
        G0 = 32
        kSize = 3 #3

        # number of RDB blocks, conv layers, out channels
        self.D, C, G =  8, 2, 32

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(3, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        # Up-sampling net
    
        self.UPNet = nn.Sequential(*[
            nn.Conv2d(G0, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.Conv2d(G, 3, kSize, padding=(kSize-1)//2, stride=1)
        ])

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x  = self.SFENet2(f__1)   #64 32 32

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            #draw_features(64, x[0],"./image/encoder{}.png".format(i))
            RDBs_out.append(x) #64 32 32

        x = self.GFF(torch.cat(RDBs_out,1))
        #draw_features(64, x[0],"./image/encoder{}.png".format(i))
        x += f__1  #64 32 32
        result = self.UPNet(x)
    
        return result,RDBs_out


class Restoretest(nn.Module):
    def __init__(self, args):
        super(Restoretest, self).__init__()
        G0 = 32
        kSize = 3 #3

        # number of RDB blocks, conv layers, out channels
        self.D, C, G =  8, 2, 32

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(3, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        # Up-sampling net
    
        self.UPNet = nn.Sequential(*[
            nn.Conv2d(G0, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.Conv2d(G, 3, kSize, padding=(kSize-1)//2, stride=1)
        ])

    def forward(self, x):
        with torch.no_grad():
            f__1 = self.SFENet1(x)
            x  = self.SFENet2(f__1)   #64 32 32

            RDBs_out = []
            for i in range(self.D):
                x = self.RDBs[i](x)
                #draw_features(64, x[0],"./image/encoder{}.png".format(i))
                RDBs_out.append(x) #64 32 32

            x = self.GFF(torch.cat(RDBs_out,1))
            #draw_features(64, x[0],"./image/encoder{}.png".format(i))
            x += f__1  #64 32 32
            result = self.UPNet(x)
    
        return result,RDBs_out

################################################################
def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis, mode=None):
        super(Attention, self).__init__()
        self.vis = vis
        self.mode = mode
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)
        self.out = Linear(config.hidden_size, config.hidden_size)

        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])
        
        self.queryd = Linear(config.hidden_size, self.all_head_size)
        self.keyd = Linear(config.hidden_size, self.all_head_size)
        self.valued = Linear(config.hidden_size, self.all_head_size)
        self.outd = Linear(config.hidden_size, config.hidden_size)

        if self.mode == 'mba':
            self.w11 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.w12 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.w21 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.w22 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.w11.data.fill_(0.5)
            self.w12.data.fill_(0.5)
            self.w21.data.fill_(0.5)
            self.w22.data.fill_(0.5)
        
            # self.gate_sx = nn.Conv1d(config.hidden_size, 1, kernel_size=1, bias=True)
            # self.gate_cx = nn.Conv1d(config.hidden_size, 1, kernel_size=1, bias=True)
            # self.gate_sy = nn.Conv1d(config.hidden_size, 1, kernel_size=1, bias=True)
            # self.gate_cy = nn.Conv1d(config.hidden_size, 1, kernel_size=1, bias=True)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_statesx, hidden_statesy):
        mixed_query_layer = self.query(hidden_statesx)
        mixed_key_layer = self.key(hidden_statesx)
        mixed_value_layer = self.value(hidden_statesx)

        mixed_queryd_layer = self.queryd(hidden_statesy)
        mixed_keyd_layer = self.keyd(hidden_statesy)
        mixed_valued_layer = self.valued(hidden_statesy)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        queryd_layer = self.transpose_for_scores(mixed_queryd_layer)
        keyd_layer = self.transpose_for_scores(mixed_keyd_layer)
        valued_layer = self.transpose_for_scores(mixed_valued_layer)

        ## Self Attention x: Qx, Kx, Vx
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_sx = self.out(context_layer)
        attention_sx = self.proj_dropout(attention_sx)
        
        ## Self Attention y: Qy, Ky, Vy
        attention_scores = torch.matmul(queryd_layer, keyd_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, valued_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_sy = self.outd(context_layer)
        attention_sy = self.proj_dropout(attention_sy)
        
        # return attention_sx, attention_sy, weights
        if self.mode == 'mba':
            # ## Cross Attention x: Qx, Ky, Vy
            attention_scores = torch.matmul(query_layer, keyd_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            attention_probs = self.softmax(attention_scores)
            weights = attention_probs if self.vis else None
            attention_probs = self.attn_dropout(attention_probs)

            context_layer = torch.matmul(attention_probs, valued_layer)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)
            attention_cx = self.out(context_layer)
            attention_cx = self.proj_dropout(attention_cx)
            
            ## Cross Attention y: Qy, Kx, Vx
            attention_scores = torch.matmul(queryd_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            attention_probs = self.softmax(attention_scores)
            weights = attention_probs if self.vis else None
            attention_probs = self.attn_dropout(attention_probs)

            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)
            attention_cy = self.outd(context_layer)
            attention_cy = self.proj_dropout(attention_cy)
        
            # return attention_cx, attention_cy, weights
            
            # ## ADD
            # attention_x = torch.div(torch.add(attention_sx, attention_cx), 2)
            # attention_y = torch.div(torch.add(attention_sy, attention_cy), 2)
            # Adaptative MBA
            attention_sx = self.w11 * attention_sx + self.w12 * attention_cx
            attention_sy = self.w21 * attention_sy + self.w22 * attention_cy
            ## Gated MBA
            # attention_x = self.w11 * attention_sx + (1 - self.w11) * attention_cx
            # attention_y = self.w21 * attention_sy + (1 - self.w21) * attention_cy
            ## SA-GATE MBA
            # attention_sx =  attention_sx.transpose(-1, -2)
            # attention_cx =  attention_cx.transpose(-1, -2)
            # attention_sy =  attention_sy.transpose(-1, -2)
            # attention_cy =  attention_cy.transpose(-1, -2)
            # attention_vector_sx = self.gate_sx(attention_sx)
            # attention_vector_cx = self.gate_cx(attention_cx)
            # attention_vector_sy = self.gate_sy(attention_sy)
            # attention_vector_cy = self.gate_cy(attention_cy)
            # attention_vector_x = torch.cat([attention_vector_sx, attention_vector_cx], dim=1)
            # attention_vector_x = self.softmax(attention_vector_x)
            # attention_vector_y = torch.cat([attention_vector_sy, attention_vector_cy], dim=1)
            # attention_vector_y = self.softmax(attention_vector_y)
            
            # attention_vector_sx, attention_vector_cx = attention_vector_x[:, 0:1, :], attention_vector_x[:, 1:2, :]
            # attention_x = (attention_sx*attention_vector_sx + attention_cx*attention_vector_cx).transpose(-1, -2)
            # attention_vector_sy, attention_vector_cy = attention_vector_y[:, 0:1, :], attention_vector_y[:, 1:2, :]
            # attention_y = (attention_sy*attention_vector_sy + attention_cy*attention_vector_cy).transpose(-1, -2)
        
        return attention_sx, attention_sy, weights

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = FuseResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.patch_embeddingsd = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])


    def forward(self, x, y):
        # print(x.shape)
        # print(y.shape)
        # y = y.unsqueeze(1)
        if self.hybrid:
            # print(x.shape)
            # print(y.shape)
            x, y, features = self.hybrid_model(x, y)
            # print(x.shape)
            # print(y.shape)
            # print(len(features))
        else:
            features = None
        x = self.patch_embeddings(x)
        # print(x.shape)
        # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        y = self.patch_embeddingsd(y)
        # print(y.shape)

        size_conv1 = x.size()[2:]

        x = x.flatten(2)
        # print(x.shape)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        # print(x.shape)
        y = y.flatten(2)
        # print(y.shape)
        y = y.transpose(-1, -2)
        # print(y.shape)
        
        embeddingsx = x + self.position_embeddings
        embeddingsx = self.dropout(embeddingsx)
        # print(embeddingsx.shape)
        embeddingsy = y + self.position_embeddings
        embeddingsy = self.dropout(embeddingsy)
        # print(embeddingsy.shape)


        res2 = embeddingsx
        return embeddingsx, embeddingsy, features,  res2, size_conv1


class Block(nn.Module):
    def __init__(self, config, vis, mode=None):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.attention_normd = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_normd = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.ffnd = Mlp(config)
        self.attn = Attention(config, vis, mode=mode)

    def forward(self, x, y):
        hx = x
        hy = y
        x = self.attention_norm(x)
        y = self.attention_normd(y)
        x, y, weights = self.attn(x, y)
        x = x + hx
        y = y + hy

        hx = x
        hy = y
        x = self.ffn_norm(x)
        y = self.ffn_normd(y)
        x = self.ffn(x)
        y = self.ffnd(y)
        x = x + hx
        y = y + hy
        return x, y, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)
            
            self.attn.queryd.weight.copy_(query_weight)
            self.attn.keyd.weight.copy_(key_weight)
            self.attn.valued.weight.copy_(value_weight)
            self.attn.outd.weight.copy_(out_weight)
            self.attn.queryd.bias.copy_(query_bias)
            self.attn.keyd.bias.copy_(key_bias)
            self.attn.valued.bias.copy_(value_bias)
            self.attn.outd.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)
            
            self.ffnd.fc1.weight.copy_(mlp_weight_0)
            self.ffnd.fc2.weight.copy_(mlp_weight_1)
            self.ffnd.fc1.bias.copy_(mlp_bias_0)
            self.ffnd.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.attention_normd.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_normd.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_normd.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_normd.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.encoder_normd = LayerNorm(config.hidden_size, eps=1e-6)
        for i in range(config.transformer["num_layers"]):
            ## 12+0
            # if i >= 0 :
            ## 3+6+3
            if i < 3 or i > 8:
            # ## 1+1+1+1...
            # if i % 2 == 0:
                layer = Block(config, vis, mode='sa')
            else:
                layer = Block(config, vis, mode='mba')
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_statesx, hidden_statesy): # FVit
        attn_weights = []
        for layer_block in self.layer:
            hidden_statesx, hidden_statesy, weights = layer_block(hidden_statesx, hidden_statesy)
            if self.vis:
                attn_weights.append(weights)
        encodedx = self.encoder_norm(hidden_statesx)
        encodedy = self.encoder_normd(hidden_statesy)
        return encodedx, encodedy, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids, dsm_ids):
        embeddingsx, embeddingsy, features, res2, size_conv1 = self.embeddings(input_ids, dsm_ids) ## Local Feature Block of Encoder
        encodedx, encodedy, attn_weights = self.encoder(embeddingsx, embeddingsy)  # (B, n_patch, hidden) ## Global Feature Block of Encoder
        # encodedx = embeddingsx #
        # encodedy = embeddingsy #
        # attn_weights = None #
        return encodedx, encodedy, attn_weights, features, res2, size_conv1
        # return features, res2, size_conv1
        # return embeddingsx, embeddingsy, None, features, res2, size_conv1


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        num_classes=2
        self.conv1 = Conv2dReLU(
            # in_channels + skip_channels,
            # num_classes*2,
            num_classes,
            # out_channels,
            num_classes*16,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            # out_channels,
            num_classes*16,
            # out_channels,
            num_classes,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv3 = Conv2dReLU(
            # in_channels + skip_channels,
            num_classes,
            # out_channels,
            num_classes*16,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        # self.conv3 = Conv2dReLU(
        #     in_channels,
        #     out_channels,
        #     kernel_size=3,
        #     padding=1,
        #     use_batchnorm=use_batchnorm,
        # )

        # self.conv3 = nn.Conv2d(
        #     in_channels,
        #     out_channels,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1,
        #     bias=not (use_batchnorm),
        # )

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        
        ################################
        # self.conv1 = conv3x3(in_channels + skip_channels, out_channels*16, stride=1)
        # self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(out_channels*16, out_channels)
        # self.conv3 = conv3x3(in_channels, out_channels*16, stride=1)
        # self.stride = stride
        # print(in_channels)
        self.predict1 = self._predict_layer(in_channels, num_classes=2)
        # print(skip_channels)
        # print()
        if skip_channels != 0:
            self.predict2 = self._predict_layer(skip_channels, num_classes=2)
        self.predict3 = self._predict_layer(num_classes+num_classes, num_classes=2)
        self.br1 = BR(num_classes=2)
        self.br2 = BR2(num_classes=2)
        ################################
        
        ################################
    def _predict_layer(self, in_channels, num_classes):
        return nn.Sequential(nn.Conv2d(in_channels, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
			nn.ReLU(True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=3, stride=1, padding=1, bias=True))
        
        ################################
        
        
        

    def forward(self, x, skip=None):
        # x = self.predict1(x)
        # x = self.up(x) #
        # print(x.shape)
        # print(abc)
        # size = x.shape
        # size[2] = size[2] * 2
        # size[3] = size[3] * 2
        # print(size)
        # print(size[3])
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        residual = x #
        # print(x.shape, 1)
        if skip is not None:
            skip = self.predict2(skip)
            # print(skip.shape, 2)
            x = torch.cat([x, skip], dim=1)
            x = self.predict3(x)
            # residual = torch.cat([residual, skip], dim=1)
            # residual = self.predict3(residual)
            # x = self.conv1(x)
            # x = self.conv2(x)
            # residual = self.conv1(residual)
            # residual = self.conv2(residual)
            x = self.br1(x)
        
        else:
            # x = self.conv3(x)
            # # x = self.relu(x) #
            # x = self.conv2(x)
            # residual = self.conv3(residual)
            # residual = self.conv2(residual)

            x = self.br1(x)
        
        # print(x.shape, 1)
        # x = x + residual
        return x #


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        # skip_channels=[0,0,0,0] #

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        # print(in_channels)
        # print(out_channels)
        # print(skip_channels)

        ################################
        def _predict_layer(in_channels, num_classes):
            return nn.Sequential(nn.Conv2d(in_channels, 256, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(256),
			    nn.ReLU(True),
                nn.Dropout2d(0.1),
                nn.Conv2d(256, num_classes, kernel_size=3, stride=1, padding=1, bias=True))

        self.fpm1 = _FPM(512, num_classes=2)
        # self.fpm1 = _FPM(in_channels[0], num_classes=2)
        # self.fpm1 = _FPM(config.hidden_size, num_classes=2)
        print(config.hidden_size)
        # self.predict1 = _predict_layer(in_channels[0], num_classes=2)
        self.predict1 = _predict_layer(512*6, num_classes=2)
        self.predict1_withoutfpm1 = _predict_layer(512, num_classes=2)
        # self.predict1 = _predict_layer(512*5+768, num_classes=2)
        self.br1 = BR(num_classes=2)
        self.br5 = BR(num_classes=2)	
        self.br6 = BR(num_classes=2)	
        self.br7 = BR(num_classes=2)

        ################################

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)

        ################################
        # print(x.shape)
        # x = self.fpm1(x)
        # print(x.shape)

        # x = self._predict_layer(in_channels, num_classes=2) 	# 1/8	
        # print(x.shape)

        # x = self.br1(x)
        # print(x.shape)
        ################################

        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None

            # skip = None #
            
            if i==0:
                # print(x.shape)
                x = self.fpm1(x) ##
                # print(x.shape)
                x = self.predict1(x) ##
                # x = self.predict1_withoutfpm1(x) ##
                # print(x.shape)
                x = self.br1(x)
                # print(x.shape)
                # print(abc)
            x = decoder_block(x, skip=skip)
            # if i==0:
            #     print(x.shape)
            #     x = self.predict1(x)
            #     x = self.br5(x)
            # elif i==1:
            #     x = self.predict1(x)
            #     x = self.br6(x)
            # elif i==1:
            # if i==1:
            #     x = self.fpm1(x)
            #     x = self.predict1(x)
            #     x = self.br7(x)
        return x


################################################################

class FLDCF_multiModal(nn.Module): #FLDCF_multiModal
    def __init__(self, args,config, block=Bottleneck, layers=[1, 2, 2, 1], num_classes=2, aux=True, \
                img_size=256, num_classes_2=6, zero_head=False, vis=False):
        super(FLDCF_multiModal, self).__init__()
        self.inplanes = 64
        self.aux = aux	
        # super(FLDCF_multiModal, self).__init__()

        ################################################################
        self.num_classes_2 = num_classes_2
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config
        # print(config.skip_channels)
        ################################################################

        self.learned = 	Restoretest(args)
        self.learned.load_state_dict(
                # torch.load(
                #     # os.path.join('./model', 'model_vi.pt'),
                #     os.path.join('/media/lscsc/nas/mading/fakedetect/model', 'model_vi_gray.pt'),  
                # ),
                torch.load(
                # os.path.join('./model', 'model_vi.pt'),
                os.path.join('/media/lscsc/nas/mading/fakedetect/model', 'model_vi_sjl_rgb.pt'),  
                ),
                strict=True
            )
        
        self.handlelern1 = nn.Conv2d(32 *4,64, kernel_size=3, stride=2, padding=1, bias=False)
        self.handlelern2 = nn.Conv2d(32 *4,64, kernel_size=3, stride=2, padding=1, bias=False)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine = affine_par)
        self.conv2 = nn.Conv2d(64*2, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, affine = affine_par)		
        self.conv3 = nn.Conv2d(64*2, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64, affine = affine_par)
		
		
        for i in self.bn1.parameters():
            i.requires_grad = False
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
					
        # self.fpm1 = _FPM(512, num_classes)
        self.fpm1 = _FPM(16, num_classes)
        #  def __init__(self, in_channels, num_classes, norm_layer=nn.BatchNorm2d):
        # self.fpm1 = _FPM(768, num_classes)
        # self.fpm1 = _FPM(8, num_classes)
        #self.fpm2 = _FPM(512, num_classes)
        #self.fpm3 = _FPM(256, num_classes)	
		
        self.br1 = BR(num_classes)	
        #self.br2 = BR(num_classes)	
        #self.br3 = BR(num_classes)			
        #self.br4 = BR(num_classes)	
        self.br5 = BR(num_classes)	
        self.br6 = BR(num_classes)	
        self.br7 = BR(num_classes)			
		

        # self.predict1 = self._predict_layer(512*6, num_classes)
        self.predict1 = self._predict_layer(16, num_classes)	
        #self.predict2 = self._predict_layer(512*6,num_classes)			
        #self.predict3 = self._predict_layer(512*5+256,num_classes)

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
            nn.Conv2d(5, 3, 1, padding=0, stride=1),
            nn.Conv2d(3, 3, 3, padding=(3-1)//2, stride=1)
        ])

        ################################
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = Conv2dReLU(
            512 + 512,
            256,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        self.conv2 = Conv2dReLU(
            256,
            256,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        ################################

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = False
		
    def _predict_layer(self, in_channels, num_classes):
        return nn.Sequential(nn.Conv2d(in_channels, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
			nn.ReLU(True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=3, stride=1, padding=1, bias=True))

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
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

    
    def base_forward(self, x,RDNout):
        # draw_features(64, RDNout[4][0],"{}/0.png".format('./image'))
        # draw_features(64, RDNout[5][0],"{}/1.png".format('./image'))
        # draw_features(64, RDNout[6][0],"{}/2.png".format('./image'))
        # draw_features(64, RDNout[7][0],"{}/3.png".format('./image'))
        rdn1 = self.handlelern1(torch.cat(RDNout[0:4], 1))
        rdn2 = self.handlelern2(torch.cat(RDNout[4:8], 1))
        x = self.relu(self.bn1(self.conv1(x)))
        size_conv1 = x.size()[2:]

        # print(x.shape)
        # print(rdn1.shape)
        # print(torch.cat([x, rdn1],dim=1).shape)
        # print(abc)

        x = self.relu(self.bn2(self.conv2(torch.cat([x, rdn1],dim=1))))
        # draw_features(64, rdn2[0],"{}/rdn.png".format('./image'))
        # draw_features(64, x[0],"{}/x.png".format('./image'))
        x = self.relu(self.bn3(self.conv3(torch.cat([x, rdn2],dim=1))))
        x = self.maxpool(x)
        x = self.layer1(x)
        res2 = x
        x = self.layer2(x)
        res3 = x
        x = self.layer3(x)
        res4 = x
        x = self.layer4(x)
        x = self.res5_con1x1(torch.cat([x, res4], dim=1))

		
        return x, res3, res2, size_conv1
			
    def forward(self, x, y):
        # Localization network
        # Encoder
        x1 = x.clone()
        # result, RDNout = self.learned(x)
        # b,c,w,h = x.size()
        size = x.size()[2:]
        # score1, score2, score3,  size_conv1 = self.base_forward(x,RDNout)
        # # return x, res3, res2, size_conv1
        # print(x.shape)
        score1, y, attn_weights, features, score3, size_conv1 = self.transformer(x, y)  # (B, n_patch, hidden)
        # print(score1.shape) #([8,256,768])
        # print(y.shape)
        score1 = score1 + y
        # print(score1.shape)
        
        # score1 = self.fpm1(score1)
        
        # score1 = score1  # 
        # print(score1.shape)
        # x, y, attn_weights, features = self.transformer(x, y)  # (B, n_patch, hidden)
        # x = x + y

        # # ########################################
        # hidden_states = score1
        # B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        # h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        # x = hidden_states.permute(0, 2, 1)
        # x = x.contiguous().view(B, hidden, h, w)
        # x = self.conv_more(x)
        # score1 = x
        # # ########################################

        # print(score1.shape)
        # score1 = self.fpm1(score1)
        # # print(score1.shape)

        # score1 = self.predict1(score1) 	# 1/8	
        # # # print(score1.shape)

        # score1 = self.br1(score1)
        # score2 = score1
        # # # print(score2.shape)


        
		
        ################################################################
		# Localization network
        # Decoder
        score3 = self.decoder(score1, features)
        # second fusion	
        # size_score3 = score3.size()[2:]

        # skip = features[0]
        # score3 = self.up(score1)
        # # score3 = F.interpolate(score2, size_score3, mode='bilinear', align_corners=True)			
        # score3 = torch.cat([score3, skip], dim=1)
        # score3 = self.conv1(score3)
        # score3 = self.conv2(score3)
        # print(score3.shape)
        # score3 = self.br5(score3)
        #draw_features(64, score3[:,0],"{}/decoder1.png".format('./image'))
        # print(score3.shape)
        # print(abc1)
		
        # upsampling + BR	
        # skip = features[1]
        # score3 = torch.cat([score3, skip], dim=1)
        # score3 = F.interpolate(score3, size_conv1, mode='bilinear', align_corners=True) 
        # score3 = torch.cat([score3, skip], dim=1)		
        # score3 = self.br6(score3)

        # skip = features[2]
        # score3 = torch.cat([score3, skip], dim=1)
        #draw_features(64, score3[:,0],"{}/decoder2.png".format('./image'))
        # score3 = F.interpolate(score3, size, mode='bilinear', align_corners=True)
        # score3 = torch.cat([score3, skip], dim=1)
        # score3 = self.br7(score3)

        # print(score3.shape)
        # print(abc)
        # score1 = self.fpm1(score3)
        # score3 = self.predict1(score3)
        # score3 = self.br1(score3)
        # score3 = self.br5(score3)
        # score3 = self.br6(score3)
        # score3 = self.br7(score3)  	
        
        ################################################################
        # Detection network
        score32 = score3.clone().detach() 
        # print(score32.shape)
        y = torch.cat([x1,score32],dim =1)
        # print(y.shape)
        y = self.GFF(y)
        out = self.res(y)

        return score3, out
    
    
    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.patch_embeddingsd.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddingsd.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))
            self.transformer.encoder.encoder_normd.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_normd.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                ws = res_weight["conv_root/kernel"]
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(ws, conv=True))
                ws = np.expand_dims(np.mean(ws, axis=2), axis=2)
                self.transformer.embeddings.hybrid_model.rootd.conv.weight.copy_(np2th(ws, conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)
                self.transformer.embeddings.hybrid_model.rootd.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.rootd.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)
                for bname, block in self.transformer.embeddings.hybrid_model.bodyd.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)
            print('Load pretrained done.')

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}


