import numpy as np
from glob import glob
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix
import random, time
import itertools
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
import torch
import torchvision
from thop import profile
from utils import *
from torch.autograd import Variable
from IPython.display import clear_output
from model.UNetFormer import UNetFormer as UNetFormer
from model.FTUNetFormer import ft_unetformer as FTUNetFormer
from model.ABCNet import ABCNet
from model.CMTFNet.CMTFNet import CMTFNet
from model.MANet import MANet
from model.UNetformer_Embed import UNetformer_Embed, load_pretrained_ckpt
#from model.UNetFormerGML import UNetFormerGML as UNetFormerGML
#from model.UNetFormerGVML import UNetFormerGVML as UNetFormerGVML
from model.RSMamba import RSMamba
from model.SIINet.models.SIIS_NET import Resnet_SIIS
from model.HST_UNet import HST_UNet as HST
from model.TransUNet.networks.vit_seg_modeling import VisionTransformer as TransUNet
from model.TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from model.FLDCF_multiModal.FLDCF_multiModal import FLDCF_multiModal

try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener

if MODEL == 'UNetformer':
    net = UNetFormer(num_classes=N_CLASSES).cuda()
elif MODEL == 'FTUNetformer':
    net = FTUNetFormer(num_classes=N_CLASSES).cuda()
elif MODEL == 'ABCNet':
    net = ABCNet(num_classes=N_CLASSES).cuda()
elif MODEL == 'CMTFNet':
    net = CMTFNet(num_classes=N_CLASSES).cuda()
elif MODEL == 'MANet':
    net = MANet(num_classes=N_CLASSES).cuda()
elif MODEL == 'UNetformer_Embed':
    net = UNetformer_Embed(num_classes=N_CLASSES).cuda()
    #net = load_pretrained_ckpt(net)
#elif MODEL == 'UNetformerGML':
#    net = UNetFormerGML(num_classes=N_CLASSES).cuda()
#elif MODEL == 'UNetformerGVML':
#    net = UNetFormerGVML(num_classes=N_CLASSES).cuda()
elif MODEL == 'RSMamba':
    net = RSMamba(num_classes=N_CLASSES).cuda()
elif MODEL == 'Resnet_SIIS':
     net = Resnet_SIIS(num_classes=N_CLASSES).cuda()
elif MODEL == 'HST':
     net = HST(num_classes=N_CLASSES).cuda()
elif MODEL == 'TransUNet':
    config_vit = CONFIGS_ViT_seg['R50-ViT-L_16']
    config_vit.n_classes = N_CLASSES
    config_vit.n_skip = 3
    config_vit.patches.grid = (int(256 / 16), int(256 / 16))
    net = TransUNet(config_vit, img_size=256, num_classes=N_CLASSES).cuda()
    # net.load_from(weights=None)


print('==> Building model..')
model = net

dummy_input = torch.randn(1, 3, 256, 256).cuda()
flops, params = profile(model, inputs=(dummy_input,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
