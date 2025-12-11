import os
from importlib import import_module

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from torchstat import stat
import torchvision.models as models

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

from .FLDCF_multiModal.FLDCF_multiModal import CONFIGS as CONFIGS_ViT_seg
# /media/lscsc/nas/mading/fakedetect/src/model/FLDCF_multiModal/FLDCF_multiModal.py
# /media/lscsc/nas/mading/fakedetect/src/model/FLDCF_multiModal/model/vitcross_seg_modeling.py

def get_model(name,args):
    if name == 'crnet':
        from .crnet_small.crnet import CDnetV1_MODEL as M
    if name == 'CDnetV2':
        from .CDnetV2.CDnetv2_model import CDnetV2_MODEL as M
    if name == 'face':
        from .faceforensics.face import TransferModel as M
    if name == 'deepfake':
        from .deepfake.deepfake import Deepfake
        return Deepfake()
    if(name =='patch'):
        from .patchfake.patch import create_model
        return create_model() 
    if(name =='capsule'):
        from .capsule.capsule import CapsuleNet as M
    if(name =='scunet'):
        from .scunet.scunet import SCSEUnet as M
    if(name =='dfcn'):
        from .dfcn.dfcn import normal_denseFCN as M
    if(name =='my2'): # FLDCF
        from .my.my import Restore2 as M
    if(name =='mylocal'):
        from .my.my import Restorelocal as M
    if(name =='restore'):
        from .my.my import Restoretrain as M
    if(name =='FLDCF2'): # not FLDCF2
        from .FLDCF2.FLDCF2 import FLDCF2 as M
    if(name=='mvss'):
        from .mvss.mvssnet import get_mvss

        print('==> Building model..')
        model = get_mvss(args, num_classes=2).cuda()
        dummy_input = torch.randn(1, 3, 256, 256).cuda()
        flops, params = profile(model, inputs=(dummy_input,))
        print('flops: ', flops, 'params: ', params)
        print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))

        return get_mvss(backbone='resnet50',
                             pretrained_base=True,
                             nclass=1,
                             sobel=True,
                             constrain=True,
                             n_input=3).cuda()
    if(name =='movenet'): # SE-Network
        from .movenet.movenet import Movenet as M
    if(name =='FLDCF_multiModal'): # FLDCF2
        from .FLDCF_multiModal.FLDCF_multiModal import FLDCF_multiModal as M
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = 6
        config_vit.n_skip = 3
        config_vit.patches.grid = (int(256 / 16), int(256 / 16))

        print('==> Building model..')
        model = M(args, config_vit, num_classes=2).cuda()
        dummy_input = torch.randn(1, 3, 256, 256).cuda()
        dummy_input1 = torch.randn(1, 1, 256, 256).cuda()
        flops, params = profile(model, inputs=(dummy_input,dummy_input1))
        print('flops: ', flops, 'params: ', params)
        print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
        return M(args, config_vit), config_vit
    if(name =='FLDCF_multiModal_TransUNet'): # FLDCF2_multiModal_TransUNet
        from .FLDCF_multiModal_TransUNet.FLDCF_multiModal_TransUNet import FLDCF_multiModal_TransUNet as M
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = 6
        config_vit.n_skip = 3
        config_vit.patches.grid = (int(256 / 16), int(256 / 16))

        print('==> Building model..')
        model = M(args, config_vit, num_classes=2).cuda()
        dummy_input = torch.randn(1, 3, 256, 256).cuda()
        dummy_input1 = torch.randn(1, 1, 256, 256).cuda()
        # flops, params = profile(model, inputs=(dummy_input,dummy_input1))
        flops, params = profile(model, inputs=(dummy_input,))
        print('flops: ', flops, 'params: ', params)
        print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
        return M(args, config_vit), config_vit
    
    

    print('==> Building model..')
    # model = M(args, num_classes=2).cuda()
    model = M(args).cuda()
    dummy_input = torch.randn(1, 3, 256, 256).cuda()
    flops, params = profile(model, inputs=(dummy_input,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))

    return M(args), None




class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Making '+args.model+'...')

        self.args = args
        self.scale = int(args.scale)
        self.self_ensemble = args.self_ensemble
        self.chop = args.chop
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models
        #choose model
        # self.model, config_vit = get_model(args.model,args).to(self.device)
        self.model, config_vit = get_model(args.model,args) #2025.3.11
        # self.model = get_model(args.model,args).to(self.device) ##
        # self.model = get_model(args.model,args)
        # print(type(self.model))
        # print(1)
        # print(self.model)
        self.model = self.model.to(self.device) #2025.3.24
        # if (args.model == 'FLDCF_multiModal' or args.model == 'FLDCF_multiModal_TransUNet'):
        #     self.model.load_from(weights=np.load(config_vit.pretrained_path))
        print('Total params: %.2fM' % (sum(p.numel() for p in self.model.parameters())/1000000.0))

        # stat(self.model, (3, 224, 224), (1, 224, 224)) ###



        # # 查看模型在GPU上的内存占用情况
        # allocated_memory = torch.cuda.memory_allocated(device=self.device)
        # cached_memory = torch.cuda.memory_cached(device=self.device)
        # print(f"Allocated memory: {allocated_memory/1024/1024}")
        # print(f"Cached memory: {cached_memory/1024/1024}")




        if args.precision == 'half': self.model.half()
        if not args.cpu and args.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(args.n_GPUs))

        self.load(
            ckp.dir,
            pre_train=args.pre_train,
            resume=args.resume,
            cpu=args.cpu
        )
        print(self.model, file=ckp.log_file)

    # def forward(self, x, y):
    def forward(self, x):
        if self.self_ensemble and not self.training:
            if self.chop:
                forward_function = self.forward_chop
            else:
                forward_function = self.model.forward

            return self.forward_x8(x, forward_function)
        elif self.chop and not self.training:
            # return self.forward_chop(x, y)  #test的正常情况
            return self.forward_chop(x)  #test的正常情况
        else:
            # # if (self.args.model == "FLDCF_multiModal" or self.args.model == "FLDCF_multiModal_TransUNet"):
            if (self.args.model == "FLDCF_multiModal" ):
                # return self.model(x, y)  #train的正常情况
                return self.model(x)  #train的正常情况
            else: return self.model(x)  #train的正常情况

    def forward(self, x, y):
    # def forward(self, x):
        if self.self_ensemble and not self.training:
            if self.chop:
                forward_function = self.forward_chop
            else:
                forward_function = self.model.forward

            return self.forward_x8(x, forward_function)
        elif self.chop and not self.training:
            return self.forward_chop(x, y)  #test的正常情况
            # return self.forward_chop(x)  #test的正常情况
        else:
            # # if (self.args.model == "FLDCF_multiModal" or self.args.model == "FLDCF_multiModal_TransUNet"):
            if (self.args.model == "FLDCF_multiModal" ):
                return self.model(x, y)  #train的正常情况
                # return self.model(x)  #train的正常情况
            else: return self.model(x)  #train的正常情况

    def get_model(self):
        if self.n_GPUs == 1:
            return self.model
        else:
            return self.model.module

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    def save(self, apath, epoch, is_best=False):
        target = self.get_model()
        torch.save(
            target.state_dict(), 
            os.path.join(apath, 'model_latest.pt')
        )
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model_best.pt')
            )
        
        if self.save_models:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model_{}.pt'.format(epoch))
            )

    def load(self, apath, pre_train='.', resume=-1, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        if resume == -1:
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model_latest.pt'),
                    **kwargs
                ),
                strict=False
            )
        elif resume == 0:
            if pre_train != '.':
                print('Loading model from {}'.format(pre_train))
                self.get_model().load_state_dict(
                    torch.load(pre_train, **kwargs),
                    strict=False
                )
        else:
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model', 'model_{}.pt'.format(resume)),
                    **kwargs
                ),
                strict=False
            )

    def forward_chop(self, x, shave=10, min_size=120000):
        scale = self.scale
        n_GPUs = min(self.n_GPUs, 4)
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        h_size +=4-h_size%4
        w_size +=8-w_size%8
        
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]

        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                sr_batch = self.model(lr_batch)
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [
                self.forward_chop(patch, shave=shave, min_size=min_size) \
                for patch in lr_list
            ]

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half  #这里默认输入输出不是一个格式，而是倍数的格式，不太好改，所以暂时关掉chop
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        output = x.new(b, c, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output

    def forward_x8(self, x, forward_function):
        def _transform(v, op):
            if self.precision != 'single': v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            if self.precision == 'half': ret = ret.half()

            return ret

        lr_list = [x]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])

        sr_list = [forward_function(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)

        return output

