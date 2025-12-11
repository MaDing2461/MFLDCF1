# import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" #在卡2调试
import torch
import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import glob
from thop import profile
import time
import torch.optim as optim

import numpy as np
from glob import glob
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix
import random
import time
import itertools
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from utils import *
from torch.autograd import Variable
from IPython.display import clear_output




torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args) 
#运行前务必改epoch！！！！！sjilu

os.environ["CUDA_VISIBLE_DEVICES"] = "1" #在卡2调试
print(torch.__version__)
def main():
    global model
    if checkpoint.ok:
        loader = data.Data(args)
        _model = model.Model(args, checkpoint)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 查看模型在GPU上的内存占用情况
        allocated_memory = torch.cuda.memory_allocated(device=device)
        cached_memory = torch.cuda.memory_cached(device=device)
        print(f"Allocated memory: {allocated_memory/1024/1024}")
        print(f"Cached memory: {cached_memory/1024/1024}")
        
        
        # input = torch.randn(16, 3, 256, 256).cuda()
        # flops, params = profile(_model, inputs=(input, ))
        # print('Complexity: %.3fM' % (flops/1000000000), end=' GFLOPs\n')
        # torch.cuda.synchronize()
        # time_start = time.time()
        # predict = _model(input)
        # torch.cuda.synchronize()
        # time_end = time.time()
        # print('Speed: %.5f FPS\n' % (1/(time_end-time_start)))
        # optimizer = optim.SGD(_model.parameters(), lr=0.9, momentum=0.9, weight_decay=0.0005)
        # for _ in range(1000):
        #     optimizer.zero_grad()
        #     _model(input)

        _loss = loss.Loss_fake() if not args.test_only else None
        t = Trainer(args, loader, _model, _loss, checkpoint)
        # def __init__(self, args, loader, my_model, my_loss, ckp):
        # def __init__(self, args, loader, my_model, my_loss, ckp):
        # def __init__(self, args, loader, my_model, my_loss, ckp):
        # def __init__(self, args, loader, my_model, my_loss, ckp):
        t.testtrain(is_train=True)
        checkpoint.done()

if __name__ == '__main__':
    main()

# cd /media/lscsc/nas/mading/fakedetect
# python src/main.py --save my3
# nohup python -u src/main.py > main_1 2>&1 &
# python src/main.py --save scseunetVDSM1
# python src/main.py --save mvssnetVDSM1
# python src/main.py --save movenetVDSM1

# python src/main.py --model capsule --save capsuleVDSM1
# python src/main.py --model face --save faceVDSM1
# python src/main.py --model deepfake --save deepfakeVDSM1

# python src/main.py --model crnet --save crnetVDSM1 
# python src/main.py --model my2 --save my2VDSM1 
# python src/main.py --model restore --save restoreVDSM1 
# python src/main.py --model my2 --save my2VDSM1 

# python src/main.py --model restore --dir_data /media/lscsc/nas/jialu/data --data_train_dir fakeV --save restoreRGB1 
# python src/main.py --model my2 --dir_data /media/lscsc/nas/jialu/data --data_train_dir fakeV --save my2VRGB1 

# python src/main.py --model restore --save restoreVRGB1
# python src/main.py --model restore --save restoreVRGB2
# python src/main.py --model my2 --save my2VRGB1

# python src/main.py --model restore --save restoreVDSM2
# python src/main.py --model my2 --save my2VDSM2
# python src/main.py --model FLDCF2 --save FLDCF2VDSM1

# python src/main.py --model my2 --save my2VDSM3
# python src/main.py --model scunet --save scseunetVDSM1
# python src/main.py --model FLDCF2 --save FLDCF2VDSM1

# python src/main.py --model scunet --save scseunetVDSM2
# python src/main.py --model my2 --save my2VDSM4
# python src/main.py --model mylocal --save mylocalVDSM1

# python src/main.py --model my2 --save my2VDSM4
# python src/main.py --model FLDCF2 --save FLDCF2VDSM3

# python src/main.py --model CDnetV2 --save CDnetV2VDSM1
# python src/main.py --model my2 --save my2VDSM5
# python src/main.py --model crnet --save crnetVDSM2

# python src/main.py --model CDnetV2 --save CDnetV2VDSM1

# python src/main.py --model FLDCF_multiModal --save FLDCF_multiModalV_NIRRG_DSM1
# python src/main.py --model FLDCF_multiModal --save FLDCF_multiModalV_NIRRG_withGTDSM_DSM1
# python src/main.py --model FLDCF_multiModal --save FLDCF_multiModalV_NIRRG_withGTDSM_withDeep_withoutShallow
# CUDA_VISIBLE_DEVICES=1 nohup python -u test.py > LOVEDA_train_2_1 2>&1 &


# python src/main.py --model FLDCF_multiModal --save FLDCF_multiModalV_NIRRG_withGTDSM_withDeep_withShallow
# python src/main.py --model FLDCF_multiModal --save FLDCF_multiModalV_NIRRG_withGTDSM_withDeep_withShallow2
# python src/main.py --model FLDCF_multiModal --save FLDCF_multiModalV_NIRRG_withGTDSM_withDeep_withShallow3
# python src/main.py --model FLDCF_multiModal --save FLDCF_multiModalV_NIRRG_withGTDSM_withDeep_withShallow_inpainted
# python src/main.py --model FLDCF_multiModal --save FLDCF_multiModalV_NIRRG_withGTDSM_withDeep_withShallow_lama

# python src/main.py --model FLDCF_multiModal --save FLDCF_multiModalV_modifiedEncoder_NIRRG_withGTDSM_withDeep_withShallow_lama
# python src/main.py --model FLDCF_multiModal --save FLDCF_multiModalV_modifiedEncoder_NIRRG_withGTDSM_withDeep_withShallow_all1
# python src/main.py --model FLDCF_multiModal --save FLDCF_multiModalV_modifiedEncoder_NIRRG_withGTDSM_withDeep_withShallow_repaint

# python src/main.py --model my2 --save my2VDSM5 --data_train Vaihingen
# python src/main.py --model FLDCF_multiModal --save FLDCF_multiModalV_modifiedEncoder_NIRRG_withGTDSM_withDeep_withShallow_lama

# python src/main.py --model capsule --save capsule_Vaihingen_CycleGAN 
# python src/main.py --model face --save face_Vaihingen_CycleGAN 
# python src/main.py --model deepfake --save deepfake_Vaihingen_CycleGAN

# python src/main.py --model FLDCF_multiModal --save FLDCF_multiModalV_modifiedEncoder7_NIRRG_withGTDSM_withDeep_withShallow_lama


# cd /media/lscsc/nas/mading/fakedetect

# python src/main.py --model FLDCF_multiModal --save FLDCF_multiModalV_modifiedEncoder7_NIRRG_withGTDSM_withDeep_withShallow_all
# python src/main.py --model FLDCF_multiModal --save FLDCF_multiModalV_modifiedEncoder7_NIRRG_withGTDSM_withDeep_withShallow_lama

# python src/main.py --model capsule --save Capsule_Vaihingen_NIRRG_all
# python src/main.py --model face --save FaceForensics_Vaihingen_NIRRG_all
# python src/main.py --model deepfake --save DeepFake-Detection_Vaihingen_NIRRG_all

# python src/main.py --model FLDCF_multiModal --save FLDCF_multiModalV_modifiedEncoder7_NIRRG_withGTDSM_withDeep_withShallow_all_withoutPyramidFeatureExtractionModule

# python src/main.py --model FLDCF_multiModal --save FLDCF_multiModalV_modifiedEncoder7_NIRRG_withGTDSM_withDeep_withShallow_all_withoutDSM

# python src/main.py --model FLDCF_multiModal --save FLDCF_multiModalV_modifiedEncoder7_NIRRG_withGTDSM_withDeep_withShallow_all_withoutGlobalFeatureBlock

# python src/main.py --model FLDCF_multiModal --save FLDCF_multiModalV_modifiedEncoder7_NIRRG_withGTDSM_withDeep_withShallow_all_withoutLocalFeatureBlock

# python src/main.py --model scunet  --save SCSE-Unet_V_all

# python src/main.py --model mvss --save MVSSNet_V_all

# python src/main.py --model movenet --save SE-Network_V_all

# python src/main.py --model my2 --save FLDCF_V_all

# python src/main.py --model FLDCF_multiModal --save FLDCF_multiModalV_modifiedEncoder7_NIRRG_withGTDSM_withDeep_withShallow_all

# python src/main.py --model FLDCF_multiModal_TransUNet --save FLDCF_multiModal_TransUNet_V_modifiedEncoder7_NIRRG_withoutGTDSM_withDeep_withShallow_all

# python src/main.py --model FLDCF_multiModal --save FLDCF_multiModalV_modifiedEncoder7_NIRRG_withFakeDSM_withDeep_withShallow_all

# python src/main.py --model FLDCF_multiModal --save FLDCF_multiModal_Potsdam_modifiedEncoder7_NIRRG_withGTDSM_withDeep_withShallow_all

# python src/main.py --model  --save FLDCF_multiModal_Potsdam_modifiedEncoder7_NIRRG_withGTDSM_withDeep_withShallow_all

# python src/main.py --model scunet  --save SCSE-Unet_Potsdam_all

# python src/main.py --model mvss --save MVSSNet_Potsdam_all

# python src/main.py --model movenet --save SE-Network_Potsdam_all

# python src/main.py --model my2 --save FLDCF_Potsdam_all

# python src/main.py --model restore --save restore_Postdam_DSM

# python src/main.py --model capsule --save Capsule_Potsdam_NIRRGAndDSM_all

# python src/main.py --model face --save FaceForensics_Potsdam_NIRRGAndDSM_all

# python src/main.py --model deepfake --save DeepFake-Detection_Potsdam_NIRRGAndDSM_all

# cd /media/lscsc/nas/mading/fakedetect
# python src/main.py --model FLDCF_multiModal_TransUNet --save FLDCF_multiModal_TransUNet_WHU-GCD_modifiedEncoder7_NIRRG_withoutGTDSM_withDeep_withShallow

# python src/main.py --model FLDCF_multiModal_TransUNet --save FLDCF_multiModal_TransUNet__Fake_Vaihingen_Vehicle__NIRRG_withoutGTDSM


# cd /media/lscsc/nas2/mading/fakedetect
# python src/main.py --model FLDCF_multiModal_TransUNet --save FLDCF_multiModal_TransUNet__Fake_Vaihingen_Vehicle__NIRRG_withoutGTDSM

# python src/main.py --dsm_option True --model FLDCF_multiModal --save FLDCF_multiModal__Fake_Vaihingen_Vehicle__NIRRG_withGTDSM

# CUDA_VISIBLE_DEVICES=1 nohup python -u test.py > fake_Vaihingen_256_CarMask_train_962 2>&1 &
# nohup python -u src/main.py --dsm_option True --model FLDCF_multiModal --save FLDCF_multiModal__Fake_Vaihingen_Vehicle__10PixelLargerMask__NIRRG_withGTDSM > FLDCF_multiModal__Fake_Vaihingen_Vehicle__10PixelLargerMask__NIRRG_withGTDSM 2>&1 &

# nohup python -u src/main.py --dsm_option False --data_train Fake_Vaihingen_Vehicle --model FLDCF_multiModal_TransUNet --save FLDCF_multiModal_TransUNet__Fake_Vaihingen_Vehicle__10PixelLargerMask__NIRRG_withoutGTDSM > FLDCF_multiModal_TransUNet__Fake_Vaihingen_Vehicle__10PixelLargerMask__NIRRG_withoutGTDSM 2>&1 &

# python src/main.py --dsm_option False --model FLDCF_multiModal_TransUNet --save FLDCF_multiModal_TransUNet__Fake_Vaihingen_Vehicle__10PixelLargerMask__NIRRG_withoutGTDSM

# nohup python -u src/main.py --dsm_option False --data_train Fake_Vaihingen_Vehicle --model FLDCF_multiModal_TransUNet --save FLDCF_multiModal_TransUNet__Fake_Vaihingen_Vehicle__5PixelLargerMask__NIRRG_withoutGTDSM > FLDCF_multiModal_TransUNet__Fake_Vaihingen_Vehicle__5PixelLargerMask__NIRRG_withoutGTDSM 2>&1 &

# nohup python -u src/main.py --dsm_option False --data_train Fake_Vaihingen_Vehicle --model FLDCF_multiModal_TransUNet --save FLDCF_multiModal_TransUNet__Fake_Vaihingen_Vehicle__15PixelLargerMask__NIRRG_withoutGTDSM > FLDCF_multiModal_TransUNet__Fake_Vaihingen_Vehicle__15PixelLargerMask__NIRRG_withoutGTDSM 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python -u src/main.py --dsm_option False --data_train Fake_Vaihingen_Vehicle --model FLDCF_multiModal_TransUNet --save FLDCF_multiModal_TransUNet__Fake_Vaihingen_Vehicle__20PixelLargerMask__NIRRG_withoutGTDSM > FLDCF_multiModal_TransUNet__Fake_Vaihingen_Vehicle__20PixelLargerMask__NIRRG_withoutGTDSM 2>&1 &

