
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
from torchstat import stat
import torchvision.models as models
from thop import profile
from thop import clever_format

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args) 


# from torchstat import stat
# import torchvision.models as models
# model = models.resnet152()
# stat(model, (3, 224, 224))


def get_model_memory(model1):
    # 获取模型中参数的总大小（字节数）
    total_params = sum(p.numel() for p in model1.parameters())
    # 每个参数默认占用4字节（float32），故计算内存占用（MB）
    return total_params * 4 / (1024 ** 2)

def get_activation_memory(model, input_size):
    # 构造输入数据
    input_tensor = torch.randn(input_size).to(device)
    # 记录中间层激活的内存
    activations = []

    def hook_fn(module, input, output):
        activations.append(output.nelement() * output.element_size())

    hooks = []
    for layer in model.children():
        hooks.append(layer.register_forward_hook(hook_fn))
    
    # 运行一次前向传播
    model(input_tensor)

    # 解除挂接
    for hook in hooks:
        hook.remove()
    
    return sum(activations) / (1024 ** 2)  # 返回结果（MB）




def main():
    global model
    if checkpoint.ok:                  
        loader = data.Data(args)
        _model = model.Model(args, checkpoint) ###

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 查看模型在GPU上的内存占用情况
        # allocated_memory = torch.cuda.memory_allocated(device=device)
        allocated_memory = torch.cuda.max_memory_allocated(device=device)
        # torch.cuda.max_memory_allocated()
        cached_memory = torch.cuda.memory_cached(device=device)
        print(f"Allocated memory: {allocated_memory/1024/1024}")
        print(f"Cached memory: {cached_memory/1024/1024}")


        # # input = torch.randn(1, 3, 224, 224)
        # # input1 = torch.randn(1, 1, 224, 224)
        # input = torch.randn(1, 3, 256, 256)
        # input1 = torch.randn(1, 1, 256, 256)
        # flops, params = profile(_model, inputs=(input, input1))
        # print(flops, params) # 1819066368.0 11689512.0
        # flops, params = clever_format([flops, params], "%.3f")
        # print(flops, params) # 1.819G 11.690M

        # model_memory = get_model_memory(model)
        # print(f"Model Parameters Memory Usage: {model_memory:.2f} MB")
        # activation_memory = get_activation_memory(model, (1, 784))
        # print(f"Activation Memory Usage: {activation_memory:.2f} MB")
        # total_memory = model_memory + activation_memory
        # print(f"Total Memory Usage: {total_memory:.2f} MB")


        # model1 = models.resnet152()
        # stat(model1, (3, 224, 224))
        # stat(_model, (3, 224, 224))

        # for name, parms in _model.named_parameters():
        #     if 'layers.0.residual_group.blocks.0.attn.relative_position_bias_table' in name:
        #         print(parms)
        #     print('%-50s' % name, '%-30s' % str(parms.shape), '%-10s' % str(parms.nelement()))
        # print('Total params: %.2fM' % (sum(p.numel() for p in _model.parameters())/1000000.0))
        # input = torch.randn(16, 3, 32, 32).cuda()
        # flops, params = profile(_model, inputs=(input, ))
        # print('Complexity: %.3fM' % (flops/1000000000), end=' GFLOPs\n')
        # predict = _model(input)  #影响speed的值
        # torch.cuda.synchronize()
        # time_start = time.time()
        # predict = _model(input)
        # torch.cuda.synchronize()
        # time_end = time.time()
        # print('Speed: %.3f FPS\n' % (1/(time_end-time_start)))
        # flops, params = profile(_model, inputs=(input, ))
        # print('Complexity: %.3fM' % (flops/1000000000), end=' GFLOPs\n')
        # optimizer = optim.SGD(_model.parameters(), lr=0.9, momentum=0.9, weight_decay=0.0005)
        # for _ in range(1000):
        #     optimizer.zero_grad()
        #     _model(input)

        _loss = loss.Loss_fake() if not args.test_only else None
        t = Trainer(args, loader, _model, _loss, checkpoint)
        t.testtrain(False)
        checkpoint.done()

if __name__ == '__main__':
    main()

# cd /media/lscsc/nas/mading/fakedetect
# python src/test.py --model your_model_name --save my10
# python src/test.py --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_myV.pt' --save my10
# python src/test.py --pre_train '/media/lscsc/nas/mading/fakedetect/experiment/my12/model/model_latest.pt' --save my13
# python src/test.py --model crnet --pre_train '/media/lscsc/nas/mading/fakedetect/experiment/crnetVDSM1/model/model_latest.pt' --save crnetVDSM_test
# python src/test.py --model my2 --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_V_DSM.pt' --save my2VDSM_test
# python src/test.py --model FLDCF2 --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF2_V_DSM.pt' --save FLDCF2VDSM_test
# python src/test.py --model FLDCF2 --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF2_V_DSM_2.pt' --save FLDCF2VDSM_test
# python src/test.py --model CDnetV2 --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_CDnetV2_V_DSM.pt' --save CDnetV2VDSM_test
# python src/test.py --model CDnetV2 --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_CDnetV2_V_DSM_contentBasedPrior.pt' --save CDnetV2VDSM_test

# python src/test.py --model my2 --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_myV.pt' --save my2VRGB_test
# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_forgedDSM.pt' --save FLDCF_multiModal_V_NIRRG_forgedDSM_test
# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_GTDSM2.pt' --save FLDCF_multiModal_V_NIRRG_GTDSM2_test
# python src/test.py --model my2 --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_V_LaMa.pt' --save FLDCF_LaMa_test
# python src/test.py --model my2 --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_V_Repaint.pt' --save FLDCF_Repaint_test
# python src/test.py --model my2 --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_V_All.pt' --save FLDCF_All_test

# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_GTDSM3_Repaint1.pt' --save FLDCF_multiModal_V_NIRRG_GTDSM3_Repaint_test
# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_GTDSM3_All.pt' --save FLDCF_multiModal_V_NIRRG_GTDSM3_All_test
# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_GTDSM3_LaMa.pt' --save FLDCF_multiModal_V_NIRRG_GTDSM3_LaMa_test
# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_GTDSM4_LaMa.pt' --save FLDCF_multiModal_V_NIRRG_GTDSM4_LaMa_test
# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_GTDSM5_LaMa.pt' --save FLDCF_multiModal_V_NIRRG_GTDSM5_LaMa_test

# python src/test.py --model my2 --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_myV.pt' --save FLDCF_LaMa_test2

# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/experiment/FLDCF_multiModalV_modifiedEncoder_NIRRG_withGTDSM_withDeep_withShallow_all/model/model_25.pt' --save FLDCF_multiModal_V_NIRRG_GTDSM5_All_test

# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_GTDSM5_All.pt' --save FLDCF_multiModal_V_NIRRG_GTDSM5_All_test
# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_GTDSM5_Repaint.pt' --save FLDCF_multiModal_V_NIRRG_GTDSM5_Repaint_test
# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_GTDSM6_LaMa.pt' --save FLDCF_multiModal_V_NIRRG_GTDSM6_LaMa_test
# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_GTDSM7_LaMa.pt' --save FLDCF_multiModal_V_NIRRG_GTDSM7_LaMa_test

# python src/test.py --model face --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FaceForensics_V_CycleGAN.pt' --save FaceForensics_V_CycleGAN_test


# python src/test.py --model scunet --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_scuV.pt' --save SCSE-Unet_V_all
# python src/test.py --model scunet --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_scuV.pt' --save SCSE-Unet_V_repaint
# python src/test.py --model scunet --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_scuV.pt' --save SCSE-Unet_V_lama

# python src/test.py --model mvss --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_mvssV.pt' --save MVSSNet_V_repaint

# python src/test.py --model movenet --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_moveV.pt' --save SE-Network_V_repaint

# python src/test.py --model my2 --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_myV.pt' --save FLDCF_V_repaint

# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_GTDSM7_LaMa.pt' --save FLDCF2_V_repaint

# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_GTDSM7_2_LaMa.pt' --save FLDCF2_V_lama

# python src/test.py --model mvss --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_mvssV.pt' --save MVSSNet_V_lama

# python src/test.py --model movenet --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_moveV.pt' --save SE-Network_V_lama

# python src/test.py --model capsule --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_Capsule_Vaihingen_all.pt' --save Capsule_Vaihingen__repaint

# python src/test.py --model face --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FaceForensics_Vaihingen_all.pt' --save FaceForensics_Vaihingen__repaint

# python src/test.py --model deepfake --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_DeepFake-Detection_Vaihingen_all.pt' --save DeepFake-Detection_Vaihingen__all

# python src/test.py --model my2 --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_myV.pt' --save FLDCF_V_all

# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_GTDSM7_All.pt' --save FLDCF2_V_all

# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_GTDSM7_All_withoutPyramidFeatureExtractionModule.pt' --save FLDCF2_V_repaint_withoutPyramidFeatureExtractionModule

# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_GTDSM7_All_withoutDSM.pt' --save FLDCF2_V_all_withoutDSM

# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_GTDSM7_All_withoutGlobalFeatureBlock.pt' --save FLDCF2_V_repaint_withoutGlobalFeatureBlock

# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_GTDSM7_All_withoutLocalFeatureBlock.pt' --save FLDCF2_V_repaint_withoutLocalFeatureBlock

# python src/test.py --model FLDCF_multiModal_TransUNet --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_TransUNet_V_NIRRG_withoutGTDSM7_All.pt' --save FLDCF2_TransUNet_V_lama_withoutDSM

# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_FakeDSM7_All.pt' --save FLDCF2_V_lama_fakeDSM

# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/experiment/FLDCF_multiModalV_modifiedEncoder7_NIRRG_withFakeDSM_withDeep_withShallow_all/model/model_34.pt' --save FLDCF2_V_lama_fakeDSM

# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_FakeDSM7_All2.pt' --save FLDCF2_V_lama_fakeDSM

# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_GTDSM7_All.pt' --save FLDCF2_V_lama

# python src/test.py --model my2 --data_train 'Vaihingen' --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_myV.pt' --save FLDCF_V_lama

# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/experiment/FLDCF_multiModalV_modifiedEncoder7_NIRRG_withGTDSM_withDeep_withShallow_all/model/model_32.pt' --save FLDCF2_V_lama

# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_GTDSM7_All2.pt' --save FLDCF2_V_lama

# MFFLDCF == FLDCF2
# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_Potsdam_NIRRG_GTDSM7_All.pt' --save MFFLDCF_Potsdam_all

# python src/test.py --model scunet --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_SCSEUnet_Potsdam.pt' --save SCSE-Unet_Potsdam_lama

# python src/test.py --model mvss --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_MVSSNet_Potsdam.pt' --save MVSSNet_Potsdam_lama

# python src/test.py --model movenet --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_SE-Network_Potsdam1.pt' --save SE-Network_Potsdam_lama

# python src/test.py --model scunet --pre_train '/media/lscsc/nas/mading/fakedetect/experiment/SCSE-Unet_Potsdam_all/model/model_20.pt' --save SCSE-Unet_Potsdam_lama

# python src/test.py --model my2 --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_Potsdam_all.pt' --save FLDCF_Potsdam_lama

# python src/test.py --model my2 --pre_train '/media/lscsc/nas/mading/fakedetect/experiment/FLDCF_Potsdam_all/model/model_20.pt' --save FLDCF_Potsdam_lama

# python src/test.py --model capsule --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_Capsule_Potsdam_all.pt' --save Capsule_Potsdam__lama

# python src/test.py --model face --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FaceForensics_Potsdam_all1.pt' --save FaceForensics_Potsdam__lama

# python src/test.py --model deepfake --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_DeepFake-Detection_Potsdam_all.pt' --save DeepFake-Detection_Potsdam__lama

# python src/test.py --model FLDCF_multiModal_TransUNet --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_TransUNet_WHUGCD_RGB_withoutGTDSM7_All6types.pt' --save FLDCF_multiModal_TransUNet_WHU-GCD_modifiedEncoder7_NIRRG_withoutGTDSM_withDeep_withShallow_2

# python src/test.py --model FLDCF_multiModal_TransUNet --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_TransUNet_WHUGCD_RGB_withoutGTDSM7_All6types.pt' --save FLDCF_multiModal_TransUNet_WHU-GCD_modifiedEncoder7_NIRRG_withoutGTDSM_withDeep_withShallow_3

# python src/test.py --model FLDCF_multiModal_TransUNet --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_TransUNet_WHUGCD_RGB_withoutGTDSM7_All6types.pt' --save FLDCF_multiModal_TransUNet_WHU-GCD_modifiedEncoder7_NIRRG_withoutGTDSM_withDeep_withShallow_4

# python src/test.py --model FLDCF_multiModal_TransUNet --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_TransUNet_WHUGCD_RGB_withoutGTDSM7_All6types.pt' --save FLDCF_multiModal_TransUNet_WHU-GCD_modifiedEncoder7_NIRRG_withoutGTDSM_withDeep_withShallow_5

# python src/test.py --model FLDCF_multiModal_TransUNet --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_TransUNet_WHUGCD_RGB_withoutGTDSM7_All6types.pt' --save FLDCF_multiModal_TransUNet_WHU-GCD_modifiedEncoder7_NIRRG_withoutGTDSM_withDeep_withShallow_6

# python src/test.py --model FLDCF_multiModal_TransUNet --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_TransUNet_WHUGCD_RGB_withoutGTDSM7_All6types.pt' --save FLDCF_multiModal_TransUNet_WHU-GCD_modifiedEncoder7_NIRRG_withoutGTDSM_withDeep_withShallow_7


# python src/test.py --model my2 --data_train_dir 'fakeV' --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_myV.pt' --save FLDCF_V_gt

# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_GTDSM7_All2.pt' --save FLDCF2_V_gt

# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_FakeDSM7_All2.pt' --save FLDCF2_V_gt_fakeDSM


# python src/test.py --model FLDCF_multiModal_TransUNet --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_TransUNet_WHUGCD_RGB_withoutGTDSM7_All6types.pt' --save FLDCF_multiModal_TransUNet_WHU-GCD_modifiedEncoder7_NIRRG_withoutGTDSM_withDeep_withShallow

# python src/test.py --model FLDCF_multiModal_TransUNet --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_TransUNet_WHUGCD_RGB_withoutGTDSM7_All6types.pt' --save FLDCF_multiModal_TransUNet_WHU-GCD_withoutGTDSM_x_3








# cd /media/lscsc/nas/mading/fakedetect
