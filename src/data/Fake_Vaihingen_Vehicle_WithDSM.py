import os
import glob
from .srdata import SRData
from .common import np2Tensor
from utils import imgproc
import numpy as np
from PIL import Image
from utils.tools import draw_spectrum

class Fake_Vaihingen_Vehicle(SRData):
    def __init__(self, args, name='Fake_Vaihingen_Vehicle', train=True, benchmark=False):
        super(Fake_Vaihingen_Vehicle, self).__init__(
            args, name, train=train, benchmark=benchmark
        )

    def _scan(self):
        if(self.train):
            names_lr = sorted(
                glob.glob(os.path.join(self.dir_lr, '*' + '.png'))
            )
            names_dsm = sorted(
                glob.glob(os.path.join(self.dir_lr_dsm, '*' + '.png'))
            )
            #names_lr=[]
            names_hr = []
            for f in names_lr:
                filename, _ = os.path.splitext(os.path.basename(f))
                #names_lr.append(f.replace('train/inpainted','copy&paste'))
                # names_hr.append(os.path.join(self.apath, 'gt_mask','gt_mask.png'))
                names_hr.append(os.path.join(
                    self.dir_hr, '{}{}'.format(
                        filename, '.png'
                    )
                ))
            names_lr_gt = sorted(
                glob.glob(os.path.join(self.dir_lr_gt, '*' + '.png'))
            )
            names_dsm_gt = sorted(
                glob.glob(os.path.join(self.dir_lr_gt_dsm, '*' + '.png'))
            )
            names_hr_gt=[]
            for f in names_lr_gt:
                names_hr_gt.append(os.path.join(self.apath, 'gt_mask','gt_mask.png'))
        else:
            names_lr = sorted(
                glob.glob(os.path.join(self.dir_test_lr, '*' + '.png'))
            )
            names_dsm = sorted(
                glob.glob(os.path.join(self.dir_test_lr_dsm, '*' + '.png'))
            )
            #names_lr=[]
            names_hr = []
            for f in names_lr:
                filename, _ = os.path.splitext(os.path.basename(f))
                #names_lr.append(f.replace('test/inpainted','copy&paste'))
                # names_hr.append('/media/lscsc/nas/jialu/fakedetect/image/50.png')
                names_hr.append(os.path.join( 
                    self.dir_test_hr, '{}{}'.format(
                        filename, '.png'
                    )
                ))
            names_lr_gt = sorted(
                glob.glob(os.path.join(self.dir_test_lr_gt, '*' + '.png'))
            )
            names_dsm_gt = sorted(
                glob.glob(os.path.join(self.dir_test_lr_gt_dsm, '*' + '.png'))
            )
            names_hr_gt = []
            for f in names_lr_gt:
                names_hr_gt.append(os.path.join(self.apath, 'gt_mask','gt_mask.png'))
        
        names_hr = names_hr_gt + names_hr
        names_lr = names_lr_gt + names_lr
        names_dsm = names_dsm_gt + names_dsm

        return names_hr, names_lr, names_dsm
        # return names_hr, names_lr

    def __getitem__(self, idx):
        list_1 = []
        # counter = 0
        lr, label, filename, dsm = self._load_file(idx)  #whc
        # lr, label, filename = self._load_file(idx)  #whc 
        
        # print(filename)
        # print('a')
        # print()
        # list_1.append(filename)
        # print(list_1, 1)
        # counter += 1
        lr, label, dsm = np2Tensor(*[lr, label, dsm], rgb_range=self.args.rgb_range) #归一化外加转成cwh
        # lr, label = np2Tensor(*[lr, label], rgb_range=self.args.rgb_range) #归一化外加转成cwh
        # print(filename)
        # print('b')
        # print()
        # list_1.remove(filename)
        # print(list_1, 2)
        # print(counter)
        real = not label.min()==0
        if(real):
            real=np.float32(1.0)
        else:
            real=np.float32(0.0)
        
        return lr, label[0,:,:],real, filename, dsm  #cwh
        # return lr, label[0,:,:],real, filename  #cwh
    # (lr, hr, real, filename, dsm)
    

    # def _set_filesystem(self, dir_data):
    #     super(Vaihingen, self)._set_filesystem(dir_data)
    #     self.dir_hr = os.path.join(self.apath, 'train/inpainted_mask')
    #     self.dir_lr = os.path.join(self.apath, 'train/inpainted')
    #     self.dir_lr_gt = os.path.join(self.apath, 'train/gt')
    #     # self.dir_test_hr = os.path.join(self.apath, 'test/diverse_mask/regular_mask30')
    #     # self.dir_test_lr = os.path.join(self.apath, 'test/diverse_mask/lama_regular')
    #     self.dir_test_hr = os.path.join(self.apath, 'test/inpainted_mask')
    #     self.dir_test_lr = os.path.join(self.apath, 'test/repaint')
    #     self.dir_test_lr_gt = os.path.join(self.apath, 'test/gt')

    # def _set_filesystem(self, dir_data):
    #     super(Vaihingen, self)._set_filesystem(dir_data)
    #     self.dir_hr = os.path.join(self.apath, 'train/inpainted_mask')
    #     self.dir_lr = os.path.join(self.apath, 'train/inpainted')
    #     self.dir_lr_gt = os.path.join(self.apath, 'train/gt')
    #     self.dir_lr_dsm = os.path.join('/media/lscsc/nas/mading/data/fakeV_dsm', 'train/inpainted_gray')
    #     self.dir_lr_gt_dsm = os.path.join('/media/lscsc/nas/mading/data/fakeV_dsm', 'train/gt_dsm')

    #     self.dir_test_hr = os.path.join(self.apath, 'test/inpainted_mask')
    #     self.dir_test_lr = os.path.join(self.apath, 'test/repaint')
    #     self.dir_test_lr_gt = os.path.join(self.apath, 'test/gt')
    #     self.dir_test_lr_dsm = os.path.join('/media/lscsc/nas/mading/data/fakeV_dsm', 'test/inpainted_gray')
    #     self.dir_test_lr_gt_dsm = os.path.join('/media/lscsc/nas/mading/data/fakeV_dsm', 'test/gt_dsm')

################################
# 双分支 NIRRG+DSM FTransUNet
# real dsm
    def _set_filesystem(self, dir_data):
        super(Fake_Vaihingen_Vehicle, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'train/repaint_mask_10pixel_enlargement_toEveryDirection')
        self.dir_lr = os.path.join(self.apath, 'train/repaint_inpaint_10pixel_enlargement_toEveryDirection') #'train/inpainted'
        self.dir_lr_gt = os.path.join(self.apath, 'train/gt')
        self.dir_lr_dsm = os.path.join('/media/lscsc/nas2/mading/data/fake_Vaihingen_vehicle_DSM', 'train/repaint_gt_dsm') #
        # self.dir_lr_dsm = os.path.join(self.apath, 'train/inpainted') ##
        # /media/lscsc/nas/mading/data/fakeV_dsm/train/gt_geanerated_place
        self.dir_lr_gt_dsm = os.path.join('/media/lscsc/nas2/mading/data/fake_Vaihingen_vehicle_DSM', 'train/gt_dsm') #
        # self.dir_lr_gt_dsm = os.path.join(self.apath, 'train/gt') ##


        self.dir_test_hr = os.path.join(self.apath, 'test/repaint_mask_10pixel_enlargement_toEveryDirection')
        self.dir_test_lr = os.path.join(self.apath, 'test/repaint_inpaint_10pixel_enlargement_toEveryDirection') #'test/lama'
        # self.dir_test_lr = os.path.join(self.apath, 'test/gt_empty') #'test/lama'
        # self.dir_test_lr = os.path.join(self.apath, 'test/lama') #'test/lama'
        self.dir_test_lr_gt = os.path.join(self.apath, 'test/gt')
        self.dir_test_lr_dsm = os.path.join('/media/lscsc/nas2/mading/data/fake_Vaihingen_vehicle_DSM', 'test/repaint_gt_dsm') # gt_generated_place gt_generated_place
        # self.dir_test_lr_dsm = os.path.join('/media/lscsc/nas/mading/data/fakeV_dsm', 'test/gt_empty') # gt_generated_place
        # self.dir_test_lr_dsm = os.path.join(self.apath, 'test/repaint') ##
        # /media/lscsc/nas/mading/data/fakeV_dsm/test/gt_generated_place
        self.dir_test_lr_gt_dsm = os.path.join('/media/lscsc/nas2/mading/data/fake_Vaihingen_vehicle_DSM', 'test/gt_dsm') #
        # self.dir_test_lr_gt_dsm = os.path.join(self.apath, 'test/gt') ##
################################

# ################################
# 单分支 NIRRG TransUNet
# real dsm
#     def _set_filesystem(self, dir_data):
#         super(Vaihingen, self)._set_filesystem(dir_data)
#         self.dir_hr = os.path.join(self.apath, 'train/inpainted_mask')
#         self.dir_lr = os.path.join(self.apath, 'train/inpainted') #'train/inpainted'
#         self.dir_lr_gt = os.path.join(self.apath, 'train/gt')
#         # self.dir_lr_dsm = os.path.join('/media/lscsc/nas/mading/data/fakeV_dsm', 'train/gt_geanerated_place') #
#         # self.dir_lr_dsm = os.path.join(self.apath, 'train/inpainted') ##
#         # /media/lscsc/nas/mading/data/fakeV_dsm/train/gt_geanerated_place
#         # self.dir_lr_gt_dsm = os.path.join('/media/lscsc/nas/mading/data/fakeV_dsm', 'train/gt_dsm') #
#         # self.dir_lr_gt_dsm = os.path.join(self.apath, 'train/gt') ##


#         self.dir_test_hr = os.path.join(self.apath, 'test/inpainted_mask')
#         self.dir_test_lr = os.path.join(self.apath, 'test/lama') #'test/lama'
#         self.dir_test_lr_gt = os.path.join(self.apath, 'test/gt')
#         # self.dir_test_lr_dsm = os.path.join('/media/lscsc/nas/mading/data/fakeV_dsm', 'test/gt_generated_place') #
#         # self.dir_test_lr_dsm = os.path.join(self.apath, 'test/repaint') ##
#         # /media/lscsc/nas/mading/data/fakeV_dsm/test/gt_generated_place
#         # self.dir_test_lr_gt_dsm = os.path.join('/media/lscsc/nas/mading/data/fakeV_dsm', 'test/gt_dsm') #
#         # self.dir_test_lr_gt_dsm = os.path.join(self.apath, 'test/gt') ##
# ################################

# ################################
# # 双分支 NIRRG+DSM FTransUNet
# # fake dsm
#     def _set_filesystem(self, dir_data):
#         super(Vaihingen, self)._set_filesystem(dir_data)
#         self.dir_hr = os.path.join(self.apath, 'train/inpainted_mask')
#         self.dir_lr = os.path.join(self.apath, 'train/inpainted') #'train/inpainted'
#         self.dir_lr_gt = os.path.join(self.apath, 'train/gt')
#         self.dir_lr_dsm = os.path.join('/media/lscsc/nas/mading/data/fakeV_dsm', 'train/inpainted_gray') #data/fakeV_dsm/train/inpainted_gray
#         # self.dir_lr_dsm = os.path.join(self.apath, 'train/inpainted') ##
#         # /media/lscsc/nas/mading/data/fakeV_dsm/train/gt_geanerated_place
#         self.dir_lr_gt_dsm = os.path.join('/media/lscsc/nas/mading/data/fakeV_dsm', 'train/gt_dsm') #
#         # self.dir_lr_gt_dsm = os.path.join(self.apath, 'train/gt') ##


#         self.dir_test_hr = os.path.join(self.apath, 'test/inpainted_mask')
#         self.dir_test_lr = os.path.join(self.apath, 'test/inpainted') #'test/lama'
#         self.dir_test_lr_gt = os.path.join(self.apath, 'test/gt')
#         self.dir_test_lr_dsm = os.path.join('/media/lscsc/nas/mading/data/fakeV_dsm', 'test/inpainted_gray') # inpainted_gray
#         # self.dir_test_lr_dsm = os.path.join(self.apath, 'test/repaint') ##
#         # /media/lscsc/nas/mading/data/fakeV_dsm/test/gt_generated_place
#         self.dir_test_lr_gt_dsm = os.path.join('/media/lscsc/nas/mading/data/fakeV_dsm', 'test/gt_dsm') #
#         # self.dir_test_lr_gt_dsm = os.path.join(self.apath, 'test/gt') ##
# ################################