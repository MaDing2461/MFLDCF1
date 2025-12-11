import os
import glob
from .srdata import SRData
from .common import np2Tensor
from utils import imgproc
import numpy as np
from PIL import Image
from utils.tools import draw_spectrum

class WHU_GCD(SRData):
    def __init__(self, args,name='Vaihingen', train=True, benchmark=False):
        super(WHU_GCD, self).__init__(
            args, name, train=train, benchmark=benchmark
        )

    def _scan(self):
        if(self.train):
            names_lr = sorted(
                glob.glob(os.path.join(self.dir_lr, '*' + '.png'))
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
            names_hr_gt=[]
            for f in names_lr_gt:
                names_hr_gt.append(os.path.join(self.apath, 'gt_mask','gt_mask.png'))
        else:
            names_lr = sorted(
                glob.glob(os.path.join(self.dir_test_lr, '*' + '.png'))
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
            names_hr_gt = []
            for f in names_lr_gt:
                names_hr_gt.append(os.path.join(self.apath, 'gt_mask','gt_mask.png'))
        names_hr=names_hr_gt+names_hr
        names_lr=names_lr_gt+names_lr
        # return names_hr, names_lr, None
        return names_hr, names_lr

    def __getitem__(self, idx):
        list_1 = []
        # counter = 0
        lr, label, filename = self._load_file(idx)  #whc 
        # print(filename)
        # print('a')
        # print()
        # list_1.append(filename)
        # print(list_1, 1)
        # counter += 1
        lr,label = np2Tensor(*[lr,label], rgb_range=self.args.rgb_range) #归一化外加转成cwh
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
        return lr, label[0,:,:],real,filename  #cwh
    

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

    def _set_filesystem(self, dir_data):
        super(WHU_GCD, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'train/gcd_mask')
        self.dir_lr = os.path.join(self.apath, 'train/gcd')
        self.dir_lr_gt = os.path.join(self.apath, 'train/gt')

        self.dir_test_hr = os.path.join(self.apath, 'test/im2_mask_x_7')
        self.dir_test_lr = os.path.join(self.apath, 'test/im2_x_7')
        self.dir_test_lr_gt = os.path.join(self.apath, 'test/gt')

# ################################

#     def _set_filesystem(self, dir_data):
#         super(Vaihingen, self)._set_filesystem(dir_data)
#         self.dir_hr = os.path.join(self.apath, 'train/inpainted_mask_gray')
#         self.dir_lr = os.path.join(self.apath, 'train/inpainted_gray')
#         self.dir_lr_gt = os.path.join(self.apath, 'train/gt')

#         self.dir_test_hr = os.path.join(self.apath, 'test/inpainted_mask_gray')
#         self.dir_test_lr = os.path.join(self.apath, 'test/inpainted_gray')
#         self.dir_test_lr_gt = os.path.join(self.apath, 'test/gt')

# ################################

# ################################

#     def _set_filesystem(self, dir_data):
#         super(Vaihingen, self)._set_filesystem(dir_data)
#         self.dir_hr = os.path.join(self.apath, 'train/inpainted_mask')
#         self.dir_lr = os.path.join(self.apath, 'train/inpainted')
#         self.dir_lr_gt = os.path.join(self.apath, 'train/gt')

#         self.dir_test_hr = os.path.join(self.apath, 'test/inpainted_mask')
#         self.dir_test_lr = os.path.join(self.apath, 'test/inpainted')
#         self.dir_test_lr_gt = os.path.join(self.apath, 'test/gt')

# ################################
