from sklearn.metrics import confusion_matrix
import os
import tifffile as tf
import tqdm
import numpy as np


tifpath = '/media/lscsc/nas/jialu/Non-Local-Sparse-Attention-main/data/OLI2MSITIF/train_hr/'
pngpath = '/media/lscsc/nas/jialu/Non-Local-Sparse-Attention-main/data/OLI2MSI/train_hr/'
for filename in os.listdir(tifpath):
    img = tf.imread(tifpath+filename).transpose(1, 2, 0)
    img = np.round(np.clip(img, 0, 0.3) / 0.3 * 255).clip(0, 255).astype('u1')[:, :, ::-1]
    p = pngpath+filename.replace('TIF', 'png')
    tf.imsave(p, img)
