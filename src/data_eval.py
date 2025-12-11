import torch
import os
import utility
import data
from option import args
import pyiqa
import numpy as np
import math

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args) 

os.environ["CUDA_VISIBLE_DEVICES"] = "0" #在卡2调试


class Rate():
    def __init__(self):
        self.rate_map = {
        0:0,
        1:0,
        2:0,
        3:0,
        4:0,
        5:0,
        6:0,
        7:0,
        8:0,
    }
    def avg_mask_calcu(self,rate):
        i = math.floor(rate*20)
        self.rate_map[i]+=1
    def print_mask(self):
        print(self.rate_map)
def main(trainloader): 
    avg_niqe= []
    avg_brisque= []
    avg_musiq= []
    avg_dbcnn= []
    avg_clipiqa= []
    av_mask = []
    rate = Rate()
    niqa_metric = pyiqa.create_metric('niqe',device='cuda')  #brisque niqe tres-flive musiq-ava clipiqa dbcnn
    brisque_metric = pyiqa.create_metric('brisque',device='cuda')  
    musiq_metric = pyiqa.create_metric('musiq-ava',device='cuda')  
    clipiqa_metric = pyiqa.create_metric('clipiqa',device='cuda')  
    dbcnn_metric = pyiqa.create_metric('dbcnn',device='cuda')  
    print("===> Trainset")
    for dataload in trainloader:
        for batch, (lr, hr,real, _,) in enumerate(dataload):
            lr_cacu = lr.cuda()
            if(1-hr.mean().item()!=0):
                av_mask.append(1-hr.mean().item())
                rate.avg_mask_calcu(1-hr.mean().item())
            avg_niqe.append(niqa_metric(lr_cacu).item())
            avg_brisque.append(brisque_metric(lr_cacu).item())
            avg_musiq.append(musiq_metric(lr_cacu).item())
            avg_dbcnn.append(dbcnn_metric(lr_cacu).item())
            avg_clipiqa.append(clipiqa_metric(lr_cacu).item())
    print(len(avg_niqe))
    print("===> Avg. niqe: {:.4f}".format(np.mean(avg_niqe)))
    print("===> Avg. brisque: {:.4f}".format(np.mean(avg_brisque)))
    print("===> Avg. musiq-ava: {:.4f}".format(np.mean(avg_musiq)))
    print("===> Avg. dbcnn: {:.4f}".format(np.mean(avg_dbcnn)))
    print("===> Avg. clipiqa: {:.4f}".format(np.mean(avg_clipiqa)))

    print("===> Avg. mask: {:.4f}".format(np.mean(av_mask)))
    rate.print_mask()

args.batch_size=1
loader = data.Data(args)
trainloader = loader.loader_train
testloader = loader.loader_test
dataload = [trainloader]
main(dataload)