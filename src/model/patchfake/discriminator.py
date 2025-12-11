import torch
import torch.nn as nn
from collections import OrderedDict
from utils import renormalize, imutil
from .base_model import BaseModel
from .networks import networks
import numpy as np
import logging
import cv2
from PIL import Image
from collections import namedtuple

class PatchDiscriminatorModel(BaseModel):

    def name(self):
        return 'PatchDiscriminatorModel'

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out. 
        self.loss_names = ['loss_D']
        self.loss_names += ['acc_D_raw', 'acc_D_voted', 'acc_D_avg']
        self.val_metric = 'acc_D_raw'

        # specify the images you want to save/display. 
        self.visual_names = ['fake_0', 'fake_1', 'fake_2', 'fake_3', 'fake_4',
                             'real_0', 'real_1', 'real_2', 'real_3', 'real_4',
                             'vfake_0', 'vfake_1', 'vfake_2', 'vfake_3', 'vfake_4',
                             'vreal_0', 'vreal_1', 'vreal_2', 'vreal_3', 'vreal_4']

        # specify the models you want to save to the disk. 
        self.model_names = ['D']

        # load/define networks
        torch.manual_seed(opt.seed) # set model seed
        self.net_D = networks.define_patch_D(opt.which_model_netD,
                                             opt.init_type, self.gpu_ids)
        self.criterionCE = nn.CrossEntropyLoss().to(self.device)
        self.softmax = torch.nn.Softmax(dim=1)

        if self.isTrain:
            self.optimizers['D'] = torch.optim.Adam(
                self.net_D.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    def set_input(self, input):
        self.ims = input['ims'].to(self.device)
        self.labels = input['labels'].to(self.device)

    def forward(self):
        self.pred_logit = self.net_D(self.ims)

    def compute_losses_D(self):
        # logit shape should be N2HW
        assert(len(self.pred_logit.shape) == 4)
        assert(self.pred_logit.shape[1] == 2)
        n, c, h, w = self.pred_logit.shape
        labels = self.labels.view(-1, 1, 1).expand(n, h, w)
        predictions = self.pred_logit
        self.loss_D = self.criterionCE(predictions, labels)
        self.acc_D_raw = torch.mean(torch.eq(labels, torch.argmax(
            predictions, dim=1)).float())
        # voted acc is forcing each patch into a 0/1 decision,
        # and taking the average
        votes = torch.mode(torch.argmax(predictions, dim=1).view(n, -1))[0]
        self.acc_D_voted = torch.mean(torch.eq(self.labels, votes).float())
        # average acc is averaging each softmaxed patch prediction, and 
        # taking the argmax
        avg_preds = torch.argmax(self.softmax(self.pred_logit)
                                 .mean(dim=(2,3)), dim=1)
        self.acc_D_avg = torch.mean(torch.eq(self.labels,
                                             avg_preds).float())

    def backward_D(self):
        self.compute_losses_D()
        self.loss_D.backward()

    def optimize_parameters(self):
        self.optimizers['D'].zero_grad()
        self.forward()
        self.backward_D()
        self.optimizers['D'].step()

    def get_current_visuals(self):
        from collections import OrderedDict
        visual_ret = OrderedDict()
        fake_ims = self.ims[self.labels == self.opt.fake_class_id]
        with torch.no_grad():
            fake_ims_overlay = self.softmax(self.pred_logit[
                self.labels == self.opt.fake_class_id])
        real_ims = self.ims[self.labels != self.opt.fake_class_id]
        with torch.no_grad():
            real_ims_overlay = self.softmax(self.pred_logit[
                self.labels != self.opt.fake_class_id])
        for i in range(min(5, len(fake_ims))):
            im = renormalize.as_tensor(
                fake_ims[[i], :, :, :], source='zc', target='pt')
            visual_ret['fake_%d' % i] = im
            visual_ret['vfake_%d' % i] = self.overlay_visual(
                im.detach().cpu().numpy().squeeze(),
                fake_ims_overlay[i, 1, :, :].detach().cpu().numpy(),
                to_tensor=True
            )
        for i in range(min(5, len(real_ims))):
            im = renormalize.as_tensor(
                real_ims[[i], :, :, :], source='zc', target='pt')
            visual_ret['real_%d' % i] = im
            visual_ret['vreal_%d' % i] = self.overlay_visual(
                im.detach().cpu().numpy().squeeze(),
                real_ims_overlay[i, 1, :, :].detach().cpu().numpy(),
                to_tensor=True
            )
        return visual_ret

    def reset(self):
        # for debugging .. clear all the cached variables
        self.loss_D = None
        self.acc_D = None
        self.acc_D_raw = None
        self.acc_D_voted = None
        self.acc_D_avg = None
        self.ims = None
        self.labels = None
        self.pred_logit = None

    def overlay_visual(self, im_np, overlay, to_tensor=False):
        # im : np array, (3, h, w)
        # overlay: np array (h', w')
        (h, w) = im_np.shape[1:]
        overlay = np.uint8(255 * overlay)
        overlay = cv2.resize(overlay, (w, h))
        heatmap = cv2.applyColorMap(overlay, cv2.COLORMAP_JET)
        heatmap = heatmap/255 # range [0, 1]
        # change heatmap to RGB, and channel to CHW
        heatmap = heatmap[:,:,::-1].transpose(2, 0, 1)
        result = heatmap * 0.3 + im_np * 0.5
        if to_tensor:
            return torch.from_numpy(result).float().to(self.device)[None]
        else:
            im_out = np.uint8(result*255)
            return im_out

    def get_predictions(self):
        Predictions = namedtuple('predictions', ['vote', 'before_softmax',
                                                 'after_softmax', 'raw'])
        with torch.no_grad():
            n = self.pred_logit.shape[0]
            # vote_predictions probability is a tally of the patch votes
            votes = torch.argmax(self.pred_logit, dim=1).view(n, -1)
            vote_predictions = torch.mean(votes.float(), axis=1)
            vote_predictions = torch.stack([1-vote_predictions,
                                            vote_predictions], axis=1)
            before_softmax_predictions = self.softmax(
                torch.mean(self.pred_logit, dim=(-1, -2)))
            after_softmax_predictions = torch.mean(
                self.softmax(self.pred_logit), dim=(-1, -2))
            patch_predictions = self.softmax(self.pred_logit)
        return Predictions(vote_predictions.cpu().numpy(),
                           before_softmax_predictions.cpu().numpy(),
                           after_softmax_predictions.cpu().numpy(),
                           patch_predictions.cpu().numpy())