import logging

import torch
import torch.nn.functional as F

from trainer.base_trainer import BaseTrainer
from models.discriminator import *

from utils.utils import get_coeff
from utils.loss import d_align_uda, sift
from torch import nn

__all__ =  ['IWANTrainer']


class IWANTrainer(BaseTrainer):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def build_model(self):
        logging.info(f'--> building models: {self.cfg.MODEL.BASENET}')
        # backbone
        self.base_net = self.build_base_model()
        fdim = self.base_net.fdim
        self.w = None # weight, the outputs of D
        # discriminator
        self.d_net = Discriminator_no_grl(
            in_feature=fdim,
            hidden_size=self.cfg.MODEL.D_HIDDEN_SIZE,
            out_feature=self.cfg.MODEL.D_OUT_DIM
        ).to(self.device)
        
        self.d0_net = eval(self.cfg.MODEL.DNET)(
            in_feature=fdim,
            hidden_size=self.cfg.MODEL.D_HIDDEN_SIZE,
            out_feature=self.cfg.MODEL.D_OUT_DIM
        ).to(self.device)

        self.registed_models = {'base_net': self.base_net, 
                                'd_net': self.d_net, 
                                'd0_net':self.d0_net, }
        self.model_parameters()
        parameter_list = self.base_net.get_parameters() + \
                         self.d_net.get_parameters() + \
                         self.d0_net.get_parameters()
        self.build_optim(parameter_list)

    def one_step(self, data_src: torch.Tensor, data_tar: torch.Tensor):
        inputs_src, labels_src = data_src[0].to(self.device), data_src[1].to(self.device)
        inputs_tar, labels_tar = data_tar[0].to(self.device), data_tar[1].to(self.device)

        # feature extract
        outputs_all_src = self.base_net(inputs_src)    # base_net: CNN
        outputs_all_tar = self.base_net(inputs_tar)


        coeff=get_coeff(self.iter, max_iter=self.cfg.TRAIN.TTL_ITE)

        # weight obtain
        d_output_source = self.d_net(outputs_all_src[0].detach())   # d_net:No grl, no sigmoid
        d_output_target = self.d_net(outputs_all_tar[0].detach())
        w = nn.Sigmoid()(d_output_source).detach()

        self.w = 1 - w
        self.w = torch.softmax(w, 0)
        # weight for source feature
        w_feature = self.w * outputs_all_src[0]

        # domain alignment
        features_all = torch.cat((w_feature, outputs_all_tar[0]), dim=0)
        loss_alg = d_align_uda(
            features_all, features_all, self.d0_net,
            coeff=coeff, ent=self.cfg.METHOD.ENT
        )

        # cls loss
        loss_cls_src = F.cross_entropy(outputs_all_src[1], labels_src)
        loss_cls_tar = F.cross_entropy(outputs_all_tar[1].data, labels_tar) 

        d_output_all = torch.cat((d_output_source, d_output_target), dim=0)

        # sift loss
        loss_sift = sift(d_output_all)
        # domain alignment loss
        loss_ttl = loss_cls_src + loss_alg * self.cfg.METHOD.W_ALG + loss_sift
        # update
        self.step(loss_ttl)

        # display
        if self.iter % self.cfg.TRAIN.PRINT_FREQ == 0:
            self.display([
                f'l_cls_src: {loss_cls_src.item():.3f}',
                f'l_cls_tar: {loss_cls_tar.item():.3f}',
                f'l_alg: {loss_alg.item():.3f}',
                f'l_ttl: {loss_ttl.item():.3f}',
                f'best_acc: {self.best_acc:.3f}',
            ])
