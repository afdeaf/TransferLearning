import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainer.base_trainer import BaseTrainer
from models.discriminator import *

from utils.utils import get_coeff
from utils.loss import d_align_uda, sift

__all__ =  ['IWAN']


class IWAN(BaseTrainer):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def build_model(self):
        logging.info(f'--> building models: {self.cfg.MODEL.BASENET}')
        # backbone
        self.base_net = self.build_base_model()
        fdim = self.base_net.fdim
        self.w = None # weight, the outputs of D
        # discriminator
        self.d_net = Sift(
            in_feature=fdim,
            hidden_size=self.cfg.MODEL.D_HIDDEN_SIZE,
            out_feature=self.cfg.MODEL.D_OUT_DIM
        ).cuda()

        self.d0_net = eval(self.cfg.MODEL.DNET)(
            in_feature=fdim,
            hidden_size=self.cfg.MODEL.D_HIDDEN_SIZE,
            out_feature=self.cfg.MODEL.D_OUT_DIM
        ).cuda()

        # feature sift
        self.fc = Sift(
            in_feature=fdim,
            hidden_size=self.cfg.MODEL.D_HIDDEN_SIZE,
            out_feature=self.cfg.DATASET.NUM_CLASSES
        ).cuda()

        self.registed_models = {'base_net': self.base_net, 
                                'd_net': self.d_net, 
                                'd0_net':self.d0_net, 
                                'classifier':self.fc}
        self.model_parameters()
        parameter_list = self.base_net.get_parameters() + \
                         self.d_net.get_parameters() + \
                         self.d0_net.get_parameters() + self.fc.get_parameters()
        self.build_optim(parameter_list)

    def one_step(self, data_src, data_tar):
        inputs_src, labels_src = data_src[0].cuda(), data_src[1].cuda()
        inputs_tar, labels_tar = data_tar[0].cuda(), data_tar[1].cuda()

        # feature extract
        feature_source = self.base_net(inputs_src)    # base_net: CNN
        feature_target = self.base_net(inputs_tar)

        # weight obtain
        d_output_source = self.d_net(feature_source.detach())   # d_net:No grl, no sigmoid
        d_output_target = self.d_net(feature_target.detach())
        w = nn.Sigmoid()(d_output_source).detach().clone()
        self.w = 1 - w

        # weight for source feature
        w_feature_source = self.w * feature_source

        # domain alignment
        features_all = torch.cat((w_feature_source, feature_target), dim=0)
        loss_alg = d_align_uda(
            features_all, features_all, self.d0_net,
            coeff=get_coeff(self.iter, max_iter=self.cfg.TRAIN.TTL_ITE), ent=self.cfg.METHOD.ENT
        )

        # classify
        logits_source = self.fc(w_feature_source)
        logits_target = self.fc(feature_target)

        # cls loss
        loss_cls_src = F.cross_entropy(logits_source, labels_src)
        loss_cls_tar = F.cross_entropy(logits_target.data, labels_tar) 

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