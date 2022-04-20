# Copyright (c) 2022 Raven Stock. email:cquptriven@qq.com

import logging

# import torch
# import torch.nn as nn
import torch.nn.functional as F

from trainer.base_trainer import BaseTrainer
# from models.mmd import *

# from utils.utils import get_coeff
# from utils.torch_utils import entropy_func
from utils.loss import MMD

__all__ =  ['MMDTrainer']


class MMDTrainer(BaseTrainer):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def build_model(self):
        logging.info(f'--> building models: {self.cfg.MODEL.BASENET}')
        # backbone
        self.base_net = self.build_base_model()
        # fdim = self.base_net.fdim

        # mmd
        self.mmd_net = MMD()

        self.registed_models = {'base_net': self.base_net, 'mmd_net': self.mmd_net}
        self.model_parameters()
        parameter_list = self.base_net.get_parameters() + self.mmd_net.get_parameters()
        self.build_optim(parameter_list)

    def one_step(self, data_src, data_tar):
        inputs_src, labels_src = data_src[0].cuda(), data_src[1].cuda()
        inputs_tar, labels_tar = data_tar[0].cuda(), data_tar[1].cuda()

        outputs_all_src = self.base_net(inputs_src)
        # print(outputs_all_src[0].shape)
        outputs_all_tar = self.base_net(inputs_tar)

        # features_all = torch.cat((outputs_all_src[0], outputs_all_tar[0]), dim=0)
        # logits_all = torch.cat((outputs_all_src[1], outputs_all_tar[1]), dim=0)
        # softmax_all = nn.Softmax(dim=1)(logits_all)

        # classificaiton
        loss_cls_src = F.cross_entropy(outputs_all_src[1], labels_src)
        loss_cls_tar = F.cross_entropy(outputs_all_tar[1].data, labels_tar)

        # mmd loss
        loss_mmd = self.mmd_net(outputs_all_src[0], outputs_all_tar[0])

        loss_ttl = loss_cls_src + loss_mmd * self.cfg.METHOD.W_ALG

        # update
        self.step(loss_ttl)

        # display
        if self.iter % self.cfg.TRAIN.PRINT_FREQ == 0:
            self.display([
                f'l_cls_src: {loss_cls_src.item():.3f}',
                f'l_cls_tar: {loss_cls_tar.item():.3f}',
                f'l_alg: {loss_mmd.item():.3f}',
                f'l_ttl: {loss_ttl.item():.3f}',
                f'best_acc: {self.best_acc:.3f}',
            ])