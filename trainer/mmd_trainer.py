import logging

import torch
import torch.nn.functional as F

from trainer.base_trainer import BaseTrainer
from models.discriminator import *

from utils.utils import get_coeff
from utils.torch_utils import entropy_func
from utils.loss import MMDLoss
from torch import nn

__all__ =  ['MMDTrainer']


class MMDTrainer(BaseTrainer):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.mmd = MMDLoss()

    def build_model(self):
        logging.info(f'--> building models: {self.cfg.MODEL.BASENET}')
        # backbone
        self.base_net = self.build_base_model()

        self.registed_models = {'base_net': self.base_net}
        self.model_parameters()
        parameter_list = self.base_net.get_parameters()
        self.build_optim(parameter_list)

    def one_step(self, data_src: torch.Tensor, data_tar: torch.Tensor):
        inputs_src, labels_src = data_src[0].to(self.device), data_src[1].to(self.device)
        inputs_tar, labels_tar = data_tar[0].to(self.device), data_tar[1].to(self.device)

        f_src, y = self.base_net(inputs_src)    # x1, x2, x3, f, y
        loss_cls_src = F.cross_entropy(y, labels_src)
        f_tar, y = self.base_net(inputs_tar)
        # loss_cls_tar = F.cross_entropy(y.data, labels_tar)

        coeff = get_coeff(self.iter, max_iter=self.cfg.TRAIN.TTL_ITE)
        loss_mmd = self.mmd(f_src, f_tar)

        loss_ttl = loss_cls_src + coeff * loss_mmd
        ent = entropy_func(nn.Softmax(dim=1)(f_tar.data)).mean()
        # print(ent)
        # exit()

        # update
        self.step(loss_ttl)

        # display
        if self.iter % self.cfg.TRAIN.PRINT_FREQ == 0:
            self.display([
                f'l_cls_src: {loss_cls_src.item():.3f}',
                f'l_alg: {loss_mmd.item():.3f}',
                f'l_ttl: {loss_ttl.item():.3f}',
                f'best_acc: {self.best_acc:.3f}',
                f'ent_tar: {ent.item():.3f}'
            ])
    