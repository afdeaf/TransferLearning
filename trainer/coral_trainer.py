import logging

import torch
import torch.nn.functional as F

from trainer.base_trainer import BaseTrainer
from models.discriminator import *

from utils.utils import get_coeff
from utils.torch_utils import entropy_func
from utils.loss import coral


__all__ =  ['CORALTrainer']


class CORALTrainer(BaseTrainer):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

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

        # source domain
        f_src, y = self.base_net(inputs_src)    # f, y
        loss_cls_src = F.cross_entropy(y, labels_src)

        # target domain
        f_tar, y = self.base_net(inputs_tar)

        coeff = get_coeff(self.iter, max_iter=self.cfg.TRAIN.TTL_ITE)
        loss_coral = coral(f_src, f_tar)

        loss_ttl = loss_cls_src + coeff * loss_coral * 0.1
        # ent = entropy_func(nn.Softmax(dim=1)(f_tar.data)).mean()
        # update
        self.step(loss_ttl)

        acc_print = f'{self.best_acc:.3f}'

        # display
        if self.iter % self.cfg.TRAIN.PRINT_FREQ == 0:
            self.display([
                f'l_cls_src: {loss_cls_src.item():.3f}',
                f'l_alg: {loss_coral.item():.3f}',
                f'l_ttl: {loss_ttl.item():.3f}',
                f'best_acc: {acc_print}',
                # f'ent_tar: {ent.item():.3f}'
            ])
    