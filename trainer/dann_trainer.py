import logging
import torch.nn.functional as F

from trainer.base_trainer import BaseTrainer
from models.discriminator import *
import torch
from utils.utils import get_coeff
from utils.loss import d_align_uda

__all__ =  ['DANNTrainer']


class DANNTrainer(BaseTrainer):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

    def build_model(self):
        logging.info(f'--> building models: {self.cfg.MODEL.BASENET}')
        # backbone
        self.base_net = self.build_base_model()
        fdim = self.base_net.fdim
        # discriminator
        self.d_net = eval(self.cfg.MODEL.DNET)(
            in_feature=fdim,
            hidden_size=self.cfg.MODEL.D_HIDDEN_SIZE,
            # hidden_size=256,
            out_feature=self.cfg.MODEL.D_OUT_DIM
        ).to(self.device)

        self.registed_models = {'base_net': self.base_net, 'd_net': self.d_net}
        self.model_parameters()
        parameter_list = self.base_net.get_parameters() + self.d_net.get_parameters()
        self.build_optim(parameter_list)

    def one_step(self, data_src: torch.Tensor, data_tar: torch.Tensor):
        inputs_src, labels_src = data_src[0].to(self.device), data_src[1].to(self.device)
        inputs_tar, labels_tar = data_tar[0].to(self.device), data_tar[1].to(self.device)

        f_src, y = self.base_net(inputs_src)    # f, y
        f_tar, y_tar = self.base_net(inputs_tar)

        features_all = torch.cat((f_src, f_tar), dim=0)
        # classificaiton
        loss_cls_src = F.cross_entropy(y, labels_src)

        # domain alignment
        # ====================
        coeff = get_coeff(self.iter, max_iter=self.cfg.TRAIN.TTL_ITE)
        loss_alg = d_align_uda(
            features_all, features_all, self.d_net,
            coeff=coeff, ent=self.cfg.METHOD.ENT
        )
        # ====================

        loss_ttl = loss_cls_src + loss_alg * self.cfg.METHOD.W_ALG
        # update
        self.step(loss_ttl)

        # display
        if self.iter % self.cfg.TRAIN.PRINT_FREQ == 0:
            self.display([
                f'l_cls_src: {loss_cls_src.item():.3f}',
                # f'l_cls_tar: {loss_cls_tar.item():.3f}',
                f'l_alg: {loss_alg.item():.3f}',
                f'l_ttl: {loss_ttl.item():.3f}',
                # f'ent_tar: {ent_tar.item():.3f}',
                f'best_acc: {self.best_acc:.3f}',
            ])
