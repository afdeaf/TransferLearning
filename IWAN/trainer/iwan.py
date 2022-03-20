import logging
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimSun']  
plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams["mathtext.fontset"] = 'stix'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from timm.utils import AverageMeter

# from utils.lr_scheduler import inv_lr_scheduler
from utils.utils import save_model

from trainer.base_trainer import BaseTrainer
from models.discriminator import *

from utils.utils import get_coeff
# from utils.torch_utils import entropy_func
from utils.loss import d_align_uda, sift, mae
import os

__all__ =  ['IWAN']


class IWAN(BaseTrainer):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.counter = 0

    def build_model(self):
        logging.info(f'--> building models: {self.cfg.MODEL.BASENET}')
        # backbone
        self.base_net = self.build_base_model()
        fdim = self.base_net.fdim
        # discriminator
        self.d_net = eval(self.cfg.MODEL.DNET)(
            in_feature=fdim,
            hidden_size=self.cfg.MODEL.D_HIDDEN_SIZE,
            out_feature=self.cfg.MODEL.D_OUT_DIM
        ).cuda()

        self.d0_net = Sift(
            in_feature=fdim,
            hidden_size=self.cfg.MODEL.D_HIDDEN_SIZE,
            out_feature=self.cfg.MODEL.D_OUT_DIM
        ).cuda()

        # classifier
        self.fc = Sift(
            in_feature=fdim,
            hidden_size=self.cfg.MODEL.D_HIDDEN_SIZE,
            out_feature=self.cfg.MODEL.D_OUT_DIM
        ).cuda()

        # 
        self.registed_models = {'base_net': self.base_net, 'd_net': self.d_net, 'd0_net':self.d0_net, 'fc': self.fc}
        self.model_parameters()
        parameter_list = self.base_net.get_parameters() + self.d_net.get_parameters() + self.fc.get_parameters()
        self.build_optim(parameter_list)  
        self.build_optim_d0(self.d0_net.get_parameters())  

    def build_optim_d0(self, parameter_list: list):
        self.optimizer_d0 = optim.Adam(
            parameter_list,
            lr=self.cfg.TRAIN.LR,
            weight_decay=self.cfg.OPTIM.WEIGHT_DECAY,
        )

    def one_step(self, data_src, data_tar):
        inputs_src, labels_src = data_src[0].cuda(), data_src[1].cuda()
        inputs_tar, labels_tar = data_tar[0].cuda(), data_tar[1].cuda()

        # feature extract
        feature_source = self.base_net(inputs_src)    # base_net: CNN + GRU
        feature_target = self.base_net(inputs_tar)


        d0_output_source = self.d0_net(feature_source)   # d0_net:Discriminator
        d0_output_target = self.d0_net(feature_target)
        w = nn.Sigmoid()(d0_output_source).detach()
        w = 1 - w

        # weight
        w_feature_source = w * feature_source

        # classify
        logits_source = self.fc(w_feature_source)
        logits_target = self.fc(feature_target)

        sigmoid_source = nn.Sigmoid()(logits_source)
        sigmoid_target = nn.Sigmoid()(logits_target)

        # loss, RMSE + MAE
        loss_cls_src = nn.MSELoss(reduction='mean')(sigmoid_source, labels_src.reshape((len(labels_src), 1)))
        loss_cls_src = torch.square(loss_cls_src)    # RMSE
        loss_cls_tar = nn.MSELoss(reduction='mean')(sigmoid_target.data, labels_tar.reshape((len(labels_tar), 1)))
        loss_cls_tar = torch.square(loss_cls_tar)    # RMSE

        loss_mae_src = nn.L1Loss(reduction='mean')(sigmoid_source, labels_src.reshape((len(labels_src), 1)))
        loss_mae_tar = nn.L1Loss(reduction='mean')(sigmoid_target.data, labels_tar.reshape((len(labels_tar), 1)))

        # feature sift, d0
        d0_features_all = torch.cat((d0_output_source, d0_output_target), dim=0)
        loss_sift = sift(d0_features_all)

        # domain alignment
        features_all = torch.cat((w_feature_source, feature_target), dim=0)
        loss_alg = d_align_uda(
            features_all, features_all, self.d_net,
            coeff=get_coeff(self.iter, max_iter=self.cfg.TRAIN.TTL_ITE), ent=self.cfg.METHOD.ENT
        )

        # RMSE + MAE + domain loss
        loss_ttl = 0.5 * loss_cls_src + 0.5 *loss_mae_src  + loss_alg * self.cfg.METHOD.W_ALG

        # update
        self.step(loss_ttl, loss_sift)

        # display
        if self.iter % self.cfg.TRAIN.PRINT_FREQ == 0 and self.iter != 0:
            self.display([
                f'mse_src: {loss_cls_src.item():.3f}',
                f'mse_tar: {loss_cls_tar.item():.3f}',
                f'mae_src: {loss_mae_src.item():.3f}',
                f'mae_tar: {loss_mae_tar.item():.3f}',
                f'l_alg: {loss_alg.item():.3f}',
                f'l_ttl: {loss_ttl.item():.3f}',
                f'best_mae: {self.best_mae:.3f}',
                f'loss_sift: {loss_sift.item():.3f}',
            ])

    def step(self, loss_ttl, loss_sift):
        self.optimizer_d0.zero_grad()
        loss_sift.backward(retain_graph=True)
        self.optimizer_d0.step()

        self.optimizer.zero_grad()
        loss_ttl.backward()
        self.optimizer.step()

    def save_model(self, is_best=False, snap=False):
        data_dict = {
            'optimizer': self.optimizer.state_dict(),
            'iter': self.iter,
            'best_mae': self.best_mae,
            'optimizer_d0': self.optimizer_d0.state_dict()
        }
        for k, v in self.registed_models.items():
            data_dict.update({k: v.state_dict()})
        save_model(self.cfg.TRAIN.OUTPUT_CKPT, data_dict=data_dict, iter=self.iter, is_best=is_best, snap=snap)

    def plot_last(self):
        logging.info('==>Ploting...')
        loader = self.dataset_loaders['target_test']
        labels = []
        labels_pred = []
        with torch.no_grad():
            iter_test = iter(loader)
            for _ in range(len(loader)):
                data = iter_test.__next__()
                labels += data[1].tolist()
                inputs, label = data[0].cuda(), data[1].cuda()
                feature = self.base_net(inputs)
                outputs = self.fc(feature)
                outputs = nn.Sigmoid()(outputs)
                labels_pred += outputs.ravel().cpu().tolist()

            plt.figure()
            plt.plot(labels[::2], color='r',  markerfacecolor='r', marker='o', label='真实RUL')
            plt.plot(labels_pred[::2], color='c',  markerfacecolor='c', marker='^', label='预测RUL')
            plt.xlabel('Points')
            plt.ylabel('RUL')
            plt.ylim(0, 1.1)
            plt.title('真实RUL-预测RUL')
            plt.legend(loc='best')
            filename = 'figure' + str(self.counter) + '.jpg'
            self.counter += 1
            plt.savefig(filename, dpi=330)
            plt.close()

    def plot(self):
        self.load_best()
        loader = self.dataset_loaders['target_test']
        labels = []
        labels_pred = []
        with torch.no_grad():
            iter_test = iter(loader)
            for _ in range(len(loader)):
                data = iter_test.__next__()
                labels += data[1].tolist()
                inputs, label = data[0].cuda(), data[1].cuda()
                feature_source = self.base_net(inputs)
                outputs = self.fc(feature_source)
                outputs = nn.Sigmoid()(outputs)
                labels_pred += outputs.ravel().cpu().tolist()
            plt.plot(labels, color='r', ls='-', label='RUL')
            plt.plot(labels_pred, color='g', ls=':', label='pred')
            plt.xlabel('Points')
            plt.ylabel('label')
            plt.ylim(0, 1.1)
            plt.title('RUL-Pred')
            plt.legend(loc='best')
            filename = 'figure.jpg'
            plt.savefig(filename, dpi=330)


    def load_best(self):
        last_ckpt = os.path.join(self.cfg.TRAIN.OUTPUT_CKPT, 'models-best.pt')
        print('===> Ploting...')
        if os.path.exists(last_ckpt):
            ckpt = torch.load(last_ckpt)
            for k, v in self.registed_models.items():
                v.load_state_dict(ckpt[k])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.start_iter = ckpt['iter']
            self.best_mae = ckpt['best_mae']
            logging.info(f'> loading ckpt from {last_ckpt} | iter: {self.start_iter} | best_mae: {self.best_mae:.3f}')
        else:
            logging.info('--> training from scratch')
