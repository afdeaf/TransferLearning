
import logging
import torch
from datasets.phm import *
from torch import optim
from utils.lr_scheduler import inv_lr_scheduler
import os
from timm.utils import AverageMeter

from utils.utils import save_model, write_log
from models import *
from abc import abstractmethod
from torch import nn

import torch.nn.functional as F


__all__ =['BaseTrainer']


class BaseTrainer(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        logging.info(f'--> trainer: {self.__class__.__name__}')
        self.setup()
        self.build_datasets()
        self.build_model()
        self.resume_from_ckpt()

    def setup(self):
        self.start_iter = 0
        self.iter = 0
        self.best_mae = 10.
    
    def build_datasets(self):
        logging.info(f'--> building dataset from {self.cfg.DATASET.NAME}')
        self.dataset_loaders = {}
        phm = PHM(self.cfg)

        self.dataset_loaders['source_train'], self.dataset_loaders['source_test'] = phm.load(domain='source')
        self.dataset_loaders['target_train'], self.dataset_loaders['target_test'] = phm.load(domain='target')

        self.len_src = len(self.dataset_loaders['source_train'])
        self.len_tar = len(self.dataset_loaders['target_train'])
        logging.info(f'    source {self.cfg.DATASET.SOURCE}: {self.len_src}'
                     f'/{len(self.dataset_loaders["source_test"])}')  
        logging.info(f'    target {self.cfg.DATASET.TARGET}: {self.len_tar}'
                     f'/{len(self.dataset_loaders["target_test"])}')

    def build_model(self):
        logging.info(f'--> building models: {self.cfg.MODEL.BASENET}')
        self.base_net = self.build_base_model()
        self.registed_models = {'base_net': self.base_net}
        param_list = self.base_net.get_parameters()
        self.model_parameters()
        self.build_optim(param_list)
        

    def build_base_model(self):
        basenet_name = self.cfg.MODEL.BASENET
        kwargs = {
            'pretrained': self.cfg.MODEL.PRETRAIN,
            'num_classes': self.cfg.DATASET.NUM_CLASSES
        }
        basenet = eval(basenet_name)(**kwargs)
        return basenet

    def model_parameters(self):
        for k, v in self.registed_models.items():
            logging.info(f'    {k} paras: '
                         f'{(sum(p.numel() for p in v.parameters()) / 1e6):.2f}M')

    def build_optim(self, parameter_list: list):
        self.optimizer = optim.Adam(
            parameter_list,
            lr=self.cfg.TRAIN.LR,
            # momentum=self.cfg.OPTIM.MOMENTUM,
            weight_decay=self.cfg.OPTIM.WEIGHT_DECAY,
            # nesterov=True
        )
        self.lr_scheduler = inv_lr_scheduler

    def resume_from_ckpt(self):
        last_ckpt = os.path.join(self.cfg.TRAIN.OUTPUT_CKPT, 'models-last.pt')
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

    def train(self):
        # start training
        for _, v in self.registed_models.items():
            v.train()
        for self.iter in range(self.start_iter, self.cfg.TRAIN.TTL_ITE):
            # test
            if self.iter % self.cfg.TRAIN.TEST_FREQ == self.cfg.TRAIN.TEST_FREQ - 1 and self.iter != self.start_iter:
                self.base_net.eval()
                self.d0_net.eval()
                self.d_net.eval()
                self.test()
                self.d_net.train()
                self.d0_net.train()
                self.base_net.train()

            self.current_lr = self.lr_scheduler(
                self.optimizer,
                ite_rate=self.iter / self.cfg.TRAIN.TTL_ITE * self.cfg.METHOD.HDA.LR_MULT,
                lr=self.cfg.TRAIN.LR,
            )
            if self.iter % self.len_src == 0 or self.iter == self.start_iter:
                iter_src = iter(self.dataset_loaders['source_train'])
            if self.iter % self.len_tar == 0 or self.iter == self.start_iter:
                iter_tar = iter(self.dataset_loaders['target_train'])

            # forward one iteration
            data_src = iter_src.__next__()
            data_tar = iter_tar.__next__()
            self.one_step(data_src, data_tar)
            if self.iter % self.cfg.TRAIN.SAVE_FREQ == 0 and self.iter != 0:
                self.save_model(is_best=False, snap=True)

    @abstractmethod
    def one_step(self, data_src, data_tar):
        pass

    def display(self, data: list):
        log_str = f'I:  {self.iter}/{self.cfg.TRAIN.TTL_ITE} | lr: {self.current_lr:.5f} '
        # update
        for _str in data:
            log_str += '| {} '.format(_str)
        logging.info(log_str)

    def step(self, loss_ttl):
        self.optimizer.zero_grad()
        loss_ttl.backward()
        self.optimizer.step()

    def test(self):
        logging.info('=================== Test ===================')
        logging.info('--> testing on source_test')
        src_mae = self.test_func_source(self.dataset_loaders['source_test'])
        logging.info('--> testing on target_test')
        tar_mae = self.test_func_target(self.dataset_loaders['target_test'])
        self.plot_last()
        # print(tar_mae)
        is_best = False
        if tar_mae < self.best_mae:
            self.best_mae = tar_mae
            is_best = True

        # display
        log_str = f'I:  {self.iter}/{self.cfg.TRAIN.TTL_ITE} | src_mae: {src_mae:.3f} | tar_mae: {tar_mae:.3f} | ' \
                  f'best_mae: {self.best_mae:.3f}'
        logging.info(log_str)
        logging.info('================= End test =================')

        # save results
        log_dict = {
            'I': self.iter,
            'src_mae': src_mae,
            'tar_mae': tar_mae,
            'best_mae': self.best_mae
        }
        write_log(self.cfg.TRAIN.OUTPUT_RESFILE, log_dict)
        self.save_model(is_best=is_best)

    def test_func_source(self, loader):
        with torch.no_grad():
            iter_test = iter(loader)
            print_freq = max(len(loader) // 5, self.cfg.TRAIN.PRINT_FREQ)
            maes = AverageMeter()
            for i in range(len(loader)):
                if i % print_freq == print_freq - 1:
                    logging.info('    I:  {}/{} | acc: {:.3f}'.format(i, len(loader), maes.avg))
                data = iter_test.__next__()
                inputs, labels = data[0].cuda(), data[1].cuda()
                feature_source = self.base_net(inputs)
                w_feature_source = (1 - self.d0_net(feature_source).detach()) * feature_source
                outputs = self.fc(w_feature_source)
                outputs = nn.Sigmoid()(outputs)

                mae = nn.MSELoss()(outputs, labels.reshape((len(labels), 1)))
                maes.update(mae.item(), labels.size(0))
        return maes.avg

    def test_func_target(self, loader):
        with torch.no_grad():
            iter_test = iter(loader)
            print_freq = max(len(loader) // 5, self.cfg.TRAIN.PRINT_FREQ)
            maes = AverageMeter()
            for i in range(len(loader)):
                if i % print_freq == print_freq - 1:
                    logging.info('    I:  {}/{} | acc: {:.3f}'.format(i, len(loader), maes.avg))
                data = iter_test.__next__()
                inputs, labels = data[0].cuda(), data[1].cuda()
                feature = self.base_net(inputs)
                outputs = self.fc(feature)
                outputs = nn.Sigmoid()(outputs)
                mae = nn.MSELoss()(outputs, labels.reshape((len(labels), 1)))
                maes.update(mae.item(), labels.size(0))
        return maes.avg

    def save_model(self, is_best=False, snap=False):
        data_dict = {
            'optimizer': self.optimizer.state_dict(),
            'iter': self.iter,
            'best_mae': self.best_mae
        }
        for k, v in self.registed_models.items():
            data_dict.update({k: v.state_dict()})
        save_model(self.cfg.TRAIN.OUTPUT_CKPT, data_dict=data_dict, iter=self.iter, is_best=is_best, snap=snap)