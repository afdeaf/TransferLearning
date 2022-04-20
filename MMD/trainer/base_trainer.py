# Copyright (c) 2022 Raven Stock. email:cquptriven@qq.com

import logging
import torch
from datasets.dds import *
from torch import optim
from utils.lr_scheduler import inv_lr_scheduler
import os
from timm.utils import accuracy, AverageMeter
from utils.utils import save_model, write_log
from models import *
from abc import abstractmethod

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
        self.best_acc = 0.
    
    def build_datasets(self):
        '''
        function:构建源域和目标域的训练集、测试集
        '''
        logging.info(f'--> building dataset from {self.cfg.DATASET.NAME}')
        self.dataset_loaders = {}
        dds = DDS(self.cfg)

        self.dataset_loaders['source_train'], self.dataset_loaders['source_test'] = dds.load(domain='source')
        self.dataset_loaders['target_train'], self.dataset_loaders['target_test'] = dds.load(domain='target')

        self.len_src = len(self.dataset_loaders['source_train'])
        self.len_tar = len(self.dataset_loaders['target_train'])
        logging.info(f'    source {self.cfg.DATASET.SOURCE}: {self.len_src}'
                     f'/{len(self.dataset_loaders["source_test"])}')
        logging.info(f'    target {self.cfg.DATASET.TARGET}: {self.len_tar}'
                     f'/{len(self.dataset_loaders["target_test"])}')

    def build_model(self):
        '''
        function:构建特征提取器和优化器
        '''
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
        '''
        function:打印模型参数数量
        '''
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
            self.best_acc = ckpt['best_acc']
            logging.info(f'> loading ckpt from {last_ckpt} | iter: {self.start_iter} | best_acc: {self.best_acc:.3f}')
        else:
            logging.info('--> training from scratch')

    def train(self):
        # start training
        for _, v in self.registed_models.items():
            v.train()
        for self.iter in range(self.start_iter, self.cfg.TRAIN.TTL_ITE):
            # test
            if self.iter % self.cfg.TRAIN.TEST_FREQ == self.cfg.TRAIN.TEST_FREQ - 1 and self.iter != self.start_iter:
                # 满足测试频率，且不是初始那一带，则测试模型
                self.base_net.eval()
                self.test()
                self.base_net.train()

            self.current_lr = self.lr_scheduler(
                self.optimizer,
                ite_rate=self.iter / self.cfg.TRAIN.TTL_ITE * self.cfg.METHOD.HDA.LR_MULT,
                lr=self.cfg.TRAIN.LR,
            )

            # dataloader
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
        return self.best_acc
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
        logging.info('--> testing on source_test')
        src_acc = self.test_func(self.dataset_loaders['source_test'], self.base_net)
        logging.info('--> testing on target_test')
        tar_acc = self.test_func(self.dataset_loaders['target_test'], self.base_net)
        is_best = False
        if tar_acc > self.best_acc:
            self.best_acc = tar_acc
            is_best = True

        # display
        log_str = f'I:  {self.iter}/{self.cfg.TRAIN.TTL_ITE} | src_acc: {src_acc:.3f} | tar_acc: {tar_acc:.3f} | ' \
                  f'best_acc: {self.best_acc:.3f}'
        logging.info(log_str)

        # save results
        log_dict = {
            'I': self.iter,
            'src_acc': round(src_acc, 3),
            'tar_acc': round(tar_acc, 3),
            'best_acc': round(self.best_acc, 3)
        }
        write_log(self.cfg.TRAIN.OUTPUT_RESFILE, log_dict)

        # tensorboard
        # self.tb_writer.add_scalar('tar_acc', tar_acc, self.iter)
        # self.tb_writer.add_scalar('src_acc', src_acc, self.iter)

        self.save_model(is_best=is_best)

    def test_func(self, loader, model):
        with torch.no_grad():
            iter_test = iter(loader)
            print_freq = max(len(loader) // 5, self.cfg.TRAIN.PRINT_FREQ)
            accs = AverageMeter()
            for i in range(len(loader)):
                if i % print_freq == print_freq - 1:
                    logging.info('    I:  {}/{} | acc: {:.3f}'.format(i, len(loader), accs.avg))
                data = iter_test.__next__()
                inputs, labels = data[0].cuda(), data[1].cuda()
                outputs_all = model(inputs)  # [f, y, ...]
                outputs = outputs_all[1]

                acc = accuracy(outputs, labels)[0]
                accs.update(acc.item(), labels.size(0))

        return accs.avg

    def save_model(self, is_best=False, snap=False):
        data_dict = {
            'optimizer': self.optimizer.state_dict(),
            'iter': self.iter,
            'best_acc': self.best_acc
        }
        for k, v in self.registed_models.items():
            data_dict.update({k: v.state_dict()})
        save_model(self.cfg.TRAIN.OUTPUT_CKPT, data_dict=data_dict, ite=self.iter, is_best=is_best, snap=snap)