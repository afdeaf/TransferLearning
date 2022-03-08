# Copyright (c) 2022 Raven Stock. email:cquptriven@qq.com

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


__all__ = ['DDS']


class DDS(object):
    def __init__(self, cfg):
        '''
        base_dir:存放故障类别的根目录
        source:源域
        target:目标域
        batch_size:batch_size
        test_size:测试集大小
        '''
        self.table = ['20R_0HP', '20R_4HP', '20R_8HP',
                      '30R_0HP', '30R_4HP', '30R_8HP',
                      '40R_0HP', '40R_4HP', '40R_8HP',]
        if cfg.DATASET.SOURCE not in self.table:
            raise ValueError("param \'soruce\' error")
        if cfg.DATASET.TARGET not in self.table:
            raise ValueError("param \'target\' error")
        if cfg.DATASET.SOURCE == cfg.DATASET.TARGET:
            Warning('source and target are the same param!')

        self.base_dir = cfg.DATASET.ROOT
        self.soruce = cfg.DATASET.SOURCE
        self.target = cfg.DATASET.TARGET
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.test_size = cfg.DATASET.TEST_SIZE
        self.shuffle = cfg.DATASET.SHUFFLE
        self.num_workers = cfg.WORKERS

    def load(self, domain: str = 'source'):
        '''
        加载数据集，返回pytorch官方提供的训练代码的DataLoader的样子(训练和测试，共两个)，具体请参考torch官方的训练示例。
        也可用于普通网络的训练集、测试集加载
        '''
        assert domain == 'source' or domain == 'target', f'domain {domain} not found'

        file_list = os.listdir(self.base_dir)
        if domain == 'source':
            file_list = list(map(lambda x:os.path.join(self.base_dir, 
                                        x, self.soruce+'.npy'), file_list))  # 所有.npy的绝对路径
        else:
            file_list = list(map(lambda x:os.path.join(self.base_dir, 
                                        x, self.target+'.npy'), file_list))  # 所有.npy的绝对路径

        x_data = None
        label = None
        for index, item in enumerate(file_list):
            temp_data = np.load(item)   # 加载数据
            if label is not None:
                label = np.append(label, np.ones(temp_data.shape[0])*index)
            else:
                label = np.ones(temp_data.shape[0])*index
                
            if x_data is not None:
                x_data = np.append(x_data, temp_data, axis=0)
            else:
                x_data = temp_data
        # 打乱数据
        if self.shuffle:
            permutation = np.random.permutation(x_data.shape[0])
            # 利用np.random.permutaion函数，获得打乱后的行数，输出permutation
            x_data = x_data[permutation]
            label = label[permutation]
                
        # 划分训练集和测试集
        x_train, x_test, y_train, y_test = train_test_split(x_data, label, test_size=self.test_size)
        
        # 转化成DataLoader
        x_train = torch.tensor(x_train)
        x_train = x_train.to(torch.float32)
        x_train = torch.unsqueeze(x_train, dim=1)    # 添加一个维度，通道数 
        
        x_test = torch.tensor(x_test)
        x_test = x_test.to(torch.float32)
        x_test = torch.unsqueeze(x_test, dim=1)      # 添加一个维度，通道数 
        
        y_train = torch.tensor(y_train)
        y_train = y_train.to(torch.long)
        
        y_test = torch.tensor(y_test)
        y_test = y_test.to(torch.long)

        combined_train = []
        for x, y in zip(x_train, y_train):
            combined_train.append((x, y))

        combined_test = []
        for x, y in zip(x_test, y_test):
            combined_test.append((x, y))

        data_train = DataLoader(combined_train, 
                                batch_size=self.batch_size, 
                                shuffle=True, 
                                num_workers=self.num_workers,
                                drop_last=True)
  
        data_test = DataLoader(combined_test, 
                               batch_size=self.batch_size, 
                               shuffle=True,
                               num_workers=self.num_workers,
                               drop_last=False)

        return data_train, data_test
