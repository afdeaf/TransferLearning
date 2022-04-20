import os
import torch
from torch.utils.data import DataLoader


__all__ = ['DDS']


class DDS(object):
    def __init__(self, cfg):
        self.base_dir = cfg.DATASET.ROOT
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.shuffle = cfg.DATASET.SHUFFLE
        self.source = cfg.DATASET.SOURCE
        self.target = cfg.DATASET.TARGET

    def load(self, domain: str = 'source'):
        '''
        加载数据集, 返回pytorch官方提供的训练代码的DataLoader的样子(训练和测试，共两个), 具体请参考torch官方的训练示例。
        也可用于普通网络的训练集、测试集加载
        '''
        assert domain == 'source' or domain == 'target', f'domain {domain} not found'
        if domain == 'source':
            x_train_path = os.path.join(self.base_dir, self.source, 'x_train.pt')
            x_test_path = os.path.join(self.base_dir, self.source, 'x_test.pt')
            y_train_path = os.path.join(self.base_dir, self.source, 'y_train.pt')
            y_test_path = os.path.join(self.base_dir, self.source, 'y_test.pt')
        else:
            x_train_path = os.path.join(self.base_dir, self.target, 'x_train.pt')
            x_test_path = os.path.join(self.base_dir, self.target, 'x_test.pt')
            y_train_path = os.path.join(self.base_dir, self.target, 'y_train.pt')
            y_test_path = os.path.join(self.base_dir, self.target, 'y_test.pt')

        x_train = torch.load(x_train_path)
        x_test = torch.load(x_test_path)
        y_train = torch.load(y_train_path)
        y_test = torch.load(y_test_path)
        
        # 转化成DataLoader
        combined_train = []
        for x, y in zip(x_train, y_train):
            combined_train.append((x, y))

        combined_test = []
        for x, y in zip(x_test, y_test):
            combined_test.append((x, y))

        data_train = DataLoader(combined_train, 
                                batch_size=self.batch_size, 
                                shuffle=self.shuffle, 
                                drop_last=True)
  
        data_test = DataLoader(combined_test, 
                               batch_size=self.batch_size, 
                               shuffle=self.shuffle,
                               drop_last=False)

        return data_train, data_test
