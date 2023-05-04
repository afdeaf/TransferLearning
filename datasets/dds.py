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

    def load(self, domain: str = '20R_0HP'):
        x_train_path = os.path.join(self.base_dir, domain, 'x_train.pt')
        x_test_path = os.path.join(self.base_dir, domain, 'x_test.pt')
        y_train_path = os.path.join(self.base_dir, domain, 'y_train.pt')
        y_test_path = os.path.join(self.base_dir, domain, 'y_test.pt')

        x_train = torch.load(x_train_path)
        x_test = torch.load(x_test_path)
        y_train = torch.load(y_train_path)
        y_test = torch.load(y_test_path)

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
