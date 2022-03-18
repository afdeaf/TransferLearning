import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os


__all__ = ['PHM']


class PHM(object):
    def __init__(self, cfg):
        self.base_dir = cfg.DATASET.ROOT
        self.source = cfg.DATASET.SOURCE
        self.target = cfg.DATASET.TARGET

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.test_size_source = cfg.DATASET.TEST_SIZE_SOURCE
        self.test_size_target = cfg.DATASET.TEST_SIZE_TRAGET

        self.shuffle = cfg.DATASET.SHUFFLE
        self.num_workers = cfg.WORKERS

        self.start = cfg.TARGET.START_POINT
        self.target_num_batches = cfg.TARGET.NUM_BATCHES
        self.end = self.start + self.start * self.target_num_batches
        self.tar_shuffle = cfg.DATASET.TEST_SHUFFLE

    def _load_source(self) -> tuple:
        x_data = np.load(os.path.join(self.base_dir, self.source, 'data.npy'))   # ?*2560
        y = np.zeros(len(x_data), dtype=np.float32)
        for i in range(len(x_data)):
            y[i] = 1. - float(i)/float(len(x_data))

        x_data, y = torch.tensor(x_data).to(torch.float32), torch.tensor(y)
        x_data = torch.unsqueeze(x_data, dim=1)
        x_train, x_test, y_train, y_test = train_test_split(x_data, y, test_size=self.test_size_source)
        combined_train = []
        for x, y in zip(x_train, y_train):
            combined_train.append((x, y))

        combined_test = []
        for x, y in zip(x_test, y_test):
            combined_test.append((x, y))

        data_train = DataLoader(combined_train, batch_size=self.batch_size, shuffle=self.shuffle, 
                                num_workers=self.num_workers, drop_last=True)
  
        data_test = DataLoader(combined_test, batch_size=self.batch_size, shuffle=self.shuffle,
                               num_workers=self.num_workers,
                               drop_last=False)

        return data_train, data_test

    def _load_target(self):
        x_data = np.load(os.path.join(self.base_dir, self.target, 'data.npy'))   # ?*2560
        y = np.zeros(len(x_data), dtype=np.float32)
        for i in range(len(x_data)):
            y[i] = 1. - float(i)/float(len(x_data))
        x_data = torch.tensor(x_data[self.start: self.end, :]).to(torch.float32)
        y = torch.tensor(y[self.start: self.end])
        x_data = torch.unsqueeze(x_data, dim=1)
        x_train, x_test, y_train, y_test = train_test_split(x_data, y, 
                                                            test_size=self.test_size_target, 
                                                            shuffle=self.tar_shuffle)
        combined_train = []
        for x, y in zip(x_train, y_train):
            combined_train.append((x, y))

        combined_test = []
        for x, y in zip(x_test, y_test):
            combined_test.append((x, y))

        data_train = DataLoader(combined_train, batch_size=self.batch_size, shuffle=self.shuffle, 
                                num_workers=self.num_workers, drop_last=True)
  
        data_test = DataLoader(combined_test, batch_size=self.batch_size, shuffle=False,
                               num_workers=self.num_workers, drop_last=False)

        return data_train, data_test

    def load(self, domain: str='source') -> tuple:
        assert domain == 'source' or domain == 'target', f'domain {domain} not found'
        if domain == 'source':
            return self._load_source()
        else:
            return self._load_target()
