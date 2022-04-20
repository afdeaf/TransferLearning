import os
from numpy import source
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
<<<<<<< HEAD
            x_train_path = os.path.join(self.base_dir, self.target, 'x_train.pt')
            x_test_path = os.path.join(self.base_dir, self.target, 'x_test.pt')
            y_train_path = os.path.join(self.base_dir, self.target, 'y_train.pt')
            y_test_path = os.path.join(self.base_dir, self.target, 'y_test.pt')

        x_train = torch.load(x_train_path)
        x_test = torch.load(x_test_path)
        y_train = torch.load(y_train_path)
        y_test = torch.load(y_test_path)
=======
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
           
        x_data = torch.tensor(x_data).to(torch.float32)       
        x_data = torch.unsqueeze(x_data, dim=1)
        
        label = torch.tensor(label).to(torch.long)
        # 划分训练集和测试集
        x_train, x_test, y_train, y_test = train_test_split(x_data, label, test_size=self.test_size)
>>>>>>> d8c7acd5ff08caba0c2506de9b082671bcd6f928
        
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
