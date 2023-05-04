# Copyright (c) 2022 Raven Stock. email:cquptriven@qq.com

import logging
import os
import random
import time
import numpy as np
import torch
from termcolor import colored
import json


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # if the following two rows are actived, the result can be ensured to be the same every time.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

def create_logger(log_dir):
    # create logger
    os.makedirs(log_dir, exist_ok=True)
    time_str = time.strftime('%m-%d-%H-%M')
    log_file = '{}.log'.format(time_str)
    final_log_file = os.path.join(log_dir, log_file)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    #
    fmt = '[%(asctime)s] %(message)s'
    color_fmt = colored('[%(asctime)s]', 'green') + ' %(message)s'

    file = logging.FileHandler(filename=final_log_file, mode='a', encoding='utf-8')
    file.setLevel(logging.INFO)
    file.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console)

    return logger

def write_log(log_path, log):
    with open(log_path, 'a') as f:
        f.write(json.dumps(log) + '\n')

def save_model(path, data_dict, ite, snap=False, is_best=False):
    ckpt = data_dict
    if snap:
        save_path = os.path.join(path, f'models-{ite:06d}.pt')
    else:
        save_path = os.path.join(path, f'models-last.pt')
    torch.save(ckpt, save_path)
    logging.info(f'    models saved to {save_path}')
    logging.info(f'    keys: {data_dict.keys()}')
    if is_best:
        save_path = os.path.join(path, f'models-best.pt')
        torch.save(ckpt, save_path)
        logging.info(f'    best models saved to {save_path}')

def get_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)
