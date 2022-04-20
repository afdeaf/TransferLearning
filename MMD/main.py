# Copyright (c) 2022 Raven Stock. email:cquptriven@qq.com

import argparse
import json
from configs.defaults import get_default_and_update_cfg
from utils.utils import set_seed, create_logger
from trainer.mmd import *
import shutil
import os

TEST = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',        default=77,    type=int)
    parser.add_argument('--source',      default='20R_0HP',      help='Source domain name')
    parser.add_argument('--target',      default='40R_8HP',      help='Target domain name')
    parser.add_argument('--output_root', default='OUTPUT', type=str, help='Output root path')
    parser.add_argument('--output_dir',  default=None, type=str, help='Output path, subdir under output_root')
    parser.add_argument('--data_root',   default=None, type=str, help='path to dataset root')
    parser.add_argument('--num_classes', default=9,    type=int, help='The number of classes')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = get_default_and_update_cfg(args)
    cfg.freeze()

    set_seed(cfg.SEED)
    logger = create_logger(cfg.TRAIN.OUTPUT_LOG)

    logger.info('============== args ==============\n' + json.dumps(vars(args), indent=4))
    logger.info('============== cfg ==============\n' + cfg.dump(indent=4))

    trainer = eval(cfg.TRAINER)(cfg)
    best_acc = trainer.train()
    if cfg.SHUTDOWN:
        os.system('shutdown -s -t 60')


if __name__ == '__main__':
    # accs = []
    # for _ in range(20):
    #     try:
    #         shutil.rmtree(r'E:\Raven\实验结果保存\源域和目标域均未加噪\MMD\OUTPUT')
    #     except:
    #         pass
    #     best_acc = main()
    #     accs.append(best_acc)
    # with open('acc.json', 'w') as f:
    #     json.dump(accs, f, indent=4)
    if TEST:
        try:
            shutil.rmtree(r'E:\Raven\实验结果保存\源域和目标域均未加噪\MMD\OUTPUT')
        except:
            pass
    main()