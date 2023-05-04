# Author: Raven Stock. Email:cquptriven@qq.com

import argparse
import json
from configs.defaults import get_default_and_update_cfg
from utils.utils import set_seed, create_logger
from trainer import *
import logging


from tasks import task


root = 'Your dataset path'
table = task.table
tasks = task.tasks 


def parse_args(src, tar):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',        default=777,      type=int)
    parser.add_argument('--basenet',     default='cnn2d',     type=str)
    parser.add_argument('--source',      default=src,       type=str, help='Source domain name.')
    parser.add_argument('--target',      default=tar,       type=str, help='Target domain name.')
    parser.add_argument('--output_root', default='OUTPUT',  type=str, help='Output root path.')
    parser.add_argument('--output_dir',  default=None,      type=str, help='Output path, subdir under output_root.')
    parser.add_argument('--data_root',   default=root,      type=str, help='path to dataset root.')
    parser.add_argument('--num_classes', default=9,         type=int, help='The number of classes.')
    parser.add_argument('--device',      default='cuda',    type=str, help='Device to train the models.')
    parser.add_argument('--trainer',     default='MMDTrainer',     type=str)
    args = parser.parse_args()
    return args


def main(args, count):
    cfg = get_default_and_update_cfg(args)
    cfg.freeze()

    set_seed(cfg.SEED)
    if count == 0:
        logger = create_logger(cfg.TRAIN.OUTPUT_LOG)
        logger.info('============== args ==============\n' + json.dumps(vars(args), indent=4, ensure_ascii=False))
        logger.info('============== cfg ==============\n' + cfg.dump(indent=4, allow_unicode=True))
        
    else:
        logging.info('============== args ==============\n' + json.dumps(vars(args), indent=4, ensure_ascii=False))
        logging.info('============== cfg ==============\n' + cfg.dump(indent=4, allow_unicode=True))

    trainer = eval(cfg.TRAINER)(cfg)
    if count == 0:
        logger.info('============== model ==============\n' + str(trainer.base_net))
    best_acc = trainer.train()
    
    return best_acc, cfg.TRAINER


if __name__ == '__main__':
    result = {}
    cnt = 0
    accs_srcs = []
    for key, val in tasks.items():
        args = parse_args(src=table[val[0]], tar=table[val[1]])

        best_acc, trainer_name = main(args, cnt)
        cnt += 1

        result[key] = best_acc

    with open('acc.txt', 'a+') as f:
        json.dump(trainer_name, f)
        json.dump(result, f, indent=2)
