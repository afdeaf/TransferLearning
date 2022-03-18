import argparse
import json
from configs.defaults import get_default_and_update_cfg
from utils.utils import set_seed, create_logger
from trainer.iwan import *
import shutil
import os

TEST = True
# TEST = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',        default=107,    type=int)
    parser.add_argument('--source',      default=None,      help='Source domain name')
    parser.add_argument('--target',      default=None,      help='Target domain name')
    parser.add_argument('--output_root', default=None, type=str, help='Output root path')
    parser.add_argument('--output_dir',  default='TEST', type=str, help='Output path, subdir under output_root')
    parser.add_argument('--data_root',   default=None, type=str, help='path to dataset root')
    parser.add_argument('--num_classes', default=1,    type=int, help='The number of classes')
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
    trainer.train()
    if cfg.SHUTDOWN:
        os.system('shutdown -s -t 60')
    trainer.plot_last()
    # trainer.plot()
    

def plot():
    args = parse_args()
    cfg = get_default_and_update_cfg(args)
    cfg.freeze()

    set_seed(cfg.SEED)
    logger = create_logger(cfg.TRAIN.OUTPUT_LOG)

    logger.info('============== args ==============\n' + json.dumps(vars(args), indent=4))
    logger.info('============== cfg ==============\n' + cfg.dump(indent=4))

    trainer = eval(cfg.TRAINER)(cfg)
    trainer.plot()
    

if __name__ == '__main__':
    if TEST:
        try:
            shutil.rmtree(r'E:\Raven\VScode\PHM\OUTPUT')
            for item in os.listdir('.'):
                if item.endswith('jpg'):
                    os.remove(item)
        except:
            pass
        main()
    else:
       plot()