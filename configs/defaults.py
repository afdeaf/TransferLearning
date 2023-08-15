# Copyright (c) 2022 Raven Stock. email:cquptriven@qq.com

import os
from yacs.config import CfgNode as CN


__all__ = ['get_default_and_update_cfg']


def get_default_and_update_cfg(args):
    with open("./configs/properties.yaml") as f:
        cfg = CN.load_cfg(f)

    cfg.SEED = args.seed

    if args.trainer:
        cfg.TRAINER = args.trainer
    if args.device:
        cfg.TRAIN.DEVICE = args.device

    # ====Datasets====
    if args.data_root:
        cfg.DATASET.ROOT = args.data_root
    if args.num_classes != 4:
        cfg.DATASET.NUM_CLASSES = args.num_classes
    if args.source:
        cfg.DATASET.SOURCE = args.source
    if args.target:
        cfg.DATASET.TARGET = args.target

    # # ====Model====
    if args.basenet:
        cfg.MODEL.BASENET = args.basenet

    # ====output====
    if args.output_root:
        cfg.TRAIN.OUTPUT_ROOT = args.output_root
    if args.output_dir:
        cfg.TRAIN.OUTPUT_DIR  = args.output_dir
    else:
        # eg:20R_0HP_To_20R_8HP_seed77
        cfg.TRAIN.OUTPUT_DIR = ''.join(cfg.DATASET.SOURCE) + '_To_' + ''.join(cfg.DATASET.TARGET) + '_seed' + str(args.seed)

    cfg.TRAIN.OUTPUT_CKPT = os.path.join(cfg.TRAIN.OUTPUT_ROOT, 'ckpt', cfg.TRAIN.OUTPUT_DIR)
    cfg.TRAIN.OUTPUT_LOG = os.path.join(cfg.TRAIN.OUTPUT_ROOT, 'log', cfg.TRAIN.OUTPUT_DIR)
    # make dirs
    os.makedirs(cfg.TRAIN.OUTPUT_CKPT, exist_ok=True)
    os.makedirs(cfg.TRAIN.OUTPUT_LOG, exist_ok=True)
    cfg.TRAIN.OUTPUT_RESFILE = os.path.join(cfg.TRAIN.OUTPUT_LOG, 'log.txt')

    return cfg
