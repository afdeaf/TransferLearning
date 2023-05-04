# Copyright (c) 2022 Raven Stock. email:cquptriven@qq.com

import os
from yacs.config import CfgNode as CN


__all__ = ['get_default_and_update_cfg']

_C = CN()
_C.SEED = 777
_C.WORKERS = 1
_C.TRAINER = 'CNNTrainer'

# ========== training ==========
_C.TRAIN = CN()
_C.TRAIN.TEST_FREQ = 100
_C.TRAIN.PRINT_FREQ = 100
_C.TRAIN.SAVE_FREQ = 300
_C.TRAIN.TTL_ITE = 3000


_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.LR = 1e-3

_C.TRAIN.OUTPUT_ROOT = 'temp'
_C.TRAIN.OUTPUT_DIR = ''
_C.TRAIN.OUTPUT_LOG = 'log'
_C.TRAIN.OUTPUT_CKPT = 'ckpt'
_C.TRAIN.OUTPUT_RESFILE = 'log.txt'
_C.TRAIN.OUTPUT_TB = 'tensorboard'
_C.TRAIN.DEVICE = 'gpu'

# ========== optim ==========
_C.OPTIM = CN()
_C.OPTIM.WEIGHT_DECAY = 5e-4
_C.OPTIM.MOMENTUM = 0.9

# ========== models ==========
_C.MODEL = CN()
_C.MODEL.PRETRAIN = False
_C.MODEL.BASENET = 'cnn'
_C.MODEL.DNET = 'Discriminator'
_C.MODEL.D_IN_DIM = 0
_C.MODEL.D_OUT_DIM = 1
_C.MODEL.D_HIDDEN_SIZE = 512

# ========== datasets ==========
_C.DATASET = CN()
_C.DATASET.NUM_CLASSES = 9
_C.DATASET.NAME = 'DDS'
_C.DATASET.SOURCE = '20R_0HP'
_C.DATASET.TARGET = '20R_8HP'
_C.DATASET.VALID = 'valid'
_C.DATASET.ROOT = ''
_C.DATASET.SHUFFLE = True

# ========== method ==========
_C.METHOD = CN()
_C.METHOD.W_ALG = 1
_C.METHOD.ENT = False


def get_default_and_update_cfg(args):
    cfg = _C.clone()
    
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