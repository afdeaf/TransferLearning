import os
from yacs.config import CfgNode as CN


__all__ = ['get_default_and_update_cfg']

_C = CN()
_C.SEED = 77
_C.WORKERS = 1
_C.TRAINER = 'IWAN'
# _C.SHUTDOWN = True
_C.SHUTDOWN = False

# ========== training ==========
_C.TRAIN = CN()
_C.TRAIN.TEST_FREQ = 100
_C.TRAIN.PRINT_FREQ = 100
_C.TRAIN.SAVE_FREQ = 300
_C.TRAIN.TTL_ITE = 5000

_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.LR = 1e-4

_C.TRAIN.OUTPUT_ROOT = 'OUTPUT'
_C.TRAIN.OUTPUT_DIR = ''
_C.TRAIN.OUTPUT_LOG = 'log'
_C.TRAIN.OUTPUT_CKPT = 'ckpt'
_C.TRAIN.OUTPUT_RESFILE = 'log.txt'

# ========== optim ==========
_C.OPTIM = CN()
_C.OPTIM.WEIGHT_DECAY = 1e-4
_C.OPTIM.MOMENTUM = 0.9

# ========== models ==========
_C.MODEL = CN()
_C.MODEL.PRETRAIN = False
_C.MODEL.BASENET = 'cnn'
_C.MODEL.DNET = 'Discriminator'
_C.MODEL.D_IN_DIM = 0
_C.MODEL.D_OUT_DIM = 1
_C.MODEL.D_HIDDEN_SIZE = 1024

# ========== datasets ==========
_C.DATASET = CN()
_C.DATASET.NUM_CLASSES = 4
_C.DATASET.NAME = 'PHM'
_C.DATASET.SOURCE = r'Learning_set\Bearing1_1'
_C.DATASET.TARGET = r'Learning_set\Bearing2_1'
_C.DATASET.ROOT = r'E:\Raven\PHM处理后'  
_C.DATASET.SHUFFLE = True
_C.DATASET.TEST_SHUFFLE = False
_C.DATASET.TEST_SIZE_SOURCE = 0.1
_C.DATASET.TEST_SIZE_TRAGET = 0.1


# ========== target ==========
_C.TARGET = CN()
_C.TARGET.START_POINT = 17
_C.TARGET.NUM_BATCHES = 20    # 32*20+77=717 < 911

# ========== method ==========
_C.METHOD = CN()
_C.METHOD.W_ALG = 1.0
_C.METHOD.ENT = False

# HDA
_C.METHOD.HDA = CN()
_C.METHOD.HDA.W_HDA = 1.0
_C.METHOD.HDA.LR_MULT = 1.0  # set as 5.0 to tune the lr_schedule to follow the setting of original HDA


def get_default_and_update_cfg(args):
    cfg = _C.clone()
    
    cfg.SEED = args.seed

    # ====Datasets====
    if args.data_root:
        cfg.DATASET_ROOT = args.data_root
    if args.num_classes != 4:
        cfg.DATASET.NUM_CLASSES = args.num_classes
    if args.source:
        cfg.DATASET.SOURCE = args.source
    if args.target:
        cfg.DATASET.TARGET = args.target

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