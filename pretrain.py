from utils import dist_utils, misc
from utils.config import *
from loguru import logger
import argparse
import time
import os
from pathlib import Path
import torch
from tensorboardX import SummaryWriter
from datasets.TeethDataset import teethDataset
from models.SSMT import SSMT
from runners.pretrain_runner import pretrain


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='yaml config file')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--deterministic', action='store_true',
                        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--save_root', type=str, required=True)
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--ckpts', type=str, default='')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--val_freq', type=int, default=1, help='test freq')
    parser.add_argument('--finetune_model', action='store_true', default=False,
                        help='finetune modelnet with pretrained weight')
    parser.add_argument('--test', action='store_true', default=False,
                        help='run test')

    args = parser.parse_args()

    if args.test:
        args.exp_name = 'test_' + args.exp_name
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    args.experiment_path = os.path.join(args.save_root, args.exp_name, timestamp)
    args.tfboard_path = os.path.join(args.experiment_path, 'TFBoard')
    args.log_name = Path(args.config).stem
    return args


def create_experiment_dir(args):
    for path in [args.experiment_path, args.tfboard_path]:
        if not os.path.exists(path):
            os.makedirs(path)
            print('Created directory: %s' % path)


def main():
    # --- Args & Config ---
    args = get_args()
    config = get_config(args, logger=logger)
    config.dataset.train.bs = config.total_bs
    config.dataset.val.bs = config.total_bs
    if config.dataset.get('test'):
        config.dataset.test.bs = config.total_bs

    # --- GPU & Seed ---
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    misc.set_random_seed(args.seed, deterministic=args.deterministic)

    # --- Dirs & Logging ---
    create_experiment_dir(args)
    logger.add(os.path.join(args.experiment_path, 'running.log'))
    log_args_to_file(args, logger, os.path.join(args.experiment_path, 'args.txt'))

    # --- TensorBoard Writer ---
    if not args.test:
        train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
        val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'val'))
    else:
        train_writer = None
        val_writer = None

    # --- Run ---
    pretrain(args, config, train_writer, val_writer, logger)


if __name__ == '__main__':
    main()
