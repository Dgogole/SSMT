from utils import dist_utils, misc
from utils.config import *
from loguru import logger
import argparse
import os
from pathlib import Path
import torch
from tensorboardX import SummaryWriter
from datasets.DentalDataset import DentalDataset
from models.SSMT import SSMT
from accelerate import Accelerator, DistributedDataParallelKwargs
from runners.train_ssmt import train_ssmt
from runners.test_ssmt import test_ssmt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='yaml config file')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='Number of batches prefetched by each DataLoader worker.')
    parser.add_argument('--persistent_workers', dest='persistent_workers', action='store_true',
                        help='Keep DataLoader workers alive between epochs.')
    parser.add_argument('--no_persistent_workers', dest='persistent_workers', action='store_false',
                        help='Disable persistent DataLoader workers.')
    parser.set_defaults(persistent_workers=True)
    parser.add_argument('--deterministic', action='store_true',
                        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--save_root', type=str, required=True)
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--run_id', type=str, required=True, help='Unique ID for this run (timestamp from shell)')
    parser.add_argument('--ckpts', type=str, default='')
    parser.add_argument('--encoder_ckpts', type=str, default='')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--val_freq', type=int, default=1, help='test freq')
    parser.add_argument('--test', action='store_true', default=False,
                        help='test mode for certain ckpt')
    parser.add_argument('--finetune_model', action='store_true', default=False,
                        help='finetune modelnet with pretrained weight')

    args = parser.parse_args()

    args.experiment_path = os.path.join(args.save_root, args.exp_name, args.run_id)
    args.tfboard_path = os.path.join(args.experiment_path, 'TFBoard')
    args.checkpoints_path = os.path.join(args.experiment_path, 'checkpoints')
    args.log_name = Path(args.config).stem
    return args


def create_experiment_dir(args):
    for path in [args.experiment_path, args.tfboard_path, args.checkpoints_path]:
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

    # --- Accelerator ---
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    # --- Main Process Setup ---
    if accelerator.is_main_process:
        create_experiment_dir(args)

        # Running log (excludes eval messages)
        logger.add(
            os.path.join(args.experiment_path, 'running.log'),
            filter=lambda record: record["extra"].get("type") != "eval"
        )

        log_args_to_file(args, logger, os.path.join(args.experiment_path, 'args.txt'))

        if not args.test:
            train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
            val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'val'))
        else:
            train_writer = None
            val_writer = None
    else:
        train_writer = None
        val_writer = None

    accelerator.wait_for_everyone()

    # --- Run ---
    if args.test:
        test_ssmt(args, config, logger)
    else:
        train_ssmt(args, config, train_writer, val_writer, logger, accelerator)


if __name__ == '__main__':
    main()
