import yaml
from easydict import EasyDict
import os
import shutil

def log_args_to_file(args, logger, file_name):
    # 使用返回的 handler_id 来精确删除，避免索引冲突
    handler_id = logger.add(file_name, level='INFO', format='{message}', filter=lambda record: record["level"].name == "INFO")
    for key, val in args.__dict__.items():
        logger.info(f'{args}.{key} : {val}')
    logger.remove(handler_id)

# def log_config_to_file(cfg, logger, pre='cfg'):
#     for key, val in cfg.items():
#         if isinstance(cfg[key], EasyDict):
#             print_log(f'{pre}.{key} = edict()', logger = logger)
#             log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
#             continue
#         print_log(f'{pre}.{key} : {val}', logger = logger)

def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                with open(new_config['_base_'], 'r') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config

def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)
    merge_new_config(config=config, new_config=new_config)        
    return config

def get_config(args, logger=None):
    if args.resume:
        cfg_path = os.path.join(args.experiment_path, 'config.yaml')
        if not os.path.exists(cfg_path):
            logger.info("Failed to resume", logger = logger)
            raise FileNotFoundError()
        logger.info(f'Resume yaml from {cfg_path}')
        args.config = cfg_path
    config = cfg_from_yaml_file(args.config)
    if not args.resume:
        save_experiment_config(args, config, logger)
    return config

def save_experiment_config(args, config, logger = None):
    config_path = os.path.join(args.experiment_path, 'config.yaml')
    os.makedirs(args.experiment_path, exist_ok=True)
    shutil.copy2(args.config, config_path)
    if logger is not None:
        logger.info(f'Copy the Config file from {args.config} to {config_path}')
