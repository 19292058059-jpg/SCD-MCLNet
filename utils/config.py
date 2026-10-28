import copy
import math
from pathlib import Path
import yaml
VARIANTS = ('full',)

def default_config():
    return {'format_version': 1, 'num_classes': 7, 'variant': 'full', 'data': {'root': 'data/SECOND', 'train_split': 'train_fit_seed42', 'test_split': 'test', 'val_split': 'val_seed42', 'augmentation_policy': 'bt_reference'}, 'model': {'profile': 'paper_graph_v3', 'checkpoint_blocks': True, 'encoder_channels': [64, 64, 128, 256, 512], 'encoder_repeats': [3, 4, 6, 3], 'encoder_expansions': [2, 2, 4, 4], 'scan_dim': 128, 'scan_depth': 2, 'scan_heads': 4, 'scan_stripe': 4, 'mlp_ratio': 4.0, 'ssae_reduction': 16}, 'train': {'epochs': 50, 'batch_size': 8, 'eval_batch_size': 8, 'lr': 0.07, 'momentum': 0.9, 'weight_decay': 0.0005, 'nesterov': True, 'poly_power': 1.5, 'poly_step_origin': 1, 'seed': 42, 'workers': 0, 'alpha': 1.0, 'amp': False, 'device': 'auto', 'log_every': 10}, 'runtime': {'output_root': 'runs', 'purpose': 'research'}}

def _merge(base, supplied, location='config'):
    if not isinstance(supplied, dict):
        raise ValueError(f'{location} must be a mapping')
    for (key, value) in supplied.items():
        if key not in base:
            raise ValueError(f'Unknown key {location}.{key}')
        if isinstance(base[key], dict):
            _merge(base[key], value, f'{location}.{key}')
        else:
            base[key] = copy.deepcopy(value)

def _number(value, name, minimum=0, integer=False, strict=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a number')
    if integer and (not isinstance(value, int)):
        raise ValueError(f'{name} must be an integer')
    if not math.isfinite(value) or value < minimum or (strict and value == minimum):
        raise ValueError(f'{name} is outside its allowed range')

def validate_config(config):
    reference = default_config()
    _merge(reference, config)
    if config != reference:
        raise ValueError('Resolved config is incomplete; use load_config or default_config')
    if type(config['format_version']) is not int or config['format_version'] != 1:
        raise ValueError('Unsupported configuration format_version')
    if type(config['num_classes']) is not int or config['num_classes'] != 7:
        raise ValueError('This data pipeline is SECOND with seven classes including unchanged')
    if config['variant'] not in VARIANTS:
        raise ValueError(f'variant must be one of {VARIANTS}')
    for key in ('root', 'train_split', 'test_split'):
        if not isinstance(config['data'][key], str) or not config['data'][key].strip():
            raise ValueError(f'data.{key} must be a nonempty string')
    val = config['data']['val_split']
    policy = config['data']['augmentation_policy']
    if not isinstance(policy, str) or policy not in ('independent_flips', 'bt_reference'):
        raise ValueError('data.augmentation_policy must be independent_flips or bt_reference')
    if val is not None and (not isinstance(val, str) or not val.strip()):
        raise ValueError('data.val_split must be null or a nonempty split name')
    if not isinstance(config['runtime']['output_root'], str) or not config['runtime']['output_root'].strip():
        raise ValueError('runtime.output_root must be a nonempty path')
    if config['runtime']['purpose'] not in ('research', 'synthetic_smoke'):
        raise ValueError('runtime.purpose must be research or synthetic_smoke')
    model = config['model']
    if model['profile'] != 'paper_graph_v3':
        raise ValueError('Only the final paper_graph_v3 model is included')
    if type(model['checkpoint_blocks']) is not bool:
        raise ValueError('model.checkpoint_blocks must be bool')
    for (key, length) in (('encoder_channels', 5), ('encoder_repeats', 4), ('encoder_expansions', 4)):
        values = model[key]
        if not isinstance(values, list) or len(values) != length:
            raise ValueError(f'model.{key} needs {length} values')
        for value in values:
            _number(value, key, strict=True, integer=True)
    for key in ('scan_dim', 'scan_depth', 'scan_heads', 'scan_stripe', 'ssae_reduction'):
        _number(model[key], key, strict=True, integer=True)
    if model['scan_heads'] % 2 or model['scan_dim'] % model['scan_heads']:
        raise ValueError('CSWin needs an even number of heads dividing scan_dim')
    _number(model['mlp_ratio'], 'mlp_ratio', strict=True)
    train = config['train']
    if type(train['poly_step_origin']) is not int or train['poly_step_origin'] not in (0, 1):
        raise ValueError('train.poly_step_origin must be integer0 or1')
    for key in ('epochs', 'batch_size', 'eval_batch_size', 'log_every'):
        _number(train[key], key, strict=True, integer=True)
    for key in ('workers', 'seed'):
        _number(train[key], key, integer=True)
    if train['seed'] >= 2 ** 32:
        raise ValueError('seed must be less than 2**32')
    for key in ('lr', 'poly_power'):
        _number(train[key], key, strict=True)
    for key in ('momentum', 'weight_decay', 'alpha'):
        _number(train[key], key)
    if train['momentum'] >= 1:
        raise ValueError('momentum must be less than one')
    for key in ('nesterov', 'amp'):
        if type(train[key]) is not bool:
            raise ValueError(f'train.{key} must be boolean')
    if train['nesterov'] and train['momentum'] <= 0:
        raise ValueError('Nesterov requires positive momentum')
    if not isinstance(train['device'], str) or not train['device']:
        raise ValueError('train.device must be auto, cpu, cuda or cuda:N')
    return config

def load_config(path=None, overrides=None):
    config = default_config()
    if path is not None:
        with Path(path).open('r', encoding='utf-8-sig') as stream:
            supplied = yaml.safe_load(stream)
        if not supplied:
            raise ValueError('Configuration YAML must contain a nonempty mapping')
        _merge(config, supplied)
    if overrides:
        _merge(config, overrides)
    return validate_config(config)
