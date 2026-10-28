import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import uuid
import numpy as np
import torch
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_VERSION = 'scd-mclnet-1'

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()

def digest_json(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()

def source_digest():
    paths = list(PROJECT_ROOT.glob('*.py'))
    for directory in ('models', 'datasets', 'utils'):
        paths.extend((PROJECT_ROOT / directory).rglob('*.py'))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode('utf-8'))
        digest.update(path.read_bytes())
    return digest.hexdigest()

def split_identity(root, splits):
    root = Path(root)
    identity = {}
    for (split, names) in splits.items():
        files = []
        for name in names:
            for folder in ('A', 'B', 'label1', 'label2'):
                path = root / folder / name
                stat = path.stat()
                files.append([folder + '/' + name, stat.st_size, stat.st_mtime_ns])
        identity[split] = {'count': len(names), 'ordered_names_sha256': digest_json(names), 'file_metadata_sha256': digest_json(files)}
    return identity

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)

def capture_rng():
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])], 'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(state):
    random.setstate(state['python'])
    numpy = state['numpy']
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), numpy[2], numpy[3], numpy[4]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Cannot restore CUDA RNG to a different GPU count')
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])

def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()

def load_checkpoint(path):
    checkpoint = torch.load(Path(path), map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get('format_version') != 1:
        raise ValueError('Expected an SCD-MCLNet checkpoint with format_version=1')
    return checkpoint

def resume_signature(config, splits, normalization, code_digest):
    train = {key: value for (key, value) in config['train'].items() if key not in ('device', 'log_every', 'eval_batch_size')}
    return {'implementation': IMPLEMENTATION_VERSION, 'variant': config['variant'], 'num_classes': config['num_classes'], 'model': config['model'], 'train': train, 'augmentation_policy': config['data']['augmentation_policy'], 'splits': splits, 'normalization': normalization, 'source_digest': code_digest}

def validate_resume(saved, current):
    differences = [key for key in sorted(set(saved) | set(current)) if saved.get(key) != current.get(key)]
    if differences:
        raise ValueError('Resume compatibility mismatch: ' + ', '.join(differences) + '. Use a new run for changed experiments; do not silently resume.')

def polynomial_lr(initial_lr, completed_steps, total_steps, power):
    if total_steps <= 0 or not 0 <= completed_steps <= total_steps or (not math.isfinite(power)) or (power <= 0):
        raise ValueError('Invalid polynomial schedule bounds')
    return initial_lr * (1 - completed_steps / total_steps) ** power

def select_device(value):
    if value == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(value)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Supported devices are cpu and cuda')
    if device.type == 'cuda' and (not torch.cuda.is_available()):
        raise ValueError('CUDA requested but is not available')
    return device

def environment_info(device):
    return {'implementation': IMPLEMENTATION_VERSION, 'python': platform.python_version(), 'torch': str(torch.__version__), 'numpy': np.__version__, 'platform': platform.platform(), 'cuda_runtime': torch.version.cuda, 'device': str(device), 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(), 'determinism': 'seeded, cuDNN deterministic; cross-device bitwise equivalence not guaranteed'}
