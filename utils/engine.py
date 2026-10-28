import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from utils.config import validate_config
from utils.experiment import PROJECT_ROOT, capture_rng, environment_info, load_checkpoint, polynomial_lr, restore_rng, resume_signature, save_checkpoint, seed_everything, seed_worker, select_device, source_digest, split_identity, validate_resume, write_json
from utils.losses import MultiTaskLoss

def _stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')

def _to_device(batch, device):
    return {key: value.to(device, non_blocking=device.type == 'cuda') if torch.is_tensor(value) else value for (key, value) in batch.items()}

def _loader(dataset, batch_size, workers, shuffle, generator=None, device=None):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, drop_last=False, pin_memory=device is not None and device.type == 'cuda', worker_init_fn=seed_worker, generator=generator, persistent_workers=False)

def _save_predictions(folder, names, first, second, change):
    for (key, values) in (('sem1', first), ('sem2', second), ('change', change)):
        destination = Path(folder) / key
        destination.mkdir(parents=True, exist_ok=True)
        for (name, value) in zip(names, values.detach().cpu().numpy()):
            Image.fromarray(value.astype(np.uint8)).save(destination / Path(name).with_suffix('.png').name)

@torch.no_grad()
def evaluate(model, loader, device, predictions_dir=None):
    from utils.metrics import SCDMetrics
    model.eval()
    metric = SCDMetrics(model.num_classes)
    for batch in loader:
        moved = _to_device(batch, device)
        outputs = model(moved['image1'], moved['image2'], return_aux=False)
        change = outputs['change'].argmax(1)
        first = outputs['sem1'].argmax(1) * change
        second = outputs['sem2'].argmax(1) * change
        metric.update(torch.cat((first, second), dim=0), torch.cat((moved['label1'], moved['label2']), dim=0))
        if predictions_dir is not None:
            _save_predictions(predictions_dir, batch['name'], first, second, change)
    return (metric.compute(), metric.matrix.copy())

def _train_epoch(model, loader, optimizer, scaler, criterion, config, device, global_step, total_steps):
    model.train()
    totals = {key: 0.0 for key in ('total', 'sem', 'cd', 'boundary', 'sim', 'logit')}
    samples = 0
    optimizer_updates = 0
    train = config['train']
    for (index, batch) in enumerate(loader):
        lr = polynomial_lr(train['lr'], global_step + train['poly_step_origin'], total_steps, train['poly_power'])
        for group in optimizer.param_groups:
            group['lr'] = lr
        moved = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=train['amp']):
            outputs = model(moved['image1'], moved['image2'], return_aux=True)
            losses = criterion(outputs, moved)
        if not torch.stack([torch.isfinite(value) for value in losses.values()]).all().item():
            raise FloatingPointError('Non-finite training loss; run stopped without changing batch size or hiding the failure')
        scaler.scale(losses['total']).backward()
        scaler.unscale_(optimizer)
        if not scaler.is_enabled():
            finite = [torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None]
            if finite and (not torch.stack(finite).all().item()):
                raise FloatingPointError('Non-finite gradient; inspect input/configuration before resuming')
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped = scaler.is_enabled() and scaler.get_scale() < old_scale
        if skipped:
            print(f'  AMP scaling overflow: optimizer update skipped; scale {old_scale:g} -> {scaler.get_scale():g}', flush=True)
        else:
            optimizer_updates += 1
        global_step += 1
        count = batch['image1'].shape[0]
        samples += count
        for (key, value) in losses.items():
            totals[key] += float(value.detach()) * count
        if index % train['log_every'] == 0 or index + 1 == len(loader):
            print(f"  step {global_step}/{total_steps} lr={lr:.7g} loss={float(losses['total'].detach()):.6f}", flush=True)
    for group in optimizer.param_groups:
        group['lr'] = polynomial_lr(train['lr'], global_step, total_steps, train['poly_power'])
    return ({key: value / samples for (key, value) in totals.items()}, global_step, {'optimizer_updates': optimizer_updates, 'amp_skipped_steps': len(loader) - optimizer_updates})

def train_run(config, run_dir=None, resume=None, init_checkpoint=None):
    from datasets import SECOND, compute_normalization, load_splits
    from models import build_model
    config = copy.deepcopy(validate_config(config))
    if resume is not None and init_checkpoint is not None:
        raise ValueError('Choose resume OR init_checkpoint, not both')
    train = config['train']
    device = select_device(train['device'])
    if train['amp'] and device.type != 'cuda':
        raise ValueError('AMP is supported only with CUDA in this implementation')
    seed_everything(train['seed'])
    root = Path(config['data']['root'])
    splits = load_splits(root, train=config['data']['train_split'], test=config['data']['test_split'], val=config['data']['val_split'])
    identity = split_identity(root, splits)
    code = source_digest()
    previous = load_checkpoint(resume) if resume is not None else None
    if previous is not None:
        normalization = previous['normalization']
    else:
        print(f"Computing train-only normalization from {len(splits['train'])} image pairs...", flush=True)
        normalization = compute_normalization(root, splits['train'])
        if identity != split_identity(root, splits):
            raise ValueError('Dataset files changed while computing normalization')
    signature = resume_signature(config, identity, normalization, code)
    if previous is not None:
        validate_resume(previous['signature'], signature)
        expected_dir = Path(resume).resolve().parent
        if run_dir is not None and Path(run_dir).resolve() != expected_dir:
            raise ValueError('Resume writes to the original reconstruction run; do not specify a different run_dir')
        run_dir = expected_dir
    else:
        if run_dir is None:
            output_root = Path(config['runtime']['output_root'])
            if not output_root.is_absolute():
                output_root = PROJECT_ROOT / output_root
            run_dir = output_root / config['variant'] / f"seed_{train['seed']}" / _stamp()
        run_dir = Path(run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
    run_dir = Path(run_dir)
    generator = torch.Generator().manual_seed(train['seed'])
    training = SECOND(root, splits['train'], normalization=normalization, augment=True, augmentation_policy=config['data']['augmentation_policy'])
    loader = _loader(training, train['batch_size'], train['workers'], True, generator, device)
    validation_loader = None
    if 'val' in splits:
        validation = SECOND(root, splits['val'], normalization=normalization, augment=False)
        validation_loader = _loader(validation, train['eval_batch_size'], train['workers'], False, device=device)
    model = build_model(config['model'], num_classes=config['num_classes'], variant=config['variant']).to(device)
    initialization = {'kind': 'random', 'source': None}
    if init_checkpoint is not None:
        initial = load_checkpoint(init_checkpoint)
        for key in ('model', 'variant', 'num_classes'):
            if initial['config'][key] != config[key]:
                raise ValueError(f'Initialization checkpoint has incompatible {key}')
        model.load_state_dict(initial['model'], strict=True)
        initialization = {'kind': 'explicit_compatible_checkpoint', 'source': str(Path(init_checkpoint).resolve())}
    optimizer = torch.optim.SGD(model.parameters(), lr=train['lr'], momentum=train['momentum'], weight_decay=train['weight_decay'], nesterov=train['nesterov'])
    scaler = torch.amp.GradScaler('cuda', enabled=train['amp'])
    criterion = MultiTaskLoss(train['alpha'])
    total_steps = len(loader) * train['epochs']
    (start_epoch, global_step, optimizer_updates, best) = (0, 0, 0, None)
    if previous is not None:
        model.load_state_dict(previous['model'], strict=True)
        optimizer.load_state_dict(previous['optimizer'])
        scaler.load_state_dict(previous['scaler'])
        (start_epoch, global_step) = (previous['completed_epochs'], previous['global_step'])
        optimizer_updates = previous.get('optimizer_updates', global_step)
        if previous['schedule']['total_steps'] != total_steps or global_step != start_epoch * len(loader):
            raise ValueError('Checkpoint progress is incompatible with this loader/schedule')
        generator.set_state(previous['loader_generator_state'])
        best = previous['best_validation_SeK']
        initialization = previous['initialization']
        restore_rng(previous['rng'])
    else:
        write_json(run_dir / 'config.json', config)
        write_json(run_dir / 'normalization.json', normalization)
        write_json(run_dir / 'splits.json', splits)
        write_json(run_dir / 'provenance.json', {'signature': signature, 'environment': environment_info(device), 'initialization': initialization, 'data_hash_scope': 'ordered split names and file sizes/mtime, not full image content'})
    selection = 'best_validation_SeK' if validation_loader is not None else 'last_epoch_no_validation'
    write_json(run_dir / 'status.json', {'status': 'running', 'completed_epochs': start_epoch, 'global_step': global_step})
    print(f"Run: {run_dir}\nVariant: {config['variant']}; device: {device}; selection: {selection}", flush=True)
    completed = start_epoch
    try:
        for epoch in range(start_epoch, train['epochs']):
            started = time.perf_counter()
            print(f"Epoch {epoch + 1}/{train['epochs']}", flush=True)
            (losses, global_step, step_counts) = _train_epoch(model, loader, optimizer, scaler, criterion, config, device, global_step, total_steps)
            optimizer_updates += step_counts['optimizer_updates']
            scores = None
            improved = False
            if validation_loader is not None:
                (scores, _) = evaluate(model, validation_loader, device)
                improved = best is None or scores['SeK'] > best
                if improved:
                    best = scores['SeK']
            completed = epoch + 1
            entry = {'epoch': completed, 'global_step': global_step, 'losses': losses, 'step_counts': step_counts, 'validation': scores, 'seconds': time.perf_counter() - started}
            checkpoint = {'format_version': 1, 'config': config, 'signature': signature, 'normalization': normalization, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(), 'schedule': {'kind': 'polynomial', 'total_steps': total_steps, 'unit': 'minibatches_including_amp_skips', 'completed_steps': global_step, 'power': train['poly_power']}, 'completed_epochs': completed, 'global_step': global_step, 'optimizer_updates': optimizer_updates, 'amp_skipped_steps': global_step - optimizer_updates, 'best_validation_SeK': best, 'selection': selection, 'rng': capture_rng(), 'loader_generator_state': generator.get_state(), 'initialization': initialization, 'last_epoch': entry}
            save_checkpoint(run_dir / 'last.pth', checkpoint)
            if improved:
                save_checkpoint(run_dir / 'best.pth', checkpoint)
            with (run_dir / 'train.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(entry, allow_nan=False) + '\n')
            write_json(run_dir / 'status.json', {'status': 'running', 'completed_epochs': completed, 'global_step': global_step})
    except BaseException as error:
        write_json(run_dir / 'status.json', {'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', 'completed_epochs': completed, 'global_step': global_step, 'error': str(error), 'resume_scope': 'last completed epoch only'})
        raise
    summary = {'status': 'completed', 'run_dir': str(run_dir), 'variant': config['variant'], 'purpose': config['runtime']['purpose'], 'seed': train['seed'], 'completed_epochs': completed, 'global_step': global_step, 'optimizer_updates': optimizer_updates, 'amp_skipped_steps': global_step - optimizer_updates, 'best_validation_SeK': best, 'selection': selection, 'last_checkpoint': str(run_dir / 'last.pth')}
    write_json(run_dir / 'status.json', summary)
    return summary

def test_run(checkpoint_path, output_dir=None, dataset_root=None, split='test', device='auto', batch_size=None, workers=None, save_predictions=False):
    from datasets import SECOND, read_split
    from models import build_model
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    config = validate_config(checkpoint['config'])
    if checkpoint['signature']['source_digest'] != source_digest():
        raise ValueError('Source code changed since this checkpoint; evaluate with its matching implementation')
    if split not in ('test', 'val'):
        raise ValueError('Independent evaluation supports test or explicitly provided val, not training data')
    split_name = config['data'][split + '_split']
    if split_name is None:
        raise ValueError('No independent validation split was supplied for this run')
    root = Path(dataset_root or config['data']['root'])
    names = read_split(root, split_name)
    dataset = SECOND(root, names, normalization=checkpoint['normalization'], augment=False)
    identity = split_identity(root, {split: names})
    if identity[split] != checkpoint['signature']['splits'][split]:
        raise ValueError('Evaluation split or dataset file metadata differs from training provenance')
    if batch_size is not None and (type(batch_size) is not int or batch_size <= 0):
        raise ValueError('batch_size must be a positive integer')
    if workers is not None and (type(workers) is not int or workers < 0):
        raise ValueError('workers must be a nonnegative integer')
    device = select_device(device)
    seed_everything(config['train']['seed'])
    model = build_model(config['model'], num_classes=config['num_classes'], variant=config['variant']).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    loader = _loader(dataset, batch_size or config['train']['eval_batch_size'], config['train']['workers'] if workers is None else workers, False, device=device)
    if output_dir is None:
        output_dir = checkpoint_path.parent / ('evaluation_' + split + '_' + _stamp())
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    (scores, matrix) = evaluate(model, loader, device, output_dir / 'predictions' if save_predictions else None)
    result = {**scores, 'status': 'measured', 'variant': config['variant'], 'seed': config['train']['seed'], 'purpose': config['runtime']['purpose'], 'checkpoint': str(checkpoint_path), 'completed_epochs': checkpoint['completed_epochs'], 'split': split, 'samples': len(dataset), 'metric_units': 'fractions, not percentages', 'matrix_orientation': 'truth rows, prediction columns; both dates pooled after change masking', 'environment': environment_info(device)}
    write_json(output_dir / 'metrics.json', result)
    np.save(output_dir / 'confusion_matrix.npy', matrix)
    print(json.dumps(result, indent=2), flush=True)
    return result
