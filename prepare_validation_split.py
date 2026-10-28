import argparse
import hashlib
import json
from pathlib import Path
from datasets.SECOND import _canonical_filename, _canonical_names, load_splits

def partition_training_names(names, val_count, seed=42):
    names = _canonical_names(names)
    if type(val_count) is not int or not 0 < val_count < len(names):
        raise ValueError('val_count must leave nonempty fit and validation sets')
    if type(seed) is not int or not 0 <= seed < 2 ** 32:
        raise ValueError('seed must be an integer in [0,2**32)')

    def rank(name):
        key = '{}:{}'.format(seed, name.casefold()).encode('utf-8')
        return (hashlib.sha256(key).digest(), name.casefold())
    held_out = set(sorted(names, key=rank)[:val_count])
    return ([name for name in names if name not in held_out], [name for name in names if name in held_out])

def prepare_validation_split(root, fit_split, val_split, val_count=475, seed=42, source_split='train', test_split='test'):
    root = Path(root).resolve()
    source_file = _canonical_filename(source_split, '.txt')
    test_file = _canonical_filename(test_split, '.txt')
    fit_file = _canonical_filename(fit_split, '.txt')
    val_file = _canonical_filename(val_split, '.txt')
    protected = {source_file.casefold(), test_file.casefold()}
    if fit_file.casefold() in protected or val_file.casefold() in protected or fit_file.casefold() == val_file.casefold():
        raise ValueError('Fit/validation names must be distinct new split names')
    splits = load_splits(root, train=source_split, test=test_split)
    (fit, val) = partition_training_names(splits['train'], val_count, seed)
    folder = root / 'list'
    if not folder.resolve().is_relative_to(root):
        raise ValueError('Split directory escapes the dataset root')
    manifest_file = Path(fit_file).stem + '_partition.json'
    paths = [folder / name for name in (fit_file, val_file, manifest_file)]
    if any((not p.resolve().is_relative_to(folder.resolve()) for p in paths)):
        raise ValueError('Output split path escapes list directory')
    report = {'format_version': 1, 'method': 'sha256_seed_casefolded_basename_rank_v1', 'seed': seed, 'counts': {'source_train': len(splits['train']), 'fit': len(fit), 'val': len(val), 'test': len(splits['test'])}, 'source_split': source_file, 'test_split': test_file, 'fit_split': fit_file, 'val_split': val_file, 'source_list_sha256': hashlib.sha256((folder / source_file).read_bytes()).hexdigest(), 'test_list_sha256': hashlib.sha256((folder / test_file).read_bytes()).hexdigest(), 'fit_names': fit, 'val_names': val, 'normalization_scope': 'fit_names_only', 'historical_paper_partition_claimed': False}
    payloads = ['\n'.join(fit) + '\n', '\n'.join(val) + '\n', json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2) + '\n']
    if any((path.exists() for path in paths)):
        if not all((path.is_file() and path.read_bytes() == text.encode('utf-8') for (path, text) in zip(paths, payloads))):
            raise FileExistsError('Partition output conflict; no existing files were changed')
        return {**report, 'created': False, 'manifest': str(paths[-1])}
    created = []
    try:
        for (path, text) in zip(paths, payloads):
            with path.open('x', encoding='utf-8', newline='\n') as stream:
                created.append(path)
                stream.write(text)
    except BaseException:
        for path in reversed(created):
            path.unlink()
        raise
    return {**report, 'created': True, 'manifest': str(paths[-1])}

def main(argv=None):
    parser = argparse.ArgumentParser(description='Hold validation out of training IDs; never change test')
    parser.add_argument('--datapath', required=True)
    parser.add_argument('--fit-name', default='train_fit_seed42')
    parser.add_argument('--val-name', default='val_seed42')
    parser.add_argument('--val-count', type=int, default=475)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    result = prepare_validation_split(args.datapath, args.fit_name, args.val_name, args.val_count, args.seed)
    print(json.dumps({key: result[key] for key in ('counts', 'seed', 'method', 'created', 'manifest')}, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
