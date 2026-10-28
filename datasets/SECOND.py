from collections.abc import Mapping
from pathlib import Path, PureWindowsPath
import random
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
PALETTE = np.array([[255, 255, 255], [0, 128, 0], [128, 128, 128], [0, 255, 0], [0, 0, 255], [128, 0, 0], [255, 0, 0]], dtype=np.uint8)
STD_FLOOR = 1e-06
_FOLDERS = ('A', 'B', 'label1', 'label2')
_RESERVED = {'CON', 'PRN', 'AUX', 'NUL', 'CLOCK$'} | {prefix + str(number) for prefix in ('COM', 'LPT') for number in range(1, 10)}

def _canonical_filename(value, extension):
    if not isinstance(value, str):
        raise ValueError('Names must be strings')
    value = value.strip()
    if not value or value in ('.', '..') or value.endswith('.') or any((character in value for character in '/\\:<>"|?*')) or any((ord(character) < 32 for character in value)):
        raise ValueError('Unsafe filename: {!r}'.format(value))
    suffix = PureWindowsPath(value).suffix
    if suffix and suffix.lower() != extension:
        raise ValueError('Expected {} filename: {!r}'.format(extension, value))
    stem = value[:-len(suffix)] if suffix else value
    if not stem or stem.endswith(('.', ' ')) or stem.split('.')[0].upper() in _RESERVED:
        raise ValueError('Unsafe filename: {!r}'.format(value))
    return stem + extension

def _canonical_names(names):
    if isinstance(names, (str, bytes)):
        raise ValueError('Expected a sequence of sample names, not one string')
    (result, seen) = ([], set())
    for name in names:
        canonical = _canonical_filename(name, '.png')
        key = canonical.casefold()
        if key in seen:
            raise ValueError('Duplicate sample name: {}'.format(canonical))
        seen.add(key)
        result.append(canonical)
    if not result:
        raise ValueError('Sample list must not be empty')
    return result

def _checked_path(root, folder, name):
    path = root / folder / name
    if not path.resolve().is_relative_to(root):
        raise ValueError('Path escapes the dataset root: {}'.format(path))
    if not path.is_file():
        raise FileNotFoundError('Missing SECOND file: {}'.format(path))
    return path

def _validate_pairs(root, names):
    for name in names:
        for folder in _FOLDERS:
            _checked_path(root, folder, name)

def read_split(root, split):
    root = Path(root).resolve()
    filename = _canonical_filename(split, '.txt')
    path = _checked_path(root, 'list', filename)
    with path.open('r', encoding='utf-8-sig') as stream:
        names = [line.strip() for line in stream if line.strip()]
    return _canonical_names(names)

def load_splits(root, train='train', test='test', val=None):
    root = Path(root).resolve()
    requested = {'train': train, 'test': test}
    if val is not None:
        requested['val'] = val
    (splits, owner) = ({}, {})
    for (role, split) in requested.items():
        names = read_split(root, split)
        for name in names:
            key = name.casefold()
            if key in owner:
                raise ValueError('Split overlap between {} and {}: {}'.format(owner[key], role, name))
            owner[key] = role
        _validate_pairs(root, names)
        splits[role] = names
    return splits

def _load_rgb(path):
    with Image.open(path) as image:
        if image.mode != 'RGB':
            raise ValueError('Expected RGB input image: {} (mode {})'.format(path, image.mode))
        return np.array(image, dtype=np.uint8)

def _decode_colors(colors, path):
    labels = np.full(colors.shape[:2], -1, dtype=np.int64)
    for (class_id, color) in enumerate(PALETTE):
        labels[np.all(colors == color, axis=2)] = class_id
    unknown = labels < 0
    if np.any(unknown):
        examples = np.unique(colors[unknown], axis=0)[:8].tolist()
        raise ValueError('Unknown label color(s) {} in {}'.format(examples, path))
    return labels

def _load_label(path):
    with Image.open(path) as image:
        if image.mode in ('RGB', 'P'):
            return _decode_colors(np.array(image.convert('RGB'), dtype=np.uint8), path)
        labels = np.array(image)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('Expected RGB, paletted, or integer class label image: {}'.format(path))
    if np.any((labels < 0) | (labels >= len(PALETTE))):
        raise ValueError('Unknown class ID(s) {} in {}'.format(np.unique(labels).tolist(), path))
    return labels.astype(np.int64)

def _normalization_vectors(normalization):
    if normalization is None:
        return None
    if not isinstance(normalization, Mapping):
        raise ValueError('Normalization must contain A and B mean/std mappings')
    result = {}
    for date in ('A', 'B'):
        statistics = normalization.get(date)
        if not isinstance(statistics, Mapping):
            raise ValueError('Missing normalization mapping for {}'.format(date))
        result[date] = {}
        for key in ('mean', 'std'):
            try:
                raw = np.asarray(statistics[key])
                if raw.dtype.kind not in 'fiu':
                    raise ValueError('Expected numeric vectors')
                vector = raw.astype(np.float64)
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                raise ValueError('Invalid {} {} normalization vector'.format(date, key)) from error
            if vector.shape != (3,) or not np.isfinite(vector).all() or (key == 'std' and np.any(vector <= 0)):
                raise ValueError('{} {} must be a finite RGB 3-vector; std must be > 0'.format(date, key))
            result[date][key] = vector
    return result

class SECOND(Dataset):

    def __init__(self, root, names, normalization=None, augment=False, augmentation_policy='independent_flips'):
        if not isinstance(augmentation_policy, str) or augmentation_policy not in ('independent_flips', 'bt_reference'):
            raise ValueError('augmentation_policy must be independent_flips or bt_reference')
        self.root = Path(root).resolve()
        self.names = _canonical_names(names)
        self.normalization = _normalization_vectors(normalization)
        self.augment = augment
        self.augmentation_policy = augmentation_policy
        _validate_pairs(self.root, self.names)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        image1 = _load_rgb(_checked_path(self.root, 'A', name))
        image2 = _load_rgb(_checked_path(self.root, 'B', name))
        label1 = _load_label(_checked_path(self.root, 'label1', name))
        label2 = _load_label(_checked_path(self.root, 'label2', name))
        shape = image1.shape[:2]
        if image2.shape[:2] != shape or label1.shape != shape or label2.shape != shape:
            raise ValueError('SECOND image/label size mismatch for {}: {}, {}, {}, {}'.format(name, image1.shape, image2.shape, label1.shape, label2.shape))
        arrays = [image1, image2, label1, label2]
        if self.augment and self.augmentation_policy == 'bt_reference':
            turns = 0 if random.random() < 0.5 else 1
            arrays = [np.rot90(array, turns, axes=(0, 1)) for array in arrays]
            flip = random.random()
            if flip >= 0.25:
                axes = (0,) if flip < 0.5 else (1,) if flip < 0.75 else (0, 1)
                arrays = [np.flip(array, axis=axes) for array in arrays]
        elif self.augment:
            turns = random.randrange(4)
            arrays = [np.rot90(array, turns, axes=(0, 1)) for array in arrays]
            for axis in (0, 1):
                if random.random() < 0.5:
                    arrays = [np.flip(array, axis=axis) for array in arrays]
        (image1, image2, label1, label2) = arrays
        first = np.ascontiguousarray(label1, dtype=np.uint8)
        second = np.ascontiguousarray(label2, dtype=np.uint8)
        boundary_sem = (cv2.Canny(first, 0, 0) | cv2.Canny(second, 0, 0)) // 255
        boundary_change = cv2.Canny((first > 0).astype(np.uint8), 0, 0) // 255
        result = {'name': name}
        for (key, image, date) in (('image1', image1, 'A'), ('image2', image2, 'B')):
            image = image.astype(np.float64) / 255.0
            if self.normalization is not None:
                stats = self.normalization[date]
                image = (image - stats['mean']) / stats['std']
            result[key] = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32))
        for (key, array, dtype) in (('label1', label1, np.int64), ('label2', label2, np.int64), ('boundary_sem', boundary_sem, np.float32), ('boundary_change', boundary_change, np.float32)):
            result[key] = torch.from_numpy(np.ascontiguousarray(array, dtype=dtype))
        return result

def compute_normalization(root, train_names):
    root = Path(root).resolve()
    names = _canonical_names(train_names)
    result = {}
    for date in ('A', 'B'):
        count = 0
        (mean, moment) = (np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64))
        for name in names:
            pixels = _load_rgb(_checked_path(root, date, name)).reshape(-1, 3).astype(np.float64) / 255.0
            batch_count = pixels.shape[0]
            batch_mean = pixels.mean(axis=0)
            batch_moment = np.square(pixels - batch_mean).sum(axis=0)
            total = count + batch_count
            difference = batch_mean - mean
            moment += batch_moment + np.square(difference) * (count * batch_count / total)
            mean += difference * (batch_count / total)
            count = total
        std = np.maximum(np.sqrt(np.maximum(moment / count, 0.0)), STD_FLOOR)
        result[date] = {'mean': mean.tolist(), 'std': std.tolist()}
    return result
