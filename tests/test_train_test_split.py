import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from datasets.SECOND import load_splits
from utils.config import default_config, load_config


def make_fixture(root, train_count, test_count, images=False):
    root = Path(root)
    for folder in ('A', 'B', 'label1', 'label2', 'list'):
        (root / folder).mkdir(parents=True)
    training = ['train{:04d}.png'.format(i) for i in range(train_count)]
    testing = ['test{:04d}.png'.format(i) for i in range(test_count)]
    held_out = 475 if train_count == 2375 else train_count // 2
    lists = {
        'train': training,
        'test': testing,
        'train_fit_seed42': training[:-held_out],
        'val_seed42': training[-held_out:],
    }
    for split, names in lists.items():
        (root / 'list' / (split + '.txt')).write_text('\n'.join(names) + '\n', encoding='utf-8')
    rng = np.random.default_rng(17)
    for index, name in enumerate(training + testing):
        for folder in ('A', 'B', 'label1', 'label2'):
            path = root / folder / name
            if not images:
                path.touch()
            elif folder in ('A', 'B'):
                # Keep the deepest feature map larger than 1x1 for BatchNorm.
                Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(path)
            else:
                labels = np.zeros((64, 64), dtype=np.uint8)
                labels[4:20, 5:19] = 1 + index % 2
                Image.fromarray(labels).save(path)
    return root


def resolve_splits(config, root):
    data = config['data']
    return load_splits(root, train=data['train_split'], test=data['test_split'], val=data['val_split'])


class FullTrainingSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.root = make_fixture(Path(cls.folder.name) / 'SECOND', 2375, 593)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def check_full_training(self, config):
        splits = resolve_splits(config, self.root)
        self.assertEqual(len(splits['train']), 2375)
        self.assertEqual(len(splits['test']), 593)
        self.assertNotIn('val', splits)
        self.assertFalse(set(splits['train']) & set(splits['test']))

    def test_python_defaults_load_every_training_pair(self):
        self.check_full_training(default_config())

    def test_shipped_yaml_cli_loads_every_training_pair(self):
        from train_SCD import main

        config_path = Path(__file__).resolve().parents[1] / 'configs' / 'second.yaml'
        with contextlib.redirect_stdout(io.StringIO()):
            config = main(['--config', str(config_path), '--datapath', str(self.root), '--dry-run'])
        self.check_full_training(config)


class NoValidationTrainingTests(unittest.TestCase):
    def test_train_last_checkpoint_then_independent_test(self):
        from utils.engine import train_run, test_run
        from utils.experiment import load_checkpoint

        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder:
            root = make_fixture(Path(folder) / 'SECOND', 4, 2, images=True)
            config = load_config(overrides={
                'data': {'root': str(root)},
                'runtime': {'purpose': 'synthetic_smoke'},
                'model': {
                    'encoder_channels': [8, 8, 16, 24, 32],
                    'encoder_repeats': [1, 1, 1, 1],
                    'scan_dim': 16, 'scan_depth': 1,
                    'scan_heads': 4, 'scan_stripe': 2,
                },
                'train': {'epochs': 2, 'batch_size': 2, 'eval_batch_size': 2, 'device': 'cpu', 'workers': 0},
            })
            run = Path(folder) / 'run'
            result = train_run(config, run_dir=run)
            state = load_checkpoint(run / 'last.pth')
            saved_splits = json.loads((run / 'splits.json').read_text(encoding='utf-8'))
            entries = [json.loads(row) for row in (run / 'train.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(saved_splits['train']), 4)
            self.assertNotIn('val', saved_splits)
            self.assertEqual(result['global_step'], 4)
            self.assertEqual(state['selection'], 'last_epoch_no_validation')
            self.assertIsNone(state['best_validation_SeK'])
            self.assertTrue(all(entry['validation'] is None for entry in entries))
            self.assertFalse((run / 'best.pth').exists())
            output = Path(folder) / 'test'
            metrics = test_run(run / 'last.pth', output_dir=output, device='cpu')
            self.assertEqual(metrics['samples'], 2)
            self.assertEqual(metrics['split'], 'test')
            self.assertTrue(all(np.isfinite(metrics[key]) for key in ('OA', 'Fscd', 'mIoU', 'SeK')))


if __name__ == '__main__':
    unittest.main()
