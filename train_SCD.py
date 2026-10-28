import argparse
import json
from pathlib import Path
from utils.config import load_config

def main(argv=None):
    parser = argparse.ArgumentParser(description='Train SCD-MCLNet on SECOND')
    parser.add_argument('--config', default=str(Path(__file__).parent / 'configs' / 'second.yaml'))
    parser.add_argument('--variant', choices=('full',))
    parser.add_argument('--datapath')
    parser.add_argument('--epochs', '--epoch', dest='epochs', type=int)
    parser.add_argument('--batch-size', '--train_batchsize', dest='batch_size', type=int)
    parser.add_argument('--eval-batch-size', dest='eval_batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--workers', '--num_workers', dest='workers', type=int)
    parser.add_argument('--device')
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--val-split', help='Explicit independent validation list name; never the test list')
    parser.add_argument('--run-dir', help='New run directory; existing runs are never overwritten')
    state = parser.add_mutually_exclusive_group()
    state.add_argument('--resume', help='Resume a compatible reconstruction last.pth at an epoch boundary')
    state.add_argument('--init-checkpoint', help='Explicit compatible reconstruction weights; starts a new optimizer')
    parser.add_argument('--dry-run', action='store_true', help='Print configuration without reading data or creating runs')
    args = parser.parse_args(argv)
    overrides = {}
    if args.variant is not None:
        overrides['variant'] = args.variant
    data = {}
    if args.datapath is not None:
        data['root'] = args.datapath
    if args.val_split is not None:
        data['val_split'] = args.val_split
    if data:
        overrides['data'] = data
    train = {key: getattr(args, key) for key in ('epochs', 'batch_size', 'eval_batch_size', 'lr', 'seed', 'workers', 'device', 'alpha', 'amp') if getattr(args, key) is not None}
    if train:
        overrides['train'] = train
    config = load_config(args.config, overrides)
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return config
    from utils.engine import train_run
    result = train_run(config, run_dir=args.run_dir, resume=args.resume, init_checkpoint=args.init_checkpoint)
    print(json.dumps(result, indent=2))
    return result
if __name__ == '__main__':
    main()
