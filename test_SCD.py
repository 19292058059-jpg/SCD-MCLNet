import argparse

def main(argv=None):
    parser = argparse.ArgumentParser(description='Independent SECOND evaluation; does not train or select checkpoints')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--datapath', help='Optional relocated SECOND root; manifest/file metadata must still match')
    parser.add_argument('--split', choices=('test', 'val'), default='test')
    parser.add_argument('--output-dir', help='New directory for actual metrics and optional predictions')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--save-predictions', action='store_true')
    args = parser.parse_args(argv)
    from utils.engine import test_run
    return test_run(args.checkpoint, output_dir=args.output_dir, dataset_root=args.datapath, split=args.split, device=args.device, batch_size=args.batch_size, workers=args.workers, save_predictions=args.save_predictions)
if __name__ == '__main__':
    main()
