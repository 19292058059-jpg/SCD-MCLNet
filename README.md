# SCD-MCLNet

Reference-based PyTorch implementation for SECOND. Only the final model is included.
No datasets, trained weights or experiment results are included.

Python 3.10 or later. Install dependencies with `pip install -r requirements.txt`.

Place SECOND at `data/SECOND` with `A`, `B`, `label1`, `label2`, and
`list/train.txt`, `list/test.txt`. Paths may be overridden with `--datapath`.

```bash
python prepare_validation_split.py --datapath data/SECOND
python train_SCD.py --config configs/second.yaml --run-dir runs/full
python test_SCD.py --checkpoint runs/full/best.pth --output-dir runs/full/test
```

The configuration uses 1900 training, 475 validation and 593 test pairs.
Train for 50 epochs and evaluate the checkpoint selected by validation SeK.
Third-party terms are in `THIRD_PARTY_NOTICES.md`.
