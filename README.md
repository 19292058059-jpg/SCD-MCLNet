# SCD-MCLNet

Reference-based PyTorch implementation for SECOND. Only the final model is included.
No datasets, trained weights or experiment results are included.

Python 3.10 or later. Install dependencies with `pip install -r requirements.txt`.

Place SECOND at `data/SECOND` with `A`, `B`, `label1`, `label2`, and
`list/train.txt`, `list/test.txt`. Paths may be overridden with `--datapath`.

```bash
python train_SCD.py --config configs/second.yaml --run-dir runs/full
python test_SCD.py --checkpoint runs/full/last.pth --output-dir runs/full/test
```

The default SECOND protocol uses all **2375 training pairs** listed in
`list/train.txt` and **593 independent test pairs** listed in `list/test.txt`,
matching the train/test split stated in the manuscript. No validation subset
is held out from the training pairs (`val_split: null`). The dataset loader
checks that the training and test lists do not overlap.

Train for 50 epochs, then evaluate the final checkpoint `last.pth` on the test
set. The default workflow does not use test scores to select checkpoints or
create a validation-selected `best.pth`. Start a new run when changing from a
previous training/validation partition; those checkpoints belong to a different
training protocol.

`prepare_validation_split.py` is an optional utility and is not part of this
2375-training/593-test workflow. Do not run it for the default protocol.
This code update does not establish that the numerical results reported in
the manuscript have been reproduced; performance must be measured by running
the training and independent evaluation.

To run the regression checks (including a small CPU training/evaluation test):

```bash
python -m unittest discover -s tests -v
```

<details>
<summary>Third-party licenses and attribution</summary>

# Third-Party Notices

## BT-SCD

Source: https://github.com/TangYJ1229/BT-SCD

Authors: Yingjie Tang, Shou Feng, Chunhui Zhao, Yongqi Chen, Zhiyong Lv and
Weiwei Sun. Paper: https://doi.org/10.1109/TNNLS.2025.3570425

The task interaction, bidirectional temporal transfer, aggregate Sobel boundary
processing and auxiliary objectives were adapted for this implementation.
Changes include the shared lightweight encoder/decoders, channel adapters,
dynamic resizing and finite-gradient boundary magnitude.

The source README licenses BT-SCD under Creative Commons
Attribution-NonCommercial 4.0 International for non-commercial use only:
https://creativecommons.org/licenses/by-nc/4.0/
Legal code and warranty disclaimer:
https://creativecommons.org/licenses/by-nc/4.0/legalcode.en
Commercial use requires permission from the original rights holders.
No source-author endorsement is implied.

## CSWin Transformer

Source: https://github.com/microsoft/CSWin-Transformer
Original implementation written by Xiaoyi Dong.

`models/cswin.py` adapts cross-shaped stripe attention, LePE and the residual
attention/MLP structure. Changes include dynamic rectangular NCHW features,
masked stripe padding and native PyTorch operations without timm/einops.

MIT License

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

These notices do not grant a blanket MIT license to the complete project.

</details>

