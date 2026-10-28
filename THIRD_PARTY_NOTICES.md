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
