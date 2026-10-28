import torch
from torch import nn
import torch.nn.functional as F
from .layers import ConvNormAct, resize_like
from .checkpointing import run_block
DECODER_CHANNELS = (256, 128, 64, 32, 16)

class CheckedSemanticDecoder(nn.Module):

    def __init__(self, encoder_channels):
        super().__init__()
        self.checkpoint_blocks = False
        self.blocks = nn.ModuleList()
        incoming = encoder_channels[-1]
        for (skip, width) in zip(reversed(encoder_channels[:-1]), DECODER_CHANNELS[:-1]):
            self.blocks.append(nn.Sequential(ConvNormAct(incoming + skip, width), ConvNormAct(width, width)))
            incoming = width
        self.blocks.append(nn.Sequential(ConvNormAct(incoming, DECODER_CHANNELS[-1]), ConvNormAct(DECODER_CHANNELS[-1], DECODER_CHANNELS[-1])))

    def forward(self, features, output_size):
        x = features[-1]
        for (block, skip) in zip(self.blocks[:-1], reversed(features[:-1])):
            x = run_block(block, resize_like(x, skip), skip, enabled=self.checkpoint_blocks, concatenate=True)
        x = F.interpolate(x, size=output_size, mode='bilinear', align_corners=False)
        return run_block(self.blocks[-1], x, enabled=self.checkpoint_blocks)
