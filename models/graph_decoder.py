import torch
from torch import nn
import torch.nn.functional as F
from .checkpointing import run_block
from .layers import ConvNormAct, ResidualChannelAttention, positive_int, resize_like
NODE_GROUPS = (('A',), ('B0', 'B1'), ('C0', 'C1', 'C2'), ('D0', 'D1', 'D2', 'D3'), ('E',))
GRAPH_PARENTS = (('A', ()), ('B0', ('A',)), ('B1', ('A', 'B0')), ('C0', ('B0', 'B1')), ('C1', ('B0', 'B1', 'C0')), ('C2', ('B1', 'C1')), ('D0', ('C0', 'C1')), ('D1', ('C0', 'C1', 'C2', 'D0')), ('D2', ('C1', 'C2', 'D1')), ('D3', ('C2', 'D2')), ('E', ('D0', 'D1', 'D2', 'D3')))

class FigureDenseChangeDecoder(nn.Module):

    def __init__(self, encoder_channels):
        super().__init__()
        if not isinstance(encoder_channels, (list, tuple)) or len(encoder_channels) != 5:
            raise ValueError('encoder_channels must contain five widths')
        self.encoder_channels = tuple((positive_int(value, 'encoder channel') for value in encoder_channels))
        self.checkpoint_blocks = False
        self.parents = dict(GRAPH_PARENTS)
        self.encoder_inputs = {'A': 4, 'B0': 3, 'C0': 2, 'D0': 1, 'E': 0}
        self.node_channels = {name: width for (group, width) in zip(NODE_GROUPS, (256, 128, 64, 32, 16)) for name in group}
        self.nodes = nn.ModuleDict()
        for (name, parents) in GRAPH_PARENTS:
            incoming = sum((self.node_channels[parent] for parent in parents))
            if name in self.encoder_inputs:
                incoming += self.encoder_channels[self.encoder_inputs[name]]
            width = self.node_channels[name]
            self.nodes[name] = nn.Sequential(ConvNormAct(incoming, width), ResidualChannelAttention(width), ConvNormAct(width, width), ResidualChannelAttention(width))

    def _validate(self, features, output_size):
        if not isinstance(features, (list, tuple)) or len(features) != 5:
            raise ValueError('Expected five encoder feature maps')
        if not isinstance(output_size, (list, tuple)) or len(output_size) != 2:
            raise ValueError('output_size must contain height and width')
        for value in output_size:
            positive_int(value, 'output dimension')
        first = features[0]
        if not isinstance(first, torch.Tensor):
            raise TypeError('Encoder features must be tensors')
        for (feature, channels) in zip(features, self.encoder_channels):
            if not isinstance(feature, torch.Tensor) or feature.ndim != 4 or min(feature.shape) < 1 or (feature.shape[1] != channels):
                raise ValueError('Encoder features must be nonempty NCHW tensors with configured channels')
            if feature.shape[0] != first.shape[0] or feature.dtype != first.dtype or feature.device != first.device or (not feature.is_floating_point()):
                raise ValueError('Encoder features must share floating dtype, device and batch size')

    def forward(self, features, output_size):
        self._validate(features, output_size)
        outputs = {}
        for (column, group) in enumerate(NODE_GROUPS):
            reference = features[4 - column]
            for name in group:
                inputs = [resize_like(outputs[parent], reference) for parent in self.parents[name]]
                if name in self.encoder_inputs:
                    inputs.append(features[self.encoder_inputs[name]])
                outputs[name] = run_block(self.nodes[name], *inputs, enabled=self.checkpoint_blocks, concatenate=True)
        return F.interpolate(outputs['E'], size=output_size, mode='bilinear', align_corners=False)
