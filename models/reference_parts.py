import torch
from torch import nn
import torch.nn.functional as F
from .layers import ConvNormAct, SafeBatchNorm2d, positive_int
from .checkpointing import run_block

class DepthwiseLinearConv(nn.Sequential):

    def __init__(self, incoming, outgoing, expansion, stride=1):
        hidden = max(1, round(incoming * expansion))
        super().__init__(ConvNormAct(incoming, hidden, 1, activation='relu6'), ConvNormAct(hidden, hidden, 3, stride, groups=hidden, activation='relu6'), nn.Conv2d(hidden, outgoing, 1, bias=False), SafeBatchNorm2d(outgoing))

class PaperResidualBlock(nn.Module):

    def __init__(self, incoming, outgoing, expansion, stride=1, linear_bottlenecks=False):
        super().__init__()
        if type(linear_bottlenecks) is not bool:
            raise ValueError('linear_bottlenecks must be bool')
        self.path = nn.Sequential(DepthwiseLinearConv(incoming, outgoing, expansion, stride), nn.Identity() if linear_bottlenecks else nn.ReLU6(inplace=False), DepthwiseLinearConv(outgoing, outgoing, expansion))
        self.shortcut = nn.Identity() if incoming == outgoing and stride == 1 else nn.Sequential(nn.Conv2d(incoming, outgoing, 1, stride, bias=False), SafeBatchNorm2d(outgoing))

    def forward(self, x):
        return self.path(x) + self.shortcut(x)

class PaperLightweightEncoder(nn.Module):

    def __init__(self, channels, repeats, expansions, linear_bottlenecks=False):
        super().__init__()
        self.checkpoint_blocks = False
        self.stem = ConvNormAct(3, channels[0], 7, 2, activation='relu6')
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.stages = nn.ModuleList()
        for (index, (repeat, expansion)) in enumerate(zip(repeats, expansions)):
            stride = 1 if index == 0 else 2
            blocks = [PaperResidualBlock(channels[index], channels[index + 1], expansion, stride, linear_bottlenecks=linear_bottlenecks)]
            blocks.extend((PaperResidualBlock(channels[index + 1], channels[index + 1], expansion, linear_bottlenecks=linear_bottlenecks) for _ in range(repeat - 1)))
            self.stages.append(nn.Sequential(*blocks))

    def forward(self, image):
        stem = run_block(self.stem, image, enabled=self.checkpoint_blocks)
        features = [stem]
        x = self.maxpool(stem)
        for stage in self.stages:
            for block in stage:
                x = run_block(block, x, enabled=self.checkpoint_blocks)
            features.append(x)
        return features

class BTCBA(nn.Module):

    def __init__(self, incoming, outgoing, kernel=3):
        super().__init__()
        self.cba = nn.Sequential(nn.Conv2d(incoming, outgoing, kernel, padding=kernel // 2), SafeBatchNorm2d(outgoing), nn.ReLU(inplace=False))

    def forward(self, x):
        return self.cba(x)

class BTECA(nn.Module):

    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, 3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        descriptor = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        weights = self.sigmoid(self.conv(descriptor).transpose(-1, -2).unsqueeze(-1))
        return x * weights.expand_as(x)

class BTResidualBlock(nn.Module):

    def __init__(self, incoming, outgoing):
        super().__init__()
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False)
        self.bn1 = SafeBatchNorm2d(outgoing)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False)
        self.bn2 = SafeBatchNorm2d(outgoing)
        self.downsample = None if incoming == outgoing else nn.Sequential(nn.Conv2d(incoming, outgoing, 1, bias=False), SafeBatchNorm2d(outgoing))

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + identity)

class BTChangeSpecificTransfer(nn.Module):

    def __init__(self, channels=128):
        super().__init__()
        positive_int(channels, 'channels')
        self.conv = BTCBA(2 * channels, channels, 1)
        self.eca = BTECA()
        self.resblock = nn.Sequential(BTResidualBlock(2 * channels, channels), *[BTResidualBlock(channels, channels) for _ in range(5)])

    def forward(self, a, b):
        forward = self.conv(torch.cat((a, b), dim=1))
        reverse = self.conv(torch.cat((b, a), dim=1))
        combined = self.eca(forward + reverse)
        return self.resblock(torch.cat((combined, torch.abs(a - b)), dim=1))

class ReferenceTemporalFusion(nn.Module):

    def __init__(self, channels, transfer_channels=128):
        super().__init__()
        self.input_projection = nn.Identity() if channels == transfer_channels else BTCBA(channels, transfer_channels, 1)
        self.transfer = BTChangeSpecificTransfer(transfer_channels)
        self.output_projection = nn.Identity() if channels == transfer_channels else BTCBA(transfer_channels, channels, 1)

    def forward(self, a, b):
        return self.output_projection(self.transfer(self.input_projection(a), self.input_projection(b)))

class BTBoundaryFeatures(nn.Module):

    def __init__(self, channels):
        super().__init__()
        positive_int(channels, 'channels')
        kx = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
        ky = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
        directions = []
        for kernel in (kx, ky):
            conv = nn.Conv2d(channels, 1, 3, padding=1, bias=False)
            conv.weight = nn.Parameter(kernel.reshape(1, 1, 3, 3).repeat(1, channels, 1, 1), requires_grad=False)
            directions.append(nn.Sequential(conv, SafeBatchNorm2d(1)))
        (self.sobel_x, self.sobel_y) = directions
        self.conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x, size):
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        (gx, gy) = (self.sobel_x(x), self.sobel_y(x))
        magnitude = torch.linalg.vector_norm(torch.stack((gx.float(), gy.float())), dim=0)
        return self.conv(torch.sigmoid(magnitude).to(x.dtype) * x)
