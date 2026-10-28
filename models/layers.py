import math
from numbers import Real
import torch
from torch import nn
import torch.nn.functional as F

def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError('{} must be a positive integer'.format(name))
    return value

def positive_real(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or (not math.isfinite(value)) or (value <= 0):
        raise ValueError('{} must be finite and positive'.format(name))
    return value

class SafeBatchNorm2d(nn.BatchNorm2d):

    def forward(self, x):
        if self.training and x.numel() // x.shape[1] == 1:
            return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, False, 0.0, self.eps)
        return super().forward(x)

class ConvNormAct(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, activation='relu'):
        if activation not in ('relu', 'relu6'):
            raise ValueError('activation must be relu or relu6')
        super().__init__(nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, groups=groups, bias=False), SafeBatchNorm2d(out_channels), nn.ReLU6(inplace=False) if activation == 'relu6' else nn.ReLU(inplace=False))

def resize_like(x, reference):
    return F.interpolate(x, size=reference.shape[-2:], mode='bilinear', align_corners=False)

class ChannelAttention(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()
        positive_int(channels, 'channels')
        positive_int(reduction, 'reduction')
        hidden = max(1, channels // reduction)
        self.channel_mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False), nn.ReLU(inplace=False), nn.Conv2d(hidden, channels, 1, bias=False))

    def forward(self, x):
        weights = torch.sigmoid(self.channel_mlp(F.adaptive_avg_pool2d(x, 1)) + self.channel_mlp(F.adaptive_max_pool2d(x, 1)))
        return x * weights

class ResidualChannelAttention(ChannelAttention):

    def forward(self, x):
        return x + super().forward(x)

class SSAE(ChannelAttention):

    def __init__(self, channels, reduction=16):
        super().__init__(channels, reduction)
        self.spatial = nn.Conv2d(2, 1, 3, padding=1)

    def forward(self, x):
        channel_features = super().forward(x)
        descriptors = torch.cat((channel_features.mean(dim=1, keepdim=True), channel_features.amax(dim=1, keepdim=True)), dim=1)
        saliency = torch.sigmoid(self.spatial(descriptors))
        return (saliency, channel_features * saliency)

class TaskInteraction(nn.Module):

    def __init__(self, channels):
        super().__init__()
        positive_int(channels, 'channels')
        self.gate = nn.Sequential(nn.Conv2d(2, 1, 3, padding=1, bias=False), nn.Sigmoid())
        self.loss_fn = nn.CosineEmbeddingLoss(margin=0.0, reduction='mean')

    def forward(self, sem1, sem2, change_features, initial_change=None, compute_loss=True):
        difference = torch.abs(sem1 - sem2)
        descriptors = torch.cat((difference.max(dim=1, keepdim=True).values, difference.mean(dim=1, keepdim=True)), dim=1)
        refined = change_features * self.gate(descriptors)
        if not compute_loss:
            return (refined, None)
        if initial_change is None:
            raise ValueError('initial change logits are required for similarity supervision')
        a = sem1.permute(0, 2, 3, 1).reshape(-1, sem1.shape[1]).float()
        b = sem2.permute(0, 2, 3, 1).reshape(-1, sem2.shape[1]).float()
        labels = (1 - 2 * initial_change.detach().argmax(dim=1)).reshape(-1).float()
        return (refined, self.loss_fn(a, b, labels))
