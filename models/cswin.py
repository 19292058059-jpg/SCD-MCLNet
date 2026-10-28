import torch
from torch import nn
import torch.nn.functional as F
from .layers import positive_int, positive_real

def _partition(x, window_height, window_width):
    (batch, channels, height, width) = x.shape
    return x.reshape(batch, channels, height // window_height, window_height, width // window_width, window_width).permute(0, 2, 4, 1, 3, 5).reshape(-1, channels, window_height, window_width)

def _reverse(windows, batch, height, width):
    (_, channels, window_height, window_width) = windows.shape
    return windows.reshape(batch, height // window_height, width // window_width, channels, window_height, window_width).permute(0, 3, 1, 4, 2, 5).reshape(batch, channels, height, width)

class StripeAttention(nn.Module):

    def __init__(self, dim, num_heads, stripe_size=4, orientation='horizontal'):
        super().__init__()
        positive_int(dim, 'dim')
        positive_int(num_heads, 'num_heads')
        positive_int(stripe_size, 'stripe_size')
        if dim % num_heads:
            raise ValueError('dim must be divisible by num_heads')
        if orientation not in ('horizontal', 'vertical'):
            raise ValueError('orientation must be horizontal or vertical')
        self.dim = dim
        self.num_heads = num_heads
        self.stripe_size = stripe_size
        self.orientation = orientation
        self.scale = (dim // num_heads) ** (-0.5)
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, q, k, v):
        if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape or (q.shape[1] != self.dim) or (min(q.shape) < 1):
            raise ValueError('Q/K/V must have equal nonempty NCHW shapes and configured channels')
        (batch, channels, height, width) = q.shape
        if self.orientation == 'horizontal':
            (wh, ww) = (min(height, self.stripe_size), width)
        else:
            (wh, ww) = (height, min(width, self.stripe_size))
        (pad_height, pad_width) = (-height % wh, -width % ww)
        padding = (0, pad_width, 0, pad_height)
        (q_windows, k_windows, v_windows) = [_partition(F.pad(x, padding), wh, ww) for x in (q, k, v)]
        valid = F.pad(torch.ones((batch, 1, height, width), device=q.device, dtype=torch.bool), padding, value=False)
        valid_windows = _partition(valid, wh, ww).flatten(2)
        head_dim = channels // self.num_heads
        (q_heads, k_heads, v_heads) = [x.reshape(-1, self.num_heads, head_dim, wh * ww).transpose(-1, -2) for x in (q_windows, k_windows, v_windows)]
        scores = q_heads * self.scale @ k_heads.transpose(-1, -2)
        scores = scores.masked_fill(~valid_windows.unsqueeze(2), float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        output = (weights @ v_heads).transpose(-1, -2).reshape(-1, channels, wh, ww)
        output = output + self.lepe(v_windows)
        output = output.masked_fill(~valid_windows.reshape(-1, 1, wh, ww), 0)
        output = _reverse(output, batch, height + pad_height, width + pad_width)
        return output[:, :, :height, :width]

class CSWinBlock(nn.Module):

    def __init__(self, dim, num_heads=4, stripe_size=4, mlp_ratio=4):
        super().__init__()
        positive_int(dim, 'dim')
        positive_int(num_heads, 'num_heads')
        positive_int(stripe_size, 'stripe_size')
        positive_real(mlp_ratio, 'mlp_ratio')
        if num_heads % 2 or dim % num_heads:
            raise ValueError('CSWin requires an even number of heads dividing dim')
        hidden = int(dim * mlp_ratio)
        if hidden < 1:
            raise ValueError('mlp_ratio produces an empty hidden dimension')
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.horizontal = StripeAttention(dim // 2, num_heads // 2, stripe_size, 'horizontal')
        self.vertical = StripeAttention(dim // 2, num_heads // 2, stripe_size, 'vertical')
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.dim or min(x.shape) < 1:
            raise ValueError('CSWin expects a nonempty NCHW map with configured channels')
        tokens = self.norm1(x.permute(0, 2, 3, 1))
        (q, k, v) = [part.permute(0, 3, 1, 2) for part in self.qkv(tokens).chunk(3, dim=-1)]
        half = self.dim // 2
        horizontal = self.horizontal(q[:, :half], k[:, :half], v[:, :half])
        vertical = self.vertical(q[:, half:], k[:, half:], v[:, half:])
        attended = torch.cat((horizontal, vertical), dim=1).permute(0, 2, 3, 1)
        x = x + self.proj(attended).permute(0, 3, 1, 2)
        return x + self.mlp(self.norm2(x.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)

class SCanFormer(nn.Module):

    def __init__(self, in_channels, dim=128, depth=2, num_heads=4, stripe_size=4, mlp_ratio=4):
        super().__init__()
        positive_int(in_channels, 'in_channels')
        positive_int(depth, 'depth')
        self.input_projection = nn.Conv2d(in_channels, dim, 1)
        self.input_norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([CSWinBlock(dim, num_heads, stripe_size, mlp_ratio) for _ in range(depth)])
        self.output_norm = nn.LayerNorm(dim)
        self.output_projection = nn.Conv2d(dim, in_channels, 1)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.input_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        for block in self.blocks:
            x = block(x)
        x = self.output_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.output_projection(x)
