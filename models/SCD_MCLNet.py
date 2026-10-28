import copy
import torch
from torch import nn
import torch.nn.functional as F
from .cswin import SCanFormer
from .layers import SSAE, TaskInteraction, positive_int, positive_real
from .source_checked import CheckedSemanticDecoder, DECODER_CHANNELS
from .reference_parts import PaperLightweightEncoder, ReferenceTemporalFusion, BTBoundaryFeatures, BTECA, BTCBA
from .checkpointing import run_block
from .graph_decoder import FigureDenseChangeDecoder
DEFAULT_SETTINGS = {'profile': 'paper_graph_v3', 'checkpoint_blocks': True, 'encoder_channels': [64, 64, 128, 256, 512], 'encoder_repeats': [3, 4, 6, 3], 'encoder_expansions': [2, 2, 4, 4], 'scan_dim': 128, 'scan_depth': 2, 'scan_heads': 4, 'scan_stripe': 4, 'mlp_ratio': 4, 'ssae_reduction': 16}
VARIANTS = ('full',)

def validate_settings(settings):
    if not isinstance(settings, dict):
        raise TypeError('model settings must be a dict')
    unknown = set(settings) - set(DEFAULT_SETTINGS)
    if unknown:
        raise ValueError('unknown model settings: {}'.format(', '.join(sorted((str(key) for key in unknown)))))
    resolved = copy.deepcopy(DEFAULT_SETTINGS)
    resolved.update(copy.deepcopy(settings))
    if resolved['profile'] != 'paper_graph_v3':
        raise ValueError('Only the final paper_graph_v3 model is included')
    if type(resolved['checkpoint_blocks']) is not bool:
        raise ValueError('checkpoint_blocks must be bool')
    for (name, length) in (('encoder_channels', 5), ('encoder_repeats', 4), ('encoder_expansions', 4)):
        values = resolved[name]
        if not isinstance(values, (list, tuple)) or len(values) != length:
            raise ValueError('{} must contain {} values'.format(name, length))
        for value in values:
            if name == 'encoder_expansions':
                positive_real(value, name)
                if value < 1:
                    raise ValueError('encoder expansion factors must be at least 1')
            else:
                positive_int(value, name)
        resolved[name] = list(values)
    for name in ('scan_dim', 'scan_depth', 'scan_heads', 'scan_stripe', 'ssae_reduction'):
        positive_int(resolved[name], name)
    positive_real(resolved['mlp_ratio'], 'mlp_ratio')
    if resolved['scan_heads'] % 2 or resolved['scan_dim'] % resolved['scan_heads']:
        raise ValueError('scan_heads must be even and divide scan_dim')
    if int(resolved['scan_dim'] * resolved['mlp_ratio']) < 1:
        raise ValueError('mlp_ratio produces an empty hidden dimension')
    return resolved

class SCDMCLNet(nn.Module):

    def __init__(self, settings=None, num_classes=7, variant='full'):
        super().__init__()
        self.settings = validate_settings({} if settings is None else settings)
        positive_int(num_classes, 'num_classes')
        if num_classes < 2:
            raise ValueError('num_classes must include unchanged and at least one semantic class')
        if variant not in VARIANTS:
            raise ValueError('variant must be one of {}'.format(', '.join(VARIANTS)))
        self.variant = variant
        self.num_classes = num_classes
        cfg = self.settings
        channels = cfg['encoder_channels']
        self.encoder = PaperLightweightEncoder(channels, cfg['encoder_repeats'], cfg['encoder_expansions'], linear_bottlenecks=True)
        with torch.random.fork_rng(devices=[]):
            self.scanformer = SCanFormer(channels[-1], cfg['scan_dim'], cfg['scan_depth'], cfg['scan_heads'], cfg['scan_stripe'], cfg['mlp_ratio'])
        with torch.random.fork_rng(devices=[]):
            self.ssae = SSAE(channels[-1], cfg['ssae_reduction'])
        self.semantic_decoder = CheckedSemanticDecoder(channels)
        output_channels = DECODER_CHANNELS[-1]
        self.semantic_classifier = nn.Conv2d(output_channels, num_classes, 1)
        self.temporal_fusion = ReferenceTemporalFusion(channels[-1])
        self.change_decoder = FigureDenseChangeDecoder(channels)
        self.change_classifier = nn.Conv2d(output_channels, 2, 1)
        self.task_interaction = TaskInteraction(output_channels)
        self.boundary_decoder = BTBoundaryFeatures(output_channels)
        self.boundary_attention = BTECA()
        self.boundary_classifier = nn.Sequential(BTCBA(output_channels, max(4, output_channels // 2)), nn.Conv2d(max(4, output_channels // 2), 1, 1), nn.Sigmoid())
        if cfg['checkpoint_blocks']:
            self.encoder.checkpoint_blocks = True
            self.semantic_decoder.checkpoint_blocks = True
            self.change_decoder.checkpoint_blocks = True

    @staticmethod
    def _validate_inputs(image1, image2, return_aux):
        if not isinstance(return_aux, bool):
            raise TypeError('return_aux must be bool')
        if not isinstance(image1, torch.Tensor) or not isinstance(image2, torch.Tensor):
            raise TypeError('images must be torch tensors')
        if image1.ndim != 4 or image2.ndim != 4 or image1.shape != image2.shape:
            raise ValueError('images must have identical NCHW shapes')
        if image1.shape[1] != 3 or min(image1.shape) < 1:
            raise ValueError('images must be nonempty RGB batches')
        if not image1.is_floating_point() or not image2.is_floating_point():
            raise TypeError('images must have floating-point dtype')
        if image1.dtype != image2.dtype or image1.device != image2.device:
            raise ValueError('images must have identical dtype and device')

    def forward(self, image1, image2, return_aux=True):
        self._validate_inputs(image1, image2, return_aux)
        output_size = image1.shape[-2:]
        (features1, features2) = (self.encoder(image1), self.encoder(image2))
        (semantic1, semantic2) = (list(features1), list(features2))
        semantic1[-1] = self.scanformer(features1[-1])
        semantic2[-1] = self.scanformer(features2[-1])
        sem1 = self.semantic_decoder(semantic1, output_size)
        sem2 = self.semantic_decoder(semantic2, output_size)
        (top1, top2) = (features1[-1], features2[-1])
        (_, top1) = self.ssae(top1)
        (_, top2) = self.ssae(top2)
        changes = [torch.abs(a - b) for (a, b) in zip(features1[:-1], features2[:-1])]
        changes.append(self.temporal_fusion(top1, top2))
        change_features = self.change_decoder(changes, output_size)
        initial_change = self.change_classifier(change_features) if return_aux else None
        (refined, similarity) = self.task_interaction(sem1, sem2, change_features, initial_change, compute_loss=return_aux)
        outputs = {'change': F.interpolate(self.change_classifier(refined), size=output_size, mode='bilinear', align_corners=False), 'sem1': F.interpolate(self.semantic_classifier(sem1), size=output_size, mode='bilinear', align_corners=False), 'sem2': F.interpolate(self.semantic_classifier(sem2), size=output_size, mode='bilinear', align_corners=False)}
        if return_aux:
            checkpoint_blocks = self.settings['checkpoint_blocks']
            boundary1 = run_block(self.boundary_decoder, sem1, output_size, enabled=checkpoint_blocks)
            boundary2 = run_block(self.boundary_decoder, sem2, output_size, enabled=checkpoint_blocks)
            boundary_change = run_block(self.boundary_decoder, refined, output_size, enabled=checkpoint_blocks)
            outputs.update({'sim_loss': similarity, 'boundary_sem': run_block(self.boundary_classifier, self.boundary_attention(boundary1 + boundary2), enabled=checkpoint_blocks), 'boundary_change': run_block(self.boundary_classifier, boundary_change, enabled=checkpoint_blocks)})
        return outputs
