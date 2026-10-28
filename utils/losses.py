import math
import torch
from torch import nn
import torch.nn.functional as F

def safe_semantic_ce(logits, labels):
    if not torch.any(labels != 0):
        return logits.float().sum() * 0.0
    return F.cross_entropy(logits, labels, ignore_index=0)

def binary_dice_loss(prediction, target, smooth=1e-08):
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError('Dice prediction and target must have matching batch/spatial shapes')
    prediction = prediction.float().flatten(1)
    target = target.float().flatten(1)
    numerator = 2 * (prediction * target).sum(1) + smooth
    denominator = prediction.sum(1) + target.sum(1) + smooth
    return (1 - numerator / denominator).mean()

def logit_interaction_loss(sem1, sem2, change):
    probability1 = sem1.float().softmax(1)
    probability2 = sem2.float().softmax(1)
    probability_change = change.float().softmax(1)
    same = probability1.argmax(1) == probability2.argmax(1)
    target_confidence = torch.where(same, probability_change[:, 0], probability_change[:, 1])
    semantic_confidence = 0.5 * (probability1.max(1).values + probability2.max(1).values)
    return F.mse_loss(semantic_confidence, target_confidence.detach())

class MultiTaskLoss(nn.Module):

    def __init__(self, alpha=1.0):
        super().__init__()
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError('alpha must be finite and nonnegative')
        self.alpha = float(alpha)

    def forward(self, outputs, batch):
        sem = 0.5 * (safe_semantic_ce(outputs['sem1'], batch['label1']) + safe_semantic_ce(outputs['sem2'], batch['label2']))
        change_target = (batch['label1'] > 0).long()
        cd = F.cross_entropy(outputs['change'], change_target)
        mask = outputs['change'].argmax(1).to(outputs['boundary_sem'].dtype)
        boundary_sem = outputs['boundary_sem'].squeeze(1) * mask
        boundary_change = outputs['boundary_change'].squeeze(1)
        boundary = 0.5 * (binary_dice_loss(boundary_sem, batch['boundary_sem']) + binary_dice_loss(boundary_change, batch['boundary_change']))
        sim = outputs['sim_loss']
        logit = logit_interaction_loss(outputs['sem1'], outputs['sem2'], outputs['change'])
        total = sem + cd + boundary + self.alpha * (sim + logit)
        return {'total': total, 'sem': sem, 'cd': cd, 'boundary': boundary, 'sim': sim, 'logit': logit}
