import math
import operator
import numpy as np
import torch

class SCDMetrics:

    def __init__(self, num_classes=7):
        try:
            if isinstance(num_classes, (bool, np.bool_)):
                raise TypeError('Boolean class count')
            num_classes = operator.index(num_classes)
        except TypeError as error:
            raise ValueError('num_classes must be an integer >= 2') from error
        if num_classes < 2:
            raise ValueError('num_classes must be an integer >= 2')
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def _class_map(self, value, name):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError('{} must contain integer class IDs'.format(name))
        if np.any(array < 0) or np.any(array >= self.num_classes):
            raise ValueError('{} class IDs must be in [0,{})'.format(name, self.num_classes))
        return array

    def update(self, prediction, target):
        prediction = self._class_map(prediction, 'prediction')
        target = self._class_map(target, 'target')
        if prediction.shape != target.shape:
            raise ValueError('Prediction and target shapes differ: {} vs {}'.format(prediction.shape, target.shape))
        indices = self.num_classes * target.ravel().astype(np.int64) + prediction.ravel().astype(np.int64)
        self.matrix += np.bincount(indices, minlength=self.num_classes ** 2).reshape(self.matrix.shape)

    def compute(self):
        hist = self.matrix.astype(np.float64)
        total = hist.sum()
        if total == 0:
            raise ValueError('Cannot compute SCD metrics without any observations')
        unchanged = hist[0, 0]
        false_changed = hist[0, 1:].sum()
        missed_changed = hist[1:, 0].sum()
        changed = hist[1:, 1:].sum()
        changed_union = changed + false_changed + missed_changed
        unchanged_union = unchanged + false_changed + missed_changed
        iou_changed = changed / changed_union if changed_union else 0.0
        iou_unchanged = unchanged / unchanged_union if unchanged_union else 0.0
        semantic_tp = np.trace(hist[1:, 1:])
        predicted_changed = hist[:, 1:].sum()
        true_changed = hist[1:, :].sum()
        precision = semantic_tp / predicted_changed if predicted_changed else 0.0
        recall = semantic_tp / true_changed if true_changed else 0.0
        fscd = 2 * semantic_tp / (predicted_changed + true_changed) if predicted_changed + true_changed else 0.0
        hist[0, 0] = 0
        nonzero_total = hist.sum()
        kappa = 0.0
        if nonzero_total:
            observed = np.trace(hist) / nonzero_total
            expected = np.dot(hist.sum(axis=1), hist.sum(axis=0)) / nonzero_total ** 2
            if expected < 1:
                kappa = (observed - expected) / (1 - expected)
        return {'OA': float((unchanged + semantic_tp) / total), 'Fscd': float(fscd), 'mIoU': float((iou_changed + iou_unchanged) / 2), 'SeK': float(kappa * math.exp(iou_changed - 1)), 'Precision': float(precision), 'Recall': float(recall), 'IoU_changed': float(iou_changed), 'IoU_unchanged': float(iou_unchanged)}
