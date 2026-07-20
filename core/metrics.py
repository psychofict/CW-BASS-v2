import numpy as np


class SegmentationMetrics:
    """Accumulating confusion-matrix-based segmentation metrics.

    Provides per-class IoU and mean IoU via add_batch / evaluate interface.
    Also backwards-compatible with the old meanIOU class interface.
    """

    def __init__(self, num_classes, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.hist = np.zeros((num_classes, num_classes), dtype=np.int64)

    def _compute_hist(self, pred, gt):
        mask = (gt >= 0) & (gt < self.num_classes)
        hist = np.bincount(
            self.num_classes * gt[mask].astype(int) + pred[mask].astype(int),
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, ground_truths):
        """Add a batch of predictions and ground truths.

        Args:
            predictions: numpy array [B, H, W] or [H, W]
            ground_truths: numpy array [B, H, W] or [H, W]
        """
        if predictions.ndim == 2:
            predictions = predictions[np.newaxis]
            ground_truths = ground_truths[np.newaxis]
        for pred, gt in zip(predictions, ground_truths):
            self.hist += self._compute_hist(pred.flatten(), gt.flatten())

    def evaluate(self):
        """Returns (per_class_iou, mean_iou).

        per_class_iou: array of shape [num_classes]
        mean_iou: scalar (ignoring classes with union==0)
        """
        intersection = np.diag(self.hist)
        union = self.hist.sum(axis=1) + self.hist.sum(axis=0) - intersection
        iou = np.where(union > 0, intersection / union, 0.0)
        miou = np.mean(iou[union > 0]) if np.any(union > 0) else 0.0
        return iou, miou

    def get_per_class_iou_dict(self):
        """Returns {class_name: iou} dict."""
        iou, _ = self.evaluate()
        return {name: float(iou[i]) for i, name in enumerate(self.class_names)}

    def reset(self):
        self.hist = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
