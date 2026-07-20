"""Loss functions with per-class weighting for class-adaptive thresholding."""

import torch
import torch.nn.functional as F


def weighted_cross_entropy_loss(prediction, pseudo_labels, confidence, gamma=1.0,
                                class_weights=None, ignore_index=255):
    """Confidence-weighted cross-entropy loss with optional per-class reweighting.

    Args:
        prediction: [B, C, H, W] logits
        pseudo_labels: [B, H, W] integer class labels
        confidence: [B, H, W] pixel confidence scores
        gamma: confidence exponent (higher = more weight to confident pixels)
        class_weights: optional [K] tensor of per-class weights
        ignore_index: label value to ignore (default 255)

    Returns:
        scalar loss
    """
    ce_loss = F.cross_entropy(prediction, pseudo_labels, reduction='none',
                              ignore_index=ignore_index)
    valid = pseudo_labels != ignore_index

    if class_weights is not None:
        # Apply per-class weights: weight_map[b,h,w] = class_weights[pseudo_labels[b,h,w]]
        # Clamp labels to valid range for indexing (ignore_index pixels will be masked out)
        safe_labels = pseudo_labels.clone()
        safe_labels[~valid] = 0
        weight_map = class_weights.to(prediction.device)[safe_labels]
        ce_loss = ce_loss * weight_map

    weighted = confidence ** gamma * ce_loss

    if valid.sum() == 0:
        return weighted.sum() * 0.0  # avoid NaN
    return weighted[valid].mean()


def compute_class_balance_weights(mask, pseudo_labels, num_classes, ignore_index=255):
    """Compute inverse-frequency class balance weights for retained pixels.

    Computes: class_weights[k] = total_retained / (K * retained_k)
    Classes with zero retained pixels get weight 0.

    Args:
        mask: [B, H, W] boolean mask of retained pixels
        pseudo_labels: [B, H, W] integer class labels
        num_classes: K
        ignore_index: label to ignore

    Returns:
        class_weights: [K] tensor
    """
    device = pseudo_labels.device
    retained_labels = pseudo_labels[mask & (pseudo_labels != ignore_index)]

    if retained_labels.numel() == 0:
        return torch.ones(num_classes, device=device)

    counts = torch.bincount(retained_labels, minlength=num_classes).float()
    total = counts.sum()

    weights = torch.zeros(num_classes, device=device)
    nonzero = counts > 0
    weights[nonzero] = total / (num_classes * counts[nonzero])

    # Normalize so mean weight is 1
    if weights[nonzero].numel() > 0:
        weights[nonzero] = weights[nonzero] / weights[nonzero].mean()

    return weights


def boundary_loss(pred, pseudo_labels, confidence, boundary_mask, gamma=1.0,
                  class_weights=None, boundary_weight=0.5, ignore_index=255):
    """Boundary-aware loss combining weighted CE and boundary-region CE.

    Args:
        pred: [B, C, H, W] logits
        pseudo_labels: [B, H, W] integer class labels
        confidence: [B, H, W] pixel confidence
        boundary_mask: [B, H, W] float boundary mask
        gamma: confidence exponent
        class_weights: optional [K] per-class weights
        boundary_weight: weight for boundary loss term (default 0.5)
        ignore_index: label to ignore

    Returns:
        scalar loss
    """
    base = weighted_cross_entropy_loss(pred, pseudo_labels, confidence, gamma,
                                       class_weights, ignore_index)

    # Boundary-specific loss: compute per-pixel CE once on the real logits,
    # then weight by boundary_mask and average over boundary pixels. The old
    # implementation multiplied the logits tensor by an [B,C,H,W] expanded mask
    # (allocating an extra ~B*C*H*W tensor) just to zero out non-boundary
    # pixels before CE -- mathematically the same as multiplying the per-pixel
    # CE by the boundary mask afterward, but uses much more memory.
    boundary_pixels = boundary_mask > 0
    valid = (pseudo_labels != ignore_index) & boundary_pixels
    if valid.sum() > 0:
        ce_per_pixel = F.cross_entropy(pred, pseudo_labels, reduction='none',
                                       ignore_index=ignore_index)
        boundary_term = (ce_per_pixel * boundary_mask)[valid].mean()
    else:
        boundary_term = base * 0.0

    return base + boundary_weight * boundary_term
