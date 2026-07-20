"""Perturbation primitives for weak-to-strong consistency training.

Two complementary perturbations, following UniMatch:
  - image-space strong augmentation (color jitter / grayscale / blur) + CutMix
  - feature-space perturbation (channel dropout), applied inside the model

The teacher sees the clean (weak) view and produces pseudo-labels; the student is
supervised on these perturbed views, enforcing prediction consistency.
"""

import math
import random

import torch
from torchvision import transforms as T

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def strong_augment(images):
    """Strong image-space augmentation on a normalized batch tensor.

    Denormalizes to [0,1], applies per-image ColorJitter / grayscale / blur, then
    re-normalizes. Geometry is untouched so pseudo-labels stay pixel-aligned.

    Args:
        images: [B, 3, H, W] ImageNet-normalized tensor.
    Returns:
        [B, 3, H, W] augmented, re-normalized tensor.
    """
    device = images.device
    mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)

    imgs = (images * std + mean).clamp(0, 1)
    out = []
    for i in range(imgs.shape[0]):
        img = imgs[i]
        if random.random() < 0.8:
            img = T.functional.adjust_brightness(img, random.uniform(0.5, 1.5))
            img = T.functional.adjust_contrast(img, random.uniform(0.5, 1.5))
            img = T.functional.adjust_saturation(img, random.uniform(0.5, 1.5))
            img = T.functional.adjust_hue(img, random.uniform(-0.25, 0.25))
        if random.random() < 0.2:
            img = img.mean(dim=0, keepdim=True).expand_as(img)
        if random.random() < 0.5:
            k = random.choice([3, 5])
            img = T.functional.gaussian_blur(img, k, random.uniform(0.1, 2.0))
        out.append(img)
    imgs = torch.stack(out).clamp(0, 1)
    return (imgs - mean) / std


def _rand_bbox(h, w, lam):
    """Random box covering (1 - lam) of the area, CutMix-style."""
    cut_rat = math.sqrt(1.0 - lam)
    cut_h, cut_w = int(h * cut_rat), int(w * cut_rat)
    cy, cx = random.randint(0, h), random.randint(0, w)
    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, h)
    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, w)
    return y1, y2, x1, x2


def cutmix(images, pseudo, confidence, mask):
    """Apply CutMix within a batch, consistently to image + pseudo-label tensors.

    A random rectangular region is copied from a shuffled version of the batch
    into every sample. Pseudo-labels, confidence, and the retention mask are
    pasted with the same box so supervision stays pixel-consistent.

    Args:
        images: [B, 3, H, W]
        pseudo: [B, H, W] long pseudo-labels
        confidence: [B, H, W] float
        mask: [B, H, W] bool retention mask
    Returns:
        mixed (images, pseudo, confidence, mask) — all cloned, inputs untouched.
    """
    b, _, h, w = images.shape
    perm = torch.randperm(b, device=images.device)
    lam = random.random()
    y1, y2, x1, x2 = _rand_bbox(h, w, lam)

    images = images.clone()
    pseudo = pseudo.clone()
    confidence = confidence.clone()
    mask = mask.clone()

    images[:, :, y1:y2, x1:x2] = images[perm][:, :, y1:y2, x1:x2]
    pseudo[:, y1:y2, x1:x2] = pseudo[perm][:, y1:y2, x1:x2]
    confidence[:, y1:y2, x1:x2] = confidence[perm][:, y1:y2, x1:x2]
    mask[:, y1:y2, x1:x2] = mask[perm][:, y1:y2, x1:x2]
    return images, pseudo, confidence, mask


def supports_feature_perturbation(model):
    """True if model.forward accepts a `perturb_feature` kwarg (DINOv2Segmentor)."""
    import inspect
    target = model.module if hasattr(model, 'module') else model
    try:
        sig = inspect.signature(target.forward)
        return 'perturb_feature' in sig.parameters
    except (ValueError, TypeError):
        return False
