"""Evaluation-time inference for DINOv2 (patch-14) segmentation.

Two modes:
  - whole_inference: reflect-pad the image so H,W are divisible by 14, run a
    single forward, then crop logits back to the original size. Suitable for
    small images (Pascal, ADE20K).
  - slide_inference: sliding-window over large images (Cityscapes), accumulating
    softmax probabilities with a count map. Each window is patch-padded.

Both return [B, C, H, W] logits/probabilities at the original input resolution.
"""

import torch
import torch.nn.functional as F

from dataset.transform import pad_to_multiple


@torch.no_grad()
def whole_inference(model, img, patch=14):
    """Single forward with reflect-padding to a multiple of `patch`."""
    padded, (h, w) = pad_to_multiple(img, patch)
    logits = model(padded)
    return logits[..., :h, :w]


@torch.no_grad()
def slide_inference(model, img, crop_size, num_classes, stride_ratio=2 / 3, patch=14):
    """Sliding-window inference for large images.

    Args:
        model: segmentation model returning [B, C, h, w] logits.
        img: [B, 3, H, W] input.
        crop_size: window size (square). Should be divisible by `patch`.
        num_classes: C.
        stride_ratio: window stride as a fraction of crop_size.
    Returns:
        [B, C, H, W] accumulated softmax probabilities.
    """
    b, _, h, w = img.shape
    stride = max(int(crop_size * stride_ratio), 1)
    n_h = max((h - crop_size + stride - 1) // stride + 1, 1)
    n_w = max((w - crop_size + stride - 1) // stride + 1, 1)

    probs = img.new_zeros((b, num_classes, h, w))
    count = img.new_zeros((b, 1, h, w))

    for i in range(n_h):
        for j in range(n_w):
            y1 = min(i * stride, max(h - crop_size, 0))
            x1 = min(j * stride, max(w - crop_size, 0))
            y2, x2 = min(y1 + crop_size, h), min(x1 + crop_size, w)
            y1, x1 = max(y2 - crop_size, 0), max(x2 - crop_size, 0)

            window = img[:, :, y1:y2, x1:x2]
            padded, (wh, ww) = pad_to_multiple(window, patch)
            logits = model(padded)[..., :wh, :ww]
            probs[:, :, y1:y2, x1:x2] += F.softmax(logits, dim=1)
            count[:, :, y1:y2, x1:x2] += 1

    return probs / count.clamp(min=1)
