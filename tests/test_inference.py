"""Smoke tests for DINOv2 eval-time inference (Phase 5).

Verifies whole-image patch-padded inference and sliding-window inference produce
outputs at the original (possibly non-patch-divisible) resolution, using the
network-free fake DINOv2 backbone.

Run: python -m tests.test_inference
"""

import torch

from model.semseg.dino_segmentor import DINOv2Segmentor
from core.inference import whole_inference, slide_inference
from tests.test_dino_segmentor import _FakeDINOBackbone

NCLASS = 21


def _model():
    m = DINOv2Segmentor(backbone=_FakeDINOBackbone(), nclass=NCLASS)
    m.eval()
    return m


def test_whole_inference_non_divisible():
    model = _model()
    # 100x130 is NOT divisible by 14; output must still match input size.
    img = torch.randn(1, 3, 100, 130)
    out = whole_inference(model, img)
    assert out.shape == (1, NCLASS, 100, 130), out.shape
    print(f'[ok] whole_inference output {tuple(out.shape)} for 100x130 input')


def test_slide_inference_large():
    model = _model()
    # Large image, window 70 (=5*14); should tile and return original size probs.
    img = torch.randn(1, 3, 180, 200)
    probs = slide_inference(model, img, crop_size=70, num_classes=NCLASS)
    assert probs.shape == (1, NCLASS, 180, 200), probs.shape
    # Accumulated softmax: every pixel sums to ~1 across classes.
    s = probs.sum(dim=1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-4), 'probs not normalized'
    print(f'[ok] slide_inference output {tuple(probs.shape)}, probs normalized')


if __name__ == '__main__':
    test_whole_inference_non_divisible()
    test_slide_inference_large()
    print('\nAll inference smoke tests passed.')
