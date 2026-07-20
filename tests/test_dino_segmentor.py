"""Network-free smoke tests for the DINOv2 segmentation path.

These do NOT download DINOv2 weights. They substitute a tiny fake ViT backbone
that mimics the DINOv2Backbone interface (channels list, .vit.blocks, and a
base_forward returning 4 same-resolution feature maps). This verifies the
decoder pyramid wiring, feature-perturbation hook, output shapes, and the
layer-wise LR-decay optimizer grouping — the parts we wrote — purely on CPU.

Run: python -m tests.test_dino_segmentor
"""

import torch
import torch.nn as nn

from model.semseg.dino_segmentor import DINOv2Segmentor
from core.optim import build_optimizer


EMBED_DIM = 768
DEPTH = 12
PATCH = 14


class _FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(EMBED_DIM, EMBED_DIM)


class _FakeViT(nn.Module):
    """Stand-in for the DINOv2 ViT: exposes .blocks and a patch_embed param."""

    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, EMBED_DIM, PATCH, stride=PATCH)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, 10, EMBED_DIM))
        self.blocks = nn.ModuleList([_FakeBlock() for _ in range(DEPTH)])


class _FakeDINOBackbone(nn.Module):
    """Mimics DINOv2Backbone: 4 feature maps at H/14, plus .vit for LLRD."""

    def __init__(self):
        super().__init__()
        self.embed_dim = EMBED_DIM
        self.patch_size = PATCH
        self.channels = [EMBED_DIM] * 4
        self.out_layers = (2, 5, 8, 11)
        self.vit = _FakeViT()

    def base_forward(self, x):
        h, w = x.shape[-2:]
        gh, gw = h // PATCH, w // PATCH
        # Push x through patch_embed so backbone params are in the graph.
        tok = self.vit.patch_embed(x)  # [B, EMBED_DIM, gh, gw]
        return [tok + i for i in range(4)]


def _build_model(nclass=21):
    backbone = _FakeDINOBackbone()
    return DINOv2Segmentor(backbone=backbone, nclass=nclass)


def test_forward_shape():
    model = _build_model(nclass=21)
    model.eval()
    x = torch.randn(2, 3, 14 * 6, 14 * 6)  # 84x84, divisible by 14
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2, 21, 84, 84), logits.shape
    print(f'[ok] forward shape {tuple(logits.shape)}')


def test_feature_perturbation_and_return():
    model = _build_model()
    model.train()
    x = torch.randn(1, 3, 70, 70)
    logits, feat = model(x, perturb_feature=True, return_feature=True)
    assert logits.shape[0] == 1 and logits.shape[2:] == (70, 70)
    assert feat.dim() == 4
    print(f'[ok] perturb+feature: logits {tuple(logits.shape)}, feat {tuple(feat.shape)}')


def test_backward_runs():
    model = _build_model()
    model.train()
    x = torch.randn(1, 3, 70, 70)
    logits = model(x)
    target = torch.randint(0, 21, (1, 70, 70))
    loss = nn.functional.cross_entropy(logits, target)
    loss.backward()
    grads = [p.grad is not None for p in model.parameters() if p.requires_grad]
    assert any(grads), 'no gradients flowed'
    print(f'[ok] backward: loss={loss.item():.3f}, params with grad={sum(grads)}/{len(grads)}')


def test_forward_unimatch_dual():
    model = _build_model(nclass=21)
    model.eval()
    # Two views concatenated along batch dim, batch B=3 each -> 6 total.
    x = torch.randn(6, 3, 84, 84)
    with torch.no_grad():
        s1, s2 = model.forward_unimatch_dual(x)
    assert s1.shape == (3, 21, 84, 84), s1.shape
    assert s2.shape == (3, 21, 84, 84), s2.shape
    # The two outputs should differ because complementary channel dropout
    # zeros disjoint feature subsets between the halves.
    assert not torch.allclose(s1, s2), 'forward_unimatch_dual streams are identical'
    print(f'[ok] forward_unimatch_dual: s1 {tuple(s1.shape)}, s2 {tuple(s2.shape)}, diverge ok')


def test_llrd_optimizer():
    model = _build_model()
    opt = build_optimizer(model, backbone_lr=1e-5, decoder_lr=1e-3, layer_decay=0.65)
    lrs = [g['lr'] for g in opt.param_groups]
    # Decoder groups should carry the largest LR.
    assert max(lrs) == 1e-3, f'expected decoder lr 1e-3, got max {max(lrs)}'
    # Deepest backbone layer < decoder; shallowest backbone layer is smallest.
    backbone_lrs = [g['lr'] for g in opt.param_groups if g['lr'] < 1e-3]
    assert min(backbone_lrs) < max(backbone_lrs), 'layer decay not applied'
    assert min(backbone_lrs) < 1e-5, 'shallow layers should be < base backbone lr'
    print(f'[ok] LLRD: {len(opt.param_groups)} groups, '
          f'lr range [{min(lrs):.2e}, {max(lrs):.2e}]')


if __name__ == '__main__':
    test_forward_shape()
    test_feature_perturbation_and_return()
    test_forward_unimatch_dual()
    test_backward_runs()
    test_llrd_optimizer()
    print('\nAll DINOv2 segmentor smoke tests passed.')
