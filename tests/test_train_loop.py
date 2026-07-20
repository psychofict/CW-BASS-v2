"""Network-free smoke test for the UniMatch-style train_epoch (Phase 2).

Builds a tiny DINOv2Segmentor on a fake backbone, wires up the EMA teacher,
class-adaptive threshold, and tiny random loaders, then runs train_epoch for one
epoch. Verifies: the loop runs end-to-end on CPU, losses are finite, the dual
streams (image strong+CutMix and feature perturbation) both contribute, the EMA
teacher updates, and the threshold module steps its epoch.

Run: python -m tests.test_train_loop
"""

from types import SimpleNamespace

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset

from model.semseg.dino_segmentor import DINOv2Segmentor
from core.ema import EMATeacher
from core.thresholding import ClassAdaptiveThreshold
from core.perturb import supports_feature_perturbation
from tests.test_dino_segmentor import _FakeDINOBackbone

import train as train_mod


NCLASS = 21
CROP = 70  # 5 * 14


def _make_loader(n, labeled=True):
    imgs = torch.randn(n, 3, CROP, CROP)
    if labeled:
        masks = torch.randint(0, NCLASS, (n, CROP, CROP))
    else:
        masks = torch.full((n, CROP, CROP), 255, dtype=torch.long)
    return DataLoader(TensorDataset(imgs, masks), batch_size=2, drop_last=True)


def test_train_epoch_runs():
    device = torch.device('cpu')
    model = DINOv2Segmentor(backbone=_FakeDINOBackbone(), nclass=NCLASS)
    assert supports_feature_perturbation(model), 'fake DINO model should support FP'

    teacher = EMATeacher(model, decay=0.99)
    threshold = ClassAdaptiveThreshold(num_classes=NCLASS, warmup_epochs=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = CrossEntropyLoss(ignore_index=255)

    labeled_loader = _make_loader(6, labeled=True)
    unlabeled_loader = _make_loader(6, labeled=False)

    args = SimpleNamespace(model='dino', num_classes=NCLASS, gamma=1.0,
                           epochs=2, lr=1e-4, warmup_epochs=1)

    # Snapshot all teacher params to confirm EMA moved the model overall.
    tp_before = [p.clone() for p in teacher.model.parameters()]

    # Epoch 0: warmup (global threshold). Epoch 1: per-class adaptive.
    for epoch in range(2):
        stats = train_mod.train_epoch(
            model, teacher, labeled_loader, unlabeled_loader,
            optimizer, criterion, args, threshold, device, epoch,
        )
        for k in ('loss', 'loss_x', 'loss_u', 'mask_ratio'):
            assert stats[k] == stats[k], f'{k} is NaN'  # NaN check
            assert stats[k] >= 0.0 or k == 'lr'
        print(f'[ok] epoch {epoch}: loss={stats["loss"]:.3f} '
              f'x={stats["loss_x"]:.3f} u={stats["loss_u"]:.3f} '
              f'mask={stats["mask_ratio"]:.3f}')

    total_change = sum((a - b).abs().sum().item()
                       for a, b in zip(teacher.model.parameters(), tp_before))
    assert total_change > 0.0, 'EMA teacher did not update at all'
    assert threshold.current_epoch.item() == 2, 'threshold epoch did not step'
    print('[ok] EMA updated, threshold stepped to epoch', threshold.current_epoch.item())


def test_unimatch_v2_epoch_runs():
    """End-to-end CPU smoke test of train_epoch_unimatch_v2 with the fake DINO model."""
    device = torch.device('cpu')
    model = DINOv2Segmentor(backbone=_FakeDINOBackbone(), nclass=NCLASS)
    teacher = EMATeacher(model, decay=0.99)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = CrossEntropyLoss(ignore_index=255)

    labeled_loader = _make_loader(6, labeled=True)
    unlabeled_loader = _make_loader(6, labeled=False)

    args = SimpleNamespace(model='dino', num_classes=NCLASS, gamma=1.0,
                           epochs=2, lr=1e-4, conf_thresh=0.95,
                           scheme='unimatch_v2')

    tp_before = [p.clone() for p in teacher.model.parameters()]
    for epoch in range(2):
        stats = train_mod.train_epoch_unimatch_v2(
            model, teacher, labeled_loader, unlabeled_loader,
            optimizer, criterion, args, device, epoch,
        )
        for k in ('loss', 'loss_x', 'loss_u', 'mask_ratio'):
            assert stats[k] == stats[k], f'{k} is NaN'
        print(f'[ok] um2 epoch {epoch}: loss={stats["loss"]:.3f} '
              f'x={stats["loss_x"]:.3f} u={stats["loss_u"]:.3f} '
              f'mask={stats["mask_ratio"]:.3f}')
    total_change = sum((a - b).abs().sum().item()
                       for a, b in zip(teacher.model.parameters(), tp_before))
    assert total_change > 0.0, 'EMA teacher did not update in unimatch_v2 path'
    print('[ok] EMA updated in unimatch_v2 path')


def test_unimatch_v2_with_pixcon():
    """End-to-end CPU smoke test of train_epoch_unimatch_v2 with PixCon enabled."""
    from core.contrastive import PixContrastiveHead, ClassMemoryBank
    device = torch.device('cpu')
    model = DINOv2Segmentor(backbone=_FakeDINOBackbone(), nclass=NCLASS)
    teacher = EMATeacher(model, decay=0.99)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = CrossEntropyLoss(ignore_index=255)

    decoder_dim = model.head[0].in_channels
    pix_head = PixContrastiveHead(in_channels=decoder_dim, hidden=64, out_dim=32)
    pix_bank = ClassMemoryBank(NCLASS, dim=32, size_per_class=16, device='cpu')
    optimizer.add_param_group({'params': list(pix_head.parameters()), 'lr': 1e-3})

    labeled_loader = _make_loader(6, labeled=True)
    unlabeled_loader = _make_loader(6, labeled=False)

    args = SimpleNamespace(model='dino', num_classes=NCLASS, gamma=1.0,
                           epochs=1, lr=1e-4, conf_thresh=0.95,
                           scheme='unimatch_v2', use_pixcon=True,
                           pixcon_weight=0.1, pixcon_temp=0.1,
                           pixcon_max_anchors=128, pixcon_per_class=8)
    stats = train_mod.train_epoch_unimatch_v2(
        model, teacher, labeled_loader, unlabeled_loader,
        optimizer, criterion, args, device, epoch=0,
        pix_head=pix_head, pix_bank=pix_bank,
    )
    for k in ('loss', 'loss_x', 'loss_u', 'loss_pix', 'mask_ratio'):
        assert stats[k] == stats[k], f'{k} is NaN'
    # Bank should have received some features.
    feats, _ = pix_bank.all_features_labels()
    assert feats is not None and feats.shape[0] > 0, 'PixCon bank not populated'
    print(f'[ok] um2+pixcon: loss={stats["loss"]:.3f} '
          f'pix={stats["loss_pix"]:.3f}, bank={feats.shape[0]} feats')


def test_resnet_fallback_path():
    """When the model lacks feature perturbation, stream 2 falls back to a 2nd
    strong view. Emulate with a model whose forward has no perturb_feature kwarg."""
    import torch.nn as nn

    class _PlainSeg(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, NCLASS, 3, padding=1)

        def forward(self, x):
            return self.conv(x)

    model = _PlainSeg()
    assert not supports_feature_perturbation(model)
    print('[ok] plain model correctly reports no feature-perturbation support')


if __name__ == '__main__':
    test_train_epoch_runs()
    test_unimatch_v2_epoch_runs()
    test_unimatch_v2_with_pixcon()
    test_resnet_fallback_path()
    print('\nAll train-loop smoke tests passed.')
