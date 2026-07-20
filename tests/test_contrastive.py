"""CPU smoke tests for PixCon (pixel contrastive auxiliary).

Verifies the projection head shapes, the per-class memory bank's enqueue +
wrap-around behaviour, and that the InfoNCE loss is finite, non-zero with
positives, and *decreases* under a small optimisation step (sanity check that
gradients point in a sensible direction).
"""

import torch

from core.contrastive import (
    PixContrastiveHead, ClassMemoryBank, pixel_contrastive_loss,
    sample_anchors_from_labeled,
)


def test_head_shapes_and_norm():
    head = PixContrastiveHead(in_channels=128, hidden=64, out_dim=32)
    x = torch.randn(2, 128, 16, 16)
    z = head(x)
    assert z.shape == (2, 32, 16, 16), z.shape
    norms = z.pow(2).sum(dim=1).sqrt()
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
        f'embeddings not unit-norm: min {norms.min()}, max {norms.max()}'
    print(f'[ok] head: shape {tuple(z.shape)}, unit-norm.')


def test_bank_enqueue_and_wrap():
    bank = ClassMemoryBank(num_classes=3, dim=8, size_per_class=4, device='cpu')
    # Push 5 features of class 1 into a size-4 queue -> should wrap.
    feats = torch.randn(5, 8)
    feats = feats / feats.pow(2).sum(dim=1, keepdim=True).sqrt()
    labels = torch.tensor([1, 1, 1, 1, 1])
    bank.enqueue(feats, labels)
    assert bool(bank.filled[1]), 'class 1 should be marked filled'
    fb, lb = bank.all_features_labels()
    assert fb.shape == (4, 8), fb.shape
    assert (lb == 1).all()
    print(f'[ok] bank: enqueue+wrap; stored {fb.shape[0]} of class 1.')


def test_loss_finite_and_positive():
    torch.manual_seed(0)
    bank = ClassMemoryBank(num_classes=3, dim=16, size_per_class=32, device='cpu')
    # Seed the bank with random class-specific features.
    for k in range(3):
        f = torch.randn(20, 16)
        f = f / f.pow(2).sum(dim=1, keepdim=True).sqrt()
        bank.enqueue(f, torch.full((20,), k, dtype=torch.long))
    anchors = torch.randn(50, 16, requires_grad=True)
    a = anchors / anchors.pow(2).sum(dim=1, keepdim=True).sqrt()
    labels = torch.randint(0, 3, (50,))
    loss = pixel_contrastive_loss(a, labels, bank, temperature=0.1)
    assert loss.dim() == 0 and torch.isfinite(loss), f'bad loss {loss}'
    assert loss.item() > 0.0, f'expected positive loss, got {loss.item()}'
    loss.backward()
    assert anchors.grad is not None and torch.isfinite(anchors.grad).all()
    print(f'[ok] loss = {loss.item():.4f}, gradient flows.')


def test_loss_decreases_under_step():
    """Sanity check: a gradient step on anchors should reduce the loss."""
    torch.manual_seed(0)
    bank = ClassMemoryBank(num_classes=3, dim=16, size_per_class=64, device='cpu')
    for k in range(3):
        # Bank features clustered near a class-specific anchor direction.
        center = torch.zeros(16); center[k] = 1.0
        f = center.unsqueeze(0) + 0.1 * torch.randn(40, 16)
        f = f / f.pow(2).sum(dim=1, keepdim=True).sqrt()
        bank.enqueue(f, torch.full((40,), k, dtype=torch.long))

    # Anchors deliberately misaligned with their class centers.
    anchors = torch.randn(60, 16, requires_grad=True)
    labels = torch.randint(0, 3, (60,))
    opt = torch.optim.SGD([anchors], lr=0.5)

    a = anchors / anchors.pow(2).sum(dim=1, keepdim=True).sqrt()
    loss0 = pixel_contrastive_loss(a, labels, bank, temperature=0.1).item()
    for _ in range(20):
        a = anchors / anchors.pow(2).sum(dim=1, keepdim=True).sqrt()
        loss = pixel_contrastive_loss(a, labels, bank, temperature=0.1)
        opt.zero_grad()
        loss.backward()
        opt.step()
    a = anchors / anchors.pow(2).sum(dim=1, keepdim=True).sqrt()
    loss_final = pixel_contrastive_loss(a, labels, bank, temperature=0.1).item()
    assert loss_final < loss0 - 0.1, f'loss did not drop meaningfully: {loss0:.3f} -> {loss_final:.3f}'
    print(f'[ok] optimisation reduces loss: {loss0:.3f} -> {loss_final:.3f}.')


def test_sample_anchors_balanced():
    feats = torch.randn(2, 8, 4, 4)
    feats = torch.nn.functional.normalize(feats, dim=1)
    labels = torch.randint(0, 3, (2, 4, 4))
    pred = labels.clone()  # all "correct" so all eligible
    af, al = sample_anchors_from_labeled(feats, labels, pred, max_per_class=3)
    assert af.dim() == 2 and af.shape[1] == 8
    # Each class should contribute at most max_per_class.
    for k in al.unique():
        assert int((al == k).sum().item()) <= 3
    print(f'[ok] anchor sampling: {af.shape[0]} anchors across {len(al.unique())} classes.')


if __name__ == '__main__':
    test_head_shapes_and_norm()
    test_bank_enqueue_and_wrap()
    test_loss_finite_and_positive()
    test_loss_decreases_under_step()
    test_sample_anchors_balanced()
    print('\nAll PixCon smoke tests passed.')
