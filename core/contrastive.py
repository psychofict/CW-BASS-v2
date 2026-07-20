"""Pixel-level contrastive auxiliary loss for semi-supervised segmentation.

A small projection head maps decoder features to a normalised embedding space.
For each high-confidence labeled pixel (anchor) we draw positives (same class)
and negatives (other classes) from a per-class memory bank, then minimise an
InfoNCE-style supervised contrastive loss. The bank is updated each iteration
with new features from the labeled batch, providing a much larger pool of
positives/negatives than the current batch alone.

References (close relatives, not direct lineage):
  - ReCo (Liu et al., 2022): regional contrast for SSSS
  - U2PL (Wang et al., 2022): unreliable pseudo-labels as informative negatives
  - SupCon (Khosla et al., 2020): supervised contrastive learning

Our novelty hook: this auxiliary is applied on top of UniMatch V2's strict-
threshold regime (conf >= 0.95), and the memory bank is updated only from
labeled pixels whose teacher prediction matches the ground truth -- a
confidence-and-correctness filter, not just confidence. This gives the bank
genuinely clean positives that pure confidence-based banks lack.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PixContrastiveHead(nn.Module):
    """Lightweight 1x1 projection head with L2 normalisation."""

    def __init__(self, in_channels, hidden=256, out_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_dim, 1),
        )
        self.out_dim = out_dim

    def forward(self, x):
        """Args:   x [B, C, H, W]
        Returns:   [B, D, H, W] unit-norm pixel embeddings.
        """
        z = self.proj(x)
        return F.normalize(z, dim=1)


class ClassMemoryBank:
    """Per-class FIFO queue of unit-norm pixel embeddings (numpy-free, on-device)."""

    def __init__(self, num_classes, dim, size_per_class=256, device='cuda'):
        self.num_classes = num_classes
        self.dim = dim
        self.size = size_per_class
        self.device = device
        # [K, size, D]
        self.queue = torch.zeros(num_classes, size_per_class, dim, device=device)
        self.ptr = torch.zeros(num_classes, dtype=torch.long, device=device)
        self.filled = torch.zeros(num_classes, dtype=torch.bool, device=device)

    @torch.no_grad()
    def enqueue(self, features, labels):
        """Push pixel embeddings into their per-class queues.

        Args:
            features: [N, D] L2-normalised features (will be detached).
            labels: [N] class indices in [0, num_classes). Out-of-range entries
                (e.g. ignore_index 255) are silently skipped.
        """
        features = features.detach()
        for k in range(self.num_classes):
            mask = labels == k
            if not mask.any():
                continue
            fk = features[mask]
            nk = fk.shape[0]
            ptr = int(self.ptr[k].item())
            if nk >= self.size:
                self.queue[k] = fk[:self.size]
                self.ptr[k] = 0
                self.filled[k] = True
            else:
                end = ptr + nk
                if end <= self.size:
                    self.queue[k, ptr:end] = fk
                else:
                    first = self.size - ptr
                    self.queue[k, ptr:] = fk[:first]
                    self.queue[k, :nk - first] = fk[first:]
                self.ptr[k] = end % self.size
                if end >= self.size:
                    self.filled[k] = True

    def all_features_labels(self):
        """Return concatenated features + labels for every class with any entry.

        Returns:
            (feats [N, D], labels [N]) or (None, None) if the bank is empty.
        """
        feats, labs = [], []
        for k in range(self.num_classes):
            n = self.size if bool(self.filled[k].item()) else int(self.ptr[k].item())
            if n > 0:
                feats.append(self.queue[k, :n])
                labs.append(torch.full((n,), k, device=self.device, dtype=torch.long))
        if not feats:
            return None, None
        return torch.cat(feats, dim=0), torch.cat(labs, dim=0)

    def state_dict(self):
        return {'queue': self.queue, 'ptr': self.ptr, 'filled': self.filled}

    def load_state_dict(self, sd):
        self.queue = sd['queue'].to(self.device)
        self.ptr = sd['ptr'].to(self.device)
        self.filled = sd['filled'].to(self.device)


def pixel_contrastive_loss(anchor_feats, anchor_labels, bank,
                           temperature=0.1, max_anchors=1024):
    """Supervised contrastive loss: each anchor pulls to same-class bank entries,
    pushes away from different-class entries.

    Args:
        anchor_feats: [N_a, D] unit-norm anchor embeddings (grad-tracked).
        anchor_labels: [N_a] class indices.
        bank: ClassMemoryBank (read-only here; the caller enqueues separately).
        temperature: InfoNCE temperature.
        max_anchors: subsample to keep compute bounded.

    Returns:
        Scalar loss; zero if the bank is empty or no anchor has same-class positives.
    """
    n = anchor_feats.shape[0]
    if n == 0:
        return anchor_feats.new_zeros(())

    if n > max_anchors:
        idx = torch.randperm(n, device=anchor_feats.device)[:max_anchors]
        anchor_feats = anchor_feats[idx]
        anchor_labels = anchor_labels[idx]

    bank_feats, bank_labels = bank.all_features_labels()
    if bank_feats is None or bank_feats.shape[0] == 0:
        return anchor_feats.new_zeros(())

    # Logits: [N_a, N_bank], scaled by 1/temperature.
    logits = anchor_feats @ bank_feats.t() / temperature
    same = anchor_labels.unsqueeze(1) == bank_labels.unsqueeze(0)  # [N_a, N_bank]

    # Numerically stable log-sum-exp.
    max_logits = logits.max(dim=1, keepdim=True).values.detach()
    exp = torch.exp(logits - max_logits)
    sum_all = exp.sum(dim=1)                       # [N_a]
    sum_pos = (exp * same.float()).sum(dim=1)      # [N_a]
    has_pos = sum_pos > 0

    if not has_pos.any():
        return anchor_feats.new_zeros(())

    # L = -log(sum_pos / sum_all) over anchors that have a positive.
    loss = -(torch.log(sum_pos[has_pos] + 1e-9) - torch.log(sum_all[has_pos] + 1e-9))
    return loss.mean()


def sample_anchors_from_labeled(features, labels, pred, max_per_class=64,
                                ignore_index=255, bank_filter='clean',
                                conf=None, conf_thresh=0.95):
    """Pick a balanced anchor set from a labeled batch.

    Two admission rules, selected by ``bank_filter`` (paper ablation):

      'clean' (default): keep pixels where
        - label != ignore_index, AND
        - student prediction matches the ground-truth label (clean positives).
      The kept pixel's *ground-truth* label is its bank label.

      'conf': ReCo/U2PL-style confidence-filtered baseline. Keep pixels where
        - label != ignore_index, AND
        - max-softmax confidence >= conf_thresh,
      WITHOUT requiring prediction == label. The kept pixel's *predicted* class
      (argmax) is its bank label, so the bank can admit confidently-wrong
      entries. This isolates the value of the clean-positive filter.

    Args:
        features: [B, D, H, W] unit-norm embeddings.
        labels: [B, H, W] ground-truth class indices.
        pred: [B, H, W] argmax of student logits (same resolution as labels).
        max_per_class: cap per-class anchors for class-balanced sampling.
        bank_filter: 'clean' or 'conf' (see above).
        conf: [B, H, W] max-softmax confidence of the student logits. Required
            when bank_filter == 'conf'.
        conf_thresh: confidence threshold for the 'conf' rule.

    Returns:
        (anchor_feats [N_a, D], anchor_labels [N_a]).
    """
    b, d, h, w = features.shape
    feats_flat = features.permute(0, 2, 3, 1).reshape(-1, d)  # [BHW, D]
    labels_flat = labels.reshape(-1)
    pred_flat = pred.reshape(-1)

    if bank_filter == 'conf':
        # ReCo/U2PL-style: confidence-only admission; label = predicted class.
        if conf is None:
            raise ValueError("bank_filter='conf' requires the per-pixel `conf` tensor")
        conf_flat = conf.reshape(-1)
        valid = (labels_flat != ignore_index) & (conf_flat >= conf_thresh)
        anchor_label_src = pred_flat
    else:
        valid = (labels_flat != ignore_index) & (pred_flat == labels_flat)
        anchor_label_src = labels_flat

    if not valid.any():
        return features.new_zeros((0, d)), labels.new_zeros((0,), dtype=torch.long)

    feats_v = feats_flat[valid]
    labels_v = anchor_label_src[valid]

    keep_feats, keep_labs = [], []
    for k in labels_v.unique():
        mask = labels_v == k
        n = int(mask.sum().item())
        if n <= max_per_class:
            keep_feats.append(feats_v[mask])
            keep_labs.append(labels_v[mask])
        else:
            idx = torch.randperm(n, device=feats_v.device)[:max_per_class]
            keep_feats.append(feats_v[mask][idx])
            keep_labs.append(labels_v[mask][idx])

    return torch.cat(keep_feats, dim=0), torch.cat(keep_labs, dim=0)
