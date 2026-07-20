"""Class-adaptive training loop for semi-supervised segmentation.

Follows UniMatch-V2 experimentation patterns:
  - Separate labeled + unlabeled dataloaders
  - Supervised loss on labeled data (always) + pseudo-label loss on unlabeled (thresholded)
  - TensorBoard logging (per-iteration train metrics, per-epoch eval metrics)
  - Console logging 8x per epoch
  - EMA teacher with ramp-up capped at 0.996
  - Dual evaluation (student + teacher)
  - latest.pth every epoch + best.pth on improvement
"""

import os
import json
import math
import time
import logging
from contextlib import nullcontext as _nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD
from torch.utils.data import DataLoader

from dataset.semi import SemiDataset
from model.semseg.deeplabv2 import DeepLabV2
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.pspnet import PSPNet
from utils import count_params

from core.thresholding import ClassAdaptiveThreshold
from core.ema import EMATeacher
from core.losses import (
    compute_class_balance_weights,
    boundary_loss,
    weighted_cross_entropy_loss,
)
from core.boundaries import detect_boundaries
from core.metrics import SegmentationMetrics
from core.perturb import strong_augment, cutmix, supports_feature_perturbation

logger = logging.getLogger(__name__)

# Class names for readable logging
PASCAL_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
    'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
    'train', 'tvmonitor'
]
CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic_light', 'traffic_sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
    'bicycle'
]


def get_class_names(dataset, num_classes):
    if dataset == 'pascal' and num_classes == 21:
        return PASCAL_CLASSES
    elif dataset == 'cityscapes' and num_classes == 19:
        return CITYSCAPES_CLASSES
    return [f'class_{i}' for i in range(num_classes)]


def compute_pixel_confidence(prediction):
    """Computes pixel-wise confidence from logits."""
    probs = F.softmax(prediction, dim=1)
    return torch.max(probs, dim=1).values


def quantile_retention_thresh(conf, pseudo, q, floor, ignore_index=255):
    """Per-batch quantile retention threshold (anti-confidence-concentration).

    A fixed absolute threshold (e.g. 0.95) barely filters at foundation-model
    strength because the teacher's confidence concentrates near 1 (>95% of
    pixels clear it), flooding training with the residual wrong pseudo-labels.
    Instead we keep only the top (1-q) fraction by confidence: tau = the
    q-quantile of the valid confidences, lower-bounded by `floor`. This
    *guarantees* the least-confident q fraction is dropped regardless of how
    concentrated the distribution is. Returns a scalar threshold.
    """
    v = conf[pseudo != ignore_index]
    if v.numel() == 0:
        return float(floor)
    if v.numel() > 1_000_000:  # subsample for a fast, accurate-enough quantile
        v = v[torch.randint(0, v.numel(), (1_000_000,), device=v.device)]
    tau = torch.quantile(v.float(), float(q)).item()
    return max(float(floor), tau)


def init_model(args, device):
    """Initialize segmentation model and optimizer.

    Returns (model, optimizer). For DINOv2 (--model dino) the optimizer is AdamW
    with layer-wise LR decay; for ResNet models it is the original SGD recipe.
    """
    if args.model == 'dino':
        from model.semseg.dino_segmentor import DINOv2Segmentor
        from core.optim import build_optimizer
        model = DINOv2Segmentor(backbone=args.backbone, nclass=args.num_classes)
        if torch.cuda.device_count() > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        optimizer = build_optimizer(
            model,
            backbone_lr=getattr(args, 'backbone_lr', 1e-5),
            decoder_lr=getattr(args, 'decoder_lr', 1e-3),
            layer_decay=getattr(args, 'layer_decay', 0.65),
            weight_decay=getattr(args, 'weight_decay', 0.01),
        )
        model = model.to(device)
        if torch.cuda.device_count() > 1:
            model = DataParallel(model)
        return model, optimizer

    model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2}
    model = model_zoo[args.model](args.backbone, args.num_classes)
    if torch.cuda.device_count() > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.model == 'deeplabv3plus' and args.backbone in ['resnet50', 'resnet101']:
        pretrain_path = f'pretrained/{args.backbone}.pth'
        if os.path.isfile(pretrain_path):
            backbone_state_dict = torch.load(pretrain_path, map_location='cpu', weights_only=False)
            model.backbone.load_state_dict(backbone_state_dict, strict=False)

    optimizer = SGD([
        {'params': model.backbone.parameters(), 'lr': args.lr},
        {'params': [p for n, p in model.named_parameters() if 'backbone' not in n],
         'lr': args.lr * 10.0}
    ], lr=args.lr, momentum=0.9, weight_decay=1e-4)

    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)

    return model, optimizer


def get_dataloaders(args):
    """Create labeled, unlabeled, and val DataLoaders.

    Following UniMatch-V2: separate labeled and unlabeled loaders.
    Labeled data is oversampled to match unlabeled length.
    """
    # Labeled trainset
    labeled_set = SemiDataset(
        args.dataset, args.data_root, 'train',
        args.crop_size, args.labeled_id_path,
    )

    # Unlabeled trainset (dummy ignore mask — teacher generates pseudo-labels)
    unlabeled_set = SemiDataset(
        args.dataset, args.data_root, 'unlabeled',
        args.crop_size, args.unlabeled_id_path,
    )

    # Hold out a small fraction of labeled examples as a calibration set used
    # only for unbiased per-class noise estimation (not for the supervised loss).
    # This is the structural fix for the collapse cause: noise measured on
    # training-labeled data is overconfident -> epsilon_hat ~ 0 -> tau collapses.
    calib_frac = float(getattr(args, 'calib_frac', 0.05))
    if calib_frac > 0 and getattr(args, 'method', 'class_adaptive') in ('class_adaptive', 'cwbass_v2'):
        rng = np.random.default_rng(getattr(args, 'seed', 0))
        ids = list(labeled_set.ids)
        rng.shuffle(ids)
        n_calib = max(8, int(round(len(ids) * calib_frac)))
        n_calib = min(n_calib, max(1, len(ids) // 4))  # never starve training
        calib_ids = ids[:n_calib]
        labeled_set.ids = ids[n_calib:]
        # Build calibration set as a 'train' SemiDataset slice so it returns
        # (img, mask) with the same transforms (random crop etc. don't hurt; the
        # teacher just needs labeled pixels to score).
        calib_set = SemiDataset(
            args.dataset, args.data_root, 'train',
            args.crop_size, args.labeled_id_path,
        )
        calib_set.ids = calib_ids
    else:
        calib_set = None

    # Oversample labeled to match unlabeled length (UniMatch-V2 pattern)
    n_labeled = len(labeled_set)
    n_unlabeled = len(unlabeled_set)
    if n_labeled < n_unlabeled:
        repeat = math.ceil(n_unlabeled / n_labeled)
        labeled_set.ids = (labeled_set.ids * repeat)[:n_unlabeled]

    num_workers = int(os.environ.get('NUM_WORKERS', min(4, os.cpu_count() or 4)))
    labeled_loader = DataLoader(
        labeled_set, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    unlabeled_loader = DataLoader(
        unlabeled_set, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    calib_loader = None
    if calib_set is not None:
        calib_loader = DataLoader(
            calib_set, batch_size=args.batch_size, shuffle=False,
            num_workers=1, pin_memory=True, drop_last=False,
        )

    # Validation
    if args.dataset == 'ade20k':
        from dataset.ade20k import ADE20KDataset
        valset = ADE20KDataset(args.data_root, 'val')
    else:
        valset = SemiDataset(args.dataset, args.data_root, 'val', None)

    val_batch = 4 if args.dataset == 'cityscapes' else 1
    valloader = DataLoader(
        valset, batch_size=val_batch, shuffle=False,
        num_workers=1, pin_memory=True, drop_last=False,
    )
    return labeled_loader, unlabeled_loader, valloader, calib_loader


def _set_lr(optimizer, args, current_iter, total_iters, is_dino):
    """Apply poly LR decay, preserving per-group base ratios.

    For DINOv2 (AdamW + layer-wise decay) each group already carries its own base
    LR, so we scale every group by the same poly factor. For ResNet (SGD, 2
    groups) we keep the original base/×10 schedule.
    """
    scale = (1 - current_iter / total_iters) ** 0.9
    if is_dino:
        for g in optimizer.param_groups:
            if 'base_lr' not in g:
                g['base_lr'] = g['lr']
            g['lr'] = g['base_lr'] * scale
        # Return the LARGEST current LR (typically the decoder's) so the log
        # shows a meaningful number rather than the deepest-layer LR (~3e-8).
        return max(g['lr'] for g in optimizer.param_groups)
    lr = args.lr * scale
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr * 10.0
    return lr


def train_epoch(model, teacher, labeled_loader, unlabeled_loader,
                optimizer, criterion, args, adaptive_threshold,
                device, epoch, tb_writer=None, global_step=0):
    """One epoch of class-adaptive semi-supervised training.

    UniMatch-style dual perturbation:
      - Labeled batch → supervised CE (always).
      - EMA teacher produces pseudo-labels + confidence on the clean unlabeled view.
      - Per-class adaptive thresholding filters pseudo-labels.
      - Student is supervised on TWO perturbed views of the same pseudo-label:
          (1) image-space strong aug + CutMix
          (2) feature-space channel dropout (DINOv2 only; falls back to a second
              strong view for ResNet)
      - Combined: loss = (loss_x + (loss_s + loss_fp) / 2) / 2
    """
    model.train()
    teacher.model.eval()

    is_dino = (args.model == 'dino')
    use_fp = supports_feature_perturbation(model) and not getattr(args, 'no_fp', False)
    use_cutmix = not getattr(args, 'no_cutmix', False)
    # DINOv2Segmentor exposes forward_dual so the strong and FP streams share
    # one backbone forward (UniMatch's design). Without that we would need a
    # second full ViT-B graph per iter, which OOMs at 518x518 batch 8 on 40GB.
    underlying = model.module if isinstance(model, DataParallel) else model
    use_dual = use_fp and hasattr(underlying, 'forward_dual')

    # bf16 autocast for DINOv2 training -- halves activation memory and matches
    # the precision DINOv2 was pretrained in. bf16 has fp32's dynamic range so no
    # GradScaler is needed (unlike fp16).
    amp_enabled = is_dino and device.type == 'cuda'
    amp_ctx = (torch.amp.autocast('cuda', dtype=torch.bfloat16)
               if amp_enabled else _nullcontext())

    total = {'loss': 0.0, 'loss_x': 0.0, 'loss_u': 0.0, 'mask_ratio': 0.0}

    total_iters = len(unlabeled_loader) * args.epochs
    base_iter = epoch * len(unlabeled_loader)
    log_interval = max(1, len(unlabeled_loader) // 8)

    labeled_iter = iter(labeled_loader)

    for i, (img_u, _) in enumerate(unlabeled_loader):
        try:
            img_l, mask_l = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(labeled_loader)
            img_l, mask_l = next(labeled_iter)

        img_l = img_l.to(device)
        mask_l = mask_l.to(device)
        img_u = img_u.to(device)

        # ---- Supervised loss on labeled data ----
        # Backward the supervised term immediately, scaled by the 0.5 weight it
        # carries in the combined loss = (loss_x + loss_u)/2. This frees the
        # labeled forward graph before the student strong-stream graph is built,
        # so the two large activation graphs never coexist. Gradient-equivalent
        # to a single combined backward (grads accumulate), but roughly halves
        # peak activation memory -- required to fit DINOv2-B dual-stream at
        # crop 518 / batch 4 on an 8 GB GPU.
        optimizer.zero_grad()
        with amp_ctx:
            logits_l = model(img_l)
            loss_x = criterion(logits_l, mask_l)
        (loss_x * 0.5).backward()
        loss_x_item = loss_x.item()
        del logits_l, loss_x

        # ---- Teacher pseudo-labels (clean weak view) ----
        with torch.no_grad(), amp_ctx:
            teacher_logits = teacher(img_u)
            pseudo_labels = torch.argmax(teacher_logits, dim=1)
            confidence = compute_pixel_confidence(teacher_logits)
            del teacher_logits
        del img_l, mask_l

        # Update self-adaptive floor + Beta fit + per-class stats from unlabeled.
        # (Per-class noise estimation is handled by the held-out calibration pass
        # run once per epoch, not from in-batch labeled data — that was the
        # overconfidence bias that caused the original collapse.)
        adaptive_threshold.update_stats(confidence, pseudo_labels, ground_truth=None)

        # Per-class adaptive retention mask on the clean view.
        mask = adaptive_threshold.apply_thresholds(confidence, pseudo_labels)
        if getattr(adaptive_threshold, 'method', None) == 'softmatch':
            # SoftMatch: no hard mask; replace the confidence with a soft
            # per-pixel weight that flows through CutMix into the loss as the
            # confidence weight (gamma=1 -> loss weight == lambda). update_stats
            # above already saw the true confidence.
            confidence = adaptive_threshold.soft_weight(confidence, pseudo_labels)
            mask_ratio = confidence.mean().item()   # mean soft weight
        else:
            mask_ratio = mask.float().mean().item()
        total['mask_ratio'] += mask_ratio

        # ---- Build the strong+CutMix view (used as the single student input) ----
        img_u_s = strong_augment(img_u)
        if use_cutmix:
            img_u_s, pseudo_s, conf_s, mask_s = cutmix(img_u_s, pseudo_labels, confidence, mask)
        else:
            pseudo_s, conf_s, mask_s = pseudo_labels, confidence, mask
        filtered_s = pseudo_s.clone()
        filtered_s[~mask_s] = 255
        class_weights = compute_class_balance_weights(mask_s, pseudo_s, args.num_classes)
        boundary_mask = detect_boundaries(pseudo_s, num_classes=args.num_classes)
        del img_u

        # ---- One student forward -> strong stream and feature-perturbation stream ----
        with amp_ctx:
            if use_dual:
                logits_s, logits_fp = (model.module if isinstance(model, DataParallel)
                                       else model).forward_dual(img_u_s)
                loss_s = boundary_loss(
                    logits_s, filtered_s, conf_s, boundary_mask,
                    gamma=args.gamma, class_weights=class_weights,
                    boundary_weight=getattr(args, 'boundary_weight', 0.5), ignore_index=255,
                )
                loss_fp = weighted_cross_entropy_loss(
                    logits_fp, filtered_s, conf_s,
                    gamma=args.gamma, class_weights=class_weights, ignore_index=255,
                )
                del logits_s, logits_fp
                loss_u = (loss_s + loss_fp) / 2.0
            else:
                # ResNet (no FP support) or --no-fp: single strong stream only.
                logits_s = model(img_u_s)
                loss_s = boundary_loss(
                    logits_s, filtered_s, conf_s, boundary_mask,
                    gamma=args.gamma, class_weights=class_weights,
                    boundary_weight=getattr(args, 'boundary_weight', 0.5), ignore_index=255,
                )
                del logits_s
                loss_fp = torch.zeros_like(loss_s)
                loss_u = loss_s
        del img_u_s

        # Unlabeled term carries the other 0.5 weight; grads accumulate on top
        # of the supervised backward done above, then a single optimizer step.
        (loss_u * 0.5).backward()
        optimizer.step()

        loss_item = (loss_x_item + loss_u.item()) / 2.0
        total['loss'] += loss_item
        total['loss_x'] += loss_x_item
        total['loss_u'] += loss_u.item()

        # LR schedule + EMA ramp-up
        current_iter = base_iter + i
        lr = _set_lr(optimizer, args, current_iter, total_iters, is_dino)
        ema_decay = min(1.0 - 1.0 / (current_iter + 1), 0.996)
        teacher.decay = ema_decay
        student = model.module if isinstance(model, DataParallel) else model
        teacher.update(student)

        if tb_writer is not None:
            step = global_step + i
            tb_writer.add_scalar('train/loss_all', loss_item, step)
            tb_writer.add_scalar('train/loss_x', loss_x_item, step)
            tb_writer.add_scalar('train/loss_s', loss_s.item(), step)
            tb_writer.add_scalar('train/loss_fp', loss_fp.item(), step)
            tb_writer.add_scalar('train/mask_ratio', mask_ratio, step)
            tb_writer.add_scalar('train/lr', lr, step)

        if i % log_interval == 0:
            logger.info(
                f'Iters: {i}/{len(unlabeled_loader)}, LR: {lr:.7f}, '
                f'Total loss: {loss_item:.3f}, Loss x: {loss_x_item:.3f}, '
                f'Loss s: {loss_s.item():.3f}, Loss fp: {loss_fp.item():.3f}, '
                f'Mask ratio: {mask_ratio:.4f}'
            )

    adaptive_threshold.step_epoch()
    n = len(unlabeled_loader)
    return {
        'loss': total['loss'] / n,
        'loss_x': total['loss_x'] / n,
        'loss_u': total['loss_u'] / n,
        'mask_ratio': total['mask_ratio'] / n,
        'lr': optimizer.param_groups[0]['lr'],
    }


def train_epoch_unimatch_v2(model, teacher, labeled_loader, unlabeled_loader,
                            optimizer, criterion, args, device, epoch,
                            tb_writer=None, global_step=0,
                            pix_head=None, pix_bank=None):
    """One epoch of UniMatch V2 training.

    Faithful to the published recipe: two independent strong-augmented views of
    each unlabeled image, both CutMix-mixed, concatenated and passed through ONE
    backbone forward with complementary channel dropout on the fused decoder
    feature. Each prediction is supervised by the weak-view pseudo-label of its
    own CutMix layout, filtered by a fixed confidence threshold (default 0.95).
    Loss is plain cross-entropy, normalised by the count of valid (non-ignored)
    pixels rather than retained-pixel count.
    """
    model.train()
    teacher.model.eval()

    is_dino = (args.model == 'dino')
    conf_thresh = float(getattr(args, 'conf_thresh', 0.95))
    use_cutmix = not getattr(args, 'no_cutmix', False)

    amp_enabled = is_dino and device.type == 'cuda'
    amp_ctx = (torch.amp.autocast('cuda', dtype=torch.bfloat16)
               if amp_enabled else _nullcontext())

    use_pixcon = (pix_head is not None) and (pix_bank is not None)
    if use_pixcon:
        from core.contrastive import (
            pixel_contrastive_loss, sample_anchors_from_labeled,
        )
        pix_head.train()
        pixcon_weight = float(getattr(args, 'pixcon_weight', 0.1))
        pixcon_temp = float(getattr(args, 'pixcon_temp', 0.1))
        pixcon_max_anchors = int(getattr(args, 'pixcon_max_anchors', 1024))
        pixcon_per_class = int(getattr(args, 'pixcon_per_class', 64))
        # Bank admission rule (ablation): 'clean' = pred==GT clean-positive filter
        # (default); 'conf' = ReCo/U2PL-style confidence-only filter.
        pixcon_bank_filter = str(getattr(args, 'pixcon_bank_filter', 'clean'))

    total = {'loss': 0.0, 'loss_x': 0.0, 'loss_u': 0.0,
             'loss_pix': 0.0, 'mask_ratio': 0.0}
    total_iters = len(unlabeled_loader) * args.epochs
    base_iter = epoch * len(unlabeled_loader)
    log_interval = max(1, len(unlabeled_loader) // 8)
    labeled_iter = iter(labeled_loader)

    underlying = model.module if isinstance(model, DataParallel) else model
    if not hasattr(underlying, 'forward_unimatch_dual'):
        raise RuntimeError('unimatch_v2 scheme requires DINOv2Segmentor (no forward_unimatch_dual on this model)')

    for i, (img_u, _) in enumerate(unlabeled_loader):
        try:
            img_l, mask_l = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(labeled_loader)
            img_l, mask_l = next(labeled_iter)

        img_l = img_l.to(device)
        mask_l = mask_l.to(device)
        img_u = img_u.to(device)

        # Supervised loss on labeled data (+ optional PixCon embeddings).
        with amp_ctx:
            if use_pixcon:
                logits_l, fused_l = model(img_l, return_feature=True)
            else:
                logits_l = model(img_l)
                fused_l = None
            loss_x = criterion(logits_l, mask_l)

        # PixCon auxiliary on the LABELED features: anchors = high-confidence
        # labeled pixels whose student prediction matches GT; positives drawn
        # from the per-class bank; negatives from other-class entries.
        loss_pix = logits_l.new_zeros(())
        if use_pixcon and fused_l is not None:
            with amp_ctx:
                # Project to embedding space at decoder resolution.
                z = pix_head(fused_l.float())   # [B, D, h', w']
                # Downsample GT + pred to embedding resolution.
                B, D, h2, w2 = z.shape
                gt_lo = F.interpolate(mask_l.unsqueeze(1).float(), size=(h2, w2),
                                      mode='nearest').squeeze(1).long()
                with torch.no_grad():
                    logits_lo = F.interpolate(logits_l.float(), size=(h2, w2),
                                              mode='bilinear', align_corners=True)
                    pred_lo = logits_lo.argmax(dim=1)
                    # Student max-softmax confidence (only used by the 'conf'
                    # ablation rule; cheap, computed unconditionally for clarity).
                    conf_lo = compute_pixel_confidence(logits_lo)
                anchor_feats, anchor_labels = sample_anchors_from_labeled(
                    z, gt_lo, pred_lo, max_per_class=pixcon_per_class,
                    bank_filter=pixcon_bank_filter, conf=conf_lo,
                    conf_thresh=conf_thresh)
            if anchor_feats.shape[0] > 0:
                loss_pix = pixel_contrastive_loss(
                    anchor_feats, anchor_labels, pix_bank,
                    temperature=pixcon_temp, max_anchors=pixcon_max_anchors)
                pix_bank.enqueue(anchor_feats, anchor_labels)
        del logits_l, fused_l, img_l, mask_l

        # Weak-view teacher pseudo-labels (no augmentation).
        with torch.no_grad(), amp_ctx:
            teacher_logits = teacher(img_u)
            pseudo_w = torch.argmax(teacher_logits, dim=1)
            conf_w = compute_pixel_confidence(teacher_logits)
            del teacher_logits
        # Two INDEPENDENT strong-augmented views.
        img_u_s1 = strong_augment(img_u)
        img_u_s2 = strong_augment(img_u)
        del img_u

        # Independent CutMix on each view: different random box + different shuffle.
        if use_cutmix:
            img_u_s1, pseudo_s1, conf_s1, _ = cutmix(
                img_u_s1, pseudo_w, conf_w, torch.ones_like(pseudo_w, dtype=torch.bool))
            img_u_s2, pseudo_s2, conf_s2, _ = cutmix(
                img_u_s2, pseudo_w, conf_w, torch.ones_like(pseudo_w, dtype=torch.bool))
        else:
            pseudo_s1, conf_s1 = pseudo_w, conf_w
            pseudo_s2, conf_s2 = pseudo_w, conf_w
        del pseudo_w, conf_w

        # Single backbone forward over the concatenation, complementary feature dropout.
        with amp_ctx:
            x_cat = torch.cat([img_u_s1, img_u_s2], dim=0)
            del img_u_s1, img_u_s2
            pred_s1, pred_s2 = underlying.forward_unimatch_dual(x_cat)
            del x_cat

            # Plain CE per pixel, weighted by conf>=tau, normalised by valid count.
            ce1 = F.cross_entropy(pred_s1, pseudo_s1, reduction='none', ignore_index=255)
            ce2 = F.cross_entropy(pred_s2, pseudo_s2, reduction='none', ignore_index=255)
            q_ret = float(getattr(args, 'retention_quantile', 0.0) or 0.0)
            if q_ret > 0.0:
                tau1 = quantile_retention_thresh(conf_s1, pseudo_s1, q_ret, conf_thresh)
                tau2 = quantile_retention_thresh(conf_s2, pseudo_s2, q_ret, conf_thresh)
            else:
                tau1 = tau2 = conf_thresh
            keep1 = (conf_s1 >= tau1).float()
            keep2 = (conf_s2 >= tau2).float()
            denom1 = max(int((pseudo_s1 != 255).sum().item()), 1)
            denom2 = max(int((pseudo_s2 != 255).sum().item()), 1)
            loss_u_s1 = (ce1 * keep1).sum() / denom1
            loss_u_s2 = (ce2 * keep2).sum() / denom2
            loss_u = (loss_u_s1 + loss_u_s2) / 2.0
            loss = (loss_x + loss_u) / 2.0
            if use_pixcon:
                loss = loss + pixcon_weight * loss_pix
            mask_ratio = 0.5 * (keep1.mean().item() + keep2.mean().item())
        del pred_s1, pred_s2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total['loss'] += loss.item()
        total['loss_x'] += loss_x.item()
        total['loss_u'] += loss_u.item()
        total['loss_pix'] += float(loss_pix.item()) if use_pixcon else 0.0
        total['mask_ratio'] += mask_ratio

        current_iter = base_iter + i
        lr = _set_lr(optimizer, args, current_iter, total_iters, is_dino)
        ema_decay = min(1.0 - 1.0 / (current_iter + 1), 0.996)
        teacher.decay = ema_decay
        student = model.module if isinstance(model, DataParallel) else model
        teacher.update(student)

        if tb_writer is not None:
            step = global_step + i
            tb_writer.add_scalar('train/loss_all', loss.item(), step)
            tb_writer.add_scalar('train/loss_x', loss_x.item(), step)
            tb_writer.add_scalar('train/loss_u_s1', loss_u_s1.item(), step)
            tb_writer.add_scalar('train/loss_u_s2', loss_u_s2.item(), step)
            tb_writer.add_scalar('train/mask_ratio', mask_ratio, step)
            tb_writer.add_scalar('train/lr', lr, step)
        if i % log_interval == 0:
            pix_str = f', Lpix: {float(loss_pix.item()):.3f}' if use_pixcon else ''
            logger.info(
                f'Iters: {i}/{len(unlabeled_loader)}, LR: {lr:.7f}, '
                f'Total: {loss.item():.3f}, Lx: {loss_x.item():.3f}, '
                f'Ls1: {loss_u_s1.item():.3f}, Ls2: {loss_u_s2.item():.3f}'
                f'{pix_str}, Mask: {mask_ratio:.4f}'
            )

    n = len(unlabeled_loader)
    return {
        'loss': total['loss'] / n,
        'loss_x': total['loss_x'] / n,
        'loss_u': total['loss_u'] / n,
        'loss_pix': total['loss_pix'] / n,
        'mask_ratio': total['mask_ratio'] / n,
        'lr': optimizer.param_groups[0]['lr'],
    }


@torch.no_grad()
def calibrate(teacher, calib_loader, adaptive_threshold, device):
    """Update the per-class noise estimator + PLS on a held-out calibration set.

    Runs once per epoch (after the noise estimator has been reset). Because the
    calibration examples are NOT in the supervised training set, teacher
    predictions on them are unbiased -- epsilon_hat_k reflects true pseudo-label
    error, not training-set memorization.
    """
    if calib_loader is None:
        return
    teacher.model.eval()
    for batch in calib_loader:
        if len(batch) == 2:
            imgs, masks = batch
        else:
            imgs, masks, *_ = batch
        imgs = imgs.to(device)
        masks = masks.to(device)
        logits = teacher(imgs)
        preds = torch.argmax(logits, dim=1)
        conf = compute_pixel_confidence(logits)
        adaptive_threshold.update_stats(conf, preds, ground_truth=masks)


def validate(model, valloader, args, device):
    """Validate and return per-class IoU and mIoU.

    DINOv2 needs inputs divisible by 14, so for --model dino we run patch-padded
    whole-image inference, switching to sliding-window for large images
    (Cityscapes). ResNet models use the original single-pass argmax.
    """
    model.eval()
    metrics = SegmentationMetrics(args.num_classes)

    is_dino = (getattr(args, 'model', None) == 'dino')
    slide = is_dino and args.dataset == 'cityscapes'
    if is_dino:
        from core.inference import whole_inference, slide_inference

    # DINOv2 was pretrained in bf16; fp16 autocast can overflow attention.
    autocast_dtype = torch.bfloat16 if is_dino else torch.float16

    with torch.no_grad(), torch.amp.autocast(
            'cuda', dtype=autocast_dtype, enabled=(device.type == 'cuda')):
        for batch in valloader:
            if len(batch) == 2:
                images, masks = batch
            else:
                images, masks, *_ = batch
            images = images.to(device)
            masks = masks.numpy() if not isinstance(masks, np.ndarray) else masks

            if slide:
                out = slide_inference(model, images, args.crop_size, args.num_classes)
            elif is_dino:
                out = whole_inference(model, images)
            else:
                out = model(images)
            preds = torch.argmax(out, dim=1).cpu().numpy()
            metrics.add_batch(preds, masks)

    per_class_iou, miou = metrics.evaluate()
    return per_class_iou, miou


def save_checkpoint(state, path):
    torch.save(state, path)


def train_full(args, device):
    """Full class-adaptive semi-supervised training pipeline."""
    os.makedirs(args.save_path, exist_ok=True)

    # Logging (file + console)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.save_path, 'out.log')),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logger.info(f'Arguments: {args}')

    # TensorBoard
    tb_writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=args.save_path)
        logger.info(f'TensorBoard: {args.save_path}')
    except ImportError:
        logger.warning('TensorBoard not available.')

    # W&B (optional)
    wandb_run = None
    if getattr(args, 'use_wandb', False):
        try:
            import wandb
            wandb_run = wandb.init(project='class-adaptive-threshold', config=vars(args))
        except ImportError:
            pass

    class_names = get_class_names(args.dataset, args.num_classes)

    # Model
    model, optimizer = init_model(args, device)
    logger.info(f'Model: {args.model} ({args.backbone}), {count_params(model):.1f}M params')

    # Supervised loss
    criterion = CrossEntropyLoss(ignore_index=255).to(device)

    # EMA teacher
    student = model.module if isinstance(model, DataParallel) else model
    teacher = EMATeacher(student, decay=0.996).to(device)

    # Adaptive threshold module. The method flag selects the regime:
    #   cwbass     -- original IJCNN 2025 dynamic threshold (never exits warmup;
    #                 no floor applied for backward compat)
    #   cwbass_v2  -- dynamic threshold + self-adaptive floor (Theorem 3.1) +
    #                 held-out calibration (gating done above in get_dataloaders)
    #   class_adaptive -- per-class adaptive thresholds (preliminary investigation;
    #                     see CW-BASS v2 Appendix for negative-result discussion)
    method = getattr(args, 'method', 'class_adaptive')
    # cwbass/cwbass_v2 hold the global dynamic-threshold mode for the whole run;
    # freematch/softmatch are themselves global self-adaptive rules with no
    # per-class-risk warmup, so they too skip the class_adaptive warmup switch.
    if method in ('cwbass', 'cwbass_v2', 'freematch', 'softmatch'):
        warmup = args.epochs
    else:
        warmup = args.warmup_epochs
    adaptive_threshold = ClassAdaptiveThreshold(
        num_classes=args.num_classes,
        warmup_epochs=warmup,
        ema_decay=args.ema_decay,
        lambda_coverage=args.lambda_coverage,
        # Legacy Hoeffding confidence parameter. The deployed per-class rule uses
        # the held-out point estimate of Eq. (5), not a concentration bound, so
        # this is inert; kept only so the archived bound utility still runs.
        delta=getattr(args, 'delta', 0.05),
        min_threshold=getattr(args, 'min_threshold', 0.3),
        max_threshold=getattr(args, 'max_threshold', 0.95),
        base_threshold=args.base_threshold,
        beta_sigmoid=args.beta,
        rarity_power=getattr(args, 'rarity_power', 1.0),
        floor_momentum=getattr(args, 'floor_momentum', 0.99),
        floor_scale=getattr(args, 'floor_scale', 0.95),
        apply_floor_in_warmup=(method == 'cwbass_v2'),
        method=method,
        freematch_momentum=getattr(args, 'freematch_momentum', 0.999),
        softmatch_ema=getattr(args, 'softmatch_ema', 0.999),
        softmatch_n_sigma=getattr(args, 'softmatch_n_sigma', 2.0),
    ).to(device)

    # PixCon auxiliary (UniMatch V2 + pixel contrastive). The head's params are
    # added to the existing optimizer at the decoder LR so layer-wise schedules
    # in _set_lr continue to apply uniformly.
    pix_head, pix_bank = None, None
    if getattr(args, 'use_pixcon', False) and getattr(args, 'scheme', 'cwbass_v2') == 'unimatch_v2':
        from core.contrastive import PixContrastiveHead, ClassMemoryBank
        underlying_model = model.module if isinstance(model, DataParallel) else model
        # Decoder fused feature width comes from the segmentor.
        decoder_dim = underlying_model.head[0].in_channels
        pix_head = PixContrastiveHead(
            in_channels=decoder_dim,
            hidden=getattr(args, 'pixcon_dim', 256),
            out_dim=getattr(args, 'pixcon_dim', 256),
        ).to(device)
        optimizer.add_param_group({
            'params': list(pix_head.parameters()),
            'lr': getattr(args, 'decoder_lr', 1e-3),
        })
        pix_bank = ClassMemoryBank(
            num_classes=args.num_classes,
            dim=getattr(args, 'pixcon_dim', 256),
            size_per_class=getattr(args, 'pixcon_bank_size', 256),
            device=device,
        )
        logger.info(f'PixCon enabled: weight={getattr(args, "pixcon_weight", 0.1)}, '
                    f'temp={getattr(args, "pixcon_temp", 0.1)}, '
                    f'bank={getattr(args, "pixcon_bank_size", 256)}/class, '
                    f'dim={getattr(args, "pixcon_dim", 256)}, '
                    f'bank_filter={getattr(args, "pixcon_bank_filter", "clean")}')

    # Data
    labeled_loader, unlabeled_loader, valloader, calib_loader = get_dataloaders(args)
    n_calib = len(calib_loader.dataset) if calib_loader is not None else 0
    logger.info(f'Labeled: {len(labeled_loader.dataset)} imgs, '
                f'Unlabeled: {len(unlabeled_loader.dataset)} imgs, '
                f'Val: {len(valloader.dataset)} imgs, '
                f'Calibration: {n_calib} imgs')
    logger.info(f'Iters/epoch: {len(unlabeled_loader)}')

    # Resume
    start_epoch = 0
    best_miou, best_miou_ema = 0.0, 0.0
    best_epoch, best_epoch_ema = 0, 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt)
        # strict=True so a wrong-architecture checkpoint fails LOUDLY instead of
        # silently leaving most of the network at random init (which would burn
        # 10h of training before anyone noticed).
        target = model.module if isinstance(model, DataParallel) else model
        try:
            target.load_state_dict(sd, strict=True)
        except RuntimeError as e:
            raise RuntimeError(
                f'Resume checkpoint at {args.resume} does not match the current '
                f'model architecture. Either delete the checkpoint dir to start '
                f'fresh, or resume from a compatible checkpoint. Underlying '
                f'error:\n{str(e)[:500]}'
            )
        if 'optimizer_state_dict' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except ValueError:
                logger.warning('Optimizer state dict mismatch — skipping optimizer resume')
        if 'adaptive_threshold' in ckpt:
            adaptive_threshold.load_state_dict(ckpt['adaptive_threshold'])
        if 'teacher' in ckpt:
            teacher.load_state_dict(ckpt['teacher'])
        start_epoch = ckpt.get('epoch', 0) + 1
        # Restore best-so-far if the checkpoint has it (added 2026-05-29 bugfix).
        # Older checkpoints stored the CURRENT-epoch mIoU under 'mIOU'/'mIOU_ema',
        # which would silently corrupt best-tracking on resume; fall back to those
        # only if the new fields are missing.
        best_miou = ckpt.get('best_miou', ckpt.get('mIOU', 0.0))
        best_miou_ema = ckpt.get('best_miou_ema', ckpt.get('mIOU_ema', 0.0))
        best_epoch = ckpt.get('best_epoch', 0)
        best_epoch_ema = ckpt.get('best_epoch_ema', 0)
        logger.info(f'Resumed from epoch {start_epoch} | '
                    f'best EMA so far {best_miou_ema:.2f}% (ep {best_epoch_ema})')

    # Training loop (with early stopping)
    patience = 20  # stop after 20 epochs without EMA improvement
    epochs_without_improvement = 0

    for epoch in range(start_epoch, args.epochs):
        logger.info(f'\n{"="*60}')
        logger.info(f'Epoch {epoch+1}/{args.epochs}  '
                     f'({"warmup" if adaptive_threshold.in_warmup else "per-class adaptive"})')
        logger.info(f'{"="*60}')

        epoch_start = time.time()
        global_step = epoch * len(unlabeled_loader)

        # Unbiased per-class noise estimation on the held-out calibration set.
        # Runs after step_epoch()'s reset (which happened at the end of the
        # previous epoch), so noise_estimator stats reflect the current model.
        # Calibration runs once per epoch when enabled. For class_adaptive it
        # runs only after warmup (since the per-class noise estimator is unused
        # during warmup). For cwbass_v2 it runs every epoch (the floor's
        # confidence-quantile tracker benefits from a fresh per-epoch update).
        run_calib = calib_loader is not None and (
            method == 'cwbass_v2' or not adaptive_threshold.in_warmup
        )
        if run_calib:
            calibrate(teacher, calib_loader, adaptive_threshold, device)

        scheme = getattr(args, 'scheme', 'cwbass_v2')
        if scheme == 'unimatch_v2':
            stats = train_epoch_unimatch_v2(
                model, teacher, labeled_loader, unlabeled_loader,
                optimizer, criterion, args,
                device, epoch, tb_writer=tb_writer, global_step=global_step,
                pix_head=pix_head, pix_bank=pix_bank,
            )
        else:
            stats = train_epoch(
                model, teacher, labeled_loader, unlabeled_loader,
                optimizer, criterion, args, adaptive_threshold,
                device, epoch, tb_writer=tb_writer, global_step=global_step,
            )

        # Evaluate both student and EMA teacher
        logger.info('Evaluating...')
        per_class_iou, miou = validate(model, valloader, args, device)
        per_class_iou_ema, miou_ema = validate(teacher, valloader, args, device)
        miou_pct = miou * 100
        miou_ema_pct = miou_ema * 100
        epoch_time = (time.time() - epoch_start) / 60.0

        logger.info(
            f'  Student mIoU: {miou_pct:.2f}%  |  EMA mIoU: {miou_ema_pct:.2f}%  |  '
            f'Loss: {stats["loss"]:.3f} (x={stats["loss_x"]:.3f} s={stats["loss_u"]:.3f})  |  '
            f'Mask: {stats["mask_ratio"]:.4f}  |  {epoch_time:.1f}min'
        )

        # Per-class breakdown
        ious = [(class_names[k], float(per_class_iou[k]) * 100) for k in range(args.num_classes)]
        ious_sorted = sorted(ious, key=lambda x: x[1])
        logger.info(f'  Worst 5: {", ".join(f"{n}={v:.1f}" for n,v in ious_sorted[:5])}')
        logger.info(f'  Best 5:  {", ".join(f"{n}={v:.1f}" for n,v in ious_sorted[-5:])}')

        # TensorBoard epoch logging
        if tb_writer is not None:
            tb_writer.add_scalar('eval/mIoU', miou_pct, epoch + 1)
            tb_writer.add_scalar('eval/mIoU_ema', miou_ema_pct, epoch + 1)
            tb_writer.add_scalar('train/epoch_loss', stats['loss'], epoch + 1)
            tb_writer.add_scalar('train/epoch_loss_x', stats['loss_x'], epoch + 1)
            tb_writer.add_scalar('train/epoch_loss_s', stats['loss_u'], epoch + 1)
            tb_writer.add_scalar('train/epoch_mask_ratio', stats['mask_ratio'], epoch + 1)
            for k in range(args.num_classes):
                tb_writer.add_scalar(f'eval/{class_names[k]}_IoU', float(per_class_iou[k]) * 100, epoch + 1)
                tb_writer.add_scalar(f'eval/{class_names[k]}_IoU_ema', float(per_class_iou_ema[k]) * 100, epoch + 1)
            threshold_stats = adaptive_threshold.get_stats_dict()
            for key, val in threshold_stats.items():
                if isinstance(val, (int, float)):
                    tb_writer.add_scalar(key, val, epoch + 1)
            tb_writer.flush()

        # W&B
        if wandb_run is not None:
            import wandb
            wandb.log({
                'epoch': epoch + 1,
                'eval/mIoU': miou_pct,
                'eval/mIoU_ema': miou_ema_pct,
                **{f'train/{k}': v for k, v in stats.items()},
            })

        # Update best-tracking FIRST so all checkpoints saved this epoch reflect
        # the true best-so-far (this avoids the resume bug where latest.pth
        # carried stale best metadata, which then overwrote best_ema.pth with
        # a worse checkpoint after a session restart).
        new_best_student = miou_pct > best_miou
        new_best_ema = miou_ema_pct > best_miou_ema
        if new_best_student:
            best_miou = miou_pct
            best_epoch = epoch + 1
        if new_best_ema:
            best_miou_ema = miou_ema_pct
            best_epoch_ema = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        ckpt_state = {
            'epoch': epoch,
            'best_miou': best_miou,
            'best_miou_ema': best_miou_ema,
            'best_epoch': best_epoch,
            'best_epoch_ema': best_epoch_ema,
            'model_state_dict': (model.module.state_dict()
                                 if isinstance(model, DataParallel) else model.state_dict()),
            'teacher': teacher.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'adaptive_threshold': adaptive_threshold.state_dict(),
            'mIOU': miou_pct,
            'mIOU_ema': miou_ema_pct,
        }

        # latest.pth every epoch
        save_checkpoint(ckpt_state, os.path.join(args.save_path, 'latest.pth'))

        if new_best_student:
            save_checkpoint(ckpt_state, os.path.join(args.save_path, 'best.pth'))
            logger.info(f'  >>> New best student: {best_miou:.2f}% (epoch {best_epoch})')

        if new_best_ema:
            save_checkpoint(ckpt_state, os.path.join(args.save_path, 'best_ema.pth'))
            logger.info(f'  >>> New best EMA: {best_miou_ema:.2f}% (epoch {best_epoch_ema})')

        if epochs_without_improvement >= patience and epoch + 1 > args.warmup_epochs + 10:
            logger.info(f'  Early stopping: no EMA improvement for {patience} epochs')
            break

    # Final results — reload best EMA checkpoint for per-class IoU
    best_ema_path = os.path.join(args.save_path, 'best_ema.pth')
    if os.path.exists(best_ema_path):
        best_ckpt = torch.load(best_ema_path, map_location='cpu', weights_only=False)
        teacher.load_state_dict(best_ckpt['teacher'])
        logger.info(f'Loaded best EMA checkpoint (epoch {best_epoch_ema}) for final evaluation')

    final_iou_ema, final_miou_ema = validate(teacher, valloader, args, device)
    _, final_miou = validate(model, valloader, args, device)
    results = {
        'best_miou': best_miou,
        'best_epoch': best_epoch,
        'best_miou_ema': best_miou_ema,
        'best_epoch_ema': best_epoch_ema,
        'final_miou': float(final_miou) * 100,
        'final_miou_ema': float(final_miou_ema) * 100,
        'per_class_iou': {class_names[k]: float(final_iou_ema[k]) * 100 for k in range(args.num_classes)},
        'args': vars(args),
    }
    with open(os.path.join(args.save_path, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    if tb_writer is not None:
        tb_writer.close()
    if wandb_run is not None:
        import wandb
        wandb.finish()

    logger.info(f'\nDone. Best student: {best_miou:.2f}% (ep {best_epoch}), '
                f'Best EMA: {best_miou_ema:.2f}% (ep {best_epoch_ema})')
    logger.info(f'TensorBoard: tensorboard --logdir {args.save_path}')
