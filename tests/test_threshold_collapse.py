"""Tests for the Phase-3 anti-collapse threshold behavior.

The original failure mode: as the model grows overconfident, per-class thresholds
drift *down*, the retention (mask) ratio climbs toward ~0.83, and pseudo-label
noise floods training -> validation mIoU collapses. These tests verify the fixes:

  1. Anti-collapse: feeding ever-more-confident pixels makes the self-adaptive
     floor RISE, so the retention ratio does NOT run away to ~1.0.
  2. Rare-class protection: a rare class with lower confidence ends up with a
     LOWER threshold than a frequent high-confidence class (so it is not
     over-filtered) -- the paper's core thesis.

Run: python -m tests.test_threshold_collapse
"""

import torch

from core.thresholding import ClassAdaptiveThreshold


NCLASS = 5


def _feed(thr, conf_by_class, gt_correct_frac=0.9, n=4000):
    """Simulate one batch: build confidence/pred/gt for given per-class conf.

    conf_by_class: dict {class_idx: mean_confidence}. Generates pixels for those
    classes, runs update_stats once as the unlabeled view (floor update) and once
    as the labeled view (noise/PLS), mimicking the train loop.
    """
    confs, preds = [], []
    for k, mu in conf_by_class.items():
        c = torch.clamp(torch.randn(n) * 0.03 + mu, 0.01, 0.999)
        confs.append(c)
        preds.append(torch.full((n,), k, dtype=torch.long))
    confidence = torch.cat(confs).reshape(1, -1)
    pred = torch.cat(preds).reshape(1, -1)
    # Unlabeled view (drives the self-adaptive floor).
    thr.update_stats(confidence, pred, ground_truth=None)
    # Labeled view: gt mostly matches pred (so noise rate ~ 1-gt_correct_frac).
    gt = pred.clone()
    flip = torch.rand_like(confidence) > gt_correct_frac
    gt[flip] = (gt[flip] + 1) % NCLASS
    thr.update_stats(confidence, pred, ground_truth=gt)


def test_floor_rises_with_confidence():
    thr = ClassAdaptiveThreshold(NCLASS, warmup_epochs=0, floor_momentum=0.9)
    floors = []
    # Confidence ramps up over "epochs", as an improving model would behave.
    # Each epoch issues several batches (a real epoch has hundreds of iters).
    for ep, mu in enumerate([0.6, 0.7, 0.8, 0.9, 0.95]):
        for _ in range(8):
            _feed(thr, {k: mu for k in range(NCLASS)})
        thr.compute_thresholds()
        floors.append(float(thr.confidence_floor))
        thr.step_epoch()
    assert floors[-1] > floors[0] + 0.1, f'floor did not rise: {floors}'
    # Retention must NOT collapse to ~everything: with floor tracking confidence,
    # the threshold stays close to the confidence mean, not at the 0.3 minimum.
    assert thr.per_class_threshold.min() > thr.min_threshold + 0.05, \
        f'thresholds collapsed to floor: {thr.per_class_threshold}'
    print(f'[ok] floor rises {floors[0]:.3f} -> {floors[-1]:.3f}; '
          f'min thr={float(thr.per_class_threshold.min()):.3f}')


def test_rare_class_gets_lower_threshold():
    thr = ClassAdaptiveThreshold(NCLASS, warmup_epochs=0,
                                 min_pixels_per_class=50, floor_momentum=0.9)
    # Class 0: frequent + high confidence. Class 4: rare + lower confidence.
    for _ in range(4):
        _feed(thr, {0: 0.95}, n=8000)       # frequent, confident
        _feed(thr, {1: 0.9, 2: 0.9, 3: 0.9}, n=4000)
        _feed(thr, {4: 0.7}, n=300)          # rare, less confident
        thr.compute_thresholds()
        thr.step_epoch()
    tau = thr.per_class_threshold
    assert tau[4] < tau[0], \
        f'rare class threshold {tau[4]:.3f} not below frequent {tau[0]:.3f}'
    print(f'[ok] rare-class thr={float(tau[4]):.3f} < frequent thr={float(tau[0]):.3f}')


def test_no_runaway_retention():
    """Directly check retention ratio stays bounded under rising confidence."""
    thr = ClassAdaptiveThreshold(NCLASS, warmup_epochs=0, floor_momentum=0.9)
    for mu in [0.7, 0.85, 0.95, 0.98]:
        for _ in range(8):
            _feed(thr, {k: mu for k in range(NCLASS)})
        thr.compute_thresholds()
        thr.step_epoch()
    # Build a fresh high-confidence batch and measure retention.
    conf = torch.clamp(torch.randn(1, 20000) * 0.03 + 0.95, 0.01, 0.999)
    pred = torch.randint(0, NCLASS, (1, 20000))
    mask = thr.apply_thresholds(conf, pred)
    ratio = mask.float().mean().item()
    assert ratio < 0.97, f'retention ran away to {ratio:.3f} (collapse)'
    print(f'[ok] bounded retention under high confidence: {ratio:.3f}')


def test_cwbass_v2_floor_in_warmup():
    """CW-BASS v2 mode: the dynamic threshold should be clipped from below by
    the self-adaptive floor even when the module is permanently in warmup mode."""
    # warmup_epochs=999 => always warmup; apply_floor_in_warmup=True activates
    # the CW-BASS v2 lower-bound mechanism.
    thr = ClassAdaptiveThreshold(NCLASS, warmup_epochs=999,
                                 floor_momentum=0.9,
                                 base_threshold=0.6, beta_sigmoid=0.5,
                                 apply_floor_in_warmup=True)
    # Push high confidence -> floor rises above the base dynamic threshold.
    for _ in range(20):
        _feed(thr, {k: 0.95 for k in range(NCLASS)})
    thr.compute_thresholds()
    t_with_floor = float(thr.per_class_threshold.min())

    # Same dynamics WITHOUT the floor flag: threshold uses base only.
    thr2 = ClassAdaptiveThreshold(NCLASS, warmup_epochs=999,
                                  floor_momentum=0.9,
                                  base_threshold=0.6, beta_sigmoid=0.5,
                                  apply_floor_in_warmup=False)
    for _ in range(20):
        _feed(thr2, {k: 0.95 for k in range(NCLASS)})
    thr2.compute_thresholds()
    t_without_floor = float(thr2.per_class_threshold.min())

    assert t_with_floor > t_without_floor, (
        f'floor should raise the dynamic threshold; '
        f'with_floor={t_with_floor:.3f} vs without_floor={t_without_floor:.3f}')
    print(f'[ok] cwbass_v2 floor raises threshold {t_without_floor:.3f} -> {t_with_floor:.3f}')


if __name__ == '__main__':
    test_floor_rises_with_confidence()
    test_rare_class_gets_lower_threshold()
    test_no_runaway_retention()
    test_cwbass_v2_floor_in_warmup()
    print('\nAll threshold-collapse tests passed.')
