"""Class-adaptive thresholding module for semi-supervised segmentation.

Wraps the theoretical components (Hoeffding bounds, Beta fitting, risk minimization)
into a training-compatible nn.Module with registered buffers.
"""

import torch
import torch.nn as nn
import numpy as np

from core.theory import PerClassNoiseEstimator, BetaDistributionFitter, TheoreticalRiskMinimizer


class ClassAdaptiveThreshold(nn.Module):
    """Per-class adaptive threshold module.

    During warmup, falls back to the global CW-BASS dynamic threshold.
    After warmup, computes per-class thresholds via Hoeffding-based risk minimization.

    Registered buffers (shape [K]):
        per_class_mean: EMA of per-class mean confidence
        per_class_var: EMA of per-class confidence variance
        per_class_count: cumulative pixel count per class
        per_class_threshold: current threshold per class

    Args:
        num_classes: K
        warmup_epochs: epochs before switching from global to per-class (default 10)
        ema_decay: decay for EMA stats (default 0.999)
        lambda_coverage: coverage penalty in risk functional (default 0.1)
        delta: Hoeffding confidence parameter (default 0.05)
        min_threshold: lower clamp (default 0.3)
        max_threshold: upper clamp (default 0.95)
        min_pixels_per_class: minimum pixels before per-class threshold activates (default 100)
        base_threshold: global threshold for warmup / fallback (default 0.6)
        beta_sigmoid: beta parameter for sigmoid in global threshold (default 0.5)
    """

    def __init__(self, num_classes, warmup_epochs=10, ema_decay=0.999,
                 lambda_coverage=0.1, delta=0.05, min_threshold=0.3,
                 max_threshold=0.95, min_pixels_per_class=100,
                 base_threshold=0.6, beta_sigmoid=0.5,
                 rarity_power=1.0, lambda_cap=10.0,
                 floor_momentum=0.99, floor_scale=0.95,
                 apply_floor_in_warmup=False,
                 method='class_adaptive',
                 freematch_momentum=0.999, softmatch_ema=0.999,
                 softmatch_n_sigma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.warmup_epochs = warmup_epochs
        self.ema_decay = ema_decay
        self.lambda_coverage = lambda_coverage
        self.delta = delta
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.min_pixels_per_class = min_pixels_per_class
        self.base_threshold = base_threshold
        self.beta_sigmoid = beta_sigmoid
        # Selection rule. The default path is the CW-BASS family
        # (class_adaptive / cwbass / cwbass_v2); 'freematch' and 'softmatch'
        # are the literature baselines run by their own definitions,
        # sharing this module's stat-tracking but using their own rule.
        self.method = method
        self.freematch_momentum = freematch_momentum
        self.softmatch_ema = softmatch_ema
        self.softmatch_n_sigma = softmatch_n_sigma
        # Rarity-scaled coverage penalty: rare classes get larger lambda -> the
        # risk minimizer prefers a lower threshold so their (scarce, lower-
        # confidence) pseudo-labels are not over-filtered.
        self.rarity_power = rarity_power
        self.lambda_cap = lambda_cap
        # Self-adaptive confidence floor (FreeMatch-style). Rises as the model
        # grows confident, preventing the global threshold collapse where mask
        # ratio drifts upward and pseudo-label noise floods training.
        self.floor_momentum = floor_momentum
        self.floor_scale = floor_scale
        # CW-BASS v2: apply the self-adaptive floor as a lower bound on the
        # dynamic threshold during warmup (or, equivalently, when the per-class
        # adaptation never engages). Eliminates the dynamic-threshold collapse
        # mode where the threshold drifts to its lower clamp.
        self.apply_floor_in_warmup = apply_floor_in_warmup

        # Registered buffers survive .to(device) and state_dict
        self.register_buffer('per_class_mean', torch.full((num_classes,), 0.5))
        self.register_buffer('per_class_var', torch.full((num_classes,), 0.1))
        self.register_buffer('per_class_count', torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer('per_class_threshold', torch.full((num_classes,), base_threshold))
        self.register_buffer('current_epoch', torch.tensor(0, dtype=torch.long))
        # Global self-adaptive confidence floor + per-class learning status (PLS).
        self.register_buffer('confidence_floor', torch.tensor(base_threshold))
        self.register_buffer('per_class_acc', torch.full((num_classes,), 0.5))
        # FreeMatch self-adaptive thresholding (Wang et al., 2023): a global EMA
        # threshold tau_t modulated per class by a "local" learning-status factor
        # MaxNorm(p_tilde_k) = p_tilde_k / max_j p_tilde_j, where p_tilde is the
        # EMA of the per-class mean max-probability over unlabeled pixels.
        self.register_buffer('freematch_tau', torch.tensor(base_threshold))
        self.register_buffer('freematch_p_class', torch.full((num_classes,), 1.0 / num_classes))
        # SoftMatch (Chen et al., 2023): EMA of the unlabeled max-confidence
        # mean/variance, used to form a truncated-Gaussian soft weight in
        # [0,1] in place of a hard mask.
        self.register_buffer('softmatch_mu', torch.tensor(1.0 / num_classes))
        self.register_buffer('softmatch_var', torch.tensor(1.0))

        # Theory components (not nn.Modules — CPU-based numpy)
        self.noise_estimator = PerClassNoiseEstimator(num_classes, delta=delta)
        self.beta_fitter = BetaDistributionFitter(num_classes, ema_decay=ema_decay)
        self.risk_minimizer = TheoreticalRiskMinimizer(
            self.noise_estimator, self.beta_fitter,
            lambda_coverage=lambda_coverage
        )

    @property
    def in_warmup(self):
        return self.current_epoch.item() < self.warmup_epochs

    def global_threshold_fallback(self, confidence):
        """Exact CW-BASS dynamic_thresholding for warmup / rare classes.

        Returns a scalar threshold tensor.
        """
        avg_conf = confidence.mean()
        threshold = self.base_threshold / (1 + torch.exp(-self.beta_sigmoid * (avg_conf - 0.5)))
        return torch.clamp(threshold, min=self.min_threshold, max=self.max_threshold)

    @torch.no_grad()
    def update_stats(self, confidence, predicted_classes, ground_truth=None):
        """Update per-class statistics from a batch.

        Args:
            confidence: [B, H, W] pixel confidence scores
            predicted_classes: [B, H, W] predicted class indices
            ground_truth: [B, H, W] optional ground truth (for noise estimation on labeled data)
        """
        conf_flat = confidence.reshape(-1)
        pred_flat = predicted_classes.reshape(-1)

        # EMA update of per-class mean and variance
        for k in range(self.num_classes):
            mask = pred_flat == k
            if mask.sum() < 10:
                continue

            c = conf_flat[mask]
            batch_mean = c.mean()
            batch_var = c.var() if c.numel() > 1 else torch.tensor(0.01)

            count = mask.sum().item()
            self.per_class_count[k] += count

            if self.per_class_count[k] == count:
                # First time seeing this class
                self.per_class_mean[k] = batch_mean
                self.per_class_var[k] = batch_var
            else:
                d = self.ema_decay
                self.per_class_mean[k] = d * self.per_class_mean[k] + (1 - d) * batch_mean
                self.per_class_var[k] = d * self.per_class_var[k] + (1 - d) * batch_var

        # Update theory components
        self.beta_fitter.update(conf_flat, pred_flat)

        if ground_truth is not None:
            gt_flat = ground_truth.reshape(-1)
            self.noise_estimator.update_from_labeled(conf_flat, pred_flat, gt_flat)
            # Per-class learning status: EMA of labeled prediction accuracy.
            valid = gt_flat < self.num_classes
            if valid.any():
                gv, pv = gt_flat[valid], pred_flat[valid]
                correct = (gv == pv).float()
                for k in range(self.num_classes):
                    km = gv == k
                    if km.sum() >= 10:
                        acc_k = correct[km].mean()
                        d = self.ema_decay
                        self.per_class_acc[k] = d * self.per_class_acc[k] + (1 - d) * acc_k
        else:
            # Self-adaptive floor: EMA of mean confidence on the unlabeled view.
            m = self.floor_momentum
            self.confidence_floor.mul_(m).add_(conf_flat.mean(), alpha=1 - m)

            if self.method == 'freematch':
                # Global self-adaptive threshold + per-class learning-status EMA.
                mf = self.freematch_momentum
                self.freematch_tau.mul_(mf).add_(conf_flat.mean(), alpha=1 - mf)
                for k in range(self.num_classes):
                    km = pred_flat == k
                    if km.sum() >= 10:
                        self.freematch_p_class[k] = (
                            mf * self.freematch_p_class[k]
                            + (1 - mf) * conf_flat[km].mean())
            elif self.method == 'softmatch':
                # EMA of the unlabeled max-confidence mean and variance.
                ms = self.softmatch_ema
                self.softmatch_mu.mul_(ms).add_(conf_flat.mean(), alpha=1 - ms)
                bv = conf_flat.var() if conf_flat.numel() > 1 else torch.tensor(
                    1.0, device=conf_flat.device)
                self.softmatch_var.mul_(ms).add_(bv, alpha=1 - ms)

    def compute_thresholds(self):
        """Compute current thresholds for all classes.

        During warmup: returns uniform global threshold.
        After warmup: returns per-class thresholds from risk minimization,
                      with fallback to global for rare classes.

        Returns:
            [K] tensor of per-class thresholds
        """
        if self.in_warmup:
            # Uniform CW-BASS-style dynamic threshold from per-class mean confidence.
            global_mean = self.per_class_mean.mean()
            threshold = self.base_threshold / (1 + torch.exp(
                -self.beta_sigmoid * (global_mean - 0.5)))
            threshold = torch.clamp(threshold, min=self.min_threshold, max=self.max_threshold)
            # CW-BASS v2: clip the dynamic threshold from below by the
            # self-adaptive confidence floor (Theorem 3.1 in the paper). Active
            # only when explicitly enabled, to preserve backward compat with the
            # original CW-BASS dynamic-threshold behaviour.
            if self.apply_floor_in_warmup:
                floor = (self.confidence_floor * self.floor_scale).clamp(
                    min=self.min_threshold, max=self.max_threshold)
                threshold = torch.maximum(threshold, floor)
            self.per_class_threshold.fill_(float(threshold.item()))
            return self.per_class_threshold.clone()

        device = self.per_class_threshold.device

        # --- Rarity-scaled coverage penalty lambda_k ---
        freq = self.per_class_count.float()
        freq_max = freq.max().clamp(min=1.0)
        lambda_k = self.lambda_coverage * (freq_max / freq.clamp(min=1.0)) ** self.rarity_power
        lambda_k = lambda_k.clamp(max=self.lambda_coverage * self.lambda_cap)

        # --- Theoretical optimal threshold (risk minimization) ---
        optimal = self.risk_minimizer.compute_all_optimal_thresholds(
            lambda_per_class=lambda_k.cpu().numpy())
        tau = torch.from_numpy(optimal).float().to(device)

        # --- Self-adaptive per-class floor (anti-collapse) ---
        # Floor rises with overall confidence; modulated by per-class mean conf so
        # rare/under-learned classes (low conf) keep a lower floor and are not
        # over-filtered, while frequent classes can't drift to the min threshold.
        denom = self.per_class_mean.max().clamp(min=1e-6)
        per_class_floor = self.confidence_floor * (self.per_class_mean / denom) * self.floor_scale
        per_class_floor = per_class_floor.clamp(min=self.min_threshold, max=self.max_threshold)

        # Final threshold: never below the self-adaptive floor.
        tau = torch.maximum(tau, per_class_floor)

        # Rare classes (too few observed pixels): rely on the floor alone.
        rare_mask = self.per_class_count < self.min_pixels_per_class
        tau[rare_mask] = per_class_floor[rare_mask]

        tau = torch.clamp(tau, min=self.min_threshold, max=self.max_threshold)
        self.per_class_threshold.copy_(tau)
        return self.per_class_threshold.clone()

    def apply_thresholds(self, confidence, predicted_classes):
        """Apply spatially-varying per-class thresholds.

        Args:
            confidence: [B, H, W] pixel confidence
            predicted_classes: [B, H, W] predicted class indices

        Returns:
            mask: [B, H, W] boolean mask (True = pixel retained)
        """
        if self.method == 'freematch':
            tau_local = self.freematch_thresholds()  # [K]
            threshold_map = tau_local[predicted_classes]
            return confidence >= threshold_map
        if self.method == 'softmatch':
            # SoftMatch imposes no hard threshold: every valid pixel is retained
            # and instead down-weighted by soft_weight(); the mask is all-True.
            return torch.ones_like(confidence, dtype=torch.bool)
        thresholds = self.compute_thresholds()  # [K]
        # Build spatial threshold map: threshold_map[b,h,w] = thresholds[predicted_classes[b,h,w]]
        threshold_map = thresholds[predicted_classes]  # fancy indexing, [B,H,W]
        mask = confidence > threshold_map
        return mask

    def freematch_thresholds(self):
        """FreeMatch self-adaptive per-class thresholds (Wang et al., 2023).

        tau_k = MaxNorm(p_tilde)_k * tau_global, clamped to [min, max], where
        tau_global is the EMA of mean confidence and p_tilde the per-class EMA of
        mean max-probability. Classes the model is less confident on (lower
        p_tilde_k) get a proportionally lower threshold.
        """
        p = self.freematch_p_class
        maxnorm = p / p.max().clamp(min=1e-6)
        tau = (maxnorm * self.freematch_tau).clamp(self.min_threshold, self.max_threshold)
        self.per_class_threshold.copy_(tau)   # expose for logging
        return tau

    @torch.no_grad()
    def soft_weight(self, confidence, predicted_classes):
        """SoftMatch truncated-Gaussian soft weight in [0,1] (Chen et al., 2023).

        Full weight for pixels at or above the running mean confidence mu;
        Gaussian decay below it with width n_sigma * sqrt(var). Used in place of a
        hard mask (gamma=1, so the loss weight equals this lambda).
        """
        mu = self.softmatch_mu
        sigma = self.softmatch_var.clamp(min=1e-8).sqrt()
        width = (self.softmatch_n_sigma * sigma).clamp(min=1e-4)
        lam = torch.exp(-((confidence - mu) ** 2) / (2.0 * width ** 2))
        lam = torch.where(confidence >= mu, torch.ones_like(lam), lam)
        return lam.clamp(0.0, 1.0)

    def step_epoch(self):
        """Increment epoch counter. Call at end of each epoch.

        Resets the per-class noise histogram so the next epoch's Hoeffding bound
        reflects the *current* model rather than stale, overconfident statistics
        accumulated early in training (a key cause of threshold collapse). The
        Beta fitter keeps its EMA (it is already decayed) and is not reset.
        """
        self.current_epoch += 1
        self.noise_estimator.reset()

    def get_threshold_dict(self, class_names=None):
        """Return {class_name: threshold} for logging."""
        if class_names is None:
            class_names = [f'class_{k}' for k in range(self.num_classes)]
        return {name: float(self.per_class_threshold[k])
                for k, name in enumerate(class_names)}

    def get_stats_dict(self, class_names=None):
        """Return per-class statistics for logging."""
        if class_names is None:
            class_names = [f'class_{k}' for k in range(self.num_classes)]
        stats = {}
        for k, name in enumerate(class_names):
            stats[f'threshold/{name}'] = float(self.per_class_threshold[k])
            stats[f'mean_conf/{name}'] = float(self.per_class_mean[k])
            stats[f'count/{name}'] = int(self.per_class_count[k])
        # Summary stats
        stats['threshold/confidence_floor'] = float(self.confidence_floor)
        active = self.per_class_count > 0
        if active.any():
            stats['threshold/mean'] = float(self.per_class_threshold[active].mean())
            stats['threshold/min'] = float(self.per_class_threshold[active].min())
            stats['threshold/max'] = float(self.per_class_threshold[active].max())
            stats['threshold/mean_coverage'] = float(active.float().mean())
            stats['pls/mean_acc'] = float(self.per_class_acc[active].mean())
        else:
            stats['threshold/mean'] = float(self.base_threshold)
            stats['threshold/min'] = float(self.base_threshold)
            stats['threshold/max'] = float(self.base_threshold)
            stats['threshold/mean_coverage'] = 0.0
        return stats
