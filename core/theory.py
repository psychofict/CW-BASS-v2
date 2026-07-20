"""Core theoretical components for class-adaptive thresholding.

Implements:
- PerClassNoiseEstimator: Hoeffding-based noise rate bounds per class
- BetaDistributionFitter: Fits Beta distributions to per-class confidence
- TheoreticalRiskMinimizer: Solves the optimal threshold problem
"""

import math
import numpy as np
from scipy import stats as sp_stats


class PerClassNoiseEstimator:
    """Tracks per-class noise rates from labeled data using Hoeffding bounds.

    For each class k, maintains empirical noise rates and computes:
        epsilon_k(tau) <= epsilon_hat_k(tau) + sqrt(ln(2K/delta) / (2 * n_k(tau)))

    Args:
        num_classes: number of semantic classes K
        delta: confidence parameter for the Hoeffding bound (default 0.05)
    """

    def __init__(self, num_classes, delta=0.05):
        self.num_classes = num_classes
        self.delta = delta
        # Per-class accumulators: (num_correct, num_total) for different threshold bins
        self.num_bins = 50  # discretize threshold range [0, 1]
        self.correct = np.zeros((num_classes, self.num_bins), dtype=np.float64)
        self.total = np.zeros((num_classes, self.num_bins), dtype=np.float64)

    def update_from_labeled(self, confidences, predicted_classes, ground_truth):
        """Update noise statistics from a labeled batch.

        Args:
            confidences: [N] tensor of pixel confidences
            predicted_classes: [N] tensor of predicted class indices
            ground_truth: [N] tensor of ground truth class indices
        """
        confidences = confidences.detach().cpu().numpy()
        predicted_classes = predicted_classes.detach().cpu().numpy()
        ground_truth = ground_truth.detach().cpu().numpy()

        # Ignore void pixels
        valid = ground_truth < self.num_classes
        confidences = confidences[valid]
        predicted_classes = predicted_classes[valid]
        ground_truth = ground_truth[valid]

        correct = (predicted_classes == ground_truth).astype(np.float64)

        for k in range(self.num_classes):
            class_mask = predicted_classes == k
            if not np.any(class_mask):
                continue
            class_conf = confidences[class_mask]
            class_correct = correct[class_mask]

            # For each threshold bin, count pixels above that threshold
            for b in range(self.num_bins):
                tau = (b + 1) / self.num_bins  # threshold from 0.02 to 1.0
                above = class_conf >= tau
                n_above = above.sum()
                if n_above > 0:
                    self.total[k, b] += n_above
                    self.correct[k, b] += class_correct[above].sum()

    def compute_empirical_noise_rate(self, class_k, threshold):
        """Compute empirical noise rate for class k at given threshold.

        Returns:
            (epsilon_hat_k, n_k): empirical noise rate and sample count
        """
        b = min(int(threshold * self.num_bins), self.num_bins - 1)
        n_k = self.total[class_k, b]
        if n_k == 0:
            return 1.0, 0
        error_rate = 1.0 - self.correct[class_k, b] / n_k
        return float(error_rate), int(n_k)

    def compute_hoeffding_bound(self, class_k, threshold, delta=None):
        """Compute Hoeffding upper bound on noise rate.

        epsilon_k(tau) <= epsilon_hat_k(tau) + sqrt(ln(2K/delta) / (2*n_k))

        Returns:
            epsilon_upper_k: upper bound on noise rate (clamped to [0, 1])
        """
        if delta is None:
            delta = self.delta
        eps_hat, n_k = self.compute_empirical_noise_rate(class_k, threshold)
        if n_k == 0:
            return 1.0
        K = self.num_classes
        hoeffding_term = math.sqrt(math.log(2.0 * K / delta) / (2.0 * n_k))
        return min(eps_hat + hoeffding_term, 1.0)

    def reset(self):
        self.correct[:] = 0
        self.total[:] = 0


class BetaDistributionFitter:
    """Fits Beta(alpha_k, beta_k) to per-class confidence distributions.

    Uses method-of-moments with EMA smoothing for online estimation.

    Args:
        num_classes: number of semantic classes K
        ema_decay: decay factor for EMA updates of mean/variance (default 0.999)
        min_samples: minimum samples needed before fitting (default 100)
    """

    def __init__(self, num_classes, ema_decay=0.999, min_samples=100):
        self.num_classes = num_classes
        self.ema_decay = ema_decay
        self.min_samples = min_samples

        # Per-class EMA of mean and variance
        self.mean = np.full(num_classes, 0.5)
        self.var = np.full(num_classes, 0.1)
        self.count = np.zeros(num_classes, dtype=np.int64)
        self.initialized = np.zeros(num_classes, dtype=bool)

    def update(self, confidences, predicted_classes):
        """Update Beta distribution parameters from a batch.

        Args:
            confidences: [N] tensor of pixel confidences
            predicted_classes: [N] tensor of predicted class indices
        """
        confidences = confidences.detach().cpu().numpy().astype(np.float64)
        predicted_classes = predicted_classes.detach().cpu().numpy()

        for k in range(self.num_classes):
            mask = predicted_classes == k
            if not np.any(mask):
                continue
            c = confidences[mask]
            # Clamp to (0, 1) open interval for Beta validity
            c = np.clip(c, 1e-6, 1.0 - 1e-6)
            batch_mean = c.mean()
            batch_var = c.var() if len(c) > 1 else 0.01

            if not self.initialized[k]:
                self.mean[k] = batch_mean
                self.var[k] = max(batch_var, 1e-6)
                self.initialized[k] = True
            else:
                d = self.ema_decay
                self.mean[k] = d * self.mean[k] + (1 - d) * batch_mean
                self.var[k] = d * self.var[k] + (1 - d) * max(batch_var, 1e-6)

            self.count[k] += len(c)

    def _get_alpha_beta(self, class_k):
        """Method-of-moments estimator for Beta(alpha, beta)."""
        mu = self.mean[class_k]
        var = self.var[class_k]
        # Ensure valid parameters
        var = min(var, mu * (1 - mu) - 1e-6)
        var = max(var, 1e-8)

        common = mu * (1 - mu) / var - 1.0
        common = max(common, 1e-4)
        alpha = mu * common
        beta = (1 - mu) * common
        return max(alpha, 0.01), max(beta, 0.01)

    def compute_threshold(self, class_k, epsilon_target):
        """Compute threshold for class k targeting epsilon_target noise rate.

        Uses Beta quantile: tau_k = F_k^{-1}(1 - epsilon_target)

        Args:
            class_k: class index
            epsilon_target: target noise rate (e.g. 0.1)

        Returns:
            tau_k: threshold for class k
        """
        if self.count[class_k] < self.min_samples:
            return None  # not enough data
        alpha, beta = self._get_alpha_beta(class_k)
        tau_k = sp_stats.beta.ppf(1.0 - epsilon_target, alpha, beta)
        return float(tau_k)

    def get_all_thresholds(self, epsilon_target):
        """Compute thresholds for all classes.

        Returns:
            thresholds: [K] numpy array (NaN for classes with insufficient data)
        """
        thresholds = np.full(self.num_classes, np.nan)
        for k in range(self.num_classes):
            tau = self.compute_threshold(k, epsilon_target)
            if tau is not None:
                thresholds[k] = tau
        return thresholds

    def get_distribution_params(self):
        """Return (alpha, beta) arrays for all classes."""
        alphas = np.zeros(self.num_classes)
        betas = np.zeros(self.num_classes)
        for k in range(self.num_classes):
            if self.initialized[k]:
                alphas[k], betas[k] = self._get_alpha_beta(k)
        return alphas, betas

    def reset(self):
        self.mean[:] = 0.5
        self.var[:] = 0.1
        self.count[:] = 0
        self.initialized[:] = False


class TheoreticalRiskMinimizer:
    """Solves the per-class optimal threshold problem.

    Minimizes the risk functional:
        R_k(tau) = epsilon_upper_k(tau) * rho_k(tau) + lambda * (1 - rho_k(tau))

    where:
        epsilon_upper_k(tau) is the Hoeffding upper bound on noise
        rho_k(tau) is the retention rate (fraction of pixels above threshold)
        lambda is the coverage penalty

    Args:
        noise_estimator: PerClassNoiseEstimator instance
        beta_fitter: BetaDistributionFitter instance
        lambda_coverage: coverage penalty weight (default 0.1)
        num_search_points: grid resolution for threshold search (default 100)
    """

    def __init__(self, noise_estimator, beta_fitter, lambda_coverage=0.1,
                 num_search_points=100):
        self.noise_estimator = noise_estimator
        self.beta_fitter = beta_fitter
        self.lambda_coverage = lambda_coverage
        self.num_search_points = num_search_points
        self.num_classes = noise_estimator.num_classes

    def _retention_rate(self, class_k, tau):
        """Compute rho_k(tau) = P(confidence_k >= tau) from fitted Beta."""
        if not self.beta_fitter.initialized[class_k]:
            return 0.5
        alpha, beta = self.beta_fitter._get_alpha_beta(class_k)
        return float(1.0 - sp_stats.beta.cdf(tau, alpha, beta))

    def compute_risk(self, class_k, tau, lambda_k=None):
        """Compute risk R_k(tau) for a given class and threshold.

        R_k(tau) = eps_upper(tau) * rho(tau) + lambda_k * (1 - rho(tau))

        A smaller lambda_k tolerates lower retention, so the minimizer prefers a
        *stricter* threshold for that class — used to protect rare classes from
        noisy pseudo-labels (see ClassAdaptiveThreshold rarity scaling).
        """
        if lambda_k is None:
            lambda_k = self.lambda_coverage
        eps_upper = self.noise_estimator.compute_hoeffding_bound(class_k, tau)
        rho = self._retention_rate(class_k, tau)
        return eps_upper * rho + lambda_k * (1.0 - rho)

    def compute_optimal_threshold(self, class_k, lambda_k=None):
        """Find tau_k* that minimizes R_k(tau) via grid search."""
        best_tau = 0.5
        best_risk = float('inf')
        for i in range(1, self.num_search_points):
            tau = i / self.num_search_points
            risk = self.compute_risk(class_k, tau, lambda_k=lambda_k)
            if risk < best_risk:
                best_risk = risk
                best_tau = tau
        return best_tau

    def compute_all_optimal_thresholds(self, lambda_per_class=None):
        """Compute optimal thresholds for all classes.

        Args:
            lambda_per_class: optional [K] array of per-class coverage penalties.
                If None, uses the scalar self.lambda_coverage for every class.
        Returns:
            thresholds: [K] numpy array of optimal thresholds
        """
        thresholds = np.zeros(self.num_classes)
        for k in range(self.num_classes):
            lam = None if lambda_per_class is None else float(lambda_per_class[k])
            thresholds[k] = self.compute_optimal_threshold(k, lambda_k=lam)
        return thresholds

    def compute_bound_tightness(self, class_k, tau=None):
        """Compute tightness metrics for the Hoeffding bound.

        Returns:
            dict with 'empirical', 'bound', 'gap' keys
        """
        if tau is None:
            tau = self.compute_optimal_threshold(class_k)

        eps_hat, n_k = self.noise_estimator.compute_empirical_noise_rate(class_k, tau)
        eps_upper = self.noise_estimator.compute_hoeffding_bound(class_k, tau)

        return {
            'empirical': eps_hat,
            'bound': eps_upper,
            'gap': eps_upper - eps_hat,
            'n_k': n_k,
            'threshold': tau,
        }
