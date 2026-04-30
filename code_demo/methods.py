"""Attack scoring functions used in the demo.

Numpy rewrite of the project's vectorised implementations for this review
bundle. Same likelihood formulae, same NIG hyperparameters; the original RMIA
used a torch outer product on GPU, which is replaced here by a numpy-batched
equivalent. The notebook's final cell checks agreement against scores
produced by the original code on the same data.

Online uses both IN and OUT shadow signals; offline uses OUT only and replaces
the IN class with a global prior estimated from the shadow population.

Methods
-------
LiRA       — Carlini et al., 2022. Gaussian on log-odds, per-sample MLE
             variance once K >= 2 * fix_var_threshold.
BASE1–4    — exponential / Gaussian variants from the paper (appendix).
             BASE4 is equivalent to LiRA online.
RMIA       — Zarifzadeh et al., 2023.
BaVarIA-n  — Gaussian LLR using the NIG posterior mean for variance.
BaVarIA-t  — NIG posterior-predictive Student-t.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, t as student_t


EPS = 1e-10


# =============================================================================
# Signal transforms
# =============================================================================

def confidence_from_logits(logits: np.ndarray) -> np.ndarray:
    """p = sigmoid(logit). Stable form."""
    return 1.0 / (1.0 + np.exp(-logits))


def loss_from_confidence(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Cross-entropy loss on the ground-truth label: -log(p)."""
    return -np.log(np.clip(p, eps, 1.0 - eps))


# =============================================================================
# Helpers
# =============================================================================

def _split_in_out(shadow_stats: np.ndarray, shadow_mask: np.ndarray):
    """Return (in_only, out_only) arrays with NaN where the other class lives."""
    mask = shadow_mask.astype(bool)
    s_in = np.where(mask, shadow_stats, np.nan)
    s_out = np.where(~mask, shadow_stats, np.nan)
    return s_in, s_out


def _neglog_mean_negexp(x: np.ndarray, axis: int = 1) -> np.ndarray:
    """-log(mean(exp(-x))), NaN-aware. Used as the pooled centre for the BASE1
    exponential model: for loss values x_j = -log p_j this equals
    -log(mean(p_j)), which is the log of the harmonic-style mean shadow signal
    that BASE1's score subtracts off.
    """
    mn = np.nanmin(x, axis=axis, keepdims=True)
    z = np.exp(mn - x)
    z = np.where(np.isnan(x), 0.0, z)
    n = np.maximum(np.sum(~np.isnan(x), axis=axis, keepdims=True).astype(float), 1.0)
    return (mn - np.log(np.sum(z, axis=axis, keepdims=True) / n)).squeeze(axis)


# =============================================================================
# LiRA
# =============================================================================

def lira(target_logodds: np.ndarray,
         shadow_logodds: np.ndarray,
         shadow_mask: np.ndarray | None = None,
         online: bool = True,
         fix_var_threshold: int = 32) -> np.ndarray:
    """LiRA: Gaussian likelihood ratio on log-odds.

    Switches from a fixed (pooled) variance to per-sample MLE variance once the
    number of shadow models exceeds ``fix_var_threshold * 2``.
    """
    if shadow_mask is None:
        shadow_mask = np.zeros_like(shadow_logodds, dtype=bool)
    mask = shadow_mask.astype(bool)
    n_samples, K = shadow_logodds.shape

    # Fixed (pooled) std fallback. Offline uses OUT-only shadows to match the
    # paper pipeline (where the offline path is fed shadow_logits_out_only).
    flat = shadow_logodds.ravel()
    flat_mask = mask.ravel()
    if online:
        out_std_fixed = np.nanstd(flat[~flat_mask])
        in_std_fixed = np.nanstd(flat[flat_mask])
    else:
        out_std_fixed = np.nanstd(np.where(flat_mask, np.nan, flat))
        in_std_fixed = out_std_fixed

    out_means = np.nanmean(np.where(~mask, shadow_logodds, np.nan), axis=1)
    in_means = np.nanmean(np.where(mask, shadow_logodds, np.nan), axis=1)
    out_stds = np.full(n_samples, out_std_fixed)
    in_stds = np.full(n_samples, in_std_fixed)
    if K >= fix_var_threshold * 2:
        out_stds = np.nanstd(np.where(~mask, shadow_logodds, np.nan), axis=1)
        if online:
            in_stds = np.nanstd(np.where(mask, shadow_logodds, np.nan), axis=1)

    if online:
        log_p_out = norm.logpdf(target_logodds, out_means, out_stds + EPS)
        log_p_in = norm.logpdf(target_logodds, in_means, in_stds + EPS)
        return log_p_in - log_p_out
    # Offline: Carlini's surrogate — score = log P(X <= target | OUT).
    return norm.logcdf(target_logodds, out_means, out_stds + EPS)


# =============================================================================
# BASE1–4 (Friends of the Exponential Family)
# =============================================================================

def base1(target_loss: np.ndarray,
          shadow_loss: np.ndarray,
          shadow_mask: np.ndarray | None = None,
          online: bool = True) -> np.ndarray:
    """BASE1: pooled exponential on the loss.

    score_i = -loss_i + log(mean(exp(-pooled_loss_j))) at sample i.
    Online pools across all shadow models; offline pools across OUT only.
    """
    if not online and shadow_mask is not None:
        shadow_loss = np.where(shadow_mask.astype(bool), np.nan, shadow_loss)
    pooled_center = _neglog_mean_negexp(shadow_loss, axis=1)
    return -target_loss + pooled_center


def base2(target_logodds: np.ndarray,
          shadow_logodds: np.ndarray,
          shadow_mask: np.ndarray | None = None,
          online: bool = True) -> np.ndarray:
    """BASE2: pooled Gaussian on log-odds.

    score_i = (target_i - mu_pool_i) / var_pool_i
    """
    if not online and shadow_mask is not None:
        shadow_logodds = np.where(shadow_mask.astype(bool), np.nan, shadow_logodds)
    mu = np.nanmean(shadow_logodds, axis=1)
    var = np.maximum(np.nanvar(shadow_logodds, axis=1), EPS)
    return (target_logodds - mu) / var


def base3(target_logodds: np.ndarray,
          shadow_logodds: np.ndarray,
          shadow_mask: np.ndarray) -> np.ndarray:
    """BASE3: separate means, pooled variance (Gaussian, online only).

    score_i = ((mu_in - mu_out) / var_pool) * (target - (mu_in + mu_out)/2).
    """
    s_in, s_out = _split_in_out(shadow_logodds, shadow_mask)
    mu_in = np.nanmean(s_in, axis=1)
    mu_out = np.nanmean(s_out, axis=1)
    n_in = np.sum(~np.isnan(s_in), axis=1).astype(float)
    n_out = np.sum(~np.isnan(s_out), axis=1).astype(float)
    var_in = np.nanvar(s_in, axis=1)
    var_out = np.nanvar(s_out, axis=1)
    var_pool = np.maximum((n_in * var_in + n_out * var_out) / np.maximum(n_in + n_out, 1.0), EPS)
    return (mu_in - mu_out) / var_pool * (target_logodds - 0.5 * (mu_in + mu_out))


def base4(target_logodds: np.ndarray,
          shadow_logodds: np.ndarray,
          shadow_mask: np.ndarray) -> np.ndarray:
    """BASE4: class-conditional Gaussian (equivalent to LiRA online)."""
    s_in, s_out = _split_in_out(shadow_logodds, shadow_mask)
    mu_in = np.nanmean(s_in, axis=1)
    mu_out = np.nanmean(s_out, axis=1)
    var_in = np.maximum(np.nanvar(s_in, axis=1), EPS)
    var_out = np.maximum(np.nanvar(s_out, axis=1), EPS)
    std_in = np.sqrt(var_in)
    std_out = np.sqrt(var_out)
    return (
        0.5 * (target_logodds - mu_out) ** 2 / var_out
        - 0.5 * (target_logodds - mu_in) ** 2 / var_in
        + np.log(std_out / std_in)
    )


# =============================================================================
# RMIA (numpy, batched — no torch required)
# =============================================================================

def rmia(target_conf: np.ndarray,
         shadow_conf: np.ndarray,
         shadow_mask: np.ndarray | None = None,
         online: bool = True,
         offline_a: float = 0.33,
         gamma: float = 1.0,
         z_sample_size: int | None = None,
         seed: int = 0,
         batch_size: int = 2048) -> np.ndarray:
    """RMIA: relativity-boosted likelihood ratio (Zarifzadeh et al., 2023).

    Uses ground-truth-label confidence as the signal. Z reference set is a
    random subsample of the audit population to keep memory bounded.
    """
    rng = np.random.default_rng(seed)
    n = target_conf.shape[0]
    if z_sample_size is None:
        z_sample_size = min(5000, n)
    z_idx = rng.choice(n, size=z_sample_size, replace=False)

    def _ratio(idx):
        p_t = target_conf[idx]
        p_s = shadow_conf[idx, :]
        if online:
            p_prior = np.mean(p_s, axis=1)
        else:
            mask_in = shadow_mask[idx, :].astype(bool)
            p_out_only = np.where(mask_in, np.nan, p_s)
            p_prior_out = np.nanmean(p_out_only, axis=1)
            p_prior = 0.5 * ((offline_a + 1.0) * p_prior_out + (1.0 - offline_a))
        return p_t / (p_prior + EPS)

    ratio_x_full = _ratio(np.arange(n))   # (n,)
    ratio_z = _ratio(z_idx)               # (n_z,)

    scores = np.empty(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        block = ratio_x_full[start:stop, None] / (ratio_z[None, :] + EPS)
        scores[start:stop] = np.mean(block > gamma, axis=1).astype(np.float32)
    return scores


# =============================================================================
# BaVarIA — NIG posterior on (μ, σ²)
# =============================================================================

def _nig_priors(shadow_logodds: np.ndarray,
                shadow_mask: np.ndarray,
                alpha0: float):
    """Empirical-Bayes NIG hyperparameters from the pooled shadow population.

    Sets μ₀ to the global IN/OUT means, and β₀ to match the global variance via
    E[σ²] = β₀ / (α₀ - 1).
    """
    mask = shadow_mask.astype(bool)
    s_in = shadow_logodds[mask].ravel()
    s_out = shadow_logodds[~mask].ravel()
    s_in = s_in[~np.isnan(s_in)]
    s_out = s_out[~np.isnan(s_out)]
    mu0_in = float(np.mean(s_in)) if s_in.size else 0.0
    mu0_out = float(np.mean(s_out)) if s_out.size else 0.0
    var_in = float(np.var(s_in)) if s_in.size else 1.0
    var_out = float(np.var(s_out)) if s_out.size else 1.0
    if alpha0 > 1.0:
        beta0_in = max(var_in * (alpha0 - 1.0), EPS)
        beta0_out = max(var_out * (alpha0 - 1.0), EPS)
    else:
        beta0_in = max(var_in, EPS)
        beta0_out = max(var_out, EPS)
    return mu0_in, mu0_out, beta0_in, beta0_out


def _nig_posterior(x_per_sample, mu0, beta0, kappa0, alpha0):
    """Vectorised posterior NIG params; rows with all-NaN fall back to prior."""
    n = np.sum(~np.isnan(x_per_sample), axis=1).astype(float)
    s = np.nansum(x_per_sample, axis=1)
    xbar = np.where(n > 0, s / np.maximum(n, 1.0), mu0)
    kappa_n = kappa0 + n
    alpha_n = alpha0 + n / 2.0
    ss = np.nansum((x_per_sample - xbar[:, None]) ** 2, axis=1)
    beta_n = beta0 + 0.5 * ss + 0.5 * kappa0 * n * (xbar - mu0) ** 2 / kappa_n
    return xbar, kappa_n, alpha_n, beta_n


def bavaria_n(target_logodds: np.ndarray,
              shadow_logodds: np.ndarray,
              shadow_mask: np.ndarray,
              online: bool = True,
              kappa0: float = 1.0,
              alpha0: float = 1.0) -> np.ndarray:
    """BaVarIA-n: Normal LLR with NIG posterior-mean variance shrinkage."""
    mu0_in, mu0_out, beta0_in, beta0_out = _nig_priors(shadow_logodds, shadow_mask, alpha0)
    s_in, s_out = _split_in_out(shadow_logodds, shadow_mask)

    if online:
        xbar_in, kappa_n_in, alpha_n_in, beta_n_in = _nig_posterior(
            s_in, mu0_in, beta0_in, kappa0, alpha0)
    else:
        # Offline: IN class collapses to the prior (no IN observations seen).
        xbar_in = np.full_like(target_logodds, mu0_in, dtype=float)
        beta_n_in = np.full_like(target_logodds, beta0_in, dtype=float)
        alpha_n_in = np.full_like(target_logodds, alpha0, dtype=float)

    xbar_out, kappa_n_out, alpha_n_out, beta_n_out = _nig_posterior(
        s_out, mu0_out, beta0_out, kappa0, alpha0)

    var_in = np.maximum(np.where(alpha_n_in > 1, beta_n_in / (alpha_n_in - 1), beta_n_in), EPS)
    var_out = np.maximum(np.where(alpha_n_out > 1, beta_n_out / (alpha_n_out - 1), beta_n_out), EPS)

    scores = (
        0.5 * np.log(var_out / var_in)
        - 0.5 * (target_logodds - xbar_in) ** 2 / var_in
        + 0.5 * (target_logodds - xbar_out) ** 2 / var_out
    )
    scores[~np.isfinite(scores)] = 0.0
    return scores


def bavaria_t(target_logodds: np.ndarray,
              shadow_logodds: np.ndarray,
              shadow_mask: np.ndarray,
              online: bool = True,
              kappa0: float = 1.0,
              alpha0: float = 2.0) -> np.ndarray:
    """BaVarIA-t: NIG posterior predictive (Student-t) LLR.

    α₀ defaults to 2 so the prior predictive has finite variance.
    """
    mu0_in, mu0_out, beta0_in, beta0_out = _nig_priors(shadow_logodds, shadow_mask, alpha0)
    s_in, s_out = _split_in_out(shadow_logodds, shadow_mask)

    def _predictive(target, mu_n, kappa_n, alpha_n, beta_n):
        nu = 2.0 * alpha_n
        scale = np.sqrt(np.maximum(beta_n * (kappa_n + 1.0) / (alpha_n * kappa_n), EPS))
        return student_t.logpdf(target, nu, loc=mu_n, scale=scale)

    if online:
        xbar_in, kappa_n_in, alpha_n_in, beta_n_in = _nig_posterior(
            s_in, mu0_in, beta0_in, kappa0, alpha0)
        n_in = np.sum(~np.isnan(s_in), axis=1).astype(float)
        sum_in = np.nansum(s_in, axis=1)
        mu_n_in = (kappa0 * mu0_in + sum_in) / kappa_n_in
        log_p_in = _predictive(target_logodds, mu_n_in, kappa_n_in, alpha_n_in, beta_n_in)
    else:
        log_p_in = _predictive(
            target_logodds,
            np.full_like(target_logodds, mu0_in, dtype=float),
            np.full_like(target_logodds, kappa0, dtype=float),
            np.full_like(target_logodds, alpha0, dtype=float),
            np.full_like(target_logodds, beta0_in, dtype=float),
        )

    xbar_out, kappa_n_out, alpha_n_out, beta_n_out = _nig_posterior(
        s_out, mu0_out, beta0_out, kappa0, alpha0)
    n_out = np.sum(~np.isnan(s_out), axis=1).astype(float)
    sum_out = np.nansum(s_out, axis=1)
    mu_n_out = (kappa0 * mu0_out + sum_out) / kappa_n_out
    log_p_out = _predictive(target_logodds, mu_n_out, kappa_n_out, alpha_n_out, beta_n_out)

    scores = log_p_in - log_p_out
    scores[~np.isfinite(scores)] = 0.0
    return scores
