"""Bundle loader, pair sampler, and metric helpers for the demo notebook.

Extracted from the project repository for this review bundle. See README
for the provenance note.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


DATA_DIR = Path(__file__).parent / "data"


def load_bundle() -> Dict[str, np.ndarray]:
    """Load the bundled CIFAR-10/WRN signals and metadata."""
    with open(DATA_DIR / "meta.json") as f:
        meta = json.load(f)
    return {
        "shadow_logits": np.load(DATA_DIR / "shadow_logits.npy"),
        "shadow_mask": np.load(DATA_DIR / "shadow_mask.npy"),
        "target_logits": np.load(DATA_DIR / "target_logits.npy"),
        "target_mask": np.load(DATA_DIR / "target_mask.npy"),
        "meta": meta,
    }


def sample_pairs(n_pairs_total: int,
                 n_pairs_to_sample: int,
                 rng: np.random.Generator,
                 exclude_pair: int | None = None) -> np.ndarray:
    """Return shadow column indices for ``n_pairs_to_sample`` randomly chosen
    antithetic pairs, optionally excluding one pair (the held-out target)."""
    available = np.arange(n_pairs_total)
    if exclude_pair is not None:
        available = np.delete(available, exclude_pair)
    chosen = rng.choice(available, size=n_pairs_to_sample, replace=False)
    return np.sort(np.concatenate([2 * chosen, 2 * chosen + 1]))


def metrics(y_true: np.ndarray, scores: np.ndarray,
            fpr_targets=(0.001, 0.01, 0.1)) -> Dict[str, float]:
    """AUC + TPR at a set of FPR thresholds (interpolated on the empirical ROC)."""
    valid = np.isfinite(scores)
    y, s = y_true[valid], scores[valid]
    fpr, tpr, _ = roc_curve(y, s)
    out = {"AUC": roc_auc_score(y, s)}
    for f in fpr_targets:
        out[f"TPR@{f}"] = float(np.interp(f, fpr, tpr))
    return out


def interp_roc(y_true: np.ndarray, scores: np.ndarray,
               fpr_grid: np.ndarray) -> np.ndarray:
    """Interpolate empirical TPR(FPR) onto a common log-spaced FPR grid."""
    valid = np.isfinite(scores)
    fpr, tpr, _ = roc_curve(y_true[valid], scores[valid])
    return np.interp(fpr_grid, fpr, tpr)
