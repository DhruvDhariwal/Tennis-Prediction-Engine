"""Metrics, reference baselines and calibration diagnostics."""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from .features import FEATURE_INDEX, FLIP_SIGN

logger = logging.getLogger(__name__)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 20) -> float:
    """Binned |confidence - accuracy|, weighted by bin population."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    total = len(p)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        ece += mask.sum() / total * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def reliability_curve(
    y: np.ndarray, p: np.ndarray, n_bins: int = 20
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    conf, acc, count = [], [], []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf.append(p[mask].mean())
        acc.append(y[mask].mean())
        count.append(int(mask.sum()))
    return np.array(conf), np.array(acc), np.array(count)


def classification_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.int8)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1 - 1e-9)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, (p >= 0.5).astype(np.int8))),
        "auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
    }


def score_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """AUC/accuracy for a ranking score that is not a calibrated probability."""
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    finite = np.isfinite(score)
    # A baseline that cannot separate (e.g. both players unranked) is scored as
    # a coin flip rather than being silently dropped from the denominator.
    filled = np.where(finite, score, 0.0)
    guess = np.where(filled > 0, 1, np.where(filled < 0, 0, -1))
    rng = np.random.default_rng(0)
    guess = np.where(guess == -1, rng.integers(0, 2, size=len(guess)), guess)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, guess)),
        "auc": float(roc_auc_score(y, filled)),
    }


def elo_probability(X: np.ndarray, feature: str = "blend_elo_diff") -> np.ndarray:
    """Closed-form Elo win probability -- a genuine probabilistic baseline."""
    diff = X[:, FEATURE_INDEX[feature]].astype(np.float64)
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def baseline_suite(X: np.ndarray, y: np.ndarray) -> dict[str, dict]:
    """Reference points every learned model should beat."""
    out: dict[str, dict] = {}

    p_elo = elo_probability(X, "elo_diff")
    out["elo_global"] = classification_metrics(y, p_elo)

    p_blend = elo_probability(X, "blend_elo_diff")
    out["elo_surface_blended"] = classification_metrics(y, p_blend)

    out["atp_rank"] = score_metrics(y, -X[:, FEATURE_INDEX["log_rank_diff"]])
    out["bookmaker_free_coinflip"] = {
        "n": int(len(y)),
        "accuracy": 0.5,
        "auc": 0.5,
        "log_loss": float(np.log(2)),
        "brier": 0.25,
    }
    return out


def symmetry_error(predict: Callable[[np.ndarray], np.ndarray], X: np.ndarray) -> dict:
    """Measure ``|p(A,B) - (1 - p(B,A))|`` on real feature rows.

    The mirrored matrix is produced by the same sign-flip the training data uses,
    so this is a genuine check of the model rather than of the flipping code.
    """
    p_forward = predict(X)
    p_mirror = predict(X * FLIP_SIGN)
    delta = np.abs(p_forward - (1.0 - p_mirror))
    return {
        "mean_abs_error": float(delta.mean()),
        "max_abs_error": float(delta.max()),
        "p99_abs_error": float(np.percentile(delta, 99)),
    }


def slice_metrics(
    y: np.ndarray,
    p: np.ndarray,
    tier: np.ndarray,
    min_prior: np.ndarray,
) -> dict[str, dict]:
    """Break performance down by tour tier and by player experience.

    Aggregate accuracy over a corpus dominated by Futures understates tour-level
    performance and overstates it on qualifiers, so both are reported.
    """
    out: dict[str, dict] = {}
    tier_names = {0: "futures", 1: "challenger", 2: "atp_main_tour"}
    for code, name in tier_names.items():
        mask = tier == code
        if mask.sum() > 50:
            out[name] = classification_metrics(y[mask], p[mask])

    for threshold in (0, 10, 25, 50):
        mask = min_prior >= threshold
        if mask.sum() > 50:
            out[f"min_prior_matches_ge_{threshold}"] = classification_metrics(
                y[mask], p[mask]
            )
    return out


def temporal_metrics(
    y: np.ndarray, p: np.ndarray, date: np.ndarray, n_periods: int = 4
) -> dict[str, dict]:
    """Metrics per equal-sized slice of the test season, oldest first.

    Ratings are frozen at the end of training, so any systematic decay across
    the season would show up here as drift. A flat profile is evidence that the
    model is not silently relying on state that goes stale.
    """
    if len(y) == 0:
        return {}
    edges = np.quantile(date, np.linspace(0, 1, n_periods + 1))
    out: dict[str, dict] = {}
    for i in range(n_periods):
        lo, hi = edges[i], edges[i + 1]
        mask = (date >= lo) & (date <= hi) if i == n_periods - 1 else (date >= lo) & (date < hi)
        if mask.sum() < 100:
            continue
        block = classification_metrics(y[mask], p[mask])
        block["start_day"] = int(lo)
        out[f"period_{i + 1}_of_{n_periods}"] = block
    return out
