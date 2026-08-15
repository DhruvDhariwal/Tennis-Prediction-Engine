"""Diagnostic figures written alongside the model artifacts.

The README claimed a reliability curve that the original notebook never
actually produced; these are the plots that back up the reported numbers.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from matplotlib import rc_context
from matplotlib.figure import Figure

from .config import display_path
from .evaluate import elo_probability, reliability_curve
from .features import FEATURE_INDEX

logger = logging.getLogger(__name__)

_STYLE = {"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.25, "font.size": 9}


def _new_figure(**kwargs) -> Figure:
    """Create a standalone figure that is never attached to a global backend.

    Using ``matplotlib.figure.Figure`` rather than ``pyplot`` keeps these
    write-to-disk plots off pyplot's global figure registry. Calling
    ``matplotlib.use("Agg")`` at import time -- the usual shortcut -- would
    hijack the backend for *any* caller, so a notebook that merely imports this
    module would find its own ``plt.show()`` silently doing nothing.
    """
    fig = Figure(**kwargs)
    fig.set_layout_engine("tight")
    return fig


def _save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")


def plot_reliability(y, p_model, p_baseline, path: Path) -> None:
    """Calibration of the model against the raw Elo probability."""
    with rc_context(_STYLE):
        fig = _new_figure(figsize=(5.5, 5.5))
        ax, ax2 = fig.subplots(2, 1, height_ratios=[3, 1], sharex=True)
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        for probs, label, style in (
            (p_baseline, "Elo baseline", "o--"),
            (p_model, "XGBoost (calibrated)", "o-"),
        ):
            conf, acc, _ = reliability_curve(y, probs)
            ax.plot(conf, acc, style, ms=4, lw=1.4, label=label)
        ax.set_ylabel("observed win rate")
        ax.set_title("Reliability on the held-out season")
        ax.legend(loc="upper left", frameon=False)

        ax2.hist(p_model, bins=40, color="#4c72b0", alpha=0.8)
        ax2.set_xlabel("predicted P(player A wins)")
        ax2.set_ylabel("matches")
        _save(fig, path)


def plot_feature_importance(importance: dict[str, float], path: Path, top_n: int = 20) -> None:
    with rc_context(_STYLE):
        items = list(importance.items())[:top_n][::-1]
        names = [k for k, _ in items]
        values = [v for _, v in items]
        fig = _new_figure(figsize=(6, 0.28 * len(items) + 1.2))
        ax = fig.subplots()
        ax.barh(names, values, color="#4c72b0")
        ax.set_xlabel("share of total gain (%)")
        ax.set_title(f"Top {len(items)} features by XGBoost gain")
        _save(fig, path)


def plot_accuracy_by_elo_gap(X, y, p, path: Path, n_bins: int = 10) -> None:
    """Accuracy as a function of how mismatched the two players are."""
    gap = np.abs(X[:, FEATURE_INDEX["blend_elo_diff"]]).astype(np.float64)
    edges = np.quantile(gap, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    idx = np.clip(np.digitize(gap, edges[1:-1]), 0, len(edges) - 2)

    correct = (p >= 0.5).astype(int) == y
    centres, model_acc, elo_acc, counts = [], [], [], []
    p_elo = elo_probability(X, "blend_elo_diff")
    elo_correct = (p_elo >= 0.5).astype(int) == y
    for b in range(len(edges) - 1):
        mask = idx == b
        if mask.sum() < 30:
            continue
        centres.append(gap[mask].mean())
        model_acc.append(correct[mask].mean())
        elo_acc.append(elo_correct[mask].mean())
        counts.append(int(mask.sum()))

    with rc_context(_STYLE):
        fig = _new_figure(figsize=(5.5, 3.4))
        ax = fig.subplots()
        ax.plot(centres, model_acc, "o-", lw=1.4, ms=4, label="XGBoost")
        ax.plot(centres, elo_acc, "o--", lw=1.2, ms=4, label="Elo baseline")
        ax.axhline(0.5, color="k", ls=":", lw=1)
        ax.set_xlabel("|surface-blended Elo gap|")
        ax.set_ylabel("accuracy")
        ax.set_title("Accuracy vs how close the match-up is")
        ax.legend(frameon=False)
        _save(fig, path)


def plot_tier_breakdown(slices: dict, path: Path) -> None:
    tiers = [k for k in ("futures", "challenger", "atp_main_tour") if k in slices]
    if not tiers:
        return
    with rc_context(_STYLE):
        fig = _new_figure(figsize=(5.5, 3.2))
        ax = fig.subplots()
        x = np.arange(len(tiers))
        width = 0.38
        ax.bar(x - width / 2, [slices[t]["accuracy"] for t in tiers], width, label="accuracy")
        ax.bar(x + width / 2, [slices[t]["auc"] for t in tiers], width, label="ROC AUC")
        ax.set_xticks(x, [t.replace("_", " ") for t in tiers])
        ax.axhline(0.5, color="k", ls=":", lw=1)
        ax.set_ylim(0.4, 1.0)
        ax.set_title("Performance by tour tier (held-out season)")
        ax.legend(frameon=False)
        for i, tier in enumerate(tiers):
            ax.annotate(f"n={slices[tier]['n']:,}", (i, 0.43), ha="center", fontsize=7)
        _save(fig, path)


def write_figures(figures_dir: Path, X, y, p_model, metrics: dict) -> list[Path]:
    """Produce the full diagnostic set. Never fatal -- a plotting failure must
    not lose a training run."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    jobs = [
        ("reliability.png", lambda path: plot_reliability(
            y, p_model, elo_probability(X, "blend_elo_diff"), path)),
        ("feature_importance.png", lambda path: plot_feature_importance(
            metrics["feature_importance_gain"], path)),
        ("accuracy_by_elo_gap.png", lambda path: plot_accuracy_by_elo_gap(X, y, p_model, path)),
        ("tier_breakdown.png", lambda path: plot_tier_breakdown(metrics["test_slices"], path)),
    ]
    for name, job in jobs:
        path = figures_dir / name
        try:
            job(path)
            written.append(path)
        except Exception:  # pragma: no cover - diagnostics only
            logger.exception("Could not render %s", name)
    logger.info("Wrote %d figures to %s", len(written), display_path(figures_dir))
    return written
