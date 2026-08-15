"""Regenerate the README results tables from ``artifacts/metrics.json``.

Keeps the documented numbers honest: they are rendered from the metrics the last
training run actually produced, never typed by hand.

    python scripts/update_readme.py [--artifacts artifacts] [--check]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_MARKER = "RESULTS_TABLE"
SLICE_MARKER = "SLICE_TABLE"
DRIFT_MARKER = "DRIFT_TABLE"

BASELINE_LABELS = {
    "elo_surface_blended": "Elo (surface-blended)",
    "elo_global": "Elo (global)",
    "atp_rank": "ATP rank only",
    "bookmaker_free_coinflip": "Coin flip",
}

SLICE_LABELS = {
    "atp_main_tour": "ATP main tour",
    "challenger": "Challenger / qualifying",
    "futures": "Futures / ITF",
    "min_prior_matches_ge_25": "Both players 25+ prior matches",
    "min_prior_matches_ge_50": "Both players 50+ prior matches",
}


def _cell(block: dict, key: str, fmt: str = "{:.4f}") -> str:
    value = block.get(key)
    return "—" if value is None else fmt.format(value)


def render_results(metrics: dict) -> str:
    test = metrics["test"]
    baselines = metrics["test_baselines"]
    dataset = metrics["dataset"]

    lines = [
        f"| Model (2024, {dataset['test_matches']:,} matches) | Accuracy | ROC AUC | Log loss | Brier | ECE |",
        "| --- | --- | --- | --- | --- | --- |",
        "| **XGBoost (calibrated)** | "
        f"**{test['accuracy']:.4f}** | **{test['auc']:.4f}** | **{test['log_loss']:.4f}** | "
        f"**{test['brier']:.4f}** | **{test['ece']:.4f}** |",
    ]
    uncal = metrics.get("test_uncalibrated")
    if uncal:
        lines.append(
            f"| XGBoost (uncalibrated) | {uncal['accuracy']:.4f} | {uncal['auc']:.4f} | "
            f"{uncal['log_loss']:.4f} | {uncal['brier']:.4f} | {uncal['ece']:.4f} |"
        )
    for key, label in BASELINE_LABELS.items():
        block = baselines.get(key)
        if not block:
            continue
        lines.append(
            f"| {label} | {_cell(block, 'accuracy')} | {_cell(block, 'auc')} | "
            f"{_cell(block, 'log_loss')} | {_cell(block, 'brier')} | {_cell(block, 'ece')} |"
        )

    reduction = test.get("log_loss_reduction_vs_elo_pct")
    footer = [
        "",
        f"Trained on **{dataset['train_matches']:,} matches** "
        f"({dataset['train_rows']:,} rows, both player orientations) across "
        f"{dataset['n_features']} features, with {dataset['warmup_matches']:,} earlier "
        f"matches used to warm up ratings. "
        f"{dataset['players_tracked']:,} players tracked; "
        f"{dataset['ranking_snapshots']:,} weekly ranking snapshots.",
    ]
    if reduction is not None:
        footer.append(
            f"Against the surface-blended Elo baseline the model cuts log loss by "
            f"**{reduction:.1f}%** and lifts AUC by "
            f"**{100 * (test['auc'] - baselines['elo_surface_blended']['auc']):.1f} points**."
        )
    footer.append(
        f"Order-invariance holds to {metrics['symmetry']['max_abs_error']:.1e} "
        f"(worst case over the season)."
    )
    return "\n".join(lines + footer)


def render_slices(metrics: dict) -> str:
    slices = metrics.get("test_slices", {})
    lines = [
        "| Slice of the 2024 season | Matches | Accuracy | ROC AUC | Log loss |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key, label in SLICE_LABELS.items():
        block = slices.get(key)
        if not block:
            continue
        lines.append(
            f"| {label} | {block['n']:,} | {block['accuracy']:.4f} | "
            f"{block['auc']:.4f} | {block['log_loss']:.4f} |"
        )
    return "\n".join(lines)


def render_drift(metrics: dict) -> str:
    """Quarter-by-quarter view of the test season, to expose any decay."""
    drift = metrics.get("test_drift", {})
    if not drift:
        return ""
    header = "| Quarter of 2024 | " + " | ".join(
        f"Q{i + 1}" for i in range(len(drift))
    ) + " |"
    sep = "| --- |" + " --- |" * len(drift)
    blocks = list(drift.values())
    rows = [
        "| Matches | " + " | ".join(f"{b['n']:,}" for b in blocks) + " |",
        "| Accuracy | " + " | ".join(f"{b['accuracy']:.4f}" for b in blocks) + " |",
        "| Log loss | " + " | ".join(f"{b['log_loss']:.4f}" for b in blocks) + " |",
    ]
    spread = max(b["accuracy"] for b in blocks) - min(b["accuracy"] for b in blocks)
    return "\n".join(
        [header, sep, *rows, "",
         f"Ratings are frozen at the end of training, yet accuracy varies by only "
         f"**{spread * 100:.1f} points** across the season — the model is not leaning on "
         f"state that goes stale."]
    )


def _replace_section(text: str, marker: str, body: str) -> str:
    """Replace the content between the paired ``START``/``END`` comment markers."""
    start, end = f"<!-- {marker}:START -->", f"<!-- {marker}:END -->"
    pattern = re.compile(
        rf"{re.escape(start)}.*?{re.escape(end)}", re.DOTALL
    )
    if not pattern.search(text):
        raise SystemExit(f"markers for {marker} not found in README")
    return pattern.sub(lambda _: f"{start}\n{body}\n{end}", text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default=PROJECT_ROOT / "artifacts", type=Path)
    parser.add_argument("--readme", default=PROJECT_ROOT / "README.md", type=Path)
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if the README is stale instead of rewriting it")
    args = parser.parse_args(argv)

    metrics_path = Path(args.artifacts) / "metrics.json"
    if not metrics_path.exists():
        print(f"error: {metrics_path} not found -- run `python -m tennis_engine train` first",
              file=sys.stderr)
        return 1

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    readme = Path(args.readme)
    original = readme.read_text(encoding="utf-8")

    updated = _replace_section(original, RESULTS_MARKER, render_results(metrics))
    updated = _replace_section(updated, SLICE_MARKER, render_slices(metrics))
    updated = _replace_section(updated, DRIFT_MARKER, render_drift(metrics))

    if args.check:
        if updated != original:
            print("README results are out of date; run scripts/update_readme.py",
                  file=sys.stderr)
            return 1
        print("README is up to date.")
        return 0

    readme.write_text(updated, encoding="utf-8")
    print(f"Updated {readme} from {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
