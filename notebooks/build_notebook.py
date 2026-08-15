"""Generate ``Tennis_Prediction_Engine.ipynb`` from a single source of truth.

Keeping the notebook in a .py generator avoids hand-editing 180 KB of JSON and
guarantees that every code cell is real, runnable text.
"""

from __future__ import annotations

import json
from pathlib import Path

MD = "markdown"
CODE = "code"

CELLS: list[tuple[str, str]] = [
    (MD, """
# Tennis Prediction Engine

**Predicting ATP match winners from 884,030 matches (1968-2024), without leaking the future.**

Author: Dhruv Dhariwal

---

This notebook is a *narrated driver* over the `tennis_engine` package in `src/`.
Every function called here is the same code that runs in production via the CLI
and that is covered by the test suite -- there is no second, notebook-only
implementation that can drift out of sync.

## The three properties the engine is built around

**1. Leak-proof.** Matches are replayed in the order they were played. For each
match the builder *reads* state, *emits* a feature row, and only then *updates*
Elo, head-to-head, form and serve statistics. No feature can depend on the match
it describes.

**2. Order-invariant.** `P(A beats B)` must equal `1 - P(B beats A)`. Every
feature is either antisymmetric (a difference, which negates when the players
swap) or symmetric (surface, best-of, tier). Swapping players is therefore a sign
flip, and the final prediction is symmetrized so the identity holds *exactly*,
not approximately.

**3. Honestly evaluated.** Three chronological splits. Ratings warm up on
1968-1990, the model trains on 1991-2022, hyperparameters and early stopping use
2023, and **2024 is not touched until the final evaluation.**
"""),

    (CODE, """
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# The package lives in src/ and is importable without installation.
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tennis_engine import evaluate, pipeline, reporting
from tennis_engine.config import Config, Paths
from tennis_engine.data import load_player_directory, prepare_matches
from tennis_engine.features import FEATURE_INDEX, FEATURE_NAMES, FLIP_SIGN
from tennis_engine.rankings import build_ranking_lookup, to_day_ordinal

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s", force=True)
pd.set_option("display.width", 150)

cfg = Config(paths=Paths(base_dir=PROJECT_ROOT, artifacts_dir=PROJECT_ROOT / "artifacts"))
print("Project root   :", cfg.paths.base_dir.name)
print("Features       :", len(FEATURE_NAMES))
print("Split          : warmup %d-%d | train %d-%d | val %s | test %s" % (
    min(cfg.split.warmup_years), max(cfg.split.warmup_years),
    min(cfg.split.train_years), max(cfg.split.train_years),
    list(cfg.split.val_years), list(cfg.split.test_years)))
"""),

    (MD, """
---
## 1. Load and clean

`prepare_matches` loads the ATP main tour, Challenger/qualifying and Futures
files, then removes what cannot be learned from and what must not be seen.

Two cleaning decisions are worth calling out, because both were bugs in the
first version of this project:

- **Outcome columns are dropped by an explicit list, not a regex.** The old
  filter was `score|min|retired|walkover|wo|after|next|post`. The `wo`
  alternative matches "**Wo**n" case-insensitively, so it silently deleted
  `w_1stWon`, `w_2ndWon`, `l_1stWon` and `l_2ndWon` -- the four columns the
  service-points-won feature is computed from. That feature was identically zero
  for all 3.2 million rows and nobody noticed.
- **Unknown surfaces stay unknown.** They used to be rewritten to `Hard`, which
  fabricated hard-court Elo history for every unlabelled match.
"""),

    (CODE, """
matches = prepare_matches(cfg.paths.base_dir, cfg.split.all_years())

print(f"{len(matches):,} matches, {matches.tourney_date.min().date()} -> {matches.tourney_date.max().date()}")
display(matches.groupby("tier").agg(
    matches=("winner_id", "size"),
    first=("tourney_date", "min"),
    last=("tourney_date", "max"),
))
display(matches.surface.value_counts().rename("matches").to_frame())

# The columns that make the fixed serve feature possible:
assert {"w_1stWon", "w_2ndWon", "l_1stWon", "l_2ndWon"} <= set(matches.columns)
assert "score" not in matches.columns and "winner_rank" not in matches.columns
print("\\nLeakage audit: outcome columns removed, box-score columns retained.")
"""),

    (MD, """
---
## 2. Point-in-time ATP rankings

3.3 million weekly ranking snapshots are collapsed into per-player sorted numpy
arrays. Looking up "what was this player ranked immediately before date D" is
then a binary search -- `O(log n)` instead of a scan over the whole table.

The lookup is strictly *as-of*: it can only ever return a snapshot dated on or
before the match. Rankings are published on Mondays and already reflect the
previous week's results, so a snapshot dated the same day a tournament starts is
legitimate pre-match information.
"""),

    (CODE, """
rankings = build_ranking_lookup(cfg.paths.base_dir)
print(f"{rankings.n_rows:,} snapshots across {len(rankings):,} players")

# Demonstrate the as-of semantics on one player across one ranking change.
pid = 104925  # Novak Djokovic
for date in ["2015-06-01", "2018-06-01", "2021-06-01", "2024-06-03"]:
    rank, points, ranked = rankings.get(pid, int(to_day_ordinal(pd.Timestamp(date))))
    print(f"  {date}: rank {int(rank):>5}  points {int(points):>6}  ranked={ranked}")

# A date before the player's first ever ranking must report 'unranked'.
rank, points, ranked = rankings.get(pid, int(to_day_ordinal(pd.Timestamp('1990-01-01'))))
print(f"  1990-01-01: rank {int(rank)} points {int(points)} ranked={ranked}  <- correctly unranked")
"""),

    (MD, """
---
## 3. Build features chronologically

One pass over all 884k matches. The warm-up seasons update ratings without
emitting rows, so the model never trains on the pre-1991 era (which has almost
no box scores) but ratings entering 1991 are not cold-started at 1500.

**Training rows are mirrored, evaluation rows are not.** Each training match
contributes both orientations, which makes the training distribution exactly
symmetric. Each *evaluation* match contributes exactly one randomly-oriented row,
so the reported match counts and metrics describe real matches rather than each
match counted twice.

This cell is the expensive one -- roughly 5-8 minutes for the full history.
"""),

    (CODE, """
%time splits, builder = pipeline.build_splits(cfg)

print()
print(f"train : {len(splits.train):>9,} rows  ({len(splits.train)//2:,} matches, both orientations)")
print(f"val   : {len(splits.val):>9,} rows  (2023)")
print(f"test  : {len(splits.test):>9,} rows  (2024)")
print(f"players tracked: {splits.n_players:,}")
"""),

    (MD, """
### 3a. Leakage and symmetry audits

These are assertions, not prose. If the feature builder ever regresses, this cell
raises rather than quietly producing a better-looking number.

One thing the audit surfaces immediately: `surface_is_unknown` is constant across
1991-2022, because every match in the training window has a labelled surface --
the 2,858 unlabelled ones all sit in the pre-1991 warm-up era. There are 51 in
2023-24, and they correctly arrive with all four real surface flags at zero, so
the model falls back on surface-agnostic signal for them instead of being told
they were played on hard court.
"""),

    (CODE, """
X = splits.train.X

# (1) Antisymmetry: mirrored training rows are exact negations on the diff block
#     and identical on the symmetric block.
forward, mirrored = X[0::2], X[1::2]
assert np.allclose(mirrored, forward * FLIP_SIGN, equal_nan=True)
print("PASS  mirrored rows are exact sign-flips")

# (2) One-hot surface encoding really is one-hot (this used to be NaN-filled).
surface_cols = [FEATURE_INDEX[c] for c in FEATURE_NAMES if c.startswith("surface_is_")]
block = X[:, surface_cols]
assert not np.isnan(block).any() and np.array_equal(block.sum(1), np.ones(len(block)))
print("PASS  surface one-hot has no NaNs and sums to 1")

# (3) No feature is constant -- a constant feature is dead code, which is exactly
#     what silently happened to srv_winrate_diff in v1.
dead = []
for name in FEATURE_NAMES:
    col = X[:, FEATURE_INDEX[name]]
    finite = col[np.isfinite(col)]
    if len(finite) == 0 or np.allclose(finite, finite[0]):
        dead.append(name)

# A *surface* flag may legitimately be constant: a category can simply not occur
# in this window. Everything else being constant is a bug.
unexpected = [n for n in dead if not n.startswith("surface_is_")]
assert not unexpected, f"constant (dead) features: {unexpected}"
print("PASS  every substantive feature varies, srv_winrate_diff included")
if dead:
    print(f"      note: {', '.join(dead)} constant in this training window")

# (4) Chronology: evaluation rows never precede training rows.
assert splits.train.date.max() < splits.val.date.min() < splits.test.date.min()
print("PASS  train < val < test in time, with no overlap")

# (5) Missingness is preserved rather than coerced to zero.
missing = pd.Series(np.isnan(X).mean(axis=0) * 100, index=list(FEATURE_NAMES))
display(missing[missing > 0].sort_values(ascending=False).round(2)
        .rename("% missing").to_frame())
"""),

    (MD, """
---
## 4. Train

XGBoost with early stopping on the **2023 validation season**.

The original version early-stopped on the 2024 test set. That is model selection
on the test data: the number of trees was chosen to minimise test loss, so the
reported 2024 metrics were optimistic by construction. Moving early stopping to a
dedicated validation season is the single most important methodological fix here,
and it is the reason the headline numbers in this notebook can be quoted honestly.
"""),

    (CODE, """
%time booster, history = pipeline.train_model(cfg, splits)

fig, ax = plt.subplots(figsize=(6, 3.4), dpi=120)
ax.plot(history["train"]["logloss"], label="train")
ax.plot(history["val"]["logloss"], label="validation (2023)")
ax.axvline(booster.best_iteration, color="k", ls="--", lw=1,
           label=f"best iter = {booster.best_iteration}")
ax.set_xlabel("boosting round"); ax.set_ylabel("log loss")
ax.set_title("Early stopping on a held-out season, not on the test set")
ax.legend(frameon=False); ax.grid(alpha=.25)
plt.show()
"""),

    (MD, """
---
## 5. Evaluate on 2024

`evaluate_all` fits the probability calibrator on the validation season, then
scores the untouched test season. It also runs the baselines that any learned
model has to beat before it is worth anything:

- **Elo (global)** and **Elo (surface-blended)** -- closed-form win probability
  from the ratings alone.
- **ATP rank** -- pick the better-ranked player.
- **Coin flip.**
"""),

    (CODE, """
metrics, calibrator = pipeline.evaluate_all(cfg, splits, booster)

rows = []
for name, block in [("XGBoost (calibrated)", metrics["test"]),
                    ("XGBoost (uncalibrated)", metrics["test_uncalibrated"]),
                    *[(f"baseline: {k}", v) for k, v in metrics["test_baselines"].items()]]:
    rows.append({
        "model": name,
        "accuracy": block.get("accuracy"),
        "auc": block.get("auc"),
        "log_loss": block.get("log_loss"),
        "brier": block.get("brier"),
        "ece": block.get("ece"),
    })
display(pd.DataFrame(rows).set_index("model").round(4))
"""),

    (MD, """
### 5a. Where the accuracy actually comes from

A single headline accuracy over a corpus that is 55% Futures is misleading: those
draws contain enormous ranking mismatches and are easy to call. Tour-level
matches are much harder and are what the number would be judged on.
"""),

    (CODE, """
display(pd.DataFrame(metrics["test_slices"]).T[["n", "accuracy", "auc", "log_loss", "brier"]].round(4))
"""),

    (MD, """
### 5b. Calibration

Accuracy alone says nothing about whether a stated 70% actually wins 70% of the
time. The calibrator is fitted on the validation season and is chosen from
identity / temperature scaling / symmetric isotonic by validation log loss.

Both non-trivial calibrators are **symmetry preserving** by construction --
temperature scaling because `logit(1-p) = -logit(p)`, and the isotonic variant
because it is fitted on the symmetry-augmented sample and averaged in both
directions. An off-the-shelf calibrator would silently break `P(A) = 1 - P(B)`.
"""),

    (CODE, """
predict = pipeline.make_predictor(booster, calibrator)
p_test = predict(splits.test.X)

reporting.plot_reliability(
    splits.test.y, p_test,
    evaluate.elo_probability(splits.test.X, "blend_elo_diff"),
    cfg.paths.figures_dir / "reliability.png",
)
from IPython.display import Image
display(Image(str(cfg.paths.figures_dir / "reliability.png")))

print("calibrator selected :", metrics["model"]["calibrator"])
print("validation log loss by method:", metrics["model"]["calibration_val_logloss"])
print(f"test ECE: {metrics['test']['ece']:.4f} "
      f"(uncalibrated {metrics['test_uncalibrated']['ece']:.4f})")
"""),

    (MD, """
### 5c. Order-invariance

The original notebook asserted `mean |p(A,B) - (1 - p(B,A))| < 0.02` and called
that order-invariant. It is a tolerance, not a guarantee -- and it says nothing
about the worst case.

Here the prediction is explicitly symmetrized, `p = ½·(p(A,B) + 1 - p(B,A))`, so
the identity holds to floating-point precision for **every** match, not on
average.
"""),

    (CODE, """
sym = evaluate.symmetry_error(predict, splits.test.X)
for k, v in sym.items():
    print(f"  {k:<18s} {v:.3e}")
assert sym["max_abs_error"] < 1e-9, "order invariance broken"
print("\\nPASS  P(A beats B) + P(B beats A) = 1 exactly, for all 2024 matches")
"""),

    (MD, """
### 5d. What the model is using
"""),

    (CODE, """
imp = pd.Series(metrics["feature_importance_gain"], name="% of total gain")
display(imp.head(15).to_frame())

reporting.plot_feature_importance(metrics["feature_importance_gain"],
                                  cfg.paths.figures_dir / "feature_importance.png")
reporting.plot_accuracy_by_elo_gap(splits.test.X, splits.test.y, p_test,
                                   cfg.paths.figures_dir / "accuracy_by_elo_gap.png")
display(Image(str(cfg.paths.figures_dir / "feature_importance.png")))
display(Image(str(cfg.paths.figures_dir / "accuracy_by_elo_gap.png")))
"""),

    (MD, """
---
## 6. Save artifacts
"""),

    (CODE, """
pipeline.save_artifacts(cfg, booster, builder, calibrator, metrics)

for path in sorted(cfg.paths.artifacts_dir.glob("*")):
    if path.is_file():
        print(f"  {path.name:<28s} {path.stat().st_size/1e6:8.1f} MB")
"""),

    (MD, """
---
## 7. Predict a future match

`PredictionEngine` rebuilds the feature builder from the saved state and calls
**the same `match_vector` used in training**. That train/serve parity matters:
the previous version hand-wrote the inference vector twice, in two different
cells, and both copies disagreed with the training code. One of them referenced a
`rank_points_diff` column that has never existed, so rank and points were fed to
the model as zeros at prediction time while being real numbers during training.
"""),

    (CODE, """
from tennis_engine.predict import PredictionEngine

engine = PredictionEngine.load(cfg)

for a, b, surface in [
    ("Jannik Sinner", "Carlos Alcaraz", "Hard"),
    ("Jannik Sinner", "Carlos Alcaraz", "Clay"),
    ("Novak Djokovic", "Carlos Alcaraz", "Grass"),
    ("Alexander Zverev", "Daniil Medvedev", "Hard"),
]:
    print(engine.predict_match(a, b, surface=surface, best_of=5))
    print()
"""),

    (MD, """
Surface matters, and it should: the same two players get different probabilities
on clay than on hard. That is the surface-Elo block doing its job.

### Sanity checks a reviewer would ask for
"""),

    (CODE, """
# 1. Order invariance at the API level.
fwd = engine.predict_match("Jannik Sinner", "Carlos Alcaraz", surface="Hard")
rev = engine.predict_match("Carlos Alcaraz", "Jannik Sinner", surface="Hard")
print(f"P(Sinner) forward = {fwd.p1_win_prob:.10f}")
print(f"P(Sinner) reversed = {rev.p2_win_prob:.10f}")
assert abs(fwd.p1_win_prob - rev.p2_win_prob) < 1e-9
print("PASS  identical to 1e-9\\n")

# 2. A top player against a journeyman should be a heavy favourite.
lopsided = engine.predict_match("Jannik Sinner", "Zizou Bergs", surface="Hard")
print(lopsided, "\\n")

# 3. Player state is inspectable, not a black box.
display(pd.Series(engine.player_card("Carlos Alcaraz", surface="Clay")).to_frame("value"))
"""),

    (CODE, """
# Current Elo leaderboard, restricted to players active in the last season.
display(engine.elo_leaderboard(top_n=15))
display(engine.elo_leaderboard(top_n=10, surface="Clay"))
"""),

    (MD, """
---
## 8. Summary of what changed from v1

| Area | Before | After |
|---|---|---|
| Early stopping | on the 2024 **test** set | on a 2023 **validation** season; 2024 untouched |
| `srv_winrate_diff` | identically 0 (regex ate `*_1stWon`/`*_2ndWon`) | real rolling serve efficiency |
| Missing serve data | `NaN`/0.0 pushed into rolling means | excluded from the mean; emitted as `NaN` for XGBoost |
| Surface one-hot | absent keys became `NaN` | true one-hot incl. an explicit `Unknown` |
| Unknown surface | rewritten to `Hard` | kept distinct |
| Order invariance | asserted `mean error < 0.02` | exact by symmetrized prediction (`<1e-9` worst case) |
| Inference features | hand-written twice, both wrong | same `match_vector` as training |
| `height` / `age` at predict time | hard-coded to 0 | resolved from the player directory |
| Calibration | none | validation-fitted, symmetry-preserving |
| Baselines | none | Elo, surface-Elo, ATP rank, coin flip |
| Evaluation rows | every match counted twice | one row per match |
| Tests | none | 57 tests covering leakage, symmetry and I/O |
| Interface | notebook cells only | installable package + CLI |

The CLI does everything this notebook does:

```bash
python -m tennis_engine train
python -m tennis_engine predict "Jannik Sinner" "Carlos Alcaraz" --surface Clay --explain
python -m tennis_engine leaderboard --surface Grass --top 10
python -m tennis_engine card "Novak Djokovic"
```
"""),
]


def build() -> dict:
    cells = []
    for kind, source in CELLS:
        text = source.strip("\n")
        cell = {
            "cell_type": kind,
            "metadata": {},
            "source": [line + "\n" for line in text.split("\n")[:-1]] + [text.split("\n")[-1]],
        }
        if kind == CODE:
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "Tennis_Prediction_Engine.ipynb"
    out.write_text(json.dumps(build(), indent=1), encoding="utf-8")
    print(f"wrote {out} ({len(CELLS)} cells)")
