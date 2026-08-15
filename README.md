# Tennis Prediction Engine 🎾

**Predicting ATP match winners from 884,030 matches (1968–2024) — leak-proof, order-invariant, and calibrated.**

An XGBoost match-winner model built on Jeff Sackmann's ATP data, with a chronological
feature engine (Elo, surface Elo, head-to-head, rolling serve/return form, point-in-time
ATP rankings), an honest three-way temporal split, probability calibration, and a CLI for
scoring arbitrary match-ups.

```bash
$ python -m tennis_engine predict "Jannik Sinner" "Carlos Alcaraz" --surface Clay --best-of 5
```

---

## Results

Trained on 1991–2022, early-stopped on 2023, evaluated on **2024 — a season the model
never saw during training, tuning, calibration or model selection.**

<!-- RESULTS_TABLE:START -->
| Model (2024, 31,339 matches) | Accuracy | ROC AUC | Log loss | Brier | ECE |
| --- | --- | --- | --- | --- | --- |
| **XGBoost (calibrated)** | **0.6995** | **0.7733** | **0.5680** | **0.1936** | **0.0082** |
| XGBoost (uncalibrated) | 0.6996 | 0.7734 | 0.5678 | 0.1936 | 0.0095 |
| Elo (surface-blended) | 0.6724 | 0.7365 | 0.6035 | 0.2083 | 0.0246 |
| Elo (global) | 0.6735 | 0.7399 | 0.5996 | 0.2067 | 0.0158 |
| ATP rank only | 0.6732 | 0.7316 | — | — | — |
| Coin flip | 0.5000 | 0.5000 | 0.6931 | 0.2500 | — |

Trained on **716,714 matches** (1,433,428 rows, both player orientations) across 33 features, with 105,974 earlier matches used to warm up ratings. 28,451 players tracked; 3,292,228 weekly ranking snapshots.
Against the surface-blended Elo baseline the model cuts log loss by **5.9%** and lifts AUC by **3.7 points**.
Order-invariance holds to 1.1e-16 (worst case over the season).
<!-- RESULTS_TABLE:END -->

Metrics are computed over **one row per match**, not one row per player-orientation.

<!-- SLICE_TABLE:START -->
| Slice of the 2024 season | Matches | Accuracy | ROC AUC | Log loss |
| --- | --- | --- | --- | --- |
| ATP main tour | 2,973 | 0.6778 | 0.7501 | 0.5853 |
| Challenger / qualifying | 10,755 | 0.6639 | 0.7306 | 0.6025 |
| Futures / ITF | 17,611 | 0.7250 | 0.7991 | 0.5441 |
| Both players 25+ prior matches | 23,521 | 0.6748 | 0.7416 | 0.5964 |
| Both players 50+ prior matches | 19,408 | 0.6662 | 0.7310 | 0.6048 |
<!-- SLICE_TABLE:END -->

<!-- DRIFT_TABLE:START -->
| Quarter of 2024 | Q1 | Q2 | Q3 | Q4 |
| --- | --- | --- | --- | --- |
| Matches | 7,639 | 7,842 | 8,017 | 7,841 |
| Accuracy | 0.6939 | 0.6924 | 0.6970 | 0.7147 |
| Log loss | 0.5703 | 0.5763 | 0.5690 | 0.5565 |

Ratings are frozen at the end of training, yet accuracy varies by only **2.2 points** across the season — the model is not leaning on state that goes stale.
<!-- DRIFT_TABLE:END -->

Aggregate numbers are dominated by Futures and Challenger draws, where ranking gaps are
enormous and outcomes are easy to call. Tour-level matches are the hard case and are
reported separately rather than being averaged away.

---

## Why this is not just another Elo wrapper

### Leak-proof by construction

Matches are replayed in the order they were played. For every match the engine

1. **reads** the current state (ratings, form, H2H, serve stats, as-of ranking),
2. **emits** the feature row,
3. **then** applies the post-match update.

A feature therefore cannot depend on the match it describes. The invariant is enforced in
code — `FeatureBuilder.build` raises if the input is not chronologically sorted — and
asserted in the test suite, which checks that a player's first-ever appearance carries no
prior information.

The 2024 test season is protected at three separate points that are commonly leaked:

| Leak vector | How it is closed |
| --- | --- |
| Feature construction | strictly chronological single pass |
| Early stopping | stops on the **2023 validation season**, never on test |
| Hyperparameter search | scored on validation only |
| Probability calibration | fitted on validation only |

### Order-invariant, exactly

`P(A beats B)` must equal `1 − P(B beats A)`. Every feature is either **antisymmetric**
(a difference — negates when players swap) or **symmetric** (surface, best-of, tier), so
swapping players is a sign flip on a fixed mask:

```python
mirrored_row = row * FLIP_SIGN     # -1 on the diff block, +1 on the rest
```

Training uses both orientations of every match, and prediction averages
`½·(p(A,B) + 1 − p(B,A))`. The identity then holds to **floating-point precision for every
match** (max error < 1e-9 across the full test season), rather than "on average, within
a tolerance".

Calibration preserves this too: temperature scaling is exactly symmetric because
`logit(1−p) = −logit(p)`, and the isotonic variant is fitted on a symmetry-augmented
sample and averaged in both directions.

### Missing means missing

A player with no recorded serve history yields `NaN`, which XGBoost routes down a learned
default branch. Encoding "no data" as `0.0` would tell the model that a debutant wins 0%
of service points — false, and far outside the real range, so highly influential.

---

## Features (33)

**Antisymmetric (25)** — negate under player swap:

| Group | Features |
| --- | --- |
| Ratings | `elo_diff`, `surf_elo_diff`, `blend_elo_diff`, `elo_expectation_diff` |
| Rankings | `rank_diff`, `log_rank_diff`, `log_points_diff`, `is_ranked_diff` |
| Form & experience | `form_diff` (last 20), `career_winrate_diff`, `log_matches_diff` |
| Surface record | `surf_winrate_diff`, `log_surf_matches_diff` |
| Head-to-head | `h2h_diff`, `h2h_surface_diff` |
| Serve / return | `ace_rate_diff`, `df_rate_diff`, `srv_winrate_diff`, `ret_winrate_diff`, `bp_save_rate_diff` |
| Schedule | `log_days_since_diff` (rest/rust), `workload_diff` (matches in last 90 days) |
| Static | `height_diff`, `age_diff`, `lefty_diff` |

**Symmetric (8)** — unchanged under swap: `h2h_total`, `best_of`, `tier_ordinal`, and the
five-way surface one-hot (`Hard`, `Clay`, `Grass`, `Carpet`, `Unknown`).

**Elo details.** K decays with experience (FiveThirtyEight-style,
`k = scale / (matches + 5)^0.4`) and is scaled by tour tier — a Futures result moves a
rating roughly a third as much as a tour-level one. Surface Elo is shrunk toward global
Elo (`blend = 0.35`) so a player with three career grass matches does not get a
meaningless grass rating.

**Rankings.** 3.29M weekly snapshots across 16,474 players are collapsed into per-player
sorted arrays; an as-of lookup is a `O(log n)` binary search.

---

## Install & run

```bash
pip install -e ".[dev]"
```

Download the ATP CSVs from [JeffSackmann/tennis_atp](https://github.com/JeffSackmann/tennis_atp)
into the project root (`atp_matches_*.csv`, `atp_rankings_*.csv`, `atp_players.csv`).

```bash
python -m tennis_engine train                 # full pipeline -> artifacts/
python -m tennis_engine train --tune 20       # + validation-scored random search
python -m tennis_engine train --no-futures    # tour + challenger only
```

Training writes `artifacts/`: `tennis_xgb_model.json`, `engine_state.pkl`,
`metrics.json`, and diagnostic figures under `artifacts/figures/`.

### Predicting

```bash
python -m tennis_engine predict "Jannik Sinner" "Carlos Alcaraz" --surface Clay --best-of 5
python -m tennis_engine predict "Novak Djokovic" "Carlos Alcaraz" --surface Grass --explain
python -m tennis_engine card "Carlos Alcaraz" --surface Clay
python -m tennis_engine leaderboard --surface Grass --top 10
```

```python
from tennis_engine import PredictionEngine

engine = PredictionEngine.load()
print(engine.predict_match("Jannik Sinner", "Carlos Alcaraz", surface="Clay", best_of=5))
print(engine.elo_leaderboard(top_n=10, surface="Clay"))
```

Names resolve case-insensitively with close-match suggestions on a miss; integer ATP
player IDs also work.

---

## Layout

```
src/tennis_engine/
  config.py       all hyperparameters and paths in one place
  data.py         loading, cleaning, leakage audit
  rankings.py     as-of ranking lookup (numpy + searchsorted)
  state.py        PlayerState, Elo engine, (de)serialization
  features.py     chronological feature builder, symmetry mask
  calibration.py  symmetry-preserving probability calibration
  evaluate.py     metrics, baselines, slices, symmetry checks
  reporting.py    diagnostic figures
  pipeline.py     splits -> tune -> train -> evaluate -> save
  predict.py      PredictionEngine (train/serve parity)
  cli.py          python -m tennis_engine ...
tests/            57 tests: leakage, symmetry, as-of lookups, round-trips, end-to-end
notebooks/        generator for the narrated notebook
Tennis_Prediction_Engine.ipynb
```

```bash
pytest            # ~40s on the synthetic fixture; no real data required
```

The test suite runs against a synthetic ATP-shaped corpus, so it needs none of the
900 MB of CSVs. Several tests are explicit regression guards for the bugs listed below.

---

## What changed in v2

The v1 notebook reported 69.52% accuracy / 0.7654 AUC on 2024. Those numbers were not
trustworthy, and several features were silently dead. The rewrite fixes the following.

**Correctness**

| Bug | Effect | Fix |
| --- | --- | --- |
| Leakage-audit regex `…\|wo\|…` matched "**Wo**n" | dropped `w_1stWon`, `w_2ndWon`, `l_1stWon`, `l_2ndWon`, so `srv_winrate_diff` was **identically 0 for all 3.2M rows** | explicit deny-list; regression test asserts no feature is constant |
| Inference feature vector hand-written separately from training (twice, in two cells, disagreeing) | one copy referenced `rank_points_diff`, a column that never existed → rank and points served as zeros | inference calls the **same** `match_vector` as training |
| `height_diff` / `age_diff` hard-coded to `0` at prediction time | train/serve skew on two real features | resolved from `atp_players.csv` (height, dob, hand) at both ends |
| Missing box scores appended as `NaN`, or coerced to `0.0` | poisoned every rolling serve average downstream | excluded from the running mean; surfaced as `NaN` |
| Surface one-hot built from a dict with only the active key | absent surfaces became `NaN`, never `0` | true one-hot, five categories |
| Unknown surface rewritten to `Hard` | fabricated hard-court Elo history | `Unknown` kept as its own category |
| Two near-duplicate "save state" cells, only one storing rankings | whichever ran last silently determined whether prediction worked | single `save_artifacts` |

**Methodology**

- **Early stopping moved off the test set.** v1 early-stopped on 2024 — the number of
  trees was chosen to minimise test loss, so the reported 2024 metrics were optimistic by
  construction. v2 introduces a 2023 validation season.
- **One row per match at evaluation.** v1 mirrored every match into the test set too, so
  "62,000 test samples" was 31,000 matches counted twice.
- **Baselines.** Elo, surface-blended Elo, ATP rank and a coin flip are all reported, so
  the model's contribution over the ratings it is built on is visible.
- **Calibration.** Fitted on validation, selected by validation log loss, symmetry-preserving.
- **Order-invariance guaranteed, not asserted within a tolerance.**
- **Hyperparameter search** over a validation-scored random search (`--tune N`).
- **Ratings warm up** on 1968–1990 without emitting training rows, instead of cold-starting
  every rating at 1500 in the first training season.

**Engineering**

- Notebook cells → installable package with a CLI and 57 tests.
- Single chronological pass instead of building train features and then `deepcopy`-ing a
  100 MB+ builder for the test split.
- Serialized state shrunk from **114.6 MB → 63.0 MB** while storing strictly more
  (surface H2H, per-surface records, schedule history): rolling histories are packed numpy
  arrays rather than nested Python lists, the two H2H tables are columnar rather than
  ~1.4M tuple-keyed dict entries, and ranking arrays use narrow dtypes.
- Feature construction for all 884k matches runs in 80-160 s.

---

## Limitations

- `tourney_date` is the tournament **start** date, so within-tournament ordering relies on
  `match_num`. Rest-day and workload features are therefore coarse inside a single event.
- No in-match, injury, retirement-risk, weather or betting-market data.
- Prediction uses each player's state as of the last processed match; it does not
  re-simulate ratings forward for matches played after the training corpus ends.
- Calibration is fitted on one season; a longer validation window would make it steadier.

---

## Acknowledgements

Match, player and ranking data from
[Jeff Sackmann's tennis_atp](https://github.com/JeffSackmann/tennis_atp), whose
meticulous curation makes this project possible.
