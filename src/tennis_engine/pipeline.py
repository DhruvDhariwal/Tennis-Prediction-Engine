"""End-to-end training pipeline: data -> features -> model -> artifacts."""

from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb

from . import calibration, evaluate, reporting
from .config import Config, ModelConfig, display_path
from .data import build_player_attributes, load_player_directory, prepare_matches
from .features import FEATURE_NAMES, BuiltFeatures, FeatureBuilder
from .rankings import build_ranking_lookup
from .state import pack_counter, serialize_player

logger = logging.getLogger(__name__)

ARTIFACT_VERSION = 2


@dataclass
class Splits:
    train: BuiltFeatures
    val: BuiltFeatures
    test: BuiltFeatures
    n_warmup_matches: int
    n_players: int
    n_ranking_rows: int


def build_splits(cfg: Config) -> tuple[Splits, FeatureBuilder]:
    """Single chronological pass producing warmup / train / val / test.

    Doing this in one pass -- rather than building a train set and then
    ``deepcopy``-ing the builder -- guarantees that ratings flow continuously
    across split boundaries and removes a 100 MB+ copy of the whole state.
    """
    t0 = time.perf_counter()
    years = cfg.split.all_years()

    matches = prepare_matches(
        cfg.paths.base_dir,
        years,
        include_challengers=cfg.include_challengers,
        include_futures=cfg.include_futures,
    )
    rankings = build_ranking_lookup(cfg.paths.base_dir)
    players = load_player_directory(cfg.paths.players_csv)
    attrs = build_player_attributes(players)
    logger.info("Player directory: %s entries", f"{len(attrs):,}")

    builder = FeatureBuilder(cfg, rankings, attrs)
    season = matches["tourney_date"].dt.year.to_numpy()

    def subset(sel_years) -> pd.DataFrame:
        return matches.loc[np.isin(season, list(sel_years))]

    warmup_df = subset(cfg.split.warmup_years)
    builder.build(warmup_df, mirror=False, emit=False)

    train = builder.build(subset(cfg.split.train_years), mirror=True)
    val = builder.build(subset(cfg.split.val_years), mirror=False)
    test = builder.build(subset(cfg.split.test_years), mirror=False)

    logger.info(
        "Splits built in %.1fs -- warmup %s matches | train %s rows | val %s rows | test %s rows",
        time.perf_counter() - t0,
        f"{len(warmup_df):,}", f"{len(train):,}", f"{len(val):,}", f"{len(test):,}",
    )
    return (
        Splits(
            train=train, val=val, test=test,
            n_warmup_matches=len(warmup_df),
            n_players=len(builder.players),
            n_ranking_rows=rankings.n_rows,
        ),
        builder,
    )


def train_model(cfg: Config, splits: Splits) -> tuple[xgb.Booster, dict]:
    """Train XGBoost with early stopping on the *validation* season.

    The original pipeline early-stopped on the 2024 test set, so the reported
    2024 numbers were optimistically biased by model selection. Here 2024 is
    untouched until the final evaluation.
    """
    names = list(FEATURE_NAMES)
    dtrain = xgb.DMatrix(
        splits.train.X, label=splits.train.y,
        weight=splits.train.weight, feature_names=names,
    )
    dval = xgb.DMatrix(splits.val.X, label=splits.val.y, feature_names=names)

    history: dict = {}
    booster = xgb.train(
        params=cfg.model.to_params(),
        dtrain=dtrain,
        num_boost_round=cfg.model.num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=cfg.model.early_stopping_rounds,
        evals_result=history,
        verbose_eval=100,
    )
    logger.info(
        "Best iteration %s (val logloss %.5f)",
        booster.best_iteration, history["val"]["logloss"][booster.best_iteration],
    )
    return booster, history


#: Search space for :func:`tune`. Deliberately centred on the defaults so the
#: search cannot wander somewhere pathological on a single validation season.
SEARCH_SPACE: dict[str, list] = {
    "max_depth": [4, 5, 6, 7, 8],
    "min_child_weight": [5.0, 10.0, 20.0, 50.0, 100.0],
    "learning_rate": [0.04, 0.05, 0.06, 0.08],
    "subsample": [0.6, 0.7, 0.8, 0.9],
    "colsample_bytree": [0.5, 0.6, 0.8, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
    "reg_alpha": [0.0, 0.1, 1.0],
}


def tune(
    cfg: Config,
    splits: Splits,
    n_trials: int = 20,
    max_rounds: int = 900,
) -> tuple[ModelConfig, list[dict]]:
    """Random search over XGBoost hyperparameters, scored on the validation season.

    The test season is never used here -- that is the whole point. The original
    project shipped whatever the first hand-picked parameter set produced.

    ``max_rounds`` caps each trial so the search stays bounded; the learning
    rates in the space all converge comfortably inside that budget, and the
    winning configuration is refit afterwards with the full round allowance.
    """
    from dataclasses import replace

    names = list(FEATURE_NAMES)
    dtrain = xgb.DMatrix(
        splits.train.X, label=splits.train.y,
        weight=splits.train.weight, feature_names=names,
    )
    dval = xgb.DMatrix(splits.val.X, label=splits.val.y, feature_names=names)

    rng = np.random.default_rng(cfg.random_state)
    trials: list[dict] = []
    best_cfg, best_loss = cfg.model, float("inf")

    # Trial 0 is always the current default, so the search can only improve on it.
    candidates = [cfg.model]
    for _ in range(n_trials - 1):
        candidates.append(replace(
            cfg.model,
            num_boost_round=max_rounds,
            **{k: type(v[0])(rng.choice(v)) for k, v in SEARCH_SPACE.items()},
        ))

    for i, candidate in enumerate(candidates):
        history: dict = {}
        booster = xgb.train(
            params=candidate.to_params(),
            dtrain=dtrain,
            num_boost_round=min(candidate.num_boost_round, max_rounds),
            evals=[(dval, "val")],
            early_stopping_rounds=cfg.model.early_stopping_rounds,
            evals_result=history,
            verbose_eval=False,
        )
        loss = float(history["val"]["logloss"][booster.best_iteration])
        auc = float(history["val"]["auc"][booster.best_iteration])
        trial = {
            "trial": i,
            "val_logloss": round(loss, 6),
            "val_auc": round(auc, 6),
            "best_iteration": int(booster.best_iteration),
            **{k: getattr(candidate, k) for k in SEARCH_SPACE},
        }
        trials.append(trial)
        if loss < best_loss:
            best_loss, best_cfg = loss, replace(
                candidate, num_boost_round=cfg.model.num_boost_round
            )
            logger.info("Trial %2d/%d  val logloss %.5f  auc %.5f  <- new best",
                        i + 1, len(candidates), loss, auc)
        else:
            logger.info("Trial %2d/%d  val logloss %.5f  auc %.5f",
                        i + 1, len(candidates), loss, auc)

    trials.sort(key=lambda t: t["val_logloss"])
    logger.info("Best hyperparameters (val logloss %.5f): %s",
                best_loss, {k: getattr(best_cfg, k) for k in SEARCH_SPACE})
    return best_cfg, trials


def make_predictor(booster: xgb.Booster, calibrator=None):
    """Symmetrized, calibrated prediction function.

    ``p = 0.5 * (p(A,B) + 1 - p(B,A))`` makes order-invariance exact rather than
    approximate, at the cost of one extra forward pass.
    """
    from .features import FLIP_SIGN

    names = list(FEATURE_NAMES)
    best = getattr(booster, "best_iteration", None)
    kwargs = {"iteration_range": (0, best + 1)} if best is not None else {}

    def predict(X: np.ndarray, symmetrize: bool = True) -> np.ndarray:
        # float64 throughout: XGBoost returns float32, and a calibrator with
        # steep segments turns a 1e-7 rounding difference between p(A,B) and
        # 1 - p(B,A) into a visible asymmetry.
        raw = booster.predict(xgb.DMatrix(X, feature_names=names), **kwargs).astype(np.float64)
        if symmetrize:
            mirrored = booster.predict(
                xgb.DMatrix(X * FLIP_SIGN, feature_names=names), **kwargs
            ).astype(np.float64)
            raw = 0.5 * (raw + (1.0 - mirrored))
        return calibrator.transform(raw) if calibrator is not None else raw

    return predict


def evaluate_all(
    cfg: Config, splits: Splits, booster: xgb.Booster
) -> tuple[dict, object]:
    """Calibrate on validation, then report honest test-season metrics."""
    raw_predict = make_predictor(booster, calibrator=None)

    p_val_raw = raw_predict(splits.val.X)
    calibrator, calib_scores = calibration.select_calibrator(p_val_raw, splits.val.y)
    logger.info("Calibrator selected: %s (val log loss by method: %s)",
                calibrator.name, calib_scores)

    predict = make_predictor(booster, calibrator)
    p_val = predict(splits.val.X)
    p_test = predict(splits.test.X)
    p_test_uncal = raw_predict(splits.test.X)

    metrics = {
        "artifact_version": ARTIFACT_VERSION,
        "split": {
            "warmup_years": _year_range(cfg.split.warmup_years),
            "train_years": _year_range(cfg.split.train_years),
            "val_years": [int(y) for y in cfg.split.val_years],
            "test_years": [int(y) for y in cfg.split.test_years],
        },
        "dataset": {
            "warmup_matches": splits.n_warmup_matches,
            "train_rows": int(len(splits.train)),
            "train_matches": int(len(splits.train) // 2),
            "val_matches": int(len(splits.val)),
            "test_matches": int(len(splits.test)),
            "players_tracked": splits.n_players,
            "ranking_snapshots": splits.n_ranking_rows,
            "n_features": len(FEATURE_NAMES),
        },
        "model": {
            "best_iteration": int(booster.best_iteration),
            "n_trees": int(booster.num_boosted_rounds()),
            "params": cfg.model.to_params(),
            "calibrator": calibrator.name,
            "calibration_val_logloss": calib_scores,
        },
        "validation": evaluate.classification_metrics(splits.val.y, p_val),
        "test": evaluate.classification_metrics(splits.test.y, p_test),
        "test_uncalibrated": evaluate.classification_metrics(splits.test.y, p_test_uncal),
        "test_baselines": evaluate.baseline_suite(splits.test.X, splits.test.y),
        "test_slices": evaluate.slice_metrics(
            splits.test.y, p_test, splits.test.tier, splits.test.min_prior
        ),
        "test_drift": evaluate.temporal_metrics(
            splits.test.y, p_test, splits.test.date
        ),
        "symmetry": evaluate.symmetry_error(predict, splits.test.X[:5000]),
        "feature_importance_gain": _importance(booster),
    }

    baseline_ll = metrics["test_baselines"]["elo_surface_blended"]["log_loss"]
    metrics["test"]["log_loss_reduction_vs_elo_pct"] = round(
        100.0 * (baseline_ll - metrics["test"]["log_loss"]) / baseline_ll, 3
    )

    reporting.write_figures(cfg.paths.figures_dir, splits.test.X, splits.test.y,
                            p_test, metrics)
    return metrics, calibrator


def _year_range(years) -> list[int]:
    """``[first, last]`` for a season range, or ``[]`` when the split is empty."""
    years = list(years)
    return [int(min(years)), int(max(years))] if years else []


def _importance(booster: xgb.Booster) -> dict[str, float]:
    gain = booster.get_score(importance_type="gain")
    total = sum(gain.values()) or 1.0
    ranked = sorted(gain.items(), key=lambda kv: kv[1], reverse=True)
    return {k: round(100.0 * v / total, 3) for k, v in ranked}


def save_artifacts(
    cfg: Config,
    booster: xgb.Booster,
    builder: FeatureBuilder,
    calibrator,
    metrics: dict,
) -> None:
    """Persist model, calibrator, engine state and metrics."""
    cfg.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(cfg.paths.model_path)

    state = {
        "artifact_version": ARTIFACT_VERSION,
        "players": {pid: serialize_player(s) for pid, s in builder.players.items()},
        "h2h": pack_counter(builder.h2h),
        "h2h_surface": pack_counter(builder.h2h_surface),
        "rankings": {
            "dates": builder.rankings.dates,
            "ranks": builder.rankings.ranks,
            "points": builder.rankings.points,
            "n_rows": builder.rankings.n_rows,
        },
        "player_attrs": builder.player_attrs,
        "calibrator": calibrator.to_dict(),
        "config": {
            "feature_names": list(FEATURE_NAMES),
            "elo": vars(cfg.elo),
            "features": vars(cfg.features),
            "last_state_date": builder.last_date,
            "n_matches_seen": builder.n_matches_seen,
        },
    }
    with open(cfg.paths.state_path, "wb") as fh:
        pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)

    with open(cfg.paths.metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    size_mb = cfg.paths.state_path.stat().st_size / 1e6
    logger.info(
        "Artifacts written to %s (state %.1f MB)",
        display_path(cfg.paths.artifacts_dir), size_mb,
    )


def run(cfg: Config, n_trials: int = 0) -> dict:
    """Train, evaluate and persist. Returns the metrics dictionary.

    ``n_trials > 0`` runs a validation-scored hyperparameter search first.
    """
    from dataclasses import replace

    splits, builder = build_splits(cfg)

    trials: list[dict] = []
    if n_trials > 0:
        best_model_cfg, trials = tune(cfg, splits, n_trials=n_trials)
        cfg = replace(cfg, model=best_model_cfg)

    booster, _ = train_model(cfg, splits)
    metrics, calibrator = evaluate_all(cfg, splits, booster)
    if trials:
        metrics["tuning"] = {"n_trials": n_trials, "trials": trials[:10]}
    save_artifacts(cfg, booster, builder, calibrator, metrics)
    return metrics
