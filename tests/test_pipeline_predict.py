"""End-to-end: train on the synthetic corpus, then load and serve predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tennis_engine import calibration, evaluate, pipeline
from tennis_engine.config import ModelConfig
from tennis_engine.predict import PlayerNotFound, PredictionEngine


@pytest.fixture(scope="module")
def _fast_model() -> ModelConfig:
    return ModelConfig(num_boost_round=40, early_stopping_rounds=10, max_depth=3)


@pytest.fixture
def trained(config, _fast_model):
    cfg = type(config)(
        paths=config.paths, split=config.split, elo=config.elo,
        features=config.features, model=_fast_model,
    )
    metrics = pipeline.run(cfg)
    return cfg, metrics


# ------------------------------------------------------------------ pipeline

def test_pipeline_produces_artifacts_and_metrics(trained):
    cfg, metrics = trained
    assert cfg.paths.model_path.exists()
    assert cfg.paths.state_path.exists()
    assert cfg.paths.metrics_path.exists()

    for split in ("validation", "test"):
        m = metrics[split]
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["auc"] <= 1.0
        assert m["log_loss"] > 0.0
        assert m["n"] > 0


def test_test_set_is_one_row_per_match(trained):
    _, metrics = trained
    # Training mirrors every match; evaluation must not, or the reported match
    # count is double the number of real matches played.
    assert metrics["dataset"]["train_rows"] == 2 * metrics["dataset"]["train_matches"]
    assert metrics["dataset"]["test_matches"] > 0


def test_baselines_are_reported(trained):
    _, metrics = trained
    baselines = metrics["test_baselines"]
    assert {"elo_global", "elo_surface_blended", "atp_rank"} <= set(baselines)
    assert baselines["bookmaker_free_coinflip"]["accuracy"] == 0.5


def test_drift_breakdown_covers_the_season(trained):
    _, metrics = trained
    drift = metrics["test_drift"]
    assert drift, "expected a temporal breakdown of the test season"
    starts = [block["start_day"] for block in drift.values()]
    assert starts == sorted(starts)
    assert sum(block["n"] for block in drift.values()) == metrics["dataset"]["test_matches"]


def test_symmetry_is_exact_after_symmetrization(trained):
    _, metrics = trained
    # Averaging p(A,B) with 1 - p(B,A) makes order invariance exact, not merely
    # "small on average" as the original assert checked.
    assert metrics["symmetry"]["max_abs_error"] < 1e-6


def test_calibrator_recorded_in_metrics(trained):
    _, metrics = trained
    assert metrics["model"]["calibrator"] in {"identity", "temperature", "isotonic"}
    assert metrics["model"]["best_iteration"] >= 0


# ----------------------------------------------------------------- inference

def test_engine_predicts_and_is_order_invariant(trained):
    cfg, _ = trained
    engine = PredictionEngine.load(cfg)
    names = list(engine.name_to_id)[:2]

    forward = engine.predict_match(names[0], names[1], surface="Hard")
    reverse = engine.predict_match(names[1], names[0], surface="Hard")
    assert forward.p1_win_prob == pytest.approx(reverse.p2_win_prob, abs=1e-6)
    assert forward.p1_win_prob + forward.p2_win_prob == pytest.approx(1.0)
    assert forward.p1_decimal_odds > 1.0


def test_inference_features_match_training_features(trained):
    """Train/serve parity: the served vector must be the training vector.

    The original notebook re-implemented the inference row by hand and got it
    wrong -- among other things it wrote ``rank_points_diff``, a column that has
    never existed, so rank and points were fed to the model as zeros.
    """
    cfg, _ = trained
    engine = PredictionEngine.load(cfg)
    ids = list(engine.builder.players)[:2]

    vec = engine.feature_vector(ids[0], ids[1], surface="Clay",
                                match_date=pd.Timestamp("2022-03-01"))
    assert vec.shape == (len(engine._names),)
    named = dict(zip(engine._names, vec))
    assert named["surface_is_clay"] == 1.0
    assert named["surface_is_hard"] == 0.0
    assert named["elo_diff"] != 0.0 or named["rank_diff"] != 0.0


def test_unknown_player_raises_with_suggestions(trained):
    cfg, _ = trained
    engine = PredictionEngine.load(cfg)
    with pytest.raises(PlayerNotFound):
        engine.predict_match("Definitely Notaplayer", "Also Notaplayer")


def test_self_match_rejected(trained):
    cfg, _ = trained
    engine = PredictionEngine.load(cfg)
    name = next(iter(engine.name_to_id))
    with pytest.raises(ValueError, match="cannot play themselves"):
        engine.predict_match(name, name)


def test_invalid_surface_rejected(trained):
    cfg, _ = trained
    engine = PredictionEngine.load(cfg)
    names = list(engine.name_to_id)[:2]
    with pytest.raises(ValueError, match="surface must be one of"):
        engine.predict_match(names[0], names[1], surface="Mud")


def test_player_card_and_leaderboard(trained):
    cfg, _ = trained
    engine = PredictionEngine.load(cfg)
    pid = next(iter(engine.builder.players))
    card = engine.player_card(pid)
    assert card["player_id"] == pid
    assert card["matches_played"] > 0

    board = engine.elo_leaderboard(top_n=5, min_matches=1, active_since=None)
    assert len(board) <= 5
    assert board["elo"].is_monotonic_decreasing


# --------------------------------------------------------------- calibration

def test_temperature_calibration_is_exactly_symmetric():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.98, 500)
    y = (rng.uniform(size=500) < p).astype(int)
    cal = calibration.TemperatureCalibrator().fit(p, y)
    np.testing.assert_allclose(
        cal.transform(p) + cal.transform(1 - p), np.ones_like(p), atol=1e-12
    )


def test_isotonic_calibration_is_symmetric():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.02, 0.98, 2000)
    y = (rng.uniform(size=2000) < p**2).astype(int)
    cal = calibration.SymmetricIsotonicCalibrator().fit(p, y)
    np.testing.assert_allclose(
        cal.transform(p) + cal.transform(1 - p), np.ones_like(p), atol=1e-9
    )


def test_calibrator_roundtrips_through_dict():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.05, 0.95, 300)
    y = (rng.uniform(size=300) < p).astype(int)
    for cal in (calibration.TemperatureCalibrator().fit(p, y),
                calibration.SymmetricIsotonicCalibrator().fit(p, y),
                calibration.IdentityCalibrator()):
        restored = calibration.from_dict(cal.to_dict())
        np.testing.assert_allclose(restored.transform(p), cal.transform(p), atol=1e-9)


def test_calibration_improves_a_miscalibrated_model():
    rng = np.random.default_rng(3)
    true_p = rng.uniform(0.05, 0.95, 5000)
    y = (rng.uniform(size=5000) < true_p).astype(int)
    overconfident = np.clip(1 / (1 + np.exp(-2.5 * np.log(true_p / (1 - true_p)))), 1e-6, 1 - 1e-6)

    best, scores = calibration.select_calibrator(overconfident, y)
    assert scores[best.name] == min(scores.values())
    assert scores[best.name] < scores["identity"]


# ------------------------------------------------------------------- metrics

def test_expected_calibration_error_bounds():
    y = np.array([0, 1, 0, 1])
    assert evaluate.expected_calibration_error(y, np.array([0.0, 1.0, 0.0, 1.0])) == pytest.approx(0.0)
    assert evaluate.expected_calibration_error(y, np.array([1.0, 0.0, 1.0, 0.0])) == pytest.approx(1.0)


def test_classification_metrics_shape():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 200)
    p = rng.uniform(0.01, 0.99, 200)
    m = evaluate.classification_metrics(y, p)
    assert set(m) == {"n", "accuracy", "auc", "log_loss", "brier", "ece"}
