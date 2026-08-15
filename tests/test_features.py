"""Feature construction: leak-proofing, symmetry and missing-value semantics."""

from __future__ import annotations

import numpy as np
import pytest

from tennis_engine.data import prepare_matches
from tennis_engine.features import (
    ANTISYMMETRIC_FEATURES,
    FEATURE_INDEX,
    FEATURE_NAMES,
    FLIP_SIGN,
    SYMMETRIC_FEATURES,
    FeatureBuilder,
)
from tennis_engine.rankings import build_ranking_lookup


@pytest.fixture
def builder(data_dir, config):
    return FeatureBuilder(config, build_ranking_lookup(data_dir))


@pytest.fixture
def matches(data_dir):
    return prepare_matches(data_dir, [2019, 2020, 2021])


# ------------------------------------------------------------------- schema

def test_schema_is_consistent():
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)), "duplicate feature name"
    assert len(FLIP_SIGN) == len(FEATURE_NAMES)
    assert set(ANTISYMMETRIC_FEATURES).isdisjoint(SYMMETRIC_FEATURES)
    for name in ANTISYMMETRIC_FEATURES:
        assert FLIP_SIGN[FEATURE_INDEX[name]] == -1.0
    for name in SYMMETRIC_FEATURES:
        assert FLIP_SIGN[FEATURE_INDEX[name]] == 1.0


def test_surface_one_hot_is_complete(builder, matches):
    """Regression: the original wrote only the active surface key into a dict,
    so every other surface column came out NaN rather than 0 -- the encoding was
    never actually one-hot."""
    built = builder.build(matches, mirror=False)
    cols = [FEATURE_INDEX[n] for n in FEATURE_NAMES if n.startswith("surface_is_")]
    block = built.X[:, cols]
    assert not np.isnan(block).any()
    assert np.array_equal(block.sum(axis=1), np.ones(len(built)))


# ------------------------------------------------------------ order invariance

def test_flip_is_an_involution(builder, matches):
    built = builder.build(matches, mirror=False)
    twice = built.X * FLIP_SIGN * FLIP_SIGN
    np.testing.assert_allclose(twice, built.X, equal_nan=True)


def test_mirrored_training_rows_are_exact_negations(builder, matches):
    built = builder.build(matches, mirror=True)
    assert len(built) == 2 * len(matches)
    forward, mirrored = built.X[0::2], built.X[1::2]
    np.testing.assert_allclose(mirrored, forward * FLIP_SIGN, equal_nan=True)
    assert (built.y[0::2] == 1).all()
    assert (built.y[1::2] == 0).all()


def test_match_vector_is_antisymmetric_in_player_order(builder, matches):
    """Building A-vs-B and B-vs-A from identical state must give exact negations
    on the antisymmetric block and identical values on the symmetric block."""
    builder.build(matches.iloc[:200], mirror=False)  # warm the state up
    a, b = int(matches.iloc[0]["winner_id"]), int(matches.iloc[0]["loser_id"])

    ab = builder.match_vector(a, b, surface_idx=0, day=18_000, tier_ord=2, best_of=3)
    ba = builder.match_vector(b, a, surface_idx=0, day=18_000, tier_ord=2, best_of=3)
    np.testing.assert_allclose(ba, ab * FLIP_SIGN, rtol=1e-5, atol=1e-5, equal_nan=True)


# -------------------------------------------------------------- leak-proofing

def test_first_appearance_has_no_history(builder, matches):
    """The very first row a player appears in must carry zero prior information."""
    built = builder.build(matches, mirror=False)
    frame = built.as_frame()
    first_row = frame.iloc[0]
    for name in ("elo_diff", "surf_elo_diff", "h2h_diff", "h2h_total",
                 "h2h_surface_diff", "workload_diff"):
        assert first_row[name] == 0.0, name
    for name in ("form_diff", "career_winrate_diff", "ace_rate_diff",
                 "srv_winrate_diff", "log_days_since_diff"):
        assert np.isnan(first_row[name]), name


def test_state_is_updated_only_after_emitting(config, data_dir, matches):
    """Replaying the same two players twice: the second meeting must see exactly
    one prior h2h match, and the first must see none."""
    builder = FeatureBuilder(config, build_ranking_lookup(data_dir))
    counts = matches.groupby(["winner_id", "loser_id"]).size()
    repeated = counts[counts >= 2]
    assert not repeated.empty, "fixture should generate repeated pairings"
    winner, loser = repeated.index[0]

    pair = matches[(matches["winner_id"] == winner) & (matches["loser_id"] == loser)]
    built = builder.build(pair, mirror=False)
    h2h_total = built.as_frame()["h2h_total"].to_numpy()
    assert h2h_total[0] == 0, "first meeting must see no prior h2h"
    assert h2h_total[1] == 1, "second meeting must see exactly one prior h2h"
    assert (np.diff(h2h_total) == 1).all()


def test_elo_diff_predicts_outcome_better_than_chance(builder, matches):
    """A sanity floor: with latent strengths in the fixture, Elo alone should
    beat a coin flip. If this fails, the state machine is not learning."""
    built = builder.build(matches, mirror=False)
    elo = built.X[:, FEATURE_INDEX["elo_diff"]]
    late = built.date > np.percentile(built.date, 50)
    agreement = ((elo[late] > 0) == (built.y[late] == 1)).mean()
    assert agreement > 0.60


def test_non_chronological_input_is_rejected(builder, matches):
    shuffled = matches.iloc[::-1]
    with pytest.raises(ValueError, match="chronologically sorted"):
        builder.build(shuffled, mirror=False)


def test_warmup_mode_updates_state_without_emitting(config, data_dir, matches):
    builder = FeatureBuilder(config, build_ranking_lookup(data_dir))
    built = builder.build(matches, mirror=False, emit=False)
    assert len(built) == 0
    assert builder.n_matches_seen == len(matches)
    assert len(builder.players) > 0


def test_no_feature_is_constant(builder, matches):
    """A feature with a single value across the corpus is dead code -- this is
    what silently happened to ``srv_winrate_diff`` in the original pipeline."""
    built = builder.build(matches, mirror=False)
    dead = []
    for name in FEATURE_NAMES:
        col = built.X[:, FEATURE_INDEX[name]]
        finite = col[np.isfinite(col)]
        if name == "surface_is_unknown" or len(finite) == 0:
            continue
        if np.allclose(finite, finite[0]):
            dead.append(name)
    assert not dead, f"constant (dead) features: {dead}"


def test_metadata_arrays_align(builder, matches):
    built = builder.build(matches, mirror=True)
    n = len(built)
    assert built.y.shape == built.weight.shape == (n,)
    assert built.date.shape == built.tier.shape == built.min_prior.shape == (n,)
    assert built.X.shape == (n, len(FEATURE_NAMES))
    assert (built.min_prior >= 0).all()
    assert np.diff(built.date).min() >= 0
