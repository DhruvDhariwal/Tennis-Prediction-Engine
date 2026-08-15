"""As-of ranking lookups, Elo updates and rolling-state bookkeeping."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tennis_engine.config import EloConfig
from tennis_engine.rankings import build_ranking_lookup, to_day_ordinal
from tennis_engine.state import (
    EloEngine,
    PlayerState,
    deserialize_player,
    expected_score,
    safe_rate,
    serialize_player,
)


# ------------------------------------------------------------------ rankings

def test_lookup_is_as_of_not_nearest(data_dir):
    lookup = build_ranking_lookup(data_dir)
    pid = next(iter(lookup.dates))
    dates = lookup.dates[pid]

    # Exactly on a snapshot date -> that snapshot is used (rankings published on
    # a Monday already reflect the previous week's results).
    rank_on, _, ok = lookup.get(pid, int(dates[2]))
    assert ok and rank_on == lookup.ranks[pid][2]

    # One day later -> still the same snapshot, never the following week's.
    rank_after, _, _ = lookup.get(pid, int(dates[2]) + 1)
    assert rank_after == lookup.ranks[pid][2]

    # One day earlier -> the previous snapshot.
    rank_before, _, _ = lookup.get(pid, int(dates[2]) - 1)
    assert rank_before == lookup.ranks[pid][1]


def test_lookup_before_first_snapshot_is_unranked(data_dir):
    lookup = build_ranking_lookup(data_dir)
    pid = next(iter(lookup.dates))
    rank, points, is_ranked = lookup.get(pid, int(lookup.dates[pid][0]) - 1)
    assert is_ranked is False
    assert rank == lookup.default_rank and points == lookup.default_points


def test_lookup_unknown_player(data_dir):
    lookup = build_ranking_lookup(data_dir)
    rank, _, is_ranked = lookup.get(-1, 20_000)
    assert is_ranked is False and rank == lookup.default_rank


def test_lookup_handles_headerless_files(tmp_path):
    """Real ranking CSVs ship both with and without a header row."""
    rows = pd.DataFrame({
        "ranking_date": [20200106, 20200113],
        "rank": [1, 2],
        "player": [104925, 104925],
        "points": [10000, 9000],
    })
    rows.to_csv(tmp_path / "atp_rankings_20s.csv", index=False)
    rows.to_csv(tmp_path / "atp_rankings_10s.csv", index=False, header=False)

    lookup = build_ranking_lookup(tmp_path)
    assert lookup.n_rows == 2  # duplicates across the two files collapse
    assert len(lookup.dates[104925]) == 2


def test_day_ordinal_roundtrip():
    ts = pd.Timestamp("2024-06-03")
    assert int(to_day_ordinal(ts)) == int(np.datetime64("2024-06-03", "D").astype(int))


# ----------------------------------------------------------------------- Elo

def test_expected_score_symmetry():
    assert expected_score(1500, 1500) == pytest.approx(0.5)
    assert expected_score(1700, 1500) + expected_score(1500, 1700) == pytest.approx(1.0)
    assert expected_score(1900, 1500) > 0.9


def test_elo_moves_toward_winner():
    engine = EloEngine(EloConfig(schedule="fixed", fixed_k=32.0))
    w, l = PlayerState(), PlayerState()
    engine.update(w, l, surface_idx=0, tier="atp")
    assert w.elo > 1500 > l.elo
    assert w.elo - 1500 == pytest.approx(1500 - l.elo)  # fixed K is zero-sum
    assert w.surf_elo[0] > 1500 > l.surf_elo[0]
    assert w.surf_elo[1] == 1500  # other surfaces untouched


def test_dynamic_k_decays_with_experience():
    engine = EloEngine(EloConfig(schedule="dynamic"))
    assert engine.k_factor(0, "atp") > engine.k_factor(200, "atp")
    assert engine.k_factor(50, "futures") < engine.k_factor(50, "atp")


def test_upset_moves_more_than_expected_win():
    engine = EloEngine(EloConfig(schedule="fixed", fixed_k=32.0))
    fav, dog = PlayerState(), PlayerState()
    fav.elo, dog.elo = 1900.0, 1500.0
    before = dog.elo
    engine.update(dog, fav, 0, "atp")          # upset
    upset_gain = dog.elo - before

    fav2, dog2 = PlayerState(), PlayerState()
    fav2.elo, dog2.elo = 1900.0, 1500.0
    before2 = fav2.elo
    engine.update(fav2, dog2, 0, "atp")        # chalk
    assert upset_gain > (fav2.elo - before2)


def test_surface_blend_shrinks_toward_global():
    state = PlayerState()
    state.elo = 1800.0
    state.surf_elo[0] = 1400.0
    assert state.blended_surface_elo(0, 0.0) == 1400.0
    assert state.blended_surface_elo(0, 1.0) == 1800.0
    assert 1400.0 < state.blended_surface_elo(0, 0.35) < 1800.0


# --------------------------------------------------------------- rolling state

def test_missing_stats_do_not_poison_the_mean():
    """Regression: NaN box scores used to be appended verbatim (or coerced to
    0.0), which made every subsequent rolling average NaN or wrong."""
    state = PlayerState(roll_n=5)
    state.record_stats((0.10, 0.05, 0.65, 0.35, 0.60))
    state.record_stats((math.nan,) * 5)
    state.record_stats((0.20, 0.05, 0.65, 0.35, 0.60))
    assert state.stat_mean(0) == pytest.approx(0.15)
    assert state.stats_count[0] == 2


def test_stat_mean_is_nan_without_history():
    assert math.isnan(PlayerState().stat_mean(0))
    assert math.isnan(PlayerState().form_rate())
    assert math.isnan(PlayerState().career_winrate())


def test_rolling_window_evicts_correctly():
    state = PlayerState(roll_n=3)
    for value in (0.1, 0.2, 0.3, 0.4):
        state.record_stats((value, math.nan, math.nan, math.nan, math.nan))
    assert state.stat_mean(0) == pytest.approx((0.2 + 0.3 + 0.4) / 3)
    assert state.stats_count[0] == 3


def test_form_tracks_last_n_only():
    state = PlayerState(roll_n=4)
    for won in (True, True, True, True, False, False):
        state.record_result(won, surface_idx=0, day=1)
    assert state.form_rate() == pytest.approx(0.5)
    assert state.career_winrate() == pytest.approx(4 / 6)


def test_workload_window_counts_without_mutating():
    state = PlayerState()
    for day in (0, 30, 60, 200):
        state.record_result(True, 0, day)

    # Counting is read-only, so an out-of-order query cannot destroy history.
    assert state.matches_in_window(day=200, window_days=90) == 1
    assert state.matches_in_window(day=60, window_days=90) == 3
    assert state.matches_in_window(day=200, window_days=90) == 1
    assert len(state.recent_dates) == 4

    # Pruning is the explicit, forward-only operation.
    state.prune_window(day=200, window_days=90)
    assert len(state.recent_dates) == 1
    assert state.days_since_last(210) == 10


def test_days_since_last_never_negative():
    state = PlayerState()
    state.record_result(True, 0, day=500)
    assert state.days_since_last(400) == 0.0
    assert state.days_since_last(510) == 10.0


def test_safe_rate_returns_nan_not_zero():
    assert math.isnan(safe_rate(5, 0))
    assert math.isnan(safe_rate(math.nan, 60))
    assert math.isnan(safe_rate(5, math.nan))
    assert safe_rate(6, 60) == pytest.approx(0.1)


def test_player_state_roundtrip():
    state = PlayerState(roll_n=4)
    state.elo = 1723.5
    for i in range(6):
        state.record_stats((0.1 * i, math.nan, 0.6, 0.3, 0.5))
        state.record_result(i % 2 == 0, surface_idx=1, day=100 + i)

    restored = deserialize_player(serialize_player(state), roll_n=4)
    assert restored.elo == state.elo
    assert restored.matches_played == state.matches_played
    assert restored.form_rate() == pytest.approx(state.form_rate())
    assert restored.stat_mean(0) == pytest.approx(state.stat_mean(0))
    assert restored.surf_elo == state.surf_elo
    assert restored.last_date == state.last_date
