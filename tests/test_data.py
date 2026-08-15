"""Cleaning, column auditing and the bugs that used to hide there."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tennis_engine.data import (
    BOXSCORE_COLUMNS,
    build_name_index,
    POST_MATCH_COLUMNS,
    SHIPPED_RANK_COLUMNS,
    audit_and_drop_leaky_columns,
    deduplicate_matches,
    drop_incomplete_matches,
    load_player_directory,
    normalize_surface,
    prepare_matches,
    sort_chronologically,
    tier_for_file,
)


def test_audit_keeps_serve_columns():
    """Regression: the old regex ``...|wo|...`` matched "Won" and deleted the
    columns that ``srv_winrate_diff`` is computed from, silently zeroing the
    feature for every row."""
    frame = pd.DataFrame(
        {c: [1] for c in ("score", "minutes", "winner_rank") + BOXSCORE_COLUMNS}
    )
    cleaned, dropped = audit_and_drop_leaky_columns(frame)

    for col in ("w_1stWon", "w_2ndWon", "l_1stWon", "l_2ndWon"):
        assert col in cleaned.columns, f"{col} must survive the leakage audit"
    assert set(dropped) == {"score", "minutes", "winner_rank"}


def test_audit_drops_outcome_columns():
    frame = pd.DataFrame({c: [1] for c in POST_MATCH_COLUMNS + SHIPPED_RANK_COLUMNS})
    cleaned, dropped = audit_and_drop_leaky_columns(frame)
    assert cleaned.empty or len(cleaned.columns) == 0
    assert set(dropped) == set(POST_MATCH_COLUMNS + SHIPPED_RANK_COLUMNS)


def test_incomplete_matches_removed():
    frame = pd.DataFrame({"score": ["6-4 6-3", "6-4 RET", "W/O", "3-6 7-6 DEF", None]})
    cleaned, n = drop_incomplete_matches(frame)
    assert n == 3
    assert len(cleaned) == 2


def test_unknown_surface_is_not_silently_hard():
    """The original mapped every unlabelled surface to Hard, fabricating
    surface-Elo history for those matches."""
    out = normalize_surface(pd.Series(["Hard", "clay", None, "", "Carpet", "H"]))
    assert list(out) == ["Hard", "Clay", "Unknown", "Unknown", "Carpet", "Hard"]


def test_tier_detection():
    assert tier_for_file("atp_matches_futures_2019.csv") == "futures"
    assert tier_for_file("atp_matches_qual_chall_2019.csv") == "challenger"
    assert tier_for_file("atp_matches_2019.csv") == "atp"


def test_dedup_and_sort():
    frame = pd.DataFrame({
        "source_file": ["a"] * 3,
        "tourney_id": ["T1", "T1", "T2"],
        "tourney_date": pd.to_datetime(["2020-01-06"] * 2 + ["2020-01-01"]),
        "match_num": [1, 1, 5],
    })
    deduped, n = deduplicate_matches(frame)
    assert n == 1 and len(deduped) == 2
    ordered = sort_chronologically(deduped)
    assert ordered["tourney_date"].is_monotonic_increasing


def test_prepare_matches_end_to_end(data_dir):
    df = prepare_matches(data_dir, [2019, 2020, 2021])
    assert len(df) > 0
    assert df["tourney_date"].is_monotonic_increasing
    assert df["winner_id"].dtype == np.int64
    assert not (df["winner_id"] == df["loser_id"]).any()
    for col in POST_MATCH_COLUMNS + SHIPPED_RANK_COLUMNS:
        assert col not in df.columns
    assert "w_1stWon" in df.columns


def test_prepare_matches_missing_data(tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare_matches(tmp_path, [1999])


def test_name_index_skips_blank_and_prefers_the_active_namesake(tmp_path):
    """`atp_players.csv` has 931 duplicate full names, some of them empty.

    Zipping the columns into a dict keeps whichever row came last, so a blank
    query string resolved to an arbitrary player.
    """
    pd.DataFrame([
        {"player_id": 1, "name_first": "Alexander", "name_last": "Zverev",
         "hand": "R", "dob": 19670422, "ioc": "GER", "height": 190},
        {"player_id": 2, "name_first": "Alexander", "name_last": "Zverev",
         "hand": "R", "dob": 19970420, "ioc": "GER", "height": 198},
        {"player_id": 3, "name_first": "", "name_last": "",
         "hand": None, "dob": None, "ioc": None, "height": None},
        {"player_id": 4, "name_first": "Solo", "name_last": "Player",
         "hand": "L", "dob": 19900101, "ioc": "USA", "height": 180},
    ]).to_csv(tmp_path / "atp_players.csv", index=False)

    players = load_player_directory(tmp_path / "atp_players.csv")
    name_to_id, id_to_name = build_name_index(players)

    assert "" not in name_to_id
    assert name_to_id["alexander zverev"] == 2  # the younger, active namesake
    assert name_to_id["solo player"] == 4
    assert id_to_name[4] == "Solo Player"
