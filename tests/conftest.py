"""Synthetic ATP-shaped fixtures so the suite runs without the real 900 MB corpus."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tennis_engine.config import Config, Paths, SplitConfig
from tennis_engine.rankings import RankingLookup

SURFACE_CYCLE = ["Hard", "Clay", "Grass", "Carpet", "Unknown"]


#: Tier -> filename stem, mirroring the real Sackmann layout.
TIER_FILES = {
    "atp": "atp_matches_{year}.csv",
    "challenger": "atp_matches_qual_chall_{year}.csv",
    "futures": "atp_matches_futures_{year}.csv",
}


def make_matches(n_players: int = 60, n_matches: int = 1800, seed: int = 0) -> pd.DataFrame:
    """Matches generated from latent player strengths, in chronological order.

    Deliberately heterogeneous -- three tiers, five surfaces, best-of-3 and
    best-of-5, some matches with no box score, and a tail of players who never
    appear in the ranking files -- so that tests exercise the real branches
    rather than a single happy path.
    """
    rng = np.random.default_rng(seed)
    ids = np.arange(100_001, 100_001 + n_players)
    strength = rng.normal(0, 1, n_players)
    tiers = ["atp", "challenger", "futures"]

    rows = []
    start = pd.Timestamp("2019-01-07")
    for i in range(n_matches):
        a, b = rng.choice(n_players, size=2, replace=False)
        p_a = 1.0 / (1.0 + np.exp(-(strength[a] - strength[b])))
        a_wins = rng.random() < p_a
        w, l = (a, b) if a_wins else (b, a)
        date = start + pd.Timedelta(days=7 * (i // 12))
        surface = SURFACE_CYCLE[i % len(SURFACE_CYCLE)]
        tier = tiers[i % 3]
        # Roughly a fifth of matches carry no box score, like pre-1991 and much
        # of the Futures data.
        has_stats = rng.random() > 0.2
        svpt_w, svpt_l = rng.integers(50, 90, size=2)
        if not has_stats:
            svpt_w = svpt_l = np.nan
        rows.append({
            "tier": tier,
            "tourney_id": f"T{i // 12:04d}",
            "tourney_name": "Synthetic Open",
            "surface": surface,
            "draw_size": 32,
            "tourney_level": "A",
            "tourney_date": int(date.strftime("%Y%m%d")),
            "match_num": i % 12,
            "best_of": 5 if i % 7 == 0 else 3,
            "winner_id": ids[w],
            "winner_name": f"Player {w}",
            "winner_hand": "R",
            "winner_ht": 180 + (w % 20),
            "winner_age": 20 + (w % 15),
            "loser_id": ids[l],
            "loser_name": f"Player {l}",
            "loser_hand": "L" if l % 3 == 0 else "R",
            "loser_ht": 180 + (l % 20),
            "loser_age": 20 + (l % 15),
            "score": "6-4 RET" if i % 97 == 0 else "6-4 6-3",
            "minutes": 95,
            "w_ace": rng.integers(0, 15), "w_df": rng.integers(0, 6), "w_svpt": svpt_w,
            "w_1stIn": svpt_w // 2, "w_1stWon": svpt_w // 3, "w_2ndWon": svpt_w // 6,
            "w_SvGms": 10, "w_bpSaved": 3, "w_bpFaced": 5,
            "l_ace": rng.integers(0, 15), "l_df": rng.integers(0, 6), "l_svpt": svpt_l,
            "l_1stIn": svpt_l // 2, "l_1stWon": svpt_l // 3, "l_2ndWon": svpt_l // 6,
            "l_SvGms": 10, "l_bpSaved": 2, "l_bpFaced": 6,
            "winner_rank": 1 + w, "winner_rank_points": 5000 - 50 * w,
            "loser_rank": 1 + l, "loser_rank_points": 5000 - 50 * l,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def data_dir(tmp_path):
    """A directory laid out like the Sackmann repo, with three seasons."""
    matches = make_matches()
    matches["year"] = (matches["tourney_date"] // 10_000).astype(int)
    for (year, tier), group in matches.groupby(["year", "tier"]):
        path = tmp_path / TIER_FILES[tier].format(year=year)
        group.drop(columns=["year", "tier"]).to_csv(path, index=False)

    ids = sorted(set(matches["winner_id"]) | set(matches["loser_id"]))
    # The last few players are never ranked, so `is_ranked_diff` and the
    # unranked-default path are actually exercised.
    ranked_ids = ids[:-8]
    rank_rows = []
    for snapshot in pd.date_range("2019-01-07", "2021-12-27", freq="7D"):
        for rank, pid in enumerate(ranked_ids, start=1):
            rank_rows.append({
                "ranking_date": int(snapshot.strftime("%Y%m%d")),
                "rank": rank,
                "player": pid,
                "points": max(10, 5000 - 40 * rank),
            })
    pd.DataFrame(rank_rows).to_csv(tmp_path / "atp_rankings_10s.csv", index=False)

    pd.DataFrame([
        {
            "player_id": pid,
            "name_first": "Player",
            "name_last": f"Number{i}",
            "hand": "L" if i % 3 == 0 else "R",
            "dob": 19950101 + i,
            "ioc": "USA",
            "height": 175 + (i % 25),
            "wikidata_id": f"Q{i}",
        }
        for i, pid in enumerate(ids)
    ]).to_csv(tmp_path / "atp_players.csv", index=False)

    return tmp_path


@pytest.fixture
def config(data_dir, tmp_path) -> Config:
    return Config(
        paths=Paths(base_dir=data_dir, artifacts_dir=tmp_path / "artifacts"),
        split=SplitConfig(
            warmup_years=(),
            train_years=(2019,),
            val_years=(2020,),
            test_years=(2021,),
        ),
    )


@pytest.fixture
def empty_rankings() -> RankingLookup:
    return RankingLookup({}, {}, {})
