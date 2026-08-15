"""As-of ATP ranking lookup.

Weekly ranking snapshots are collapsed into per-player sorted arrays so that
"what was this player's rank immediately before date D" is an O(log n)
``searchsorted`` instead of a scan. The arrays are numpy, not Python lists, which
makes the lookup roughly an order of magnitude cheaper than the original
``bisect``-over-``list[Timestamp]`` implementation and cuts the pickled state
size substantially.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import DEFAULT_POINTS, DEFAULT_RANK

logger = logging.getLogger(__name__)

RANKING_FILES: tuple[str, ...] = (
    "atp_rankings_70s.csv",
    "atp_rankings_80s.csv",
    "atp_rankings_90s.csv",
    "atp_rankings_00s.csv",
    "atp_rankings_10s.csv",
    "atp_rankings_20s.csv",
    "atp_rankings_current.csv",
)

_COLUMNS = ["ranking_date", "rank", "player", "points"]


class RankingLookup:
    """Point-in-time ranking store.

    A single lookup covering the whole history is safe: every query is an as-of
    query bounded by the match date, so a 2024 snapshot can never be returned
    for a 2019 match. The original notebook maintained two separate lookups for
    train and test, which was redundant and easy to get wrong.
    """

    __slots__ = ("dates", "ranks", "points", "default_rank", "default_points", "n_rows")

    def __init__(
        self,
        dates: dict[int, np.ndarray],
        ranks: dict[int, np.ndarray],
        points: dict[int, np.ndarray],
        default_rank: int = DEFAULT_RANK,
        default_points: int = DEFAULT_POINTS,
        n_rows: int = 0,
    ) -> None:
        self.dates = dates
        self.ranks = ranks
        self.points = points
        self.default_rank = default_rank
        self.default_points = default_points
        self.n_rows = n_rows

    def __len__(self) -> int:
        return len(self.dates)

    def get(self, player_id: int, when: np.int64 | int) -> tuple[float, float, bool]:
        """Return ``(rank, points, is_ranked)`` as of ``when``.

        ``when`` is a day ordinal (``datetime64[D]`` cast to int). Players with
        no snapshot on or before that date are reported unranked, and the caller
        gets an explicit ``is_ranked`` flag rather than having to reverse-engineer
        the sentinel rank.
        """
        arr = self.dates.get(player_id)
        if arr is None:
            return float(self.default_rank), float(self.default_points), False
        idx = int(np.searchsorted(arr, when, side="right")) - 1
        if idx < 0:
            return float(self.default_rank), float(self.default_points), False
        return float(self.ranks[player_id][idx]), float(self.points[player_id][idx]), True


def _read_ranking_file(path: Path) -> pd.DataFrame:
    """Read one ranking CSV, tolerating both headed and headerless variants."""
    head = pd.read_csv(path, nrows=1, header=None)
    has_header = str(head.iloc[0, 0]).strip().lower() == "ranking_date"
    return pd.read_csv(
        path,
        header=0 if has_header else None,
        names=_COLUMNS,
        usecols=range(4),
        low_memory=False,
    )


def build_ranking_lookup(
    base_dir: Path | str,
    files: Iterable[str] = RANKING_FILES,
    max_date: pd.Timestamp | None = None,
) -> RankingLookup:
    """Build a :class:`RankingLookup` from the weekly ranking CSVs."""
    base_dir = Path(base_dir)
    frames = []
    for name in files:
        path = base_dir / name
        if path.exists():
            frames.append(_read_ranking_file(path))
        else:
            logger.debug("Ranking file not found: %s", path)

    if not frames:
        raise FileNotFoundError(f"No ranking CSVs found in {base_dir}")

    rankings = pd.concat(frames, ignore_index=True)
    rankings["ranking_date"] = pd.to_datetime(
        pd.to_numeric(rankings["ranking_date"], errors="coerce").astype("Int64").astype(str),
        format="%Y%m%d",
        errors="coerce",
    )
    rankings["player"] = pd.to_numeric(rankings["player"], errors="coerce")
    rankings["rank"] = pd.to_numeric(rankings["rank"], errors="coerce")
    rankings["points"] = pd.to_numeric(rankings["points"], errors="coerce").fillna(0)
    rankings = rankings.dropna(subset=["ranking_date", "player", "rank"])

    if max_date is not None:
        rankings = rankings[rankings["ranking_date"] <= max_date]

    rankings = rankings.astype(
        {"player": np.int64, "rank": np.int32, "points": np.int32}
    )
    rankings = rankings.drop_duplicates(subset=["player", "ranking_date"], keep="last")
    rankings = rankings.sort_values(["player", "ranking_date"], kind="stable")

    # Narrow dtypes: day ordinals and ATP ranks/points all fit comfortably, and
    # this store dominates the size of the pickled engine state.
    def _narrow(column: str):
        limit = int(rankings[column].max()) if len(rankings) else 0
        return np.int16 if limit <= np.iinfo(np.int16).max else np.int32

    day = rankings["ranking_date"].to_numpy("datetime64[D]").astype(np.int32)
    players = rankings["player"].to_numpy()
    ranks = rankings["rank"].to_numpy(_narrow("rank"))
    points = rankings["points"].to_numpy(_narrow("points"))

    # Split once at the group boundaries instead of a Python-level groupby.
    boundaries = np.flatnonzero(np.diff(players)) + 1
    dates_by_player = dict(zip(players[np.r_[0, boundaries]], np.split(day, boundaries)))
    ranks_by_player = dict(zip(players[np.r_[0, boundaries]], np.split(ranks, boundaries)))
    points_by_player = dict(zip(players[np.r_[0, boundaries]], np.split(points, boundaries)))

    logger.info(
        "Ranking lookup: %s snapshots across %s players (%s - %s)",
        f"{len(rankings):,}",
        f"{len(dates_by_player):,}",
        rankings["ranking_date"].min().date(),
        rankings["ranking_date"].max().date(),
    )
    return RankingLookup(
        {int(k): v for k, v in dates_by_player.items()},
        {int(k): v for k, v in ranks_by_player.items()},
        {int(k): v for k, v in points_by_player.items()},
        n_rows=len(rankings),
    )


def to_day_ordinal(dates: pd.Series | pd.Timestamp | Sequence) -> np.ndarray | np.int64:
    """Convert timestamps to the integer day ordinals used by the lookup."""
    if isinstance(dates, pd.Timestamp):
        return np.datetime64(dates.date(), "D").astype(np.int64)
    return pd.to_datetime(pd.Series(dates)).to_numpy("datetime64[D]").astype(np.int64)
