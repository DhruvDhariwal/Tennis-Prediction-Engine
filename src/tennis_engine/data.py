"""Loading, cleaning and leakage-auditing of the Jeff Sackmann ATP CSVs."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import UNKNOWN_SURFACE

logger = logging.getLogger(__name__)

#: Columns that describe what happened *during or after* the match. They must
#: never reach the feature matrix.
#:
#: The original notebook selected these with the regex
#: ``score|min|retired|walkover|wo|after|next|post``. The ``wo`` alternative
#: matches "Won" case-insensitively, so it silently deleted ``w_1stWon``,
#: ``w_2ndWon``, ``l_1stWon`` and ``l_2ndWon`` -- which made the
#: ``srv_winrate_diff`` feature identically zero for every row. An explicit list
#: cannot fail that way.
POST_MATCH_COLUMNS: tuple[str, ...] = ("score", "minutes")

#: Post-match box-score columns. These are *not* features, but they are used
#: after each match to update a player's rolling form, so they are kept on the
#: frame and consumed only by the state update step.
BOXSCORE_COLUMNS: tuple[str, ...] = (
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
    "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
    "l_SvGms", "l_bpSaved", "l_bpFaced",
)

#: Pre-computed ranking columns shipped with the match files. We build our own
#: as-of ranking lookup instead (it also works at prediction time), so these are
#: dropped to guarantee a single source of truth.
SHIPPED_RANK_COLUMNS: tuple[str, ...] = (
    "winner_rank", "winner_rank_points", "loser_rank", "loser_rank_points",
)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "tourney_id", "tourney_name", "surface", "draw_size", "tourney_level",
    "tourney_date", "match_num", "best_of",
    "winner_id", "winner_name", "winner_hand", "winner_ht", "winner_age",
    "loser_id", "loser_name", "loser_hand", "loser_ht", "loser_age",
)

_SURFACE_ALIASES = {
    "h": "Hard", "hard": "Hard",
    "c": "Clay", "clay": "Clay",
    "g": "Grass", "grass": "Grass",
    "car": "Carpet", "carpet": "Carpet",
}

_INCOMPLETE_SCORE = re.compile(r"RET|W/O|W\.O\.|DEF|ABD|WEA|Def\.", re.IGNORECASE)


def tier_for_file(filename: str) -> str:
    """Map a source CSV name onto a tour tier."""
    name = str(filename).lower()
    if "futures" in name:
        return "futures"
    if "qual_chall" in name:
        return "challenger"
    return "atp"


def _file_patterns(year: int, include_challengers: bool, include_futures: bool) -> list[str]:
    patterns = [f"atp_matches_{year}.csv"]
    if include_challengers:
        patterns.append(f"atp_matches_qual_chall_{year}.csv")
    if include_futures:
        patterns.append(f"atp_matches_futures_{year}.csv")
    return patterns


def load_years(
    base_dir: Path | str,
    years: Iterable[int],
    include_challengers: bool = True,
    include_futures: bool = True,
) -> pd.DataFrame:
    """Load every available match file for ``years`` into one frame."""
    base_dir = Path(base_dir)
    frames: list[pd.DataFrame] = []
    missing: list[str] = []

    for year in years:
        for pattern in _file_patterns(year, include_challengers, include_futures):
            path = base_dir / pattern
            if not path.exists():
                missing.append(pattern)
                continue
            frame = pd.read_csv(path, low_memory=False)
            frame["source_file"] = pattern
            frame["tier"] = tier_for_file(pattern)
            frames.append(frame)

    if missing:
        logger.debug("Missing %d match files (e.g. %s)", len(missing), missing[:3])
    if not frames:
        raise FileNotFoundError(
            f"No match CSVs found in {base_dir} for years {list(years)!r}. "
            "Download the Jeff Sackmann tennis_atp data into this directory."
        )
    return pd.concat(frames, ignore_index=True)


def normalize_surface(series: pd.Series) -> pd.Series:
    """Map raw surface labels onto the canonical set, keeping 'Unknown' distinct.

    The original pipeline silently rewrote every unknown surface to 'Hard',
    which fabricated surface-Elo history for ~7% of matches. Unknown is now its
    own category with its own indicator column.
    """
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.lower()
        .map(_SURFACE_ALIASES)
    )
    return cleaned.fillna(UNKNOWN_SURFACE).astype(str)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns exist and carry sane dtypes."""
    df = df.copy()

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df["tourney_date"] = pd.to_datetime(
        df["tourney_date"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce"
    )
    df["surface"] = normalize_surface(df["surface"])

    for col in ("winner_ht", "loser_ht", "winner_age", "loser_age", "best_of", "draw_size"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in BOXSCORE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ("winner_id", "loser_id"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "tier" not in df.columns:
        df["tier"] = df.get("source_file", "atp").map(tier_for_file)

    return df


def drop_incomplete_matches(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove retirements, walkovers and defaults.

    Those results are poor evidence about relative strength and their box scores
    are truncated, so they distort both Elo and the rolling serve stats.
    """
    if "score" not in df.columns:
        return df, 0
    bad = df["score"].astype("string").str.contains(_INCOMPLETE_SCORE, na=False)
    return df.loc[~bad].copy(), int(bad.sum())


def audit_and_drop_leaky_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns that encode the outcome, using an explicit allow/deny list."""
    to_drop = [c for c in POST_MATCH_COLUMNS + SHIPPED_RANK_COLUMNS if c in df.columns]
    return df.drop(columns=to_drop), to_drop


def deduplicate_matches(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop repeated (tournament, match) keys, keeping the first occurrence."""
    key = ["source_file", "tourney_id", "tourney_date", "match_num"]
    key = [k for k in key if k in df.columns]
    if not key:
        return df, 0
    dupes = int(df.duplicated(subset=key, keep="first").sum())
    return df.drop_duplicates(subset=key, keep="first"), dupes


def sort_chronologically(df: pd.DataFrame) -> pd.DataFrame:
    """Order matches the way they were actually played.

    Within a tournament, ``match_num`` is monotonic in round for the Sackmann
    files, so ``(date, tourney_id, match_num)`` reproduces the real sequence.
    A stable sort keeps the file order as the final tie-break.
    """
    keys = [k for k in ("tourney_date", "tourney_id", "match_num") if k in df.columns]
    return df.sort_values(keys, kind="stable").reset_index(drop=True)


def prepare_matches(
    base_dir: Path | str,
    years: Sequence[int],
    include_challengers: bool = True,
    include_futures: bool = True,
) -> pd.DataFrame:
    """Full load -> clean -> audit -> sort pipeline for a set of years."""
    raw = load_years(base_dir, years, include_challengers, include_futures)
    n_raw = len(raw)

    df = normalize_columns(raw)
    df, n_incomplete = drop_incomplete_matches(df)
    df, dropped_cols = audit_and_drop_leaky_columns(df)

    valid = df["winner_id"].notna() & df["loser_id"].notna() & df["tourney_date"].notna()
    n_invalid = int((~valid).sum())
    df = df.loc[valid].copy()
    df["winner_id"] = df["winner_id"].astype(np.int64)
    df["loser_id"] = df["loser_id"].astype(np.int64)

    same_player = df["winner_id"] == df["loser_id"]
    n_self = int(same_player.sum())
    df = df.loc[~same_player].copy()

    df, n_dupes = deduplicate_matches(df)
    df = sort_chronologically(df)

    logger.info(
        "Loaded %s matches for %s-%s (dropped: %s incomplete, %s invalid, "
        "%s self-matches, %s duplicates; leaky columns removed: %s)",
        f"{len(df):,}", min(years), max(years), f"{n_incomplete:,}", f"{n_invalid:,}",
        n_self, f"{n_dupes:,}", dropped_cols,
    )
    assert df["tourney_date"].notna().all(), "NaT survived date normalization"
    assert len(df) <= n_raw
    return df


def load_player_directory(players_csv: Path | str) -> pd.DataFrame:
    """Load ``atp_players.csv`` with a normalized full-name column and dob.

    Height, handedness and date of birth come from here rather than from the
    match rows, so the exact same static features are available at prediction
    time. Previously ``height_diff`` and ``age_diff`` were real numbers during
    training but hard-coded to 0 during inference -- a train/serve skew.
    """
    players = pd.read_csv(players_csv, low_memory=False)
    players["player_id"] = pd.to_numeric(players["player_id"], errors="coerce")
    players = players.dropna(subset=["player_id"]).copy()
    players["player_id"] = players["player_id"].astype(np.int64)

    players["full_name"] = (
        players["name_first"].fillna("").astype(str).str.strip()
        + " "
        + players["name_last"].fillna("").astype(str).str.strip()
    ).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)

    players["height"] = pd.to_numeric(players.get("height"), errors="coerce")
    players.loc[(players["height"] < 140) | (players["height"] > 230), "height"] = np.nan

    dob = pd.to_numeric(players.get("dob"), errors="coerce")
    players["dob"] = pd.to_datetime(
        dob.astype("Int64").astype(str), format="%Y%m%d", errors="coerce"
    )

    hand = players.get("hand", pd.Series(index=players.index, dtype=object))
    players["hand"] = hand.astype("string").str.upper().where(lambda s: s.isin(["L", "R"]))

    return players[["player_id", "full_name", "height", "dob", "hand", "ioc"]]


def build_name_index(players: pd.DataFrame) -> tuple[dict[str, int], dict[int, str]]:
    """Build ``name -> id`` and ``id -> name`` maps for prediction lookups.

    ``atp_players.csv`` contains 931 duplicate ``full_name`` values, including a
    block of rows with no name at all. Zipping the columns straight into a dict
    silently keeps whichever row came last, so an empty query string would
    resolve to an arbitrary player. Blank names are dropped, and genuine
    namesakes resolve to the most recently born player -- the one a user asking
    about "today's match" almost certainly means.
    """
    named = players[players["full_name"].str.len() > 0].copy()
    named = named.sort_values("dob", na_position="first", kind="stable")

    name_to_id = {
        name: int(pid)
        for name, pid in zip(named["full_name"], named["player_id"])
    }
    id_to_name = {
        int(pid): str(name).title()
        for pid, name in zip(players["player_id"], players["full_name"])
    }
    return name_to_id, id_to_name


def build_player_attributes(players: pd.DataFrame) -> dict[int, dict]:
    """Compact ``player_id -> {height, dob_ordinal, is_left}`` mapping."""
    hand_to_flag = {"L": 1.0, "R": 0.0}
    attrs: dict[int, dict] = {}
    for row in players.itertuples(index=False):
        # `hand` is a nullable string column, so pd.NA must be normalized before
        # any comparison -- `pd.NA == "L"` is itself NA and raises on truth-test.
        hand = row.hand if pd.notna(row.hand) else None
        attrs[int(row.player_id)] = {
            "height": float(row.height) if pd.notna(row.height) else np.nan,
            "dob": row.dob.to_pydatetime() if pd.notna(row.dob) else None,
            "is_left": hand_to_flag.get(hand, np.nan),
        }
    return attrs
