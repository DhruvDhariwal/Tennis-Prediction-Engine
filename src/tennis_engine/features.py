"""Chronological, order-invariant feature construction.

Design notes
------------
*Leak-proof.* Matches are walked in played order. For each match the builder
reads state, emits the feature row, and only then applies the post-match update.
No feature can depend on the match it describes.

*Order-invariant by construction.* Each match produces exactly one
"winner-perspective" vector. Swapping the two players is a sign flip on the
antisymmetric block and a no-op on the symmetric block, so the mirrored row is
derived arithmetically rather than recomputed. The original notebook built both
rows independently inside the loop, doubling the work and leaving room for the
two halves to drift apart.

*Missing means missing.* A player with no recorded serve history yields ``NaN``,
which XGBoost routes down a learned default branch. Encoding "no data" as 0.0 --
as the original did -- tells the model that a debutant has a 0% service-points-won
rate, which is both false and, because 0 is far outside the real range, highly
influential.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .config import ALL_SURFACES, TIER_ORDINAL, Config
from .data import BOXSCORE_COLUMNS
from .rankings import RankingLookup, to_day_ordinal
from .state import (
    EloEngine,
    PlayerState,
    expected_score,
    safe_rate,
    surface_index,
)

logger = logging.getLogger(__name__)

#: Features that negate when the two players are swapped.
ANTISYMMETRIC_FEATURES: tuple[str, ...] = (
    "elo_diff",
    "surf_elo_diff",
    "blend_elo_diff",
    "elo_expectation_diff",
    "rank_diff",
    "log_rank_diff",
    "log_points_diff",
    "is_ranked_diff",
    "form_diff",
    "career_winrate_diff",
    "log_matches_diff",
    "surf_winrate_diff",
    "log_surf_matches_diff",
    "h2h_diff",
    "h2h_surface_diff",
    "ace_rate_diff",
    "df_rate_diff",
    "srv_winrate_diff",
    "ret_winrate_diff",
    "bp_save_rate_diff",
    "log_days_since_diff",
    "workload_diff",
    "height_diff",
    "age_diff",
    "lefty_diff",
)

#: Features that are unchanged when the two players are swapped.
SYMMETRIC_FEATURES: tuple[str, ...] = (
    "h2h_total",
    "best_of",
    "tier_ordinal",
) + tuple(f"surface_is_{s.lower()}" for s in ALL_SURFACES)

FEATURE_NAMES: tuple[str, ...] = ANTISYMMETRIC_FEATURES + SYMMETRIC_FEATURES
N_FEATURES = len(FEATURE_NAMES)

#: ``+1`` keep, ``-1`` negate. Multiplying a row by this vector produces the
#: same match seen from the other player's perspective.
FLIP_SIGN: np.ndarray = np.array(
    [-1.0] * len(ANTISYMMETRIC_FEATURES) + [1.0] * len(SYMMETRIC_FEATURES),
    dtype=np.float32,
)

FEATURE_INDEX: dict[str, int] = {name: i for i, name in enumerate(FEATURE_NAMES)}

_TIER_NAMES = {v: k for k, v in TIER_ORDINAL.items()}


def _diff(a: float, b: float) -> float:
    """Difference that propagates missingness instead of silently zeroing it."""
    if a != a or b != b:
        return math.nan
    return a - b


def _log_diff(a: float, b: float) -> float:
    """Log-scaled difference for non-negative counts.

    Ranks, match counts and rest days are all heavy-tailed: the gap between rank
    1 and 10 matters far more than between 500 and 509, and a raw difference
    forces the trees to spend splits recovering that.
    """
    if a != a or b != b or a < 0 or b < 0:
        return math.nan
    return math.log1p(a) - math.log1p(b)


@dataclass
class BuiltFeatures:
    """Feature matrix plus the metadata needed to slice and audit it."""

    X: np.ndarray
    y: np.ndarray
    weight: np.ndarray
    match_index: np.ndarray          # row -> index of the source match
    date: np.ndarray                 # day ordinal per row
    tier: np.ndarray                 # tier ordinal per row
    min_prior: np.ndarray            # min prior matches across the two players
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def __len__(self) -> int:
        return len(self.y)

    def as_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.X, columns=list(self.feature_names))
        frame["y"] = self.y
        frame["date"] = self.date
        return frame


class FeatureBuilder:
    """Walks matches chronologically, emitting features and updating state."""

    def __init__(
        self,
        cfg: Config,
        rankings: RankingLookup,
        player_attrs: dict[int, dict] | None = None,
    ) -> None:
        self.cfg = cfg
        self.rankings = rankings
        self.player_attrs = player_attrs or {}
        self.elo = EloEngine(cfg.elo)
        self.players: dict[int, PlayerState] = {}
        self.h2h: dict[tuple[int, int], int] = {}
        self.h2h_surface: dict[tuple[int, int, int], int] = {}
        self.n_matches_seen = 0
        self.last_date: int | None = None

    # ------------------------------------------------------------------ util

    def player(self, player_id: int) -> PlayerState:
        state = self.players.get(player_id)
        if state is None:
            state = PlayerState(self.cfg.elo.initial, self.cfg.features.roll_n)
            self.players[player_id] = state
        return state

    def _age(self, player_id: int, day: int, fallback: float) -> float:
        dob = self.player_attrs.get(player_id, {}).get("dob")
        if dob is not None:
            dob_day = np.datetime64(dob.date(), "D").astype(np.int64)
            return (day - float(dob_day)) / 365.2425
        return fallback if fallback == fallback else math.nan

    def _height(self, player_id: int, fallback: float) -> float:
        height = self.player_attrs.get(player_id, {}).get("height", math.nan)
        if height == height:
            return height
        return fallback if fallback == fallback else math.nan

    def _is_left(self, player_id: int) -> float:
        return self.player_attrs.get(player_id, {}).get("is_left", math.nan)

    # -------------------------------------------------------------- features

    def match_vector(
        self,
        a_id: int,
        b_id: int,
        surface_idx: int,
        day: int,
        tier_ord: int,
        best_of: float,
        a_height: float = math.nan,
        b_height: float = math.nan,
        a_age: float = math.nan,
        b_age: float = math.nan,
    ) -> np.ndarray:
        """Pre-match feature vector from A's perspective. Reads state only."""
        cfg = self.cfg
        A = self.player(a_id)
        B = self.player(b_id)

        a_rank, a_pts, a_ranked = self.rankings.get(a_id, day)
        b_rank, b_pts, b_ranked = self.rankings.get(b_id, day)

        blend = cfg.elo.surface_blend
        a_blend = A.blended_surface_elo(surface_idx, blend)
        b_blend = B.blended_surface_elo(surface_idx, blend)

        h_ab = self.h2h.get((a_id, b_id), 0)
        h_ba = self.h2h.get((b_id, a_id), 0)
        hs_ab = self.h2h_surface.get((a_id, b_id, surface_idx), 0)
        hs_ba = self.h2h_surface.get((b_id, a_id, surface_idx), 0)

        vec = np.empty(N_FEATURES, dtype=np.float32)
        vec[0] = A.elo - B.elo
        vec[1] = A.surf_elo[surface_idx] - B.surf_elo[surface_idx]
        vec[2] = a_blend - b_blend
        vec[3] = 2.0 * expected_score(a_blend, b_blend) - 1.0
        vec[4] = a_rank - b_rank
        vec[5] = math.log(a_rank) - math.log(b_rank)
        vec[6] = math.log1p(a_pts) - math.log1p(b_pts)
        vec[7] = float(a_ranked) - float(b_ranked)
        vec[8] = _diff(A.form_rate(), B.form_rate())
        vec[9] = _diff(A.career_winrate(), B.career_winrate())
        vec[10] = _log_diff(A.matches_played, B.matches_played)
        vec[11] = _diff(A.surface_winrate(surface_idx), B.surface_winrate(surface_idx))
        vec[12] = _log_diff(A.surf_matches[surface_idx], B.surf_matches[surface_idx])
        vec[13] = float(h_ab - h_ba)
        vec[14] = float(hs_ab - hs_ba)
        vec[15] = _diff(A.stat_mean(0), B.stat_mean(0))
        vec[16] = _diff(A.stat_mean(1), B.stat_mean(1))
        vec[17] = _diff(A.stat_mean(2), B.stat_mean(2))
        vec[18] = _diff(A.stat_mean(3), B.stat_mean(3))
        vec[19] = _diff(A.stat_mean(4), B.stat_mean(4))
        vec[20] = _log_diff(A.days_since_last(day), B.days_since_last(day))
        window = cfg.features.workload_window_days
        vec[21] = float(A.matches_in_window(day, window) - B.matches_in_window(day, window))
        vec[22] = _diff(self._height(a_id, a_height), self._height(b_id, b_height))
        vec[23] = _diff(self._age(a_id, day, a_age), self._age(b_id, day, b_age))
        vec[24] = _diff(self._is_left(a_id), self._is_left(b_id))

        base = len(ANTISYMMETRIC_FEATURES)
        vec[base + 0] = float(h_ab + h_ba)
        vec[base + 1] = best_of if best_of == best_of else 3.0
        vec[base + 2] = float(tier_ord)
        for i in range(len(ALL_SURFACES)):
            vec[base + 3 + i] = 1.0 if i == surface_idx else 0.0
        return vec

    # ----------------------------------------------------------------- update

    def apply_match(
        self,
        winner_id: int,
        loser_id: int,
        surface_idx: int,
        day: int,
        tier: str,
        winner_stats: tuple[float, ...],
        loser_stats: tuple[float, ...],
    ) -> None:
        """Post-match state update. Must run *after* the feature row is emitted."""
        W = self.player(winner_id)
        L = self.player(loser_id)

        self.elo.update(W, L, surface_idx, tier)

        self.h2h[(winner_id, loser_id)] = self.h2h.get((winner_id, loser_id), 0) + 1
        key = (winner_id, loser_id, surface_idx)
        self.h2h_surface[key] = self.h2h_surface.get(key, 0) + 1

        W.record_stats(winner_stats)
        L.record_stats(loser_stats)
        W.record_result(True, surface_idx, day)
        L.record_result(False, surface_idx, day)

        # Safe here (and only here): `day` is non-decreasing across the pass, so
        # nothing pruned could fall inside a later workload window.
        window = self.cfg.features.workload_window_days
        W.prune_window(day, window)
        L.prune_window(day, window)

        self.n_matches_seen += 1
        self.last_date = day

    # ------------------------------------------------------------------ build

    def build(
        self,
        df: pd.DataFrame,
        mirror: bool,
        emit: bool = True,
        rng_seed: int | None = None,
    ) -> BuiltFeatures:
        """Walk ``df`` chronologically.

        Parameters
        ----------
        mirror
            ``True`` for training: emit both orientations of every match, which
            makes the training distribution exactly symmetric.
            ``False`` for evaluation: emit one randomly-oriented row per match,
            so metrics are computed over real matches rather than over each match
            counted twice.
        emit
            ``False`` runs the state machine without materializing features --
            used to warm up ratings over history that precedes the training
            window.
        """
        n = len(df)
        rows_per_match = 2 if mirror else 1
        capacity = n * rows_per_match if emit else 0

        X = np.empty((capacity, N_FEATURES), dtype=np.float32)
        y = np.empty(capacity, dtype=np.int8)
        weight = np.empty(capacity, dtype=np.float32)
        match_index = np.empty(capacity, dtype=np.int64)
        row_date = np.empty(capacity, dtype=np.int64)
        row_tier = np.empty(capacity, dtype=np.int8)
        row_prior = np.empty(capacity, dtype=np.int32)

        cols = _extract_columns(df)
        days = cols["day"]
        rng = np.random.default_rng(
            self.cfg.random_state if rng_seed is None else rng_seed
        )
        orientation = rng.integers(0, 2, size=n) if not mirror else None

        tier_weights = _tier_weight_array(cols["tier_ord"])
        out = 0
        prev_day = -(2**60)

        for i in range(n):
            day = int(days[i])
            # Chronological invariant: state may never be updated out of order.
            if day < prev_day:
                raise ValueError(
                    f"Matches are not chronologically sorted at row {i}: "
                    f"{day} < {prev_day}"
                )
            prev_day = day

            w_id = int(cols["winner_id"][i])
            l_id = int(cols["loser_id"][i])
            s_idx = int(cols["surface_idx"][i])
            tier_ord = int(cols["tier_ord"][i])
            tier = _TIER_NAMES[tier_ord]

            if emit:
                vec = self.match_vector(
                    w_id, l_id, s_idx, day, tier_ord,
                    float(cols["best_of"][i]),
                    a_height=float(cols["winner_ht"][i]),
                    b_height=float(cols["loser_ht"][i]),
                    a_age=float(cols["winner_age"][i]),
                    b_age=float(cols["loser_age"][i]),
                )
                w = tier_weights[i]
                prior = min(
                    self.players[w_id].matches_played,
                    self.players[l_id].matches_played,
                )
                if mirror:
                    X[out] = vec
                    y[out] = 1
                    X[out + 1] = vec * FLIP_SIGN
                    y[out + 1] = 0
                    weight[out] = weight[out + 1] = w
                    match_index[out] = match_index[out + 1] = i
                    row_date[out] = row_date[out + 1] = day
                    row_tier[out] = row_tier[out + 1] = tier_ord
                    row_prior[out] = row_prior[out + 1] = prior
                    out += 2
                else:
                    if orientation[i]:
                        X[out] = vec
                        y[out] = 1
                    else:
                        X[out] = vec * FLIP_SIGN
                        y[out] = 0
                    weight[out] = w
                    match_index[out] = i
                    row_date[out] = day
                    row_tier[out] = tier_ord
                    row_prior[out] = prior
                    out += 1

            self.apply_match(
                w_id, l_id, s_idx, day, tier,
                _stats_tuple(cols, i, "w"),
                _stats_tuple(cols, i, "l"),
            )

        logger.info(
            "Built %s rows from %s matches (mirror=%s, emit=%s); "
            "%s players tracked, %s h2h pairs",
            f"{out:,}", f"{n:,}", mirror, emit,
            f"{len(self.players):,}", f"{len(self.h2h):,}",
        )
        return BuiltFeatures(
            X=X[:out], y=y[:out], weight=weight[:out],
            match_index=match_index[:out], date=row_date[:out], tier=row_tier[:out],
            min_prior=row_prior[:out],
        )


def _tier_weight_array(tier_ord: np.ndarray) -> np.ndarray:
    from .config import TIER_SAMPLE_WEIGHT

    table = np.array(
        [TIER_SAMPLE_WEIGHT["futures"], TIER_SAMPLE_WEIGHT["challenger"], TIER_SAMPLE_WEIGHT["atp"]],
        dtype=np.float32,
    )
    return table[tier_ord]


def _column(df: pd.DataFrame, name: str, dtype=np.float64, fill=np.nan) -> np.ndarray:
    if name not in df.columns:
        return np.full(len(df), fill, dtype=dtype)
    return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=dtype, na_value=fill)


def _extract_columns(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Pull every column the loop needs into contiguous numpy arrays up front."""
    cols: dict[str, np.ndarray] = {
        "winner_id": df["winner_id"].to_numpy(np.int64),
        "loser_id": df["loser_id"].to_numpy(np.int64),
        "day": to_day_ordinal(df["tourney_date"]),
        "surface_idx": df["surface"].map(surface_index).to_numpy(np.int8),
        "tier_ord": df["tier"].map(TIER_ORDINAL).fillna(2).to_numpy(np.int8),
        "best_of": _column(df, "best_of", fill=3.0),
        "winner_ht": _column(df, "winner_ht"),
        "loser_ht": _column(df, "loser_ht"),
        "winner_age": _column(df, "winner_age"),
        "loser_age": _column(df, "loser_age"),
    }
    for name in BOXSCORE_COLUMNS:
        cols[name] = _column(df, name)
    return cols


def _stats_tuple(cols: dict[str, np.ndarray], i: int, side: str) -> tuple[float, ...]:
    """Serve/return rates for one player in one match.

    ``ret_win`` is derived from the *opponent's* service columns, which is the
    only way to get return performance out of these files.
    """
    opp = "l" if side == "w" else "w"
    svpt = cols[f"{side}_svpt"][i]
    opp_svpt = cols[f"{opp}_svpt"][i]
    opp_won = cols[f"{opp}_1stWon"][i] + cols[f"{opp}_2ndWon"][i]

    return (
        safe_rate(cols[f"{side}_ace"][i], svpt),
        safe_rate(cols[f"{side}_df"][i], svpt),
        safe_rate(cols[f"{side}_1stWon"][i] + cols[f"{side}_2ndWon"][i], svpt),
        safe_rate(opp_svpt - opp_won, opp_svpt),
        safe_rate(cols[f"{side}_bpSaved"][i], cols[f"{side}_bpFaced"][i]),
    )


def flip(X: np.ndarray) -> np.ndarray:
    """Return the same rows seen from the other player's perspective."""
    return X * FLIP_SIGN
