"""Player rating / form state and its chronological update rules.

Every quantity here is a *pre-match* summary: the feature builder reads the
state, emits a row, and only then applies the post-match update. That ordering
is the entire leak-proof guarantee, so the update methods are deliberately kept
separate from the read accessors.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Iterable

import numpy as np

from .config import ALL_SURFACES, TIER_K_SCALE, EloConfig

#: Rolling serve/return statistics tracked per match, in fixed order.
STAT_KEYS: tuple[str, ...] = ("ace", "df", "srv_win", "ret_win", "bp_save")
N_STATS = len(STAT_KEYS)

_SURFACE_INDEX: dict[str, int] = {s: i for i, s in enumerate(ALL_SURFACES)}
N_SURFACES = len(ALL_SURFACES)


class PlayerState:
    """Mutable per-player state.

    Uses ``__slots__`` and incremental sums rather than recomputing means over a
    deque on every access; the builder touches this object ~2M times, so the
    difference is minutes, not microseconds.
    """

    __slots__ = (
        "elo", "surf_elo", "matches_played", "wins",
        "surf_matches", "surf_wins",
        "form", "form_sum",
        "stats", "stats_sum", "stats_count",
        "last_date", "recent_dates",
    )

    def __init__(self, initial_elo: float = 1500.0, roll_n: int = 20) -> None:
        self.elo = initial_elo
        self.surf_elo = [initial_elo] * N_SURFACES
        self.matches_played = 0
        self.wins = 0
        self.surf_matches = [0] * N_SURFACES
        self.surf_wins = [0] * N_SURFACES
        self.form: deque[int] = deque(maxlen=roll_n)
        self.form_sum = 0
        self.stats: deque[tuple[float, ...]] = deque(maxlen=roll_n)
        self.stats_sum = [0.0] * N_STATS
        self.stats_count = [0] * N_STATS
        self.last_date: int | None = None
        self.recent_dates: deque[int] = deque()

    # ---------------------------------------------------------------- reads

    def blended_surface_elo(self, surface_idx: int, blend: float) -> float:
        """Shrink surface Elo toward global Elo.

        A player with three career grass matches has a nearly meaningless grass
        Elo; blending keeps the feature usable without a hard minimum-sample cut.
        """
        return blend * self.elo + (1.0 - blend) * self.surf_elo[surface_idx]

    def form_rate(self) -> float:
        return self.form_sum / len(self.form) if self.form else math.nan

    def career_winrate(self) -> float:
        return self.wins / self.matches_played if self.matches_played else math.nan

    def surface_winrate(self, surface_idx: int) -> float:
        played = self.surf_matches[surface_idx]
        return self.surf_wins[surface_idx] / played if played else math.nan

    def stat_mean(self, stat_idx: int) -> float:
        count = self.stats_count[stat_idx]
        return self.stats_sum[stat_idx] / count if count else math.nan

    def days_since_last(self, day: int) -> float:
        """Days of rest before ``day``.

        Clamped at zero: at prediction time the caller may ask about a date that
        precedes the last match in the saved state, and "negative rest" is not a
        meaningful quantity.
        """
        if self.last_date is None:
            return math.nan
        return float(max(0, day - self.last_date))

    def matches_in_window(self, day: int, window_days: int) -> int:
        """Matches played in the ``window_days`` before ``day``.

        Read-only: it counts rather than pruning. Pruning here would mean a
        *read* mutates state, which is fine during a monotonic training pass but
        corrupts the saved state at prediction time -- asking about a past date
        would permanently discard entries a later query still needs.
        """
        cutoff = day - window_days
        count = 0
        for stamp in reversed(self.recent_dates):
            if stamp > day:
                continue  # state may run ahead of the queried date at inference
            if stamp < cutoff:
                break
            count += 1
        return count

    # --------------------------------------------------------------- writes

    def prune_window(self, day: int, window_days: int) -> None:
        """Drop entries that can no longer fall inside any future window.

        Only ever called from the post-match update, where ``day`` is
        non-decreasing.
        """
        cutoff = day - window_days
        dates = self.recent_dates
        while dates and dates[0] < cutoff:
            dates.popleft()

    def record_result(self, won: bool, surface_idx: int, day: int) -> None:
        self.matches_played += 1
        self.surf_matches[surface_idx] += 1
        if won:
            self.wins += 1
            self.surf_wins[surface_idx] += 1

        if len(self.form) == self.form.maxlen:
            self.form_sum -= self.form[0]
        self.form.append(1 if won else 0)
        self.form_sum += 1 if won else 0

        self.last_date = day
        self.recent_dates.append(day)

    def record_stats(self, values: tuple[float, ...]) -> None:
        """Append one match's serve/return rates.

        ``nan`` entries mean "this box score did not report it" and are excluded
        from the running mean instead of being coerced to 0.0. The original code
        pushed raw ``NaN`` (and, for matches without box scores, silent zeros)
        into the history, which poisoned every downstream rolling average.
        """
        if len(self.stats) == self.stats.maxlen:
            evicted = self.stats[0]
            for i in range(N_STATS):
                v = evicted[i]
                if v == v:  # not NaN
                    self.stats_sum[i] -= v
                    self.stats_count[i] -= 1
        self.stats.append(values)
        for i in range(N_STATS):
            v = values[i]
            if v == v:
                self.stats_sum[i] += v
                self.stats_count[i] += 1


def surface_index(surface: str) -> int:
    return _SURFACE_INDEX.get(surface, _SURFACE_INDEX["Unknown"])


def expected_score(rating_a: float, rating_b: float) -> float:
    """Standard logistic Elo expectation."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


class EloEngine:
    """Applies Elo updates under the configured K schedule."""

    __slots__ = ("cfg",)

    def __init__(self, cfg: EloConfig) -> None:
        self.cfg = cfg

    def k_factor(self, matches_played: int, tier: str) -> float:
        cfg = self.cfg
        scale = TIER_K_SCALE.get(tier, 1.0)
        if cfg.schedule == "fixed":
            return cfg.fixed_k * scale
        # Decaying K: a player's rating stabilizes as evidence accumulates.
        return scale * cfg.k_scale / ((matches_played + cfg.k_shift) ** cfg.k_decay)

    def update(
        self,
        winner: PlayerState,
        loser: PlayerState,
        surface_idx: int,
        tier: str,
    ) -> None:
        """Apply global and surface Elo updates for a completed match."""
        exp_w = expected_score(winner.elo, loser.elo)
        k_w = self.k_factor(winner.matches_played, tier)
        k_l = self.k_factor(loser.matches_played, tier)
        winner.elo += k_w * (1.0 - exp_w)
        loser.elo -= k_l * (1.0 - exp_w)

        w_surf = winner.surf_elo[surface_idx]
        l_surf = loser.surf_elo[surface_idx]
        exp_ws = expected_score(w_surf, l_surf)
        k_ws = self.k_factor(winner.surf_matches[surface_idx], tier)
        k_ls = self.k_factor(loser.surf_matches[surface_idx], tier)
        winner.surf_elo[surface_idx] = w_surf + k_ws * (1.0 - exp_ws)
        loser.surf_elo[surface_idx] = l_surf - k_ls * (1.0 - exp_ws)


def serialize_player(state: PlayerState) -> dict:
    """Convert a :class:`PlayerState` into a compact, picklable payload.

    The rolling histories go out as packed numpy arrays rather than nested
    Python lists. A list of twenty 5-tuples of Python floats costs ~5 KB per
    player; across ~28k tracked players that alone accounted for most of the
    original 114 MB state file.
    """
    return {
        "elo": np.float32(state.elo),
        "surf_elo": np.asarray(state.surf_elo, dtype=np.float32),
        "matches_played": state.matches_played,
        "wins": state.wins,
        "surf_matches": np.asarray(state.surf_matches, dtype=np.int32),
        "surf_wins": np.asarray(state.surf_wins, dtype=np.int32),
        "form": np.asarray(state.form, dtype=np.int8),
        "stats": np.asarray(list(state.stats), dtype=np.float32).reshape(-1, N_STATS),
        "last_date": state.last_date,
        "recent_dates": np.asarray(state.recent_dates, dtype=np.int32),
    }


def deserialize_player(payload: dict, roll_n: int) -> PlayerState:
    state = PlayerState(initial_elo=float(payload["elo"]), roll_n=roll_n)
    state.surf_elo = [float(v) for v in payload["surf_elo"]]
    state.matches_played = int(payload["matches_played"])
    state.wins = int(payload["wins"])
    state.surf_matches = [int(v) for v in payload["surf_matches"]]
    state.surf_wins = [int(v) for v in payload["surf_wins"]]
    for result in payload["form"]:
        result = int(result)
        if len(state.form) == state.form.maxlen:
            state.form_sum -= state.form[0]
        state.form.append(result)
        state.form_sum += result
    for values in payload["stats"]:
        state.record_stats(tuple(float(v) for v in values))
    state.last_date = payload["last_date"]
    state.recent_dates = deque(int(d) for d in payload["recent_dates"])
    return state


def pack_counter(mapping: dict[tuple, int]) -> dict[str, np.ndarray]:
    """Pack a ``tuple -> count`` map into columnar arrays for pickling.

    The head-to-head tables hold ~700k entries each. Pickled as a dict of tuple
    keys that is well over 100 MB of Python object overhead; as columns it is a
    few MB, and rebuilding the dict on load takes well under a second.
    """
    if not mapping:
        return {"keys": np.empty((0, 0), dtype=np.int32), "counts": np.empty(0, np.int16)}
    keys = np.fromiter(
        (component for key in mapping for component in key),
        dtype=np.int64,
        count=len(mapping) * len(next(iter(mapping))),
    ).reshape(len(mapping), -1)
    counts = np.fromiter(mapping.values(), dtype=np.int64, count=len(mapping))

    # ATP player ids and head-to-head tallies both fit in narrow types; widen
    # only if a corpus ever proves otherwise.
    key_dtype = np.int32 if keys.max(initial=0) <= np.iinfo(np.int32).max else np.int64
    count_dtype = np.int16 if counts.max(initial=0) <= np.iinfo(np.int16).max else np.int32
    return {"keys": keys.astype(key_dtype), "counts": counts.astype(count_dtype)}


def unpack_counter(payload: dict[str, np.ndarray]) -> dict[tuple, int]:
    keys, counts = payload["keys"], payload["counts"]
    if len(counts) == 0:
        return {}
    return dict(zip(map(tuple, keys.tolist()), counts.tolist()))


def safe_rate(numerator: float, denominator: float) -> float:
    """Rate that returns ``nan`` (not 0.0) when the box score is unusable."""
    if denominator is None or numerator is None:
        return math.nan
    if not (denominator > 0):
        return math.nan
    if numerator != numerator or denominator != denominator:
        return math.nan
    return numerator / denominator


def mean_ignoring_nan(values: Iterable[float]) -> float:
    vals = [v for v in values if v == v]
    return float(np.mean(vals)) if vals else math.nan
