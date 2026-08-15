"""Inference: score any hypothetical match from the saved engine state.

The single most important property here is *train/serve parity*: the feature
vector is produced by the very same :meth:`FeatureBuilder.match_vector` used
during training, driven by the same deserialized state. The original notebook
hand-wrote the inference feature vector twice, in two different cells, and both
copies disagreed with the training code -- one of them referenced a
``rank_points_diff`` column that does not exist, so rank and points silently fed
the model zeros.
"""

from __future__ import annotations

import difflib
import logging
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb

from . import calibration
from .config import ALL_SURFACES, TIER_ORDINAL, Config, EloConfig, FeatureConfig
from .data import build_name_index, load_player_directory
from .features import FLIP_SIGN, FEATURE_NAMES, FeatureBuilder
from .rankings import RankingLookup, to_day_ordinal
from .state import deserialize_player, surface_index, unpack_counter

logger = logging.getLogger(__name__)


class PlayerNotFound(LookupError):
    """Raised when a name cannot be resolved, with close-match suggestions."""

    def __init__(self, query: str, suggestions: list[str]) -> None:
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        super().__init__(f"No player matching {query!r}.{hint}")
        self.query = query
        self.suggestions = suggestions


@dataclass
class MatchPrediction:
    player1: str
    player2: str
    surface: str
    match_date: str
    p1_win_prob: float
    p2_win_prob: float
    p1_decimal_odds: float
    p2_decimal_odds: float
    features: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "player1": self.player1,
            "player2": self.player2,
            "surface": self.surface,
            "match_date": self.match_date,
            "p1_win_prob": round(self.p1_win_prob, 4),
            "p2_win_prob": round(self.p2_win_prob, 4),
            "p1_decimal_odds": round(self.p1_decimal_odds, 3),
            "p2_decimal_odds": round(self.p2_decimal_odds, 3),
        }

    def __str__(self) -> str:
        return (
            f"{self.player1} vs {self.player2} ({self.surface}, {self.match_date})\n"
            f"  {self.player1:<28s} {self.p1_win_prob:6.1%}  "
            f"(fair odds {self.p1_decimal_odds:.2f})\n"
            f"  {self.player2:<28s} {self.p2_win_prob:6.1%}  "
            f"(fair odds {self.p2_decimal_odds:.2f})"
        )


class PredictionEngine:
    """Loads model + state and scores arbitrary match-ups."""

    def __init__(
        self,
        booster: xgb.Booster,
        builder: FeatureBuilder,
        calibrator,
        name_to_id: dict[str, int],
        id_to_name: dict[int, str],
    ) -> None:
        self.booster = booster
        self.builder = builder
        self.calibrator = calibrator
        self.name_to_id = name_to_id
        self.id_to_name = id_to_name
        self._names = list(FEATURE_NAMES)
        best = getattr(booster, "best_iteration", None)
        self._kwargs = {"iteration_range": (0, best + 1)} if best is not None else {}

    # --------------------------------------------------------------- loading

    @classmethod
    def load(cls, cfg: Config | None = None) -> "PredictionEngine":
        cfg = cfg or Config()
        if not cfg.paths.model_path.exists():
            raise FileNotFoundError(
                f"No model at {cfg.paths.model_path}. Run `python -m tennis_engine train` first."
            )
        booster = xgb.Booster()
        booster.load_model(cfg.paths.model_path)

        with open(cfg.paths.state_path, "rb") as fh:
            state = pickle.load(fh)

        saved_features = state["config"]["feature_names"]
        if saved_features != list(FEATURE_NAMES):
            raise ValueError(
                "Saved state was built with a different feature schema "
                f"({len(saved_features)} features vs {len(FEATURE_NAMES)}). Retrain."
            )

        elo_cfg = EloConfig(**state["config"]["elo"])
        feat_cfg = FeatureConfig(**state["config"]["features"])
        runtime_cfg = Config(paths=cfg.paths, elo=elo_cfg, features=feat_cfg)

        rank_payload = state["rankings"]
        rankings = RankingLookup(
            rank_payload["dates"], rank_payload["ranks"], rank_payload["points"],
            n_rows=rank_payload.get("n_rows", 0),
        )
        builder = FeatureBuilder(runtime_cfg, rankings, state["player_attrs"])
        builder.players = {
            pid: deserialize_player(payload, feat_cfg.roll_n)
            for pid, payload in state["players"].items()
        }
        builder.h2h = unpack_counter(state["h2h"])
        builder.h2h_surface = unpack_counter(state["h2h_surface"])
        builder.last_date = state["config"].get("last_state_date")

        calibrator = calibration.from_dict(state.get("calibrator", {}))

        players = load_player_directory(cfg.paths.players_csv)
        name_to_id, id_to_name = build_name_index(players)
        logger.info(
            "Engine loaded: %s players, %s h2h pairs, state through day %s",
            f"{len(builder.players):,}", f"{len(builder.h2h):,}", builder.last_date,
        )
        return cls(booster, builder, calibrator, name_to_id, id_to_name)

    # ------------------------------------------------------------ resolution

    def resolve_player(self, player: int | str) -> int:
        if isinstance(player, (int, np.integer)):
            return int(player)
        text = str(player).strip()
        if not text:
            raise PlayerNotFound(player, [])
        if text.isdigit():
            return int(text)
        key = " ".join(text.lower().split())
        if key in self.name_to_id:
            return int(self.name_to_id[key])
        # Accept "Last First" before falling back to fuzzy suggestions.
        parts = key.split()
        if len(parts) == 2:
            swapped = f"{parts[1]} {parts[0]}"
            if swapped in self.name_to_id:
                return int(self.name_to_id[swapped])
        suggestions = difflib.get_close_matches(key, self.name_to_id, n=5, cutoff=0.75)
        raise PlayerNotFound(player, [s.title() for s in suggestions])

    # ------------------------------------------------------------ prediction

    def _raw_predict(self, X: np.ndarray) -> np.ndarray:
        forward = self.booster.predict(
            xgb.DMatrix(X, feature_names=self._names), **self._kwargs
        ).astype(np.float64)
        mirrored = self.booster.predict(
            xgb.DMatrix(X * FLIP_SIGN, feature_names=self._names), **self._kwargs
        ).astype(np.float64)
        return 0.5 * (forward + (1.0 - mirrored))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.calibrator.transform(self._raw_predict(X))

    def feature_vector(
        self,
        p1: int,
        p2: int,
        surface: str = "Hard",
        match_date: pd.Timestamp | None = None,
        best_of: int = 3,
        tier: str = "atp",
    ) -> np.ndarray:
        when = match_date or pd.Timestamp.now().normalize()
        day = int(to_day_ordinal(when))
        return self.builder.match_vector(
            p1, p2, surface_index(surface), day,
            TIER_ORDINAL.get(tier, 2), float(best_of),
        )

    def predict_match(
        self,
        player1: int | str,
        player2: int | str,
        surface: str = "Hard",
        match_date: str | pd.Timestamp = "today",
        best_of: int = 3,
        tier: str = "atp",
    ) -> MatchPrediction:
        """Probability that ``player1`` beats ``player2``."""
        if surface not in ALL_SURFACES:
            raise ValueError(f"surface must be one of {ALL_SURFACES}, got {surface!r}")

        p1 = self.resolve_player(player1)
        p2 = self.resolve_player(player2)
        if p1 == p2:
            raise ValueError("A player cannot play themselves.")

        when = (
            pd.Timestamp.now().normalize()
            if isinstance(match_date, str) and match_date == "today"
            else pd.to_datetime(match_date)
        )
        vec = self.feature_vector(p1, p2, surface, when, best_of, tier)
        prob = float(self.predict_proba(vec.reshape(1, -1))[0])
        prob = min(max(prob, 1e-6), 1 - 1e-6)

        return MatchPrediction(
            player1=self.id_to_name.get(p1, str(p1)),
            player2=self.id_to_name.get(p2, str(p2)),
            surface=surface,
            match_date=str(when.date()),
            p1_win_prob=prob,
            p2_win_prob=1.0 - prob,
            p1_decimal_odds=1.0 / prob,
            p2_decimal_odds=1.0 / (1.0 - prob),
            features=dict(zip(FEATURE_NAMES, vec.tolist())),
        )

    def player_card(self, player: int | str, surface: str = "Hard") -> dict:
        """Human-readable snapshot of a player's current state."""
        pid = self.resolve_player(player)
        state = self.builder.players.get(pid)
        if state is None:
            raise PlayerNotFound(str(player), [])
        s_idx = surface_index(surface)
        rank, points, is_ranked = self.builder.rankings.get(
            pid, int(to_day_ordinal(pd.Timestamp.now().normalize()))
        )
        return {
            "player": self.id_to_name.get(pid, str(pid)),
            "player_id": pid,
            "elo": round(state.elo, 1),
            f"elo_{surface.lower()}": round(state.surf_elo[s_idx], 1),
            "atp_rank": int(rank) if is_ranked else None,
            "atp_points": int(points) if is_ranked else None,
            "matches_played": state.matches_played,
            "career_win_pct": round(100 * (state.career_winrate() or 0), 1),
            f"{surface.lower()}_win_pct": _pct(state.surface_winrate(s_idx)),
            "last_20_win_pct": _pct(state.form_rate()),
            "ace_rate": _pct(state.stat_mean(0)),
            "df_rate": _pct(state.stat_mean(1)),
            "serve_pts_won": _pct(state.stat_mean(2)),
            "return_pts_won": _pct(state.stat_mean(3)),
            "bp_save_rate": _pct(state.stat_mean(4)),
        }

    def elo_leaderboard(self, top_n: int = 20, surface: str | None = None,
                        min_matches: int = 20, active_since: str | None = "2024-01-01") -> pd.DataFrame:
        """Top players by (surface) Elo in the current state."""
        s_idx = surface_index(surface) if surface else None
        cutoff = int(to_day_ordinal(pd.to_datetime(active_since))) if active_since else None
        rows = []
        for pid, st in self.builder.players.items():
            if st.matches_played < min_matches:
                continue
            if cutoff is not None and (st.last_date is None or st.last_date < cutoff):
                continue
            rows.append({
                "player": self.id_to_name.get(pid, str(pid)),
                "elo": round(st.elo, 1),
                "surface_elo": round(st.surf_elo[s_idx], 1) if s_idx is not None else None,
                "matches": st.matches_played,
            })
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        sort_col = "surface_elo" if s_idx is not None else "elo"
        return frame.sort_values(sort_col, ascending=False).head(top_n).reset_index(drop=True)


def _pct(value: float) -> float | None:
    return None if value != value else round(100 * value, 1)
