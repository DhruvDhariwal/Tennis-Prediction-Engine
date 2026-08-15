"""Central configuration for the tennis prediction engine.

Everything that used to be a magic number scattered across notebook cells lives
here, so that training, evaluation and inference all read the same values.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def display_path(path: Path) -> str:
    """Render a path relative to the working directory when possible.

    Log lines end up in committed notebook output, so absolute paths there are
    both noisy and needlessly machine-specific.
    """
    try:
        return str(Path(path).resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)

SURFACES: tuple[str, ...] = ("Hard", "Clay", "Grass", "Carpet")
UNKNOWN_SURFACE = "Unknown"
ALL_SURFACES: tuple[str, ...] = SURFACES + (UNKNOWN_SURFACE,)

#: Tour tiers, ordered weakest -> strongest. Used for Elo K scaling and for
#: sample weights during training.
TIERS: tuple[str, ...] = ("futures", "challenger", "atp")

TIER_ORDINAL: dict[str, int] = {"futures": 0, "challenger": 1, "atp": 2}

#: Elo K-factor multiplier per tier. A Futures result should move a rating far
#: less than a Masters 1000 result.
TIER_K_SCALE: dict[str, float] = {"futures": 0.35, "challenger": 0.65, "atp": 1.0}

#: Training sample weight per tier (data quality / relevance).
TIER_SAMPLE_WEIGHT: dict[str, float] = {"futures": 0.4, "challenger": 0.7, "atp": 1.0}

#: Rank assigned to a player with no ATP ranking on the match date. Ranks are
#: used on a log scale, so this is "deep outside the ranked field" rather than a
#: sentinel that a tree has to learn to special-case.
DEFAULT_RANK = 2500
DEFAULT_POINTS = 0


@dataclass(frozen=True)
class EloConfig:
    """Elo rating configuration.

    Two K schedules are supported:

    ``fixed``
        Classic constant K (what the original notebook used).
    ``dynamic``
        FiveThirtyEight-style decaying K: ``k = k_scale / (m + k_shift) ** k_decay``
        where ``m`` is the number of matches the player has already played. New
        players move quickly, established players move slowly. Because K depends
        on each player's own history the update is not strictly zero-sum, which
        is standard for this family of ratings.
    """

    initial: float = 1500.0
    schedule: str = "dynamic"  # "dynamic" | "fixed"
    fixed_k: float = 32.0
    k_scale: float = 250.0
    k_shift: float = 5.0
    k_decay: float = 0.4
    #: Surface Elo is seeded from, and shrunk toward, global Elo. 0.0 = pure
    #: surface Elo, 1.0 = pure global Elo.
    surface_blend: float = 0.35


@dataclass(frozen=True)
class FeatureConfig:
    roll_n: int = 20  # window for rolling form / serve stats
    workload_window_days: int = 90
    #: Matches with fewer than this many prior matches for *both* players carry
    #: almost no signal; they are kept for state building but can be dropped
    #: from the training matrix.
    min_prior_matches: int = 0


@dataclass(frozen=True)
class SplitConfig:
    """Strictly chronological three-way split.

    The original notebook early-stopped on the 2024 test set, which leaks test
    information into model selection. A dedicated validation season fixes that:
    2024 is never seen until final evaluation.
    """

    #: Seasons used to warm up Elo/H2H/form only. No rows are emitted, so the
    #: model never trains on the era before serve statistics were recorded, but
    #: ratings entering the training window are not cold-started at 1500.
    warmup_years: Sequence[int] = tuple(range(1968, 1991))
    train_years: Sequence[int] = tuple(range(1991, 2023))
    val_years: Sequence[int] = (2023,)
    test_years: Sequence[int] = (2024,)

    def all_years(self) -> list[int]:
        return sorted(
            {*self.warmup_years, *self.train_years, *self.val_years, *self.test_years}
        )


@dataclass(frozen=True)
class ModelConfig:
    """XGBoost settings.

    These defaults are the winner of a 12-trial random search scored on the 2023
    validation season (`python -m tennis_engine train --tune N`); the 2024 test
    season played no part in selecting them. Re-running the search reproduces
    them because it is seeded.
    """

    objective: str = "binary:logistic"
    eval_metric: tuple[str, ...] = ("logloss", "auc")
    tree_method: str = "hist"
    learning_rate: float = 0.05
    max_depth: int = 6
    min_child_weight: float = 20.0
    subsample: float = 0.6
    colsample_bytree: float = 0.5
    reg_lambda: float = 2.0
    reg_alpha: float = 1.0
    num_boost_round: int = 3000
    early_stopping_rounds: int = 100
    seed: int = 42

    def to_params(self) -> dict:
        return {
            "objective": self.objective,
            "eval_metric": list(self.eval_metric),
            "tree_method": self.tree_method,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "lambda": self.reg_lambda,
            "alpha": self.reg_alpha,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class Paths:
    base_dir: Path = PROJECT_ROOT
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"

    @property
    def players_csv(self) -> Path:
        return self.base_dir / "atp_players.csv"

    @property
    def model_path(self) -> Path:
        return self.artifacts_dir / "tennis_xgb_model.json"

    @property
    def calibrator_path(self) -> Path:
        return self.artifacts_dir / "calibrator.pkl"

    @property
    def state_path(self) -> Path:
        return self.artifacts_dir / "engine_state.pkl"

    @property
    def metrics_path(self) -> Path:
        return self.artifacts_dir / "metrics.json"

    @property
    def figures_dir(self) -> Path:
        return self.artifacts_dir / "figures"


@dataclass(frozen=True)
class Config:
    paths: Paths = field(default_factory=Paths)
    split: SplitConfig = field(default_factory=SplitConfig)
    elo: EloConfig = field(default_factory=EloConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    random_state: int = 42
    #: Include Futures / ITF results when building ratings and training.
    include_futures: bool = True
    include_challengers: bool = True

    def summary(self) -> dict:
        d = asdict(self)
        d["paths"] = {k: str(v) for k, v in d["paths"].items()}
        d["split"] = {k: list(v) for k, v in d["split"].items()}
        return d


DEFAULT_CONFIG = Config()
