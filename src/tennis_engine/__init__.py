"""Leak-proof ATP tennis match prediction engine."""

from __future__ import annotations

from .config import Config, DEFAULT_CONFIG
from .features import FEATURE_NAMES, FeatureBuilder
from .predict import MatchPrediction, PlayerNotFound, PredictionEngine

__version__ = "2.0.0"

__all__ = [
    "Config",
    "DEFAULT_CONFIG",
    "FEATURE_NAMES",
    "FeatureBuilder",
    "MatchPrediction",
    "PlayerNotFound",
    "PredictionEngine",
    "__version__",
]
