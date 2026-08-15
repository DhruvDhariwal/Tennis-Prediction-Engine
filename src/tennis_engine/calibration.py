"""Probability calibration that preserves order-invariance.

A match predictor must satisfy ``p(A beats B) == 1 - p(B beats A)``. An
arbitrary post-hoc calibrator breaks that: isotonic regression fitted on raw
``(p, y)`` pairs has no reason to map ``p`` and ``1 - p`` to complementary
values.

Two calibrators are provided:

``TemperatureCalibrator``
    ``p' = sigmoid(logit(p) / T)``. With no intercept term this is *exactly*
    symmetric for every ``T``, because ``logit(1 - p) = -logit(p)``.
``SymmetricIsotonicCalibrator``
    Isotonic regression fitted on the symmetry-augmented sample
    ``{(p, y)} u {(1 - p, 1 - y)}``, then explicitly symmetrized at predict time.
    More flexible, symmetric to within the resolution of the fit.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class IdentityCalibrator:
    name = "identity"

    def fit(self, p: np.ndarray, y: np.ndarray) -> "IdentityCalibrator":
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return np.asarray(p, dtype=np.float64)

    def to_dict(self) -> dict:
        return {"kind": "identity"}


class TemperatureCalibrator:
    """Single-parameter logit scaling. Exactly symmetric by construction."""

    name = "temperature"

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature

    def fit(self, p: np.ndarray, y: np.ndarray) -> "TemperatureCalibrator":
        z = _logit(np.asarray(p, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64)

        def objective(log_t: float) -> float:
            return float(log_loss(y, _sigmoid(z / np.exp(log_t)), labels=[0, 1]))

        result = minimize_scalar(objective, bounds=(-2.0, 2.0), method="bounded")
        self.temperature = float(np.exp(result.x))
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return _sigmoid(_logit(np.asarray(p, dtype=np.float64)) / self.temperature)

    def to_dict(self) -> dict:
        return {"kind": "temperature", "temperature": self.temperature}


class SymmetricIsotonicCalibrator:
    """Isotonic regression fitted on a symmetry-augmented sample."""

    name = "isotonic"

    def __init__(self) -> None:
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, p: np.ndarray, y: np.ndarray) -> "SymmetricIsotonicCalibrator":
        p = np.asarray(p, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        p_aug = np.concatenate([p, 1.0 - p])
        y_aug = np.concatenate([y, 1.0 - y])
        self.iso.fit(p_aug, y_aug)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        forward = self.iso.predict(p)
        backward = 1.0 - self.iso.predict(1.0 - p)
        # Averaging the two directions removes any residual asymmetry left by
        # the piecewise-constant fit.
        return np.clip(0.5 * (forward + backward), EPS, 1.0 - EPS)

    def to_dict(self) -> dict:
        return {
            "kind": "isotonic",
            "x": self.iso.X_thresholds_.tolist(),
            "y": self.iso.y_thresholds_.tolist(),
        }


def from_dict(payload: dict):
    kind = payload.get("kind", "identity")
    if kind == "temperature":
        return TemperatureCalibrator(payload["temperature"])
    if kind == "isotonic":
        cal = SymmetricIsotonicCalibrator()
        cal.iso.fit(np.asarray(payload["x"]), np.asarray(payload["y"]))
        return cal
    return IdentityCalibrator()


def select_calibrator(p_val: np.ndarray, y_val: np.ndarray):
    """Fit every candidate on the validation split and keep the best log loss."""
    candidates = [
        IdentityCalibrator(),
        TemperatureCalibrator().fit(p_val, y_val),
        SymmetricIsotonicCalibrator().fit(p_val, y_val),
    ]
    scored = [
        (float(log_loss(y_val, c.transform(p_val), labels=[0, 1])), c)
        for c in candidates
    ]
    scored.sort(key=lambda t: t[0])
    return scored[0][1], {c.name: round(ll, 6) for ll, c in scored}
