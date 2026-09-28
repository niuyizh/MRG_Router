from __future__ import annotations

import numpy as np


def quality_choice(predicted_quality: np.ndarray) -> np.ndarray:
    return np.asarray(predicted_quality).argmax(1)


def cost_aware_choice(
    predicted_quality: np.ndarray, normalized_cost: np.ndarray, coefficient: float
) -> np.ndarray:
    quality = np.asarray(predicted_quality, dtype=np.float64)
    cost = np.asarray(normalized_cost, dtype=np.float64)
    if quality.shape != cost.shape or coefficient < 0.0:
        raise ValueError("quality and cost must match; coefficient must be non-negative")
    return (quality - coefficient * cost).argmax(1)


def near_best_cheapest_choice(
    predicted_quality: np.ndarray, cost: np.ndarray, tolerance: float = 0.03
) -> np.ndarray:
    quality = np.asarray(predicted_quality, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    if quality.shape != cost.shape or tolerance < 0.0:
        raise ValueError("quality and cost must match; tolerance must be non-negative")
    near_best = quality >= quality.max(1, keepdims=True) - tolerance
    return np.where(near_best, cost, np.inf).argmin(1)

