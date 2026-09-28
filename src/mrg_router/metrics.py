from __future__ import annotations

import numpy as np


def routing_metrics(predicted: np.ndarray, quality: np.ndarray, cost: np.ndarray) -> dict[str, float]:
    if predicted.shape != quality.shape or quality.shape != cost.shape:
        raise ValueError("predicted, quality, and cost must share [N,M] shape")
    selected = predicted.argmax(1)
    rows = np.arange(len(selected))
    chosen_quality = quality[rows, selected]
    return {
        "average_quality": float(chosen_quality.mean()),
        "average_cost": float(cost[rows, selected].mean()),
        "oracle_quality": float(quality.max(1).mean()),
        "mean_regret": float((quality.max(1) - chosen_quality).mean()),
    }


def bootstrap_mean(values: np.ndarray, replicates: int = 5000, seed: int = 1729) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("values must be a non-empty finite vector")
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 256):
        count = min(256, replicates - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        samples[start:start + count] = values[indices].mean(1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"mean": float(values.mean()), "ci95_low": float(low), "ci95_high": float(high)}

