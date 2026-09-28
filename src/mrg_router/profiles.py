from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CapabilityProfiles:
    generator_ids: tuple[str, ...]
    prior: np.ndarray
    probe_difficulty: np.ndarray
    residual: np.ndarray
    valid: np.ndarray

    def reconstruct_probe_quality(self) -> np.ndarray:
        return self.prior[None, :] + self.probe_difficulty[:, None] + self.residual

    def append_generator(
        self, generator_id: str, probe_quality: np.ndarray, weights: np.ndarray | None = None
    ) -> "CapabilityProfiles":
        if generator_id in self.generator_ids:
            raise ValueError(f"generator already exists: {generator_id}")
        quality = np.asarray(probe_quality, dtype=np.float64)
        if quality.shape != (len(self.probe_difficulty),) or not np.isfinite(quality).all():
            raise ValueError("new generator needs one finite quality value per probe")
        w = _normalized_weights(len(quality), weights)
        prior = float(np.sum(w * (quality - self.probe_difficulty)))
        residual = quality - prior - self.probe_difficulty
        return CapabilityProfiles(
            self.generator_ids + (generator_id,),
            np.concatenate((self.prior, [prior])),
            self.probe_difficulty.copy(),
            np.column_stack((self.residual, residual)),
            np.column_stack((self.valid, np.ones(len(quality), dtype=bool))),
        )


def _normalized_weights(size: int, weights: np.ndarray | None) -> np.ndarray:
    value = np.ones(size, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if value.shape != (size,) or np.any(value < 0.0) or value.sum() <= 0.0:
        raise ValueError("weights must be non-negative [P] with positive sum")
    return value / value.sum()


def build_capability_profiles(
    probe_quality: np.ndarray,
    generator_ids: list[str] | tuple[str, ...],
    *,
    valid: np.ndarray | None = None,
    probe_weights: np.ndarray | None = None,
) -> CapabilityProfiles:
    """Decompose probe outcomes into global capability, difficulty, and residual.

    The coordinate system is estimated only from capability-probe outcomes.
    """
    quality = np.asarray(probe_quality, dtype=np.float64)
    if quality.ndim != 2 or quality.shape[1] != len(generator_ids):
        raise ValueError("probe_quality must be [P, M] and match generator_ids")
    mask = np.isfinite(quality) if valid is None else np.asarray(valid, dtype=bool)
    if mask.shape != quality.shape or np.any(mask.sum(0) == 0) or np.any(mask.sum(1) == 0):
        raise ValueError("every probe and generator needs at least one valid outcome")
    weights = _normalized_weights(len(quality), probe_weights)
    denominator = (mask * weights[:, None]).sum(0)
    prior = np.where(mask, quality * weights[:, None], 0.0).sum(0) / denominator
    centered = quality - prior[None, :]
    difficulty = np.where(mask, centered, 0.0).sum(1) / mask.sum(1)
    residual = np.where(mask, quality - prior[None, :] - difficulty[:, None], np.nan)
    return CapabilityProfiles(tuple(generator_ids), prior, difficulty, residual, mask)

