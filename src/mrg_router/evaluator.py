from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable


def _unit(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


@dataclass(frozen=True)
class EvaluatorComponents:
    reference_fidelity_30b: float
    reference_fidelity_8b: float
    prompt_following: float
    atomic_satisfaction: float
    perceptual_quality: float

    def validate(self) -> None:
        for name, value in asdict(self).items():
            _unit(value, name)


def aggregate_quality(
    components: EvaluatorComponents,
    *,
    large_judge_weight: float = 0.75,
    atomic_floor: float = 0.1,
) -> dict[str, float]:
    """Reconstruct the factorized evaluator reported in the paper.

    R combines the two reference-fidelity judges, S is the geometric mean of
    holistic prompt following and smoothed atomic satisfaction, and V is blind
    perceptual quality. Final-Q is their product.
    """
    components.validate()
    if not 0.0 <= large_judge_weight <= 1.0:
        raise ValueError("large_judge_weight must be in [0, 1]")
    if not 0.0 <= atomic_floor < 1.0:
        raise ValueError("atomic_floor must be in [0, 1)")
    reference = (
        large_judge_weight * components.reference_fidelity_30b
        + (1.0 - large_judge_weight) * components.reference_fidelity_8b
    )
    atomic_smoothed = atomic_floor + (1.0 - atomic_floor) * components.atomic_satisfaction
    semantic = math.sqrt(components.prompt_following * atomic_smoothed)
    quality = reference * semantic * components.perceptual_quality
    return {
        "reference_fidelity": reference,
        "atomic_satisfaction_smoothed": atomic_smoothed,
        "semantic_alignment": semantic,
        "perceptual_quality": components.perceptual_quality,
        "quality": quality,
    }


def atomic_satisfaction(answers: Iterable[str]) -> float:
    normalized = [answer.strip().upper() for answer in answers]
    if not normalized or any(answer not in {"YES", "NO"} for answer in normalized):
        raise ValueError("atomic answers must be a non-empty YES/NO sequence")
    return sum(answer == "YES" for answer in normalized) / len(normalized)

