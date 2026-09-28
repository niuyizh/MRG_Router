import numpy as np
import torch

from mrg_router.evaluator import EvaluatorComponents, aggregate_quality
from mrg_router.model import QPRMatcher, predict_quality
from mrg_router.profiles import build_capability_profiles


def test_evaluator_factorization():
    result = aggregate_quality(EvaluatorComponents(0.8, 0.4, 0.81, 0.5, 0.7))
    assert np.isclose(result["reference_fidelity"], 0.7)
    assert np.isclose(result["atomic_satisfaction_smoothed"], 0.55)
    assert np.isclose(result["quality"], 0.7 * np.sqrt(0.81 * 0.55) * 0.7)


def test_profile_reconstruction_and_append():
    quality = np.asarray([[0.4, 0.7], [0.6, 0.3], [0.8, 0.5]])
    profile = build_capability_profiles(quality, ["a", "b"])
    assert np.allclose(profile.reconstruct_probe_quality(), quality)
    expanded = profile.append_generator("c", np.asarray([0.2, 0.4, 0.6]))
    assert expanded.residual.shape == (3, 3)
    assert np.allclose(expanded.reconstruct_probe_quality()[:, 2], [0.2, 0.4, 0.6])


def test_qpr_is_generator_agnostic():
    matcher = QPRMatcher(8, projection_dim=4, hidden_dim=6)
    query = torch.randn(2, 8)
    probes = torch.randn(2, 3, 8)
    similarity = torch.randn(2, 3)
    base = torch.full((2, 3), 1 / 3)
    result = matcher(query, probes, similarity, base)
    residual = torch.randn(5, 4)
    selected = torch.tensor([[0, 1, 2], [2, 3, 4]])
    quality = predict_quality(torch.randn(4), residual, selected, result["weights"])
    assert result["weights"].shape == (2, 3)
    assert torch.allclose(result["weights"].sum(1), torch.ones(2))
    assert quality.shape == (2, 4)
