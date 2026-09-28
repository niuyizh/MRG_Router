from __future__ import annotations

from dataclasses import dataclass
import copy

import numpy as np
import torch
from torch import nn

from .model import QPRMatcher, predict_quality


@dataclass(frozen=True)
class TrainingConfig:
    utility_epochs: int = 5
    routing_epochs: int = 20
    batch_size: int = 192
    utility_learning_rate: float = 3e-4
    routing_learning_rate: float = 1.5e-4
    weight_decay: float = 1e-3
    ranking_weight: float = 0.2
    prior_weight: float = 0.02
    utility_weight: float = 0.1
    decision_temperature: float = 0.03
    ranking_margin: float = 0.03


def utility_targets(
    train_quality: np.ndarray,
    prior: np.ndarray,
    probe_residual: np.ndarray,
    temperature: float = 0.05,
) -> np.ndarray:
    probe_policy = _softmax((prior[None, :] + probe_residual) / 0.03, axis=1)
    utility = train_quality @ probe_policy.T
    regret = train_quality.max(1, keepdims=True) - utility
    return np.exp(-regret / temperature).astype(np.float32)


def _softmax(value: np.ndarray, axis: int) -> np.ndarray:
    shifted = value - value.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def routing_loss(score: torch.Tensor, quality: torch.Tensor, temperature: float) -> torch.Tensor:
    policy = torch.softmax(score / temperature, dim=1)
    return (quality.max(1).values - (policy * quality).sum(1)).mean()


def ranking_loss(
    score: torch.Tensor, quality: torch.Tensor, margin: float, temperature: float
) -> torch.Tensor:
    score_difference = score[:, :, None] - score[:, None, :]
    quality_difference = quality[:, :, None] - quality[:, None, :]
    usable = quality_difference.abs() >= margin
    if not usable.any():
        return score.sum() * 0.0
    signed = quality_difference.sign()[usable] * score_difference[usable]
    return (temperature * torch.nn.functional.softplus(-signed / temperature)).mean()


def prior_regularization(weights: torch.Tensor, base_weights: torch.Tensor) -> torch.Tensor:
    return (
        weights
        * (weights.clamp_min(1e-12).log() - base_weights.clamp_min(1e-12).log())
    ).sum(1).mean()


class RouterTrainer:
    def __init__(self, model: QPRMatcher, config: TrainingConfig, device: str = "cuda") -> None:
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device)

    def fit(
        self,
        *,
        train_features: np.ndarray,
        train_quality: np.ndarray,
        probe_features: np.ndarray,
        probe_residual: np.ndarray,
        prior: np.ndarray,
        neighbor_indices: np.ndarray,
        neighbor_similarity: np.ndarray,
        validation: dict[str, np.ndarray],
        seed: int,
    ) -> dict[str, object]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        config = self.config
        q = torch.as_tensor(train_features, dtype=torch.float32, device=self.device)
        p = torch.as_tensor(probe_features, dtype=torch.float32, device=self.device)
        truth = torch.as_tensor(train_quality, dtype=torch.float32, device=self.device)
        residual = torch.as_tensor(probe_residual, dtype=torch.float32, device=self.device)
        mu = torch.as_tensor(prior, dtype=torch.float32, device=self.device)
        indices = torch.as_tensor(neighbor_indices, dtype=torch.long, device=self.device)
        similarity = torch.as_tensor(neighbor_similarity, dtype=torch.float32, device=self.device)
        base = torch.full_like(similarity, 1.0 / similarity.shape[1])
        target = torch.as_tensor(
            utility_targets(train_quality, prior, probe_residual), dtype=torch.float32, device=self.device
        )
        rng = np.random.default_rng(seed)
        best_quality = -float("inf")
        best_state = copy.deepcopy(self.model.state_dict())

        phases = (
            ("utility", config.utility_epochs, config.utility_learning_rate),
            ("routing", config.routing_epochs, config.routing_learning_rate),
        )
        for phase, epochs, learning_rate in phases:
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=learning_rate, weight_decay=config.weight_decay
            )
            for _ in range(epochs):
                self.model.train()
                order = rng.permutation(len(q))
                for start in range(0, len(order), config.batch_size):
                    batch = torch.as_tensor(order[start:start + config.batch_size], device=self.device)
                    selected = indices[batch]
                    result = self.model(q[batch], p[selected], similarity[batch], base[batch])
                    local_target = target[batch].gather(1, selected)
                    utility = torch.nn.functional.smooth_l1_loss(
                        torch.sigmoid(result["logits"]), local_target, beta=0.1
                    )
                    if phase == "utility":
                        loss = utility
                    else:
                        score = predict_quality(mu, residual, selected, result["weights"])
                        loss = (
                            routing_loss(score, truth[batch], config.decision_temperature)
                            + config.ranking_weight
                            * ranking_loss(
                                score, truth[batch], config.ranking_margin, config.decision_temperature
                            )
                            + config.prior_weight * prior_regularization(result["weights"], base[batch])
                            + config.utility_weight * utility
                        )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                value = self.evaluate(**validation)
                if value > best_quality:
                    best_quality = value
                    best_state = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(best_state)
        return {"validation_quality": best_quality, "state_dict": best_state, "seed": seed}

    @torch.inference_mode()
    def evaluate(
        self,
        *,
        features: np.ndarray,
        quality: np.ndarray,
        probe_features: np.ndarray,
        probe_residual: np.ndarray,
        prior: np.ndarray,
        neighbor_indices: np.ndarray,
        neighbor_similarity: np.ndarray,
    ) -> float:
        self.model.eval()
        q = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        p = torch.as_tensor(probe_features, dtype=torch.float32, device=self.device)
        indices = torch.as_tensor(neighbor_indices, dtype=torch.long, device=self.device)
        similarity = torch.as_tensor(neighbor_similarity, dtype=torch.float32, device=self.device)
        base = torch.full_like(similarity, 1.0 / similarity.shape[1])
        result = self.model(q, p[indices], similarity, base)
        score = predict_quality(
            torch.as_tensor(prior, dtype=torch.float32, device=self.device),
            torch.as_tensor(probe_residual, dtype=torch.float32, device=self.device),
            indices,
            result["weights"],
        ).cpu().numpy()
        selected = score.argmax(1)
        return float(np.mean(quality[np.arange(len(quality)), selected]))

