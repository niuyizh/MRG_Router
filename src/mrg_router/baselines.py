from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import nn


class ZOOTERAdapt(nn.Module):
    """Reward-distribution distillation adapted to dense image quality."""

    def __init__(self, input_dim: int, generator_count: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, generator_count),
        )

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        return self.network(query)

    @staticmethod
    def loss(scores: torch.Tensor, quality: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
        target = torch.softmax(quality / temperature, dim=1)
        return F.kl_div(torch.log_softmax(scores, dim=1), target, reduction="batchmean")


class RouterDCAdapt(nn.Module):
    """Query-generator contrastive routing adapted to the shared outcome matrix."""

    def __init__(self, input_dim: int, generator_count: int, embedding_dim: int = 128) -> None:
        super().__init__()
        self.query = nn.Sequential(nn.Linear(input_dim, 256), nn.GELU(), nn.Linear(256, embedding_dim))
        self.generator_embeddings = nn.Parameter(torch.randn(generator_count, embedding_dim) * 0.02)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        q = F.normalize(self.query(query), dim=-1)
        g = F.normalize(self.generator_embeddings, dim=-1)
        return self.logit_scale.exp().clamp(max=100.0) * q @ g.T


class SCOPERouterAdapt(nn.Module):
    """Cost-conditioned query/profile dual tower for open-set routing."""

    def __init__(self, query_dim: int, profile_dim: int, embedding_dim: int = 128) -> None:
        super().__init__()
        self.query_tower = nn.Sequential(
            nn.Linear(query_dim + 1, 256), nn.GELU(), nn.Linear(256, embedding_dim)
        )
        self.profile_tower = nn.Sequential(
            nn.Linear(profile_dim, 256), nn.GELU(), nn.Linear(256, embedding_dim)
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def forward(
        self, query: torch.Tensor, profiles: torch.Tensor, cost_weight: torch.Tensor
    ) -> torch.Tensor:
        if cost_weight.ndim == 1:
            cost_weight = cost_weight[:, None]
        q = F.normalize(self.query_tower(torch.cat((query, cost_weight), dim=1)), dim=-1)
        p = F.normalize(self.profile_tower(profiles), dim=-1)
        return self.logit_scale.exp().clamp(max=100.0) * q @ p.T

