from __future__ import annotations

import torch
from torch import nn


class QPRMatcher(nn.Module):
    """Shared query-probe matcher used for Query-Conditioned Probe Reweighting.

    The module has no generator-specific parameters. It learns a bounded local
    correction to the semantic retrieval weights and therefore remains valid
    when a new capability-profile column is appended.
    """

    def __init__(
        self,
        feature_dim: int,
        projection_dim: int = 128,
        hidden_dim: int = 256,
        correction_bound: float = 1.0,
    ) -> None:
        super().__init__()
        if min(feature_dim, projection_dim, hidden_dim) <= 0 or correction_bound <= 0:
            raise ValueError("dimensions and correction_bound must be positive")
        self.query_projection = nn.Linear(feature_dim, projection_dim, bias=False)
        self.probe_projection = nn.Linear(feature_dim, projection_dim, bias=False)
        self.pair_mlp = nn.Sequential(
            nn.Linear(2 * projection_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.correction_bound = float(correction_bound)
        nn.init.zeros_(self.pair_mlp[-1].weight)
        nn.init.zeros_(self.pair_mlp[-1].bias)

    def pair_logits(
        self, query: torch.Tensor, probes: torch.Tensor, semantic_similarity: torch.Tensor
    ) -> torch.Tensor:
        if query.ndim != 2 or probes.ndim != 3 or semantic_similarity.ndim != 2:
            raise ValueError("expected query [B,D], probes [B,K,D], similarity [B,K]")
        if probes.shape[:2] != semantic_similarity.shape or probes.shape[0] != query.shape[0]:
            raise ValueError("query-probe batch dimensions do not match")
        q = self.query_projection(query)[:, None, :]
        p = self.probe_projection(probes)
        q = q.expand_as(p)
        cosine = torch.nn.functional.cosine_similarity(q, p, dim=-1).unsqueeze(-1)
        pair = torch.cat((q * p, (q - p).abs(), cosine, semantic_similarity.unsqueeze(-1)), dim=-1)
        return self.pair_mlp(pair).squeeze(-1)

    def forward(
        self,
        query: torch.Tensor,
        probes: torch.Tensor,
        semantic_similarity: torch.Tensor,
        base_weights: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if base_weights.shape != semantic_similarity.shape or torch.any(base_weights <= 0):
            raise ValueError("base_weights must be positive and match semantic_similarity")
        logits = self.pair_logits(query, probes, semantic_similarity)
        correction = self.correction_bound * torch.tanh(logits)
        weights = torch.softmax(base_weights.log() + correction, dim=-1)
        return {"weights": weights, "logits": logits, "correction": correction}


def predict_quality(
    prior: torch.Tensor,
    probe_residual: torch.Tensor,
    neighbor_indices: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if neighbor_indices.shape != weights.shape:
        raise ValueError("neighbor_indices and weights must have equal [B,K] shape")
    local = probe_residual[neighbor_indices]
    return prior[None, :] + torch.einsum("bk,bkm->bm", weights, local)

