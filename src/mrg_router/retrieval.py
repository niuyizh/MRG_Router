from __future__ import annotations

import numpy as np
import torch


def l2_normalize(value: np.ndarray, axis: int = -1) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return value / np.maximum(np.linalg.norm(value, axis=axis, keepdims=True), 1e-12)


def topk_neighbors(query: np.ndarray, probes: np.ndarray, k: int = 32) -> tuple[np.ndarray, np.ndarray]:
    if query.ndim != 2 or probes.ndim != 2 or query.shape[1] != probes.shape[1]:
        raise ValueError("query and probes must be [N,D] and [P,D]")
    if not 1 <= k <= len(probes):
        raise ValueError("k must be within the probe-bank size")
    similarity = l2_normalize(query) @ l2_normalize(probes).T
    indices = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
    selected = np.take_along_axis(similarity, indices, axis=1)
    order = np.argsort(-selected, axis=1)
    return np.take_along_axis(indices, order, axis=1), np.take_along_axis(selected, order, axis=1)


def uniform_base_weights(shape: tuple[int, int], device: torch.device | str) -> torch.Tensor:
    return torch.full(shape, 1.0 / shape[1], dtype=torch.float32, device=device)

