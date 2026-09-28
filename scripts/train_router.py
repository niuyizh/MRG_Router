#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from mrg_router.io import read_jsonl
from mrg_router.model import QPRMatcher
from mrg_router.retrieval import topk_neighbors
from mrg_router.training import RouterTrainer, TrainingConfig


def quality_matrix(path: Path, query_ids: list[str], generators: list[str]) -> np.ndarray:
    qi = {value: index for index, value in enumerate(query_ids)}
    gi = {value: index for index, value in enumerate(generators)}
    matrix = np.full((len(query_ids), len(generators)), np.nan, dtype=np.float32)
    for row in read_jsonl(path):
        if row["query_id"] in qi and row["generator_id"] in gi:
            matrix[qi[row["query_id"]], gi[row["generator_id"]]] = float(row["quality"])
    if not np.isfinite(matrix).all():
        raise RuntimeError("requested quality matrix is incomplete")
    return matrix


def main() -> int:
    parser = argparse.ArgumentParser(description="Train MRG-Router QPR.")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, default=Path("data/mr_routebench/outcomes.jsonl.gz"))
    parser.add_argument("--split", type=Path, default=Path("data/mr_routebench/split.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("configs/router.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    feature_file = np.load(args.features)
    feature_by_id = dict(zip(feature_file["query_ids"].astype(str), feature_file["features"]))
    profile_file = np.load(args.profiles)
    probe_ids = profile_file["query_ids"].astype(str).tolist()
    generators = profile_file["generator_ids"].astype(str).tolist()
    roles = {row["query_id"]: row["role"] for row in read_jsonl(args.split)}
    train_ids = sorted(query_id for query_id, role in roles.items() if role == "train")
    validation_ids = sorted(query_id for query_id, role in roles.items() if role == "validation")
    probe_features = np.stack([feature_by_id[query_id] for query_id in probe_ids])
    train_features = np.stack([feature_by_id[query_id] for query_id in train_ids])
    validation_features = np.stack([feature_by_id[query_id] for query_id in validation_ids])
    feature_mean = train_features.mean(0)
    feature_std = np.maximum(train_features.std(0), 1e-5)
    def normalize(value: np.ndarray) -> np.ndarray:
        value = (value - feature_mean) / feature_std
        return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-12)
    train_features = normalize(train_features)
    validation_features = normalize(validation_features)
    probe_features = normalize(probe_features)
    k = int(config["retrieval"]["top_k"])
    train_indices, train_similarity = topk_neighbors(train_features, probe_features, k)
    val_indices, val_similarity = topk_neighbors(validation_features, probe_features, k)
    train_quality = quality_matrix(args.outcomes, train_ids, generators)
    val_quality = quality_matrix(args.outcomes, validation_ids, generators)

    qpr = config["qpr"]
    model = QPRMatcher(
        train_features.shape[1], qpr["projection_dim"], qpr["hidden_dim"], qpr["correction_bound"]
    )
    training = TrainingConfig(**{
        name: value for name, value in config["training"].items()
        if name in TrainingConfig.__dataclass_fields__
    })
    trainer = RouterTrainer(model, training, args.device)
    result = trainer.fit(
        train_features=train_features,
        train_quality=train_quality,
        probe_features=probe_features,
        probe_residual=profile_file["residual"],
        prior=profile_file["prior"],
        neighbor_indices=train_indices,
        neighbor_similarity=train_similarity,
        validation={
            "features": validation_features,
            "quality": val_quality,
            "probe_features": probe_features,
            "probe_residual": profile_file["residual"],
            "prior": profile_file["prior"],
            "neighbor_indices": val_indices,
            "neighbor_similarity": val_similarity,
        },
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": result["state_dict"],
        "feature_dim": train_features.shape[1],
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "projection_dim": qpr["projection_dim"],
        "hidden_dim": qpr["hidden_dim"],
        "correction_bound": qpr["correction_bound"],
        "top_k": k,
        "generator_ids": generators,
        "seed": args.seed,
        "validation_quality": result["validation_quality"],
    }, args.output)
    print(json.dumps({"validation_quality": result["validation_quality"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
