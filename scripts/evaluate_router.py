#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from mrg_router.costs import CostRegistry
from mrg_router.io import read_jsonl
from mrg_router.metrics import routing_metrics
from mrg_router.model import QPRMatcher, predict_quality
from mrg_router.policy import cost_aware_choice
from mrg_router.retrieval import topk_neighbors


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate MRG-Router on a frozen split.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, default=Path("data/mr_routebench/outcomes.jsonl.gz"))
    parser.add_argument("--split", type=Path, default=Path("data/mr_routebench/split.jsonl"))
    parser.add_argument("--queries", type=Path, default=Path("data/mr_routebench/queries.jsonl.gz"))
    parser.add_argument("--costs", type=Path, default=Path("configs/costs.yaml"))
    parser.add_argument("--role", choices=("validation", "test"), default="test")
    parser.add_argument("--cost-coefficient", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    profiles = np.load(args.profiles)
    features = np.load(args.features)
    feature_by_id = dict(zip(features["query_ids"].astype(str), features["features"]))
    roles = {row["query_id"]: row["role"] for row in read_jsonl(args.split)}
    query_rows = {row["query_id"]: row for row in read_jsonl(args.queries)}
    query_ids = sorted(query_id for query_id, role in roles.items() if role == args.role)
    generators = profiles["generator_ids"].astype(str).tolist()
    gi = {value: index for index, value in enumerate(generators)}
    qi = {value: index for index, value in enumerate(query_ids)}
    truth = np.full((len(query_ids), len(generators)), np.nan, dtype=np.float32)
    for row in read_jsonl(args.outcomes):
        if row["query_id"] in qi and row["generator_id"] in gi:
            truth[qi[row["query_id"]], gi[row["generator_id"]]] = float(row["quality"])
    if not np.isfinite(truth).all():
        raise RuntimeError("evaluation outcome matrix is incomplete")

    query_features = np.stack([feature_by_id[query_id] for query_id in query_ids])
    probe_features = np.stack([feature_by_id[query_id] for query_id in profiles["query_ids"].astype(str)])
    def normalize(value: np.ndarray) -> np.ndarray:
        value = (value - checkpoint["feature_mean"]) / checkpoint["feature_std"]
        return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-12)
    query_features = normalize(query_features)
    probe_features = normalize(probe_features)
    indices, similarity = topk_neighbors(query_features, probe_features, int(checkpoint["top_k"]))
    model = QPRMatcher(
        checkpoint["feature_dim"],
        checkpoint["projection_dim"],
        checkpoint["hidden_dim"],
        checkpoint["correction_bound"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(args.device).eval()
    with torch.inference_mode():
        q = torch.as_tensor(query_features, dtype=torch.float32, device=args.device)
        p = torch.as_tensor(probe_features, dtype=torch.float32, device=args.device)
        ix = torch.as_tensor(indices, dtype=torch.long, device=args.device)
        sim = torch.as_tensor(similarity, dtype=torch.float32, device=args.device)
        base = torch.full_like(sim, 1.0 / sim.shape[1])
        weights = model(q, p[ix], sim, base)["weights"]
        prediction = predict_quality(
            torch.as_tensor(profiles["prior"], dtype=torch.float32, device=args.device),
            torch.as_tensor(profiles["residual"], dtype=torch.float32, device=args.device),
            ix,
            weights,
        ).cpu().numpy()

    registry = CostRegistry.load(args.costs)
    costs = np.asarray([
        [registry.cost(generator, int(query_rows[query_id]["reference_count"])) for generator in generators]
        for query_id in query_ids
    ])
    normalized = (costs - costs.min(1, keepdims=True)) / np.maximum(
        costs.max(1, keepdims=True) - costs.min(1, keepdims=True), 1e-12
    )
    if args.cost_coefficient > 0:
        selected = cost_aware_choice(prediction, normalized, args.cost_coefficient)
        adjusted = np.full_like(prediction, -np.inf)
        adjusted[np.arange(len(selected)), selected] = 0.0
    else:
        adjusted = prediction
    print(json.dumps(routing_metrics(adjusted, truth, costs), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
