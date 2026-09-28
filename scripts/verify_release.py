#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
import csv
import gzip
from pathlib import Path

from mrg_router.evaluator import EvaluatorComponents, aggregate_quality
from mrg_router.io import read_jsonl


ROOT = Path(__file__).resolve().parents[1]

def main() -> int:
    split = list(read_jsonl(ROOT / "data/mr_routebench/split.jsonl"))
    roles = Counter(row["role"] for row in split)
    required = {"probe": 512, "train": 6888, "validation": 600, "test": 1000}
    if roles != required or len({row["query_id"] for row in split}) != 9000:
        raise RuntimeError(f"invalid split: {roles}")

    pairs = set()
    generators = set()
    maximum_error = 0.0
    for row in read_jsonl(ROOT / "data/mr_routebench/outcomes.jsonl.gz"):
        key = (row["query_id"], row["generator_id"])
        if key in pairs:
            raise RuntimeError(f"duplicate outcome: {key}")
        pairs.add(key)
        generators.add(row["generator_id"])
        reconstructed = aggregate_quality(EvaluatorComponents(
            row["reference_fidelity_30b"],
            row["reference_fidelity_8b"],
            row["prompt_following"],
            row["atomic_satisfaction"],
            row["perceptual_quality"],
        ))["quality"]
        maximum_error = max(maximum_error, abs(reconstructed - row["quality"]))
    if len(pairs) != 90000 or len(generators) != 10 or maximum_error > 1e-10:
        raise RuntimeError(
            f"invalid matrix: pairs={len(pairs)}, generators={len(generators)}, error={maximum_error}"
        )
    cost_pairs = set()
    for row in read_jsonl(ROOT / "data/mr_routebench/cost_matrix.jsonl.gz"):
        key = (row["query_id"], row["generator_id"])
        if key in cost_pairs or not 0.0 < float(row["paper_cost_usd"]):
            raise RuntimeError(f"invalid cost record: {key}")
        cost_pairs.add(key)
    if cost_pairs != pairs:
        raise RuntimeError(f"quality/cost pair mismatch: {len(pairs ^ cost_pairs)}")

    timing_counts = Counter()
    timing_pairs = set()
    timing_path = ROOT / "data/costbench300/records.csv.gz"
    with gzip.open(timing_path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["generator_id"], row["query_id"])
            if key in timing_pairs or float(row["runtime_seconds"]) <= 0:
                raise RuntimeError(f"invalid CostBench record: {key}")
            timing_pairs.add(key)
            timing_counts[(row["generator_id"], int(row["reference_count"]))] += 1
    timing_generators = {generator for generator, _ in timing_pairs}
    expected_timing = Counter(
        {(generator, reference_count): 75
         for generator in timing_generators for reference_count in (2, 3, 4, 5)}
    )
    if len(timing_pairs) != 2700 or len(timing_generators) != 9 or timing_counts != expected_timing:
        raise RuntimeError(
            f"invalid CostBench matrix: records={len(timing_pairs)}, "
            f"generators={len(timing_generators)}"
        )
    print(
        "PASS: 9,000 queries, 10 generators, 90,000 quality/cost outcomes; "
        "CostBench=9 generators x 300 records; "
        f"max reconstruction error={maximum_error:.3g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
