#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mrg_router.io import read_jsonl
from mrg_router.profiles import build_capability_profiles


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Capability Profile Table.")
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    roles = {row["query_id"]: row["role"] for row in read_jsonl(args.split)}
    rows = [row for row in read_jsonl(args.outcomes) if roles[row["query_id"]] == "probe"]
    query_ids = sorted({row["query_id"] for row in rows})
    generator_ids = sorted({row["generator_id"] for row in rows})
    qi = {value: index for index, value in enumerate(query_ids)}
    gi = {value: index for index, value in enumerate(generator_ids)}
    matrix = np.full((len(query_ids), len(generator_ids)), np.nan, dtype=np.float64)
    for row in rows:
        matrix[qi[row["query_id"]], gi[row["generator_id"]]] = float(row["quality"])
    if not np.isfinite(matrix).all():
        raise RuntimeError("probe outcome matrix is incomplete")
    profiles = build_capability_profiles(matrix, generator_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        query_ids=np.asarray(query_ids),
        generator_ids=np.asarray(generator_ids),
        quality=matrix,
        prior=profiles.prior,
        probe_difficulty=profiles.probe_difficulty,
        residual=profiles.residual,
    )
    print(json.dumps({"probes": len(query_ids), "generators": len(generator_ids), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

