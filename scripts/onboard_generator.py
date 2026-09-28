#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mrg_router.profiles import CapabilityProfiles


def main() -> int:
    parser = argparse.ArgumentParser(description="Append one generator without retraining QPR.")
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--generator-id", required=True)
    parser.add_argument("--probe-quality", type=Path, required=True, help="NPY vector in profile probe order")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = np.load(args.profiles)
    profiles = CapabilityProfiles(
        tuple(source["generator_ids"].astype(str)),
        source["prior"],
        source["probe_difficulty"],
        source["residual"],
        np.isfinite(source["residual"]),
    )
    expanded = profiles.append_generator(args.generator_id, np.load(args.probe_quality))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        query_ids=source["query_ids"],
        generator_ids=np.asarray(expanded.generator_ids),
        prior=expanded.prior,
        probe_difficulty=expanded.probe_difficulty,
        residual=expanded.residual,
    )
    print(json.dumps({
        "generator_id": args.generator_id,
        "probe_queries": len(source["query_ids"]),
        "qpr_parameters_updated": 0,
        "output": str(args.output),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

