#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mrg_router.features import FrozenMultimodalEncoder
from mrg_router.io import read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract frozen multimodal query features.")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--bge", type=Path, required=True)
    parser.add_argument("--siglip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    rows = list(read_jsonl(args.queries))
    encoder = FrozenMultimodalEncoder(args.bge, args.siglip, args.device)
    features = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        features.append(encoder.encode(
            [row["instruction"] for row in batch],
            [[reference["image_path"] for reference in row["references"]] for row in batch],
        ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        query_ids=np.asarray([row["query_id"] for row in rows]),
        features=np.concatenate(features),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

