#!/usr/bin/env python3
"""Prepare README-friendly figure assets.

The script intentionally takes all source paths as arguments so the repository
does not record private filesystem layouts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def save_resized(src: Path, dst: Path, max_width: int) -> None:
    image = Image.open(src)
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA")
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        background.alpha_composite(image)
        image = background.convert("RGB")

    if image.width > max_width:
        height = round(image.height * max_width / image.width)
        image = image.resize((max_width, height), Image.Resampling.LANCZOS)

    dst.parent.mkdir(parents=True, exist_ok=True)
    image.save(dst, optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-source", type=Path, required=True)
    parser.add_argument("--pipeline-source", type=Path, required=True)
    parser.add_argument("--architecture-source", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("assets/figures"))
    parser.add_argument("--max-width", type=int, default=1800)
    args = parser.parse_args()

    save_resized(args.demo_source, args.out_dir / "qualitative_examples.png", args.max_width)
    save_resized(args.pipeline_source, args.out_dir / "benchmark_pipeline.png", args.max_width)
    if args.architecture_source is not None:
        save_resized(args.architecture_source, args.out_dir / "architecture.png", args.max_width)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
