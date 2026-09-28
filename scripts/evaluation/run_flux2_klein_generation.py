#!/usr/bin/env python3
"""Run FLUX.2 Klein on MRrouter routing benchmark samples."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps


FIELDS = [
    "model",
    "sample_id",
    "num_references",
    "seed",
    "runtime_seconds",
    "model_load_time_seconds",
    "peak_gpu_memory_mb",
    "success",
    "error_message",
    "output_path",
    "gpu",
    "model_config",
]

DEFAULT_MODEL_NAME = "FLUX2-Klein-4B"
DEFAULT_MODEL_PATH = Path(os.environ.get("FLUX2_KLEIN_MODEL_PATH", "models/flux2-klein-4b"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-jsonl", type=Path, required=True)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--max-samples", type=int, default=999999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--records-csv", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--warmup-first-sample", action="store_true", default=True)
    parser.add_argument("--no-warmup-first-sample", action="store_false", dest="warmup_first_sample")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    return parser.parse_args()


def now() -> float:
    return time.perf_counter()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def select_samples(samples: list[dict[str, Any]], sample_id: str | None, max_samples: int) -> list[dict[str, Any]]:
    if sample_id:
        samples = [sample for sample in samples if sample["sample_id"] == sample_id]
    if max_samples is not None and max_samples >= 0:
        samples = samples[:max_samples]
    return samples


def read_successes(path: Path) -> set[tuple[str, int]]:
    if not path.is_file():
        return set()
    successes: set[tuple[str, int]] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            output_path = row.get("output_path") or ""
            if row.get("success") == "True" and output_path and Path(output_path).is_file():
                try:
                    seed = int(row.get("seed") or 0)
                except ValueError:
                    seed = 0
                successes.add((row["sample_id"], seed))
    return successes


def append_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: record.get(field, "") for field in FIELDS})


def image_path(path: str | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path)
    return candidate if candidate.is_file() else None


def ordered_image_paths(sample: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    if sample.get("unified_mode") == "editing":
        source = image_path(sample.get("source_path"))
        if source is not None:
            paths.append(source)
    refs = sorted(sample.get("references", []), key=lambda ref: ref.get("ref_id", 0))
    for ref in refs:
        path = image_path(ref.get("crop_path") or ref.get("source_path"))
        if path is not None:
            paths.append(path)
    return paths


def load_images(sample: dict[str, Any]) -> list[Image.Image]:
    images = []
    for path in ordered_image_paths(sample):
        images.append(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))
    return images


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def reset_cuda_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_cuda_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def generate_image(
    pipe: Any,
    sample: dict[str, Any],
    args: argparse.Namespace,
) -> Image.Image:
    input_images = load_images(sample)
    generator_device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(args.seed)
    with torch.inference_mode():
        result = pipe(
            image=input_images if input_images else None,
            prompt=sample["prompt"],
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=1,
            generator=generator,
        )
    return result.images[0]


def main() -> int:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.records_csv.parent.mkdir(parents=True, exist_ok=True)

    samples = select_samples(read_jsonl(args.samples_jsonl), args.sample_id, args.max_samples)
    samples = [sample for index, sample in enumerate(samples) if index % args.num_shards == args.shard_index]
    done = read_successes(args.records_csv) if args.skip_existing else set()
    selected = [sample for sample in samples if (sample["sample_id"], args.seed) not in done]
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    config = (
        f"model_path={args.model_path}; steps={args.num_inference_steps}; "
        f"guidance_scale={args.guidance_scale}; size={args.width}x{args.height}; "
        f"dtype={args.dtype}; cpu_offload={args.cpu_offload}; "
        f"shard={args.shard_index}/{args.num_shards}"
    )
    if not selected:
        print(f"[{args.model_name}] no pending samples for seed {args.seed} shard {args.shard_index}/{args.num_shards}", flush=True)
        return 0

    load_start = now()
    try:
        from diffusers import Flux2KleinPipeline

        pipe = Flux2KleinPipeline.from_pretrained(
            str(args.model_path),
            torch_dtype=dtype_from_name(args.dtype),
            local_files_only=True,
        )
        if args.cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(args.device)
        pipe.set_progress_bar_config(disable=None)
        load_time = now() - load_start
        print(
            f"[{args.model_name}] model loaded in {load_time:.3f}s; "
            f"pending={len(selected)}; {config}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        load_time = now() - load_start
        error = f"model_load_failed: {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        for sample in selected:
            append_record(
                args.records_csv,
                {
                    "model": args.model_name,
                    "sample_id": sample["sample_id"],
                    "num_references": sample.get("num_references", ""),
                    "seed": args.seed,
                    "runtime_seconds": "0.000",
                    "model_load_time_seconds": f"{load_time:.3f}",
                    "peak_gpu_memory_mb": "0.0",
                    "success": False,
                    "error_message": error.replace("\n", "\\n"),
                    "output_path": "",
                    "gpu": gpu,
                    "model_config": config,
                },
        )
        raise

    if args.warmup_first_sample and selected:
        warmup_sample = selected[0]
        try:
            reset_cuda_peak()
            warmup_start = now()
            _ = generate_image(pipe, warmup_sample, args)
            warmup_runtime = now() - warmup_start
            print(
                f"[{args.model_name}] warmup {warmup_sample['sample_id']} "
                f"runtime={warmup_runtime:.3f}s not_recorded",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[{args.model_name}] warmup {warmup_sample['sample_id']} failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

    for index, sample in enumerate(selected, 1):
        sample_dir = args.output_dir / sample["sample_id"]
        sample_dir.mkdir(parents=True, exist_ok=True)
        output_path = sample_dir / f"seed_{args.seed}.png"
        runtime = 0.0
        peak_mb = 0.0
        success = False
        error = ""
        try:
            print(
                f"[{args.model_name} {index:03d}/{len(selected):03d}] "
                f"{sample['sample_id']} start",
                flush=True,
            )
            reset_cuda_peak()
            start = now()
            image = generate_image(pipe, sample, args)
            runtime = now() - start
            peak_mb = peak_cuda_mb()
            image.save(output_path)
            success = True
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

        append_record(
            args.records_csv,
            {
                "model": args.model_name,
                "sample_id": sample["sample_id"],
                "num_references": sample.get("num_references", ""),
                "seed": args.seed,
                "runtime_seconds": f"{runtime:.3f}",
                "model_load_time_seconds": f"{load_time:.3f}",
                "peak_gpu_memory_mb": f"{peak_mb:.1f}",
                "success": success,
                "error_message": error.replace("\n", "\\n"),
                "output_path": str(output_path) if success else "",
                "gpu": gpu,
                "model_config": config,
            },
        )
        print(
            f"[{args.model_name} {index:03d}/{len(selected):03d}] "
            f"{sample['sample_id']} success={success} runtime={runtime:.3f}s peak={peak_mb:.1f}MB",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
