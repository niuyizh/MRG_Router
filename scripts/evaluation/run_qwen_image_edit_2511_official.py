#!/usr/bin/env python3
"""Run Qwen-Image-Edit-2511 with the official DiffSynth inference settings."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
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

DEFAULT_DYREF_SFT = Path(os.environ.get("DYREF_SFT_ROOT", "models/DyRef/sft"))
DEFAULT_HF_HOME = Path(os.environ.get("HF_HOME", "models/huggingface"))
DEFAULT_MODEL_PATH = (
    DEFAULT_HF_HOME
    / "models--Qwen--Qwen-Image-Edit-2511"
    / "snapshots"
    / "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9"
)
MODEL_NAME = "Qwen-Image-Edit-2511"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-jsonl", type=Path, required=True)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--max-samples", type=int, default=999999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--height", type=int, default=1152)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--negative-prompt", default="blurry, distorted, low quality, ugly")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument("--dyref-sft-root", type=Path, default=DEFAULT_DYREF_SFT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--records-csv", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--low-vram", action="store_true")
    parser.add_argument(
        "--vram-recipe",
        choices=("disk_fp8", "cpu_fp8"),
        default="disk_fp8",
        help="Low-VRAM recipe used when --low-vram is set.",
    )
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


def image_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_file() else None


def ordered_ref_paths(sample: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    refs = sorted(sample.get("references", []), key=lambda ref: ref.get("ref_id", 0))
    for ref in refs:
        path = image_path(ref.get("crop_path") or ref.get("source_path"))
        if path is not None:
            paths.append(path)
    return paths


def load_ref_images(sample: dict[str, Any]) -> list[Image.Image]:
    return [ImageOps.exif_transpose(Image.open(path)).convert("RGB") for path in ordered_ref_paths(sample)]


def reset_cuda_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_cuda_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def glob_required(pattern: str) -> list[str]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(pattern)
    return paths


def official_low_vram_config(recipe: str = "disk_fp8") -> dict[str, Any]:
    if recipe == "disk_fp8":
        return {
            "offload_dtype": "disk",
            "offload_device": "disk",
            "onload_dtype": torch.float8_e4m3fn,
            "onload_device": "cpu",
            "preparing_dtype": torch.float8_e4m3fn,
            "preparing_device": "cuda",
            "computation_dtype": torch.bfloat16,
            "computation_device": "cuda",
        }
    if recipe == "cpu_fp8":
        return {
            "offload_dtype": torch.float8_e4m3fn,
            "offload_device": "cpu",
            "onload_dtype": torch.float8_e4m3fn,
            "onload_device": "cpu",
            "preparing_dtype": torch.float8_e4m3fn,
            "preparing_device": "cuda",
            "computation_dtype": torch.bfloat16,
            "computation_device": "cuda",
        }
    raise ValueError(f"Unsupported low-VRAM recipe: {recipe}")


def build_pipeline(args: argparse.Namespace) -> Any:
    sys.path.insert(0, str(args.dyref_sft_root))
    from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline

    model_path = args.model_path
    transformer_paths = glob_required(str(model_path / "transformer" / "diffusion_pytorch_model*.safetensors"))
    text_encoder_paths = glob_required(str(model_path / "text_encoder" / "model*.safetensors"))
    vae_path = glob_required(str(model_path / "vae" / "diffusion_pytorch_model.safetensors"))[0]
    processor_path = model_path / "processor"
    tokenizer_path = model_path / "tokenizer"

    vram_config = official_low_vram_config(args.vram_recipe) if args.low_vram else {}
    return QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=[
            ModelConfig(path=transformer_paths, **vram_config),
            ModelConfig(path=text_encoder_paths, **vram_config),
            ModelConfig(path=vae_path, **vram_config),
        ],
        tokenizer_config=ModelConfig(path=str(tokenizer_path)),
        processor_config=ModelConfig(path=str(processor_path)),
    )


def generate(pipe: Any, sample: dict[str, Any], args: argparse.Namespace) -> Image.Image:
    edit_images = load_ref_images(sample)
    with torch.inference_mode():
        return pipe(
            sample["prompt"],
            negative_prompt=args.negative_prompt,
            cfg_scale=args.cfg_scale,
            edit_image=edit_images,
            seed=args.seed,
            num_inference_steps=args.num_inference_steps,
            height=args.height,
            width=args.width,
            edit_image_auto_resize=True,
            zero_cond_t=True,
        )


def main() -> int:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")

    os.environ.setdefault("HF_HOME", str(args.hf_home))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.records_csv.parent.mkdir(parents=True, exist_ok=True)

    samples = select_samples(read_jsonl(args.samples_jsonl), args.sample_id, args.max_samples)
    samples = [sample for index, sample in enumerate(samples) if index % args.num_shards == args.shard_index]
    done = read_successes(args.records_csv) if args.skip_existing else set()
    selected = [sample for sample in samples if (sample["sample_id"], args.seed) not in done]
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    config = (
        f"pipeline=DiffSynth.QwenImagePipeline; model_path={args.model_path}; "
        f"steps={args.num_inference_steps}; cfg_scale={args.cfg_scale}; "
        f"negative_prompt={args.negative_prompt!r}; size={args.width}x{args.height}; dtype=bf16; "
        f"low_vram={args.low_vram}; vram_recipe={args.vram_recipe if args.low_vram else 'none'}; "
        f"edit_image_auto_resize=True; zero_cond_t=True; shard={args.shard_index}/{args.num_shards}"
    )
    if not selected:
        print(f"[{MODEL_NAME}] no pending samples for seed {args.seed} shard {args.shard_index}/{args.num_shards}", flush=True)
        return 0

    load_start = now()
    try:
        pipe = build_pipeline(args)
        load_time = now() - load_start
        print(f"[{MODEL_NAME}] model loaded in {load_time:.3f}s; pending={len(selected)}; {config}", flush=True)
    except Exception as exc:  # noqa: BLE001
        load_time = now() - load_start
        error = f"model_load_failed: {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        for sample in selected:
            append_record(
                args.records_csv,
                {
                    "model": MODEL_NAME,
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
            _ = generate(pipe, warmup_sample, args)
            print(
                f"[{MODEL_NAME}] warmup {warmup_sample['sample_id']} "
                f"runtime={now() - warmup_start:.3f}s not_recorded",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{MODEL_NAME}] warmup {warmup_sample['sample_id']} failed: {type(exc).__name__}: {exc}", flush=True)

    for index, sample in enumerate(selected, 1):
        sample_dir = args.output_dir / sample["sample_id"]
        sample_dir.mkdir(parents=True, exist_ok=True)
        output_path = sample_dir / f"seed_{args.seed}.png"
        runtime = 0.0
        peak_mb = 0.0
        success = False
        error = ""
        try:
            print(f"[{MODEL_NAME} {index:04d}/{len(selected):04d}] {sample['sample_id']} start", flush=True)
            reset_cuda_peak()
            start = now()
            image = generate(pipe, sample, args)
            runtime = now() - start
            peak_mb = peak_cuda_mb()
            image.save(output_path)
            success = True
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

        append_record(
            args.records_csv,
            {
                "model": MODEL_NAME,
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
            f"[{MODEL_NAME} {index:04d}/{len(selected):04d}] "
            f"{sample['sample_id']} success={success} runtime={runtime:.3f}s peak={peak_mb:.1f}MB",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
