#!/usr/bin/env python3
"""Run SenseNova-U1.5 image editing on MRrouter pilot samples."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch


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

SENSENOVA_ROOT = Path(os.environ.get("SENSENOVA_ROOT", "models/SenseNova-U1"))
DEFAULT_MODEL_PATH = Path(
    os.environ.get(
        "SENSENOVA_MODEL_PATH",
        "models/SenseNova-U1.5-8B-MoT",
    )
)
DEFAULT_LORA_DIR = Path(
    os.environ.get(
        "SENSENOVA_LORA_DIR",
        "models/SenseNova-U1.5-8B-MoT-LoRA-8step",
    )
)
DEFAULT_TARGET_PIXELS = 2048 * 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--records-csv", type=Path, required=True)
    parser.add_argument("--model-name", default="SenseNova-U1.5-8B-MoT")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--lora-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=999999)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--target-pixels", type=int, default=DEFAULT_TARGET_PIXELS)
    parser.add_argument("--input-max-pixels", default="auto")
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--img-cfg-scale", type=float, default=1.0)
    parser.add_argument("--cfg-norm", default="none", choices=["none", "global", "channel"])
    parser.add_argument("--timestep-shift", type=float, default=3.0)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn-backend", default="auto", choices=["auto", "flash", "sdpa"])
    parser.add_argument("--vram-mode", default="balanced", choices=["full", "fast", "low", "balanced"])
    parser.add_argument("--fast-vram-budget-gib", type=float, default=None)
    parser.add_argument("--warmup-first-sample", action="store_true", default=True)
    parser.add_argument("--no-warmup-first-sample", action="store_false", dest="warmup_first_sample")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    args = parser.parse_args()
    if (args.width is None) != (args.height is None):
        parser.error("--width and --height must be given together, or both omitted for official auto size.")
    return args


def load_editing_module() -> Any:
    sys.path.insert(0, str(SENSENOVA_ROOT / "src"))
    spec = importlib.util.spec_from_file_location(
        "sensenova_u1_editing_inference",
        SENSENOVA_ROOT / "examples" / "editing" / "inference.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load SenseNova editing inference module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: record.get(field, "") for field in FIELDS})


def read_successes(path: Path) -> set[tuple[str, int]]:
    if not path.is_file():
        return set()
    successes: set[tuple[str, int]] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            output_path = row.get("output_path") or ""
            if row.get("success") == "True" and output_path and Path(output_path).is_file():
                successes.add((row["sample_id"], int(row.get("seed") or 0)))
    return successes


def ordered_ref_paths(sample: dict[str, Any]) -> list[Path]:
    refs = sorted(sample.get("references", []), key=lambda ref: int(ref.get("ref_id") or 0))
    return [Path(ref.get("crop_path") or ref["source_path"]) for ref in refs]


def select_samples(samples: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.sample_id:
        wanted = set(args.sample_id)
        samples = [sample for sample in samples if sample["sample_id"] in wanted]
        missing = sorted(wanted - {sample["sample_id"] for sample in samples})
        if missing:
            raise ValueError(f"sample_id not found: {missing}")
    samples = samples[: args.max_samples]
    return [sample for index, sample in enumerate(samples) if index % args.num_shards == args.shard_index]


def dtype_from_name(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def reset_cuda_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_cuda_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def resolve_lora_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_file():
        return path
    if path.is_dir():
        matches = sorted(path.glob("*.safetensors"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"LoRA safetensors not found: {path}")


def generate(engine: Any, edit_mod: Any, sample: dict[str, Any], args: argparse.Namespace) -> Any:
    paths = ordered_ref_paths(sample)
    input_max_pixels = edit_mod._resolve_input_max_pixels(args.input_max_pixels, len(paths))
    images = [
        edit_mod._load_input_image(path, do_resize=True, input_max_pixels=input_max_pixels)
        for path in paths
    ]
    explicit_size = (args.width, args.height) if args.width is not None else None
    image_size = edit_mod._resolve_output_size(
        images,
        explicit=explicit_size,
        target_pixels=args.target_pixels,
    )
    outputs, _think = engine.edit(
        sample.get("instruction") or sample["prompt"],
        images,
        image_size=image_size,
        cfg_scale=args.cfg_scale,
        img_cfg_scale=args.img_cfg_scale,
        cfg_norm=args.cfg_norm,
        timestep_shift=args.timestep_shift,
        num_steps=args.num_steps,
        batch_size=1,
        think_mode=False,
        seed=args.seed,
    )
    return outputs[0]


def main() -> int:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard settings")
    if not args.model_path.exists():
        raise FileNotFoundError(f"SenseNova model path not found: {args.model_path}")
    lora_path = resolve_lora_path(args.lora_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.records_csv.parent.mkdir(parents=True, exist_ok=True)
    samples = select_samples(read_jsonl(args.samples_jsonl), args)
    done = read_successes(args.records_csv) if args.skip_existing else set()
    selected = [sample for sample in samples if (sample["sample_id"], args.seed) not in done]
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    config = json.dumps(
        {
            "model_path": str(args.model_path),
            "lora_path": str(lora_path) if lora_path else None,
            "seed": args.seed,
            "width": args.width,
            "height": args.height,
            "target_pixels": args.target_pixels,
            "cfg_scale": args.cfg_scale,
            "img_cfg_scale": args.img_cfg_scale,
            "cfg_norm": args.cfg_norm,
            "timestep_shift": args.timestep_shift,
            "num_steps": args.num_steps,
            "dtype": args.dtype,
            "attn_backend": args.attn_backend,
            "vram_mode": args.vram_mode,
            "fast_vram_budget_gib": args.fast_vram_budget_gib,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
        },
        ensure_ascii=True,
        sort_keys=True,
    )
    if not selected:
        print(f"[{args.model_name}] no pending samples", flush=True)
        return 0

    edit_mod = load_editing_module()
    import sensenova_u1

    sensenova_u1.set_attn_backend(args.attn_backend)
    load_start = time.perf_counter()
    engine = edit_mod.SenseNovaU1Editing(
        str(args.model_path),
        device=args.device,
        dtype=dtype_from_name(args.dtype),
        vram_mode=args.vram_mode,
        fast_vram_budget_gib=args.fast_vram_budget_gib,
    )
    if lora_path:
        engine.model = edit_mod.load_and_merge_lora_weight_from_safetensors(engine.model, str(lora_path))
    load_time = time.perf_counter() - load_start
    print(f"[{args.model_name}] loaded in {load_time:.3f}s pending={len(selected)} config={config}", flush=True)

    if args.warmup_first_sample and selected:
        try:
            warmup_start = time.perf_counter()
            _ = generate(engine, edit_mod, selected[0], args)
            print(
                f"[{args.model_name}] warmup {selected[0]['sample_id']} "
                f"runtime={time.perf_counter() - warmup_start:.3f}s not_recorded",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{args.model_name}] warmup failed: {type(exc).__name__}: {exc}", flush=True)

    for index, sample in enumerate(selected, 1):
        sample_id = sample["sample_id"]
        output_path = args.output_dir / sample_id / f"seed_{args.seed}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        success = False
        error = ""
        runtime = 0.0
        peak_mb = 0.0
        try:
            print(f"[{args.model_name} {index:03d}/{len(selected):03d}] {sample_id} start", flush=True)
            reset_cuda_peak()
            start = time.perf_counter()
            image = generate(engine, edit_mod, sample, args)
            runtime = time.perf_counter() - start
            peak_mb = peak_cuda_mb()
            image.save(output_path)
            success = True
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        append_record(
            args.records_csv,
            {
                "model": args.model_name,
                "sample_id": sample_id,
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
        print(f"[{args.model_name} {index:03d}/{len(selected):03d}] {sample_id} success={success} runtime={runtime:.3f}s peak={peak_mb:.1f}MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
