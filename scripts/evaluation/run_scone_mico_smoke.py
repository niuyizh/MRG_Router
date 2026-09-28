#!/usr/bin/env python3
"""Run Scone on a small MICo sample set and record generation metadata."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from PIL import Image


SCONE_ROOT = Path(os.environ.get("SCONE_ROOT", "models/Scone"))
DEFAULT_MODEL_PATH = Path(os.environ.get("SCONE_MODEL_PATH", str(SCONE_ROOT / "ckpts" / "Scone")))

sys.path.insert(0, str(SCONE_ROOT))
from inference_single_case import load_model  # noqa: E402


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--records-csv", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="Use the full JSONL order instead of the legacy two-reference smoke sample.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--num-timesteps", type=int, default=50)
    parser.add_argument("--cfg-text-scale", type=float, default=4.0)
    parser.add_argument("--cfg-img-scale", type=float, default=2.0)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    return parser.parse_args()


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


def ordered_ref_paths(sample: dict[str, Any]) -> list[Path]:
    refs = sorted(sample["references"], key=lambda ref: int(ref["ref_id"]))
    return [Path(ref.get("crop_path") or ref["source_path"]) for ref in refs]


def select_samples(samples: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")

    if args.sample_id:
        wanted = set(args.sample_id)
        selected = [sample for sample in samples if sample["sample_id"] in wanted]
        missing = sorted(wanted - {sample["sample_id"] for sample in selected})
        if missing:
            raise ValueError(f"sample_id not found: {missing}")
    elif args.all_samples:
        selected = samples
    else:
        two_ref_samples = [sample for sample in samples if len(sample.get("references", [])) == 2]
        rng = random.Random(args.seed)
        rng.shuffle(two_ref_samples)
        selected = sorted(two_ref_samples[: args.max_samples], key=lambda sample: sample["sample_id"])

    if args.all_samples and args.max_samples is not None:
        selected = selected[: args.max_samples]
    return [
        sample
        for index, sample in enumerate(selected)
        if index % args.num_shards == args.shard_index
    ]


def flash_attention_backend() -> str:
    try:
        from flash_attn import flash_attn_varlen_func  # type: ignore

        return getattr(flash_attn_varlen_func, "__module__", "flash_attn")
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {type(exc).__name__}: {exc}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_image(inferencer: Any, input_list: list[Any], args: argparse.Namespace) -> Image.Image:
    output_dict = inferencer.interleave_inference(
        input_lists=input_list,
        cfg_text_scale=args.cfg_text_scale,
        cfg_img_scale=args.cfg_img_scale,
        cfg_interval=[0.0, 1.0],
        timestep_shift=3.0,
        num_timesteps=args.num_timesteps,
        cfg_renorm_min=0.0,
        cfg_renorm_type="text_channel",
        image_shapes=(args.width, args.height),
    )
    return output_dict[0]


def main() -> None:
    args = parse_args()
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Scone checkpoint snapshot not found: {args.model_path}")

    os.environ.setdefault("PYTHONPATH", str(SCONE_ROOT))
    set_seed(args.seed)
    samples = select_samples(read_jsonl(args.samples_jsonl), args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    load_start = time.perf_counter()
    inferencer = load_model(argparse.Namespace(model_path=str(args.model_path)))
    load_time = time.perf_counter() - load_start

    model_config = json.dumps(
        {
            "checkpoint": str(args.model_path),
            "seed": args.seed,
            "width": args.width,
            "height": args.height,
            "num_timesteps": args.num_timesteps,
            "cfg_text_scale": args.cfg_text_scale,
            "cfg_img_scale": args.cfg_img_scale,
            "cfg_interval": [0.0, 1.0],
            "timestep_shift": 3.0,
            "cfg_renorm_type": "text_channel",
            "all_samples": args.all_samples,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "selected_samples": len(samples),
            "flash_attention_backend": flash_attention_backend(),
            "protocol": "Scone official single-case inference hyperparameters on MICo inputs",
        },
        ensure_ascii=True,
        sort_keys=True,
    )

    for sample in samples:
        sample_id = sample["sample_id"]
        output_path = args.output_dir / sample_id / f"seed_{args.seed}.png"
        if args.skip_existing and output_path.is_file():
            append_record(
                args.records_csv,
                {
                    "model": "Scone",
                    "sample_id": sample_id,
                    "num_references": len(sample.get("references", [])),
                    "seed": args.seed,
                    "runtime_seconds": 0.0,
                    "model_load_time_seconds": load_time,
                    "peak_gpu_memory_mb": "",
                    "success": True,
                    "error_message": "skipped_existing",
                    "output_path": str(output_path),
                    "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    "model_config": model_config,
                },
            )
            continue

        start = time.perf_counter()
        peak_mb = ""
        success = False
        error_message = ""
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            refs = [Image.open(path).convert("RGB") for path in ordered_ref_paths(sample)]
            input_list = refs + [sample.get("instruction") or sample["prompt"]]
            image = generate_image(inferencer, input_list, args)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)
            if torch.cuda.is_available():
                peak_mb = f"{torch.cuda.max_memory_allocated() / 1024 / 1024:.1f}"
            success = True
            print(f"[ok] {sample_id} -> {output_path}", flush=True)
        except Exception as exc:  # noqa: BLE001
            error_message = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}"
            print(f"[error] {sample_id}: {error_message}", flush=True)

        append_record(
            args.records_csv,
            {
                "model": "Scone",
                "sample_id": sample_id,
                "num_references": len(sample.get("references", [])),
                "seed": args.seed,
                "runtime_seconds": f"{time.perf_counter() - start:.3f}",
                "model_load_time_seconds": f"{load_time:.3f}",
                "peak_gpu_memory_mb": peak_mb,
                "success": success,
                "error_message": error_message,
                "output_path": str(output_path) if success else "",
                "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "model_config": model_config,
            },
        )


if __name__ == "__main__":
    main()
