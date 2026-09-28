#!/usr/bin/env python3
"""Batch runner: Qwen-Image-Edit-2511 + official Lightning 4-step LoRA.

Drop-in compatible with the other 8 runners (same records CSV schema, same
<output-dir>/<sample_id>/seed_42.png layout), so it can be driven by
run_multibanana_9models.py or a plain sequential loop.

Official config (ModelTC/Qwen-Image-Lightning):
    steps=4, true CFG off (cfg 1.0), LoRA alpha 1.0, bf16, seed 42.

Two memory plans for a 24 GB card:
    --gguf-path <Q4_K_M.gguf>   -> quantized DiT (13.2 GB on disk) + --vram-plan model-offload
                                   peak ~18 GB VRAM
    (omitted)                   -> bf16 DiT + --vram-plan model-offload  (fits 24 GB too,
                                   peak ~34 GB -> NOT for 24 GB; use GGUF there)
"""
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

FIELDS = ["model", "sample_id", "num_references", "seed", "runtime_seconds",
          "model_load_time_seconds", "peak_gpu_memory_mb", "success", "error_message",
          "output_path", "gpu", "model_config"]
MODEL_NAME = "Qwen-Image-Edit-2511+Lightning-4step"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples-jsonl", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--records-csv", type=Path, required=True)
    p.add_argument("--model-path", type=Path, required=True, help="base repo dir (TE/VAE/tokenizer)")
    p.add_argument("--gguf-path", type=Path, default=None, help="GGUF Q4_K_M transformer (optional)")
    p.add_argument("--lora-path", type=Path, required=True)
    p.add_argument("--lora-alpha", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--true-cfg-scale", type=float, default=1.0)
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--negative-prompt", default=" ")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=896)
    p.add_argument("--height", type=int, default=1152)
    p.add_argument("--vram-plan", choices=["resident", "model-offload", "sequential-offload"],
                   default="model-offload")
    p.add_argument("--fp8-transformer-layerwise", action="store_true", default=False,
                   help="fp8_e4m3fn storage + bf16 compute, weights resident (no offload)")
    p.add_argument("--max-samples", type=int, default=999999)
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def ordered_ref_paths(sample: dict[str, Any]) -> list[Path]:
    refs = sorted(sample.get("references", []), key=lambda r: r.get("ref_id", 0))
    out = []
    for r in refs:
        p = r.get("crop_path") or r.get("source_path")
        if p and Path(p).is_file():
            out.append(Path(p))
    return out


def sample_prompt(sample: dict[str, Any]) -> str:
    for k in ("prompt", "instruction", "edit_prompt", "target_prompt"):
        v = sample.get(k)
        if isinstance(v, str) and v.strip():
            return v
    raise KeyError("prompt")


def read_successes(path: Path, seed: int) -> set[str]:
    if not path.is_file():
        return set()
    done = set()
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("success") == "True" and Path(row.get("output_path", "")).is_file():
                done.add(row["sample_id"])
    return done


def append_record(path: Path, rec: dict[str, Any]) -> None:
    exists = path.is_file()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, restval="", extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow({k: rec.get(k, "") for k in FIELDS})


def build_pipe(args: argparse.Namespace):
    from diffusers import QwenImageEditPlusPipeline

    transformer = None
    if args.gguf_path is not None:
        from diffusers import GGUFQuantizationConfig
        from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
        transformer = QwenImageTransformer2DModel.from_single_file(
            str(args.gguf_path),
            config=str(Path(args.model_path) / "transformer"),   # GGUF has no config metadata
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16, local_files_only=True,
        )
    kw = dict(torch_dtype=torch.bfloat16, local_files_only=True)
    if transformer is not None:          # passing transformer=None would SKIP loading it
        kw["transformer"] = transformer
    pipe = QwenImageEditPlusPipeline.from_pretrained(str(args.model_path), **kw)
    # offline mode requires an explicit weight_name when loading a LoRA file
    _lp = Path(args.lora_path)
    pipe.load_lora_weights(str(_lp.parent), weight_name=_lp.name, adapter_name="lightning")
    if args.fp8_transformer_layerwise and transformer is None:
        # fp8 storage + bf16 compute, weights resident on GPU (no offload).
        # Fuse the LoRA FIRST: layerwise casting would otherwise turn the LoRA
        # A/B Linears fp8 too, and F.linear on fp8 weights is not implemented.
        pipe.fuse_lora(adapter_names=["lightning"], lora_scale=args.lora_alpha)
        pipe.unload_lora_weights()
        pipe.transformer.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
    else:
        pipe.set_adapters(["lightning"], adapter_weights=[args.lora_alpha])
    if args.vram_plan == "model-offload":
        pipe.enable_model_cpu_offload()
    elif args.vram_plan == "sequential-offload":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to(args.device)
    pipe.set_progress_bar_config(disable=True)
    n_lora = sum(1 for n, _ in pipe.transformer.named_modules() if "lora_A" in n or "lora_B" in n)
    print(f"[{MODEL_NAME}] alpha={args.lora_alpha} residual_lora_modules={n_lora} "
          f"fp8_layerwise={args.fp8_transformer_layerwise} plan={args.vram_plan}", flush=True)
    return pipe


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.records_csv.parent.mkdir(parents=True, exist_ok=True)

    samples = [json.loads(l) for l in args.samples_jsonl.open(encoding="utf-8") if l.strip()]
    samples = samples[: args.max_samples]
    done = read_successes(args.records_csv, args.seed) if args.skip_existing else set()
    todo = [s for s in samples if s["sample_id"] not in done]
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    config = (f"pipeline=diffusers.QwenImageEditPlusPipeline+Lightning; base={args.model_path}; "
              f"gguf={args.gguf_path}; lora={args.lora_path}; alpha={args.lora_alpha}; "
              f"steps={args.steps}; true_cfg={args.true_cfg_scale}; guidance={args.guidance_scale}; "
              f"size={args.width}x{args.height}; dtype=bf16; vram_plan={args.vram_plan}")
    if not todo:
        print(f"[{MODEL_NAME}] no pending samples", flush=True)
        return 0

    t0 = time.perf_counter()
    pipe = build_pipe(args)
    load_s = time.perf_counter() - t0
    print(f"[{MODEL_NAME}] loaded in {load_s:.1f}s; pending={len(todo)}; {config}", flush=True)

    for i, s in enumerate(todo, 1):
        sid = s["sample_id"]
        out_dir = args.output_dir / sid
        out_dir.mkdir(parents=True, exist_ok=True)
        png = out_dir / f"seed_{args.seed}.png"
        rec = {"model": MODEL_NAME, "sample_id": sid, "seed": args.seed,
               "num_references": s.get("num_references", ""), "output_path": str(png),
               "gpu": gpu, "model_config": config, "model_load_time_seconds": round(load_s, 2)}
        t = time.perf_counter()
        try:
            refs = [ImageOps.exif_transpose(Image.open(p)).convert("RGB")
                    for p in ordered_ref_paths(s)]
            if not refs:
                raise FileNotFoundError("no readable reference images")
            gen = torch.Generator(device="cpu").manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                out = pipe(image=refs, prompt=sample_prompt(s),
                           negative_prompt=args.negative_prompt,
                           true_cfg_scale=args.true_cfg_scale,
                           guidance_scale=args.guidance_scale,
                           num_inference_steps=args.steps,
                           height=args.height, width=args.width,
                           num_images_per_prompt=1, generator=gen)
            out.images[0].save(png)
            rec.update({"runtime_seconds": round(time.perf_counter() - t, 3), "success": True,
                        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1)})
            print(f"[{MODEL_NAME} {i:04d}/{len(todo):04d}] {sid} success "
                  f"runtime={rec['runtime_seconds']}s peak={rec['peak_gpu_memory_mb']}MB", flush=True)
        except Exception as exc:  # noqa: BLE001
            rec.update({"runtime_seconds": round(time.perf_counter() - t, 3), "success": False,
                        "error_message": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"[:4000]})
            print(f"[{MODEL_NAME} {i:04d}/{len(todo):04d}] {sid} FAILED {type(exc).__name__}: {exc}",
                  flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        append_record(args.records_csv, rec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
