#!/usr/bin/env python3
"""Qwen-Image-Edit-2511 + Lightning 4-step LoRA — multi-reference latency test.

Official configuration (ModelTC/Qwen-Image-Lightning README, "Run 4-step
Edit-2511 Model"):

    --model_name  Qwen/Qwen-Image-Edit-2511
    --lora_path   Qwen-Image-Edit-2511-Lightning-4steps-V1.0-*.safetensors
    --base_seed 42 --steps 4 --cfg 1.0

i.e. num_inference_steps=4, true CFG off (cfg 1.0), LoRA strength 1.0, bf16.
This script runs the *same* diffusers runner family as the cost_benchmark_v1
Qwen-Image-Edit-2511 40-step entry (QwenImageEditPlusPipeline, fp8 layerwise
transformer storage, cpu offload optional) so the two are directly comparable,
and it sweeps 1..5 reference images to establish the real multi-reference limit.
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

DEFAULT_HF_HOME = Path(os.environ.get("HF_HOME", "models/huggingface"))
DEFAULT_MODEL_PATH = (
    DEFAULT_HF_HOME
    / "models--Qwen--Qwen-Image-Edit-2511"
    / "snapshots"
    / "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9"
)
DEFAULT_LORA = (
    DEFAULT_HF_HOME
    / "models--lightx2v--Qwen-Image-Edit-2511-Lightning"
    / "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
)
DEFAULT_SAMPLES = Path(
    "benchmark/samples_costbench_300.jsonl"
)
MODEL_NAME = "Qwen-Image-Edit-2511+Lightning-4step"


def now() -> float:
    return time.perf_counter()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ordered_ref_paths(sample: dict[str, Any]) -> list[Path]:
    refs = sorted(sample.get("references", []), key=lambda r: r.get("ref_id", 0))
    out = []
    for ref in refs:
        p = ref.get("crop_path") or ref.get("source_path")
        if p and Path(p).is_file():
            out.append(Path(p))
    return out


def load_ref_images(paths: list[Path]) -> list[Image.Image]:
    return [ImageOps.exif_transpose(Image.open(p)).convert("RGB") for p in paths]


def sample_prompt(sample: dict[str, Any]) -> str:
    for key in ("prompt", "instruction", "edit_prompt", "target_prompt", "prompt_original"):
        v = sample.get(key)
        if isinstance(v, str) and v.strip():
            return v
    raise KeyError("prompt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples-jsonl", type=Path, default=DEFAULT_SAMPLES)
    # pick one sample per reference count (its own refs + prompt) for the sweep
    p.add_argument("--ref-counts", default="1,2,3,4,5")
    p.add_argument("--sample-id", default=None, help="override: run only this sample")
    p.add_argument("--mode", choices=["smoke", "latency"], default="smoke")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--true-cfg-scale", type=float, default=1.0)
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--negative-prompt", default=" ")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    p.add_argument("--lora-path", type=Path, default=DEFAULT_LORA)
    p.add_argument("--lora-alpha", type=float, default=1.0)
    p.add_argument("--fuse-lora", action="store_true", help="fuse instead of hot-load adapters")
    p.add_argument("--fp8-transformer-layerwise", action="store_true", default=True)
    p.add_argument("--no-fp8-transformer-layerwise", action="store_false",
                   dest="fp8_transformer_layerwise")
    p.add_argument("--gguf-path", type=Path, default=None,
                   help="GGUF-quantized transformer (e.g. Q4_K_M); replaces the bf16 DiT")
    p.add_argument("--vram-plan", choices=["resident", "model-offload", "sequential-offload"],
                   default="resident",
                   help="resident = whole pipeline on GPU; model-offload = per-component "
                        "CPU offload; sequential-offload = per-submodule CPU offload")
    p.add_argument("--max-vram-gb", type=float, default=None,
                   help="hard cap: torch.cuda.set_per_process_memory_fraction (simulates a smaller card)")
    p.add_argument("--cpu-offload", action="store_true", default=False,
                   help="alias for --vram-plan model-offload (kept for the earlier runs)")
    p.add_argument("--no-cpu-offload", action="store_false", dest="cpu_offload")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--records", type=Path, required=True)
    return p.parse_args()


def build_pipeline(args: argparse.Namespace) -> Any:
    from diffusers import QwenImageEditPlusPipeline

    t_load = now()
    transformer = None
    if args.gguf_path is not None:
        from diffusers import GGUFQuantizationConfig
        from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

        if not Path(args.gguf_path).is_file():
            raise FileNotFoundError(f"missing GGUF transformer: {args.gguf_path}")
        transformer = QwenImageTransformer2DModel.from_single_file(
            str(args.gguf_path),
            # this GGUF carries no diffusers config metadata (only 6 GGUF fields),
            # so without an explicit config diffusers falls back to the SD1.5
            # default config and fails offline.
            config=str(Path(args.model_path) / "transformer"),
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
    kw = dict(torch_dtype=torch.bfloat16, local_files_only=True)
    if transformer is not None:          # passing transformer=None would SKIP loading it
        kw["transformer"] = transformer
    pipe = QwenImageEditPlusPipeline.from_pretrained(str(args.model_path), **kw)
    if not (args.lora_path and Path(args.lora_path).is_file()):
        raise FileNotFoundError(f"missing Lightning LoRA: {args.lora_path}")
    pipe.load_lora_weights(str(args.lora_path), adapter_name="lightning")
    if args.fuse_lora and args.gguf_path is None:
        # fuse into the bf16 base weights FIRST, then cast: the layerwise fp8
        # casting would otherwise turn the LoRA A/B Linears fp8 too, and
        # F.linear on fp8 weights is not implemented on CUDA.
        pipe.fuse_lora(adapter_names=["lightning"], lora_scale=args.lora_alpha)
        pipe.unload_lora_weights()
    else:
        # quantized (GGUF) layers cannot be fused -> keep the adapter hot
        pipe.set_adapters(["lightning"], adapter_weights=[args.lora_alpha])
    # evidence that the Lightning LoRA is really attached (not a base-model run)
    try:
        adapters = pipe.get_active_adapters()
    except Exception:  # noqa: BLE001
        adapters = "?"
    n_lora = sum(1 for n, _ in pipe.transformer.named_modules()
                 if "lora_A" in n or "lora_B" in n or n.endswith(".lora"))
    print(f"[lora] file={Path(args.lora_path).name} alpha={args.lora_alpha} "
          f"active_adapters={adapters} lora_modules={n_lora}", flush=True)
    if args.fp8_transformer_layerwise and args.gguf_path is None:
        kwargs = {}
        if not args.fuse_lora:
            # keep the (bf16) LoRA adapter layers out of the fp8 storage policy
            kwargs["skip_modules_pattern"] = (r".*lora.*", r".*lora_A.*", r".*lora_B.*")
        pipe.transformer.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16, **kwargs,
        )
    plan = "model-offload" if args.cpu_offload else args.vram_plan
    if plan == "model-offload":
        pipe.enable_model_cpu_offload()
    elif plan == "sequential-offload":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to(args.device)
    pipe.set_progress_bar_config(disable=True)
    return pipe, now() - t_load


def instrument(pipe: Any) -> dict[str, float]:
    """Time the coarse phases of one pipe() call (prompt/ref encoding vs latents vs denoise)."""
    acc: dict[str, float] = {}
    for name in ("encode_prompt", "prepare_latents", "_encode_vae_image"):
        fn = getattr(pipe, name, None)
        if fn is None:
            continue

        def wrap(*a, _fn=fn, _n=name, **kw):
            t = now()
            try:
                return _fn(*a, **kw)
            finally:
                acc[_n] = acc.get(_n, 0.0) + (now() - t)

        setattr(pipe, name, wrap)
    return acc


def run_once(pipe: Any, sample: dict[str, Any], args: argparse.Namespace, refs_n: int,
             acc: dict[str, float]) -> tuple[Image.Image, dict[str, Any]]:
    paths = ordered_ref_paths(sample)[:refs_n]
    if not paths:
        raise FileNotFoundError("no readable reference images")
    images = load_ref_images(paths)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    for k in list(acc):
        acc[k] = 0.0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t = now()
    with torch.inference_mode():
        out = pipe(
            image=images,
            prompt=sample_prompt(sample),
            negative_prompt=args.negative_prompt,
            true_cfg_scale=args.true_cfg_scale,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            height=args.height,
            width=args.width,
            num_images_per_prompt=1,
            generator=gen,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total = now() - t
    info = {
        "total_s": round(total, 3),
        "refs": refs_n,
        "ref_paths": [str(p) for p in paths],
        "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1)
        if torch.cuda.is_available() else 0.0,
        "phase_s": {k: round(v, 3) for k, v in acc.items()},
    }
    return out.images[0], info


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.records.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))

    if args.max_vram_gb and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        frac = min(1.0, (args.max_vram_gb * 1e9) / total)
        torch.cuda.set_per_process_memory_fraction(frac, 0)
        print(f"[cap] hard VRAM cap = {args.max_vram_gb} GB "
              f"(fraction {frac:.3f} of {total/1e9:.1f} GB)", flush=True)

    counts = [int(x) for x in args.ref_counts.split(",") if x]
    samples = read_jsonl(args.samples_jsonl)
    if args.sample_id:
        samples = [s for s in samples if s["sample_id"] == args.sample_id]
        cases = {c: samples[0] for c in counts}
    else:
        by_k = {}
        for s in samples:
            by_k.setdefault(int(s.get("num_references", len(s.get("references", [])))), s)
        cases = {}
        for c in counts:
            if c in by_k:
                cases[c] = by_k[c]
            elif c == 1 and 2 in by_k:
                cases[c] = by_k[2]          # 1-ref case derived from a 2-ref sample
            else:
                print(f"[warn] no sample with {c} references in {args.samples_jsonl}", flush=True)

    summary: dict[str, Any] = {
        "model": MODEL_NAME,
        "base_model": str(args.model_path),
        "lora_path": str(args.lora_path),
        "lora_alpha": args.lora_alpha,
        "fuse_lora": args.fuse_lora,
        "steps": args.steps,
        "true_cfg_scale": args.true_cfg_scale,
        "guidance_scale": args.guidance_scale,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "width": args.width,
        "height": args.height,
        "mode": args.mode,
        "fp8_transformer_layerwise": args.fp8_transformer_layerwise and args.gguf_path is None,
        "gguf_path": str(args.gguf_path) if args.gguf_path else None,
        "gguf_bytes": Path(args.gguf_path).stat().st_size if args.gguf_path and Path(args.gguf_path).is_file() else None,
        "vram_plan": "model-offload" if args.cpu_offload else args.vram_plan,
        "max_vram_gb": args.max_vram_gb,
        "cpu_offload": args.cpu_offload,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        if torch.cuda.is_available() else 0,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "runs": [],
    }

    try:
        pipe, load_s = build_pipeline(args)
        summary["load_s"] = round(load_s, 2)
        print(f"[{MODEL_NAME}] loaded in {load_s:.1f}s", flush=True)
    except Exception as exc:  # noqa: BLE001
        summary["load_error"] = f"{type(exc).__name__}: {exc}"
        summary["load_traceback"] = traceback.format_exc()
        (args.records.parent / "lightning_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[{MODEL_NAME}] LOAD FAILED: {type(exc).__name__}: {exc}", flush=True)
        print(traceback.format_exc(), flush=True)
        return 2

    acc = instrument(pipe)
    for c in counts:
        sample = cases.get(c)
        if sample is None:
            continue
        run: dict[str, Any] = {"refs": c, "sample_id": sample["sample_id"]}
        try:
            if args.mode == "smoke":
                img, info = run_once(pipe, sample, args, c, acc)
                png = args.out / f"K{c}_seed{args.seed}.png"
                img.save(png)
                run.update(info, status="success", output_path=str(png))
                print(f"[{MODEL_NAME}] smoke K={c}: OK {info['total_s']}s "
                      f"peak={info['peak_gpu_mb']}MB -> {png}", flush=True)
            else:
                for _ in range(args.warmup):
                    run_once(pipe, sample, args, c, acc)
                reps = []
                for i in range(args.reps):
                    img, info = run_once(pipe, sample, args, c, acc)
                    reps.append(info)
                png = args.out / f"K{c}_seed{args.seed}.png"
                img.save(png)
                tot = [r["total_s"] for r in reps]
                run.update({
                    "status": "success", "output_path": str(png), "reps": reps,
                    "mean_s": round(sum(tot) / len(tot), 3), "min_s": min(tot), "max_s": max(tot),
                    "peak_gpu_mb": max(r["peak_gpu_mb"] for r in reps),
                    "phase_s_mean": {k: round(sum(r["phase_s"].get(k, 0.0) for r in reps) / len(reps), 3)
                                     for k in reps[0]["phase_s"]},
                })
                print(f"[{MODEL_NAME}] latency K={c}: mean={run['mean_s']}s "
                      f"reps={tot} peak={run['peak_gpu_mb']}MB", flush=True)
        except Exception as exc:  # noqa: BLE001
            run.update({"status": "error", "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()})
            print(f"[{MODEL_NAME}] K={c} FAILED: {type(exc).__name__}: {exc}", flush=True)
            print(traceback.format_exc(), flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        summary["runs"].append(run)

    (args.records.parent / "lightning_summary.json").write_text(json.dumps(summary, indent=2))
    with args.records.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["refs", "sample_id", "status", "mean_s", "min_s", "max_s",
                    "peak_gpu_mb", "output_path", "error"])
        for r in summary["runs"]:
            w.writerow([r.get("refs"), r.get("sample_id"), r.get("status"),
                        r.get("mean_s", r.get("total_s")), r.get("min_s", ""), r.get("max_s", ""),
                        r.get("peak_gpu_mb", ""), r.get("output_path", ""),
                        (r.get("error", "") or "")[:200]])
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
