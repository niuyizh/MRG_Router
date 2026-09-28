#!/usr/bin/env python3
"""Run the 9 candidate models on one 1k part of mico fourth3k, using 8 GPUs.

The per-model generation recipes below are copied verbatim from
`scripts/evaluation/nightly_next3k_third3k_watchdog.py` (the established,
provenance-clean recipes for next3k/third3k); only the sample manifest is
pointed at
`experiments/mico_fourth3k_highref_generation/samples_mico_fourth3k_highref_3000_part{N}_1000.jsonl`.

Scheduling: the 1k manifest is split into chunks of --chunk samples; chunks are
handed to the 8 GPUs round-robin and each GPU runs its chunk list sequentially in
its own tmux session.  A model must finish before the next model starts.

Usage:
  run_fourth3k_1k_9models_8gpu.py --part 0 [--dry-run] [--models A,B] [--chunk 32]
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("MRG_ROUTER_ROOT", ".")).resolve()
OI = Path(os.environ.get("MODEL_ASSET_ROOT", "models")).resolve()
MB = Path(os.environ.get("MULTIBANANA_ROOT", "benchmark/multibanana")).resolve()
PY = Path(os.environ.get("PYTHON", sys.executable))
FLUX_PY = Path(os.environ.get("FLUX_PYTHON", sys.executable))

SAMPLES_ROOT = ROOT / "experiments/mico_fourth3k_highref_generation"
SEED = 42

SENSENOVA_BASE = "SenseNova-U1.5-8B-MoT"
SENSENOVA_LORA = "SenseNova-U1.5-8B-MoT-LoRA-8step"
SENSENOVA_MODEL_PATH = OI / "model_weights/huggingface/local_dirs/SenseNova-U1.5-8B-MoT"
SENSENOVA_LORA_PATH = OI / "model_weights/huggingface/local_dirs/SenseNova-U1.5-8B-MoT-LoRA-8step"
OMNI_MODEL_PATH = (
    OI / "model_weights/huggingface/hub/models--OmniGen2--OmniGen2/snapshots"
    / "df5dca8a981d74e6c3af214c145f5c735fe72367"
)
DREAM_BASE_MODEL = (
    OI / "model_weights/huggingface/hub/models--black-forest-labs--FLUX.1-Kontext-dev/snapshots"
    / "24e9dedc4ef646698dc8eb4e18ae2cec3c9fea0d"
)
DREAM_VLM = (
    OI / "model_weights/huggingface/models--xiabs--DreamOmni2/snapshots"
    / "38eb68e8f546bec36680180fd922e61b7112a7e1/vlm-model"
)
FLUX4_MODEL = Path(os.environ.get("FLUX2_KLEIN_4B_MODEL_PATH", "models/flux2-klein-4b"))
FLUX9_MODEL_FILE = ROOT / "experiments/flux2_klein_9b_model_path.txt"

MODEL_ORDER = [
    "FLUX2-Klein-4B",
    "FLUX.2-Klein-9B",
    "OmniGen2",
    "Scone",
    "DreamOmni2_official",
    "UNO",
    "SenseNova-U1.5-8B-MoT",
    "SenseNova-U1.5-8B-MoT-LoRA-8step",
    "Qwen-Image-Edit-2511",
]


def common_env(gpu: int) -> dict[str, str]:
    return {
        "HF_HOME": str(OI / "model_weights/huggingface"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "CUDA_VISIBLE_DEVICES": str(gpu),
    }


def build_recipe(model: str, chunk: Path, out_dir: Path, records: Path) -> tuple[dict[str, str], list[str]]:
    """Return (extra env, argv) for one chunk of one model."""
    base = ["--samples-jsonl", str(chunk), "--output-dir", str(out_dir), "--records-csv", str(records)]
    if model == "UNO":
        env = common_env(0) | {"PYTHONPATH": f"{OI}/models/UNO:{OI}"}
        argv = [
            str(OI / "model_envs/uno_eval/bin/python"),
            str(OI / "routing_probe_100/scripts/run_uno_probe.py"), *base,
            "--seed", str(SEED), "--width", "704", "--height", "704",
            "--num-steps", "25", "--guidance", "4.0", "--skip-existing",
        ]
    elif model == "OmniGen2":
        env = common_env(0) | {"PYTHONPATH": f"{OI}/models/OmniGen2:{OI}"}
        argv = [
            str(OI / "model_envs/omnigen2_eval/bin/python"),
            str(OI / "routing_probe_100/scripts/run_omnigen2_probe.py"), *base,
            "--model-path", str(OMNI_MODEL_PATH),
            "--seed", str(SEED), "--width", "1024", "--height", "1024",
            "--num-inference-step", "50", "--text-guidance-scale", "5.0",
            "--image-guidance-scale", "3.0", "--skip-existing",
        ]
    elif model == "DreamOmni2_official":
        env = common_env(0)
        argv = [
            str(OI / "model_envs/dyref_sft/bin/python"),
            str(MB / "experiments/six_model_pilot/scripts/run_dreamomni2_official_multibanana.py"), *base,
            "--base-model-path", str(DREAM_BASE_MODEL),
            "--gen-lora-path", "xiabs/DreamOmni2",
            "--vlm-path", str(DREAM_VLM),
            "--seed", str(SEED), "--width", "1024", "--height", "1024",
            "--num-inference-steps", "30", "--guidance-scale", "3.5", "--skip-existing",
        ]
    elif model == "Qwen-Image-Edit-2511":
        env = common_env(0) | {"PYTHONPATH": str(OI / "models/DyRef/sft")}
        argv = [
            str(OI / "model_envs/dyref_sft/bin/python"),
            str(ROOT / "scripts/evaluation/run_qwen_image_edit_2511_official.py"), *base,
            "--seed", str(SEED), "--width", "896", "--height", "1152",
            "--num-inference-steps", "40", "--cfg-scale", "4.0",
            "--low-vram", "--vram-recipe", "disk_fp8", "--skip-existing",
        ]
    elif model == "FLUX2-Klein-4B":
        env = common_env(0)
        argv = [
            str(FLUX_PY), str(ROOT / "scripts/evaluation/run_flux2_klein_generation.py"), *base,
            "--model-name", "FLUX2-Klein-4B", "--model-path", str(FLUX4_MODEL),
            "--seed", str(SEED), "--width", "1024", "--height", "1024",
            "--num-inference-steps", "4", "--guidance-scale", "1.0", "--dtype", "bf16",
            "--cpu-offload", "--no-warmup-first-sample", "--skip-existing",
        ]
    elif model == "FLUX.2-Klein-9B":
        env = common_env(0)
        model_path = FLUX9_MODEL_FILE.read_text(encoding="utf-8").strip()
        argv = [
            str(FLUX_PY), str(ROOT / "scripts/evaluation/run_flux2_klein_generation.py"), *base,
            "--model-name", "FLUX.2-Klein-9B", "--model-path", model_path,
            "--seed", str(SEED), "--width", "1024", "--height", "1024",
            "--num-inference-steps", "4", "--guidance-scale", "1.0", "--dtype", "bf16",
            "--cpu-offload", "--skip-existing",
        ]
    elif model == "Scone":
        env = common_env(0) | {"PYTHONPATH": str(OI / "models/Scone")}
        argv = [
            str(OI / "model_envs/scone/bin/python"),
            str(ROOT / "scripts/evaluation/run_scone_mico_smoke.py"), *base,
            "--seed", str(SEED), "--all-samples", "--max-samples", "999999",
            "--width", "1024", "--height", "1024", "--num-timesteps", "50",
            "--cfg-text-scale", "4.0", "--cfg-img-scale", "2.0", "--skip-existing",
        ]
    elif model == SENSENOVA_BASE:
        env = common_env(0) | {"PYTHONPATH": str(OI / "models/SenseNova-U1/src")}
        argv = [
            str(PY), str(ROOT / "scripts/evaluation/run_sensenova_u15_mico_pilot.py"), *base,
            "--model-name", SENSENOVA_BASE, "--model-path", str(SENSENOVA_MODEL_PATH),
            "--seed", str(SEED), "--cfg-scale", "4.0", "--img-cfg-scale", "1.0",
            "--cfg-norm", "none", "--timestep-shift", "3.0", "--num-steps", "50",
            "--target-pixels", "4194304", "--attn-backend", "auto", "--vram-mode", "balanced",
            "--no-warmup-first-sample", "--skip-existing",
        ]
    elif model == SENSENOVA_LORA:
        env = common_env(0) | {"PYTHONPATH": str(OI / "models/SenseNova-U1/src")}
        argv = [
            str(PY), str(ROOT / "scripts/evaluation/run_sensenova_u15_mico_pilot.py"), *base,
            "--model-name", SENSENOVA_LORA, "--model-path", str(SENSENOVA_MODEL_PATH),
            "--lora-path", str(SENSENOVA_LORA_PATH),
            "--seed", str(SEED), "--cfg-scale", "1.0", "--img-cfg-scale", "1.0",
            "--cfg-norm", "none", "--timestep-shift", "3.0", "--num-steps", "8",
            "--target-pixels", "4194304", "--attn-backend", "auto", "--vram-mode", "balanced",
            "--no-warmup-first-sample", "--skip-existing",
        ]
    else:
        raise SystemExit(f"unknown model: {model}")
    return env, argv


def tmux(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["tmux", *args], capture_output=True, text=True)


def session_alive(name: str) -> bool:
    return tmux("has-session", "-t", name).returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, choices=[0, 1, 2], required=True)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--models", default=",".join(MODEL_ORDER))
    ap.add_argument("--run-root", type=Path,
                    default=ROOT / "experiments/mico_fourth3k_1k_9models_20260917")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model-timeout-hours", type=float, default=24.0)
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(",") if g != ""]
    models = [m for m in args.models.split(",") if m]
    unknown = [m for m in models if m not in MODEL_ORDER]
    if unknown:
        raise SystemExit(f"unknown models: {unknown}")

    samples = SAMPLES_ROOT / f"samples_mico_fourth3k_highref_3000_part{args.part}_1000.jsonl"
    if not samples.is_file():
        raise SystemExit(f"missing manifest: {samples}")
    lines = [ln for ln in samples.read_text(encoding="utf-8").splitlines() if ln.strip()]

    run_root: Path = args.run_root
    chunks_dir = run_root / f"chunks/part{args.part}"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    (run_root / "logs").mkdir(parents=True, exist_ok=True)
    (run_root / "records").mkdir(parents=True, exist_ok=True)
    (run_root / "workers").mkdir(parents=True, exist_ok=True)

    # 1. chunk the manifest (shared queue for all GPUs)
    chunk_dir = run_root / f"chunks/part{args.part}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    for old in chunk_dir.glob("chunk_*.jsonl"):
        old.unlink()
    chunk_paths: list[Path] = []
    for i in range(0, len(lines), args.chunk):
        cp = chunk_dir / f"chunk_{len(chunk_paths):04d}.jsonl"
        cp.write_text("\n".join(lines[i:i + args.chunk]) + "\n", encoding="utf-8")
        chunk_paths.append(cp)

    print(f"manifest={samples.name} samples={len(lines)} chunks={len(chunk_paths)} "
          f"chunk_size={args.chunk} gpus={gpus} part={args.part} (dynamic queue)")

    status_path = run_root / f"status_part{args.part}.json"
    status: dict = {"part": args.part, "samples": str(samples), "mode": "dynamic_queue", "models": {}}

    def save_status() -> None:
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def merge_prev_records(model: str) -> tuple[list[str], list[dict]]:
        """Union of success rows from previous per-gpu record files for (model, part)."""
        import csv as _csv
        fields: list[str] = []
        rows: dict[str, dict] = {}
        for f in sorted((run_root / "records").glob(
                f"generation_records_{model}_fourth3k_part{args.part}_gpu*.csv")):
            with f.open(encoding="utf-8", newline="") as fh:
                r = _csv.DictReader(fh)
                if not fields and r.fieldnames:
                    fields = list(r.fieldnames)
                for row in r:
                    sid = row.get("sample_id", "")
                    if not sid or row.get("success") != "True":
                        continue
                    if not Path(row.get("output_path", "")).is_file():
                        continue
                    rows[sid] = row
        return fields, list(rows.values())

    # 2. run model by model
    for model in models:
        out_dir = run_root / "generations" / model
        out_dir.mkdir(parents=True, exist_ok=True)
        fields, prev_rows = merge_prev_records(model)

        # per-model queue + claim dir (fresh each run; chunk files are hardlinked in)
        queue = run_root / f"queue/part{args.part}/{model}"
        claims = run_root / f"claims/part{args.part}/{model}"
        for d in (queue, claims):
            if args.dry_run:
                break
            if d.exists():
                for f in d.iterdir():
                    f.unlink()
            d.mkdir(parents=True, exist_ok=True)
        for cp in chunk_paths:
            dst = queue / cp.name
            if not dst.exists():
                os.link(cp, dst)

        sessions = []
        for g in gpus:
            records = run_root / "records" / f"generation_records_{model}_fourth3k_part{args.part}_gpu{g}_seed42.csv"
            # pre-seed this gpu's records file with every sample already done for (model, part),
            # so --skip-existing works no matter which gpu produced it before.
            if fields and not args.dry_run:
                import csv as _csv
                with records.open("w", encoding="utf-8", newline="") as fh:
                    w = _csv.DictWriter(fh, fieldnames=fields, restval="", extrasaction="ignore")
                    w.writeheader()
                    for row in prev_rows:
                        w.writerow({k: row.get(k, "") for k in fields})
            log = run_root / "logs" / f"{model}_part{args.part}_gpu{g}.log"
            worker = run_root / "workers" / f"{model}_part{args.part}_gpu{g}.sh"
            env, _ = build_recipe(model, chunk_paths[0], out_dir, records)
            env = dict(env); env["CUDA_VISIBLE_DEVICES"] = str(g)
            exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
            lines_ = [
                "#!/usr/bin/env bash",
                "set -uo pipefail",
                f"cd {shlex.quote(str(ROOT))}",
                f"QUEUE={shlex.quote(str(queue))}",
                f"CLAIMS={shlex.quote(str(claims))}",
                f"LOG={shlex.quote(str(log))}",
                "mkdir -p \"$CLAIMS\"",
                "while :; do",
                "  f=$(ls \"$QUEUE\"/*.jsonl 2>/dev/null | head -1)",
                "  [ -z \"$f\" ] && break",
                "  if mv \"$f\" \"$CLAIMS\"/ 2>/dev/null; then",
                "    chunk=\"$CLAIMS\"/$(basename \"$f\")",
                "    echo \"[$(date '+%F %T')] chunk $(basename \"$f\")\" >> \"$LOG\"",
            ]
            for chunk in chunk_paths:   # one argv template per chunk is not needed; build inside loop
                break
            # emit the command with $chunk so it works for every claimed chunk
            _, argv = build_recipe(model, Path("/PLACEHOLDER"), out_dir, records)
            argv = ['"$chunk"' if a == "/PLACEHOLDER" else a for a in argv]
            lines_.append(
                f"    env {exports} " + " ".join(shlex.quote(a) if not a.startswith('"$') else a for a in argv)
                + " >> \"$LOG\" 2>&1"
            )
            lines_ += [
                "    echo \"[$(date '+%F %T')] chunk $(basename \"$f\") rc=$?\" >> \"$LOG\"",
                "  fi",
                "done",
                "echo \"[$(date '+%F %T')] worker done\" >> \"$LOG\"",
            ]
            worker.write_text("\n".join(lines_) + "\n", encoding="utf-8")
            worker.chmod(0o755)
            name = f"mr_f4k_{model.replace('.', '').replace('-', '_')}_p{args.part}_g{g}"
            sessions.append((g, name, worker, log))

        status["models"][model] = {"state": "starting", "gpus": len(sessions),
                                   "out_dir": str(out_dir), "started": time.strftime("%F %T")}
        save_status()
        print(f"[{time.strftime('%F %T')}] === {model}: launching {len(sessions)} GPU workers (queue) ===")
        if args.dry_run:
            for g, name, worker, log in sessions:
                print(f"  (dry-run) {name}: {worker}  log={log}")
            status["models"][model]["state"] = "dry_run"
            save_status()
            continue

        for g, name, worker, log in sessions:
            tmux("kill-session", "-t", name)
            r = tmux("new-session", "-d", "-s", name, f"bash {worker}")
            if r.returncode != 0:
                print(f"  !! failed to start {name}: {r.stderr.strip()}")
        status["models"][model]["state"] = "running"
        save_status()

        deadline = time.time() + args.model_timeout_hours * 3600
        while True:
            alive = [n for _, n, _, _ in sessions if session_alive(n)]
            if not alive:
                break
            if time.time() > deadline:
                print(f"  !! {model}: timeout after {args.model_timeout_hours} h; killing {alive}")
                for n in alive:
                    tmux("kill-session", "-t", n)
                status["models"][model]["state"] = "timeout"
                save_status()
                break
            time.sleep(60)
        if status["models"][model]["state"] == "running":
            status["models"][model]["state"] = "done"
            status["models"][model]["finished"] = time.strftime("%F %T")
            save_status()
        print(f"[{time.strftime('%F %T')}] === {model}: {status['models'][model]['state']} ===")

    print(f"status: {status_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
