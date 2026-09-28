#!/usr/bin/env python3
"""Build runner-compatible sample jsonls for the two official external benchmarks.

A) MICON-Bench (1043 cases / 2518 reference images, 6 tasks, 2-3 refs per case)
   -> experiments/external_ood/micon_bench_v1/samples/micon_bench_1043.jsonl

B) MultiBanana, every query with 2-5 reference images (1390 of 2230)
   -> experiments/external_ood/multibanana_v1/samples/multibanana_k2to5.jsonl
   (the already-generated Main300 stay in the same output dir and are skipped by
    --skip-existing, so they are not regenerated)

Both emit the same schema the existing runners consume:
  sample_id, prompt, instruction, num_references,
  references: [{ref_id, role, crop_path, source_path}]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("MRG_ROUTER_ROOT", ".")).resolve()
MICON = Path(os.environ.get("MICON_BENCH_ROOT", "benchmark/MICON-Bench")).resolve()
MB_DATA = Path(os.environ.get("MULTIBANANA_DATA_ROOT", "benchmark/multibanana/data")).resolve()
MB_EXP = ROOT / "experiments/external_ood/multibanana_v1"


def build_micon(out: Path) -> None:
    import yaml

    rows = []
    data = MICON / "data"
    for task in sorted(p for p in data.iterdir() if p.is_dir()):
        for case_file in sorted((task / "cases").glob("*.yaml")):
            d = yaml.safe_load(case_file.read_text(encoding="utf-8"))
            refs = d.get("refs") or {}
            ordered = sorted(refs.items())          # RA, RB, RC ... = official order
            prompt = (d.get("generation_prompt") or "").strip()
            if not prompt or not ordered:
                continue
            rows.append({
                "sample_id": f"micon_{task.name}_{d.get('case_id')}",
                "task_id": str(d.get("case_id")),
                "task": task.name,
                "source_benchmark": "MICON-Bench",
                "prompt": prompt,
                "instruction": prompt,
                "num_references": len(ordered),
                "references": [
                    {"ref_id": i + 1, "role": role,
                     "role_name": role,
                     "crop_path": str(MICON / rel),
                     "source_path": str(MICON / rel)}
                    for i, (role, rel) in enumerate(ordered)
                ],
                "goal": d.get("goal"),
                "case_yaml": str(case_file),
            })
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    kdist: dict[int, int] = {}
    for r in rows:
        kdist[r["num_references"]] = kdist.get(r["num_references"], 0) + 1
    print(f"MICON-Bench: {len(rows)} cases -> {out}")
    print(f"  K distribution: {dict(sorted(kdist.items()))}")
    print(f"  ref images: {sum(r['num_references'] for r in rows)}")


def build_multibanana(out: Path) -> None:
    cells = []
    for d in sorted(p for p in MB_DATA.iterdir() if p.is_dir()):
        name = d.name
        if name.endswith("_prompt.txt") or not name:
            continue
        # K = leading integer for the multi-ref cells; the category cells are all K=2
        k = int(name.split("_")[0]) if name.split("_")[0].isdigit() else 2
        if 2 <= k <= 5:
            cells.append((k, name, d))

    done_ids = set()
    main = MB_EXP / "manifest/main_300.jsonl"
    if main.is_file():
        for line in main.open(encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                done_ids.add(d.get("query_id") or d.get("sample_id"))

    rows = []
    skipped_other_k = 0
    for k, name, d in cells:
        for pf in sorted(d.glob("*_prompt.txt")):
            tid = pf.name[: -len("_prompt.txt")]
            prompt = pf.read_text(encoding="utf-8").strip()
            if not prompt:
                continue
            refs = sorted(d.glob(f"{tid}_*.jpg")) + sorted(d.glob(f"{tid}_*.png"))
            if not refs:
                continue
            if not (2 <= len(refs) <= 5):      # filter on the REAL reference count
                skipped_other_k += 1
                continue
            sid = f"mb_{len(refs)}_{name.split('_', 1)[1] if '_' in name else name}_{tid}"
            if sid in done_ids:
                continue
            rows.append({
                "sample_id": sid,
                "task_id": tid,
                "cell": name,
                "source_benchmark": "MultiBanana",
                "prompt": prompt,
                "instruction": prompt,
                "num_references": len(refs),
                "references": [
                    {"ref_id": i + 1, "role": f"image{i+1}", "role_name": f"image{i+1}",
                     "crop_path": str(p), "source_path": str(p)}
                    for i, p in enumerate(refs)
                ],
            })
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    kdist: dict[int, int] = {}
    for r in rows:
        kdist[r["num_references"]] = kdist.get(r["num_references"], 0) + 1
    print(f"MultiBanana K=2..5 (Main300 excluded): {len(rows)} queries -> {out}")
    print(f"  K distribution: {dict(sorted(kdist.items()))}")
    print(f"  skipped (refs outside 2-5): {skipped_other_k}")
    print(f"  already-generated Main300 ids excluded: {len(done_ids)}")


def _resolve_micobench_ref(img_root: Path, rel: str) -> Path:
    """Resolve a MICo-Bench input path.

    The official HF release stores the inputs of the single case
    mico_bench_dere/10 as __tmp_input_<n>.png while its annotation refers to
    <n>.png; repair that one packaging slip here (documented in the report).
    """
    p = img_root / rel
    if p.is_file():
        return p
    alt = p.with_name(f"__tmp_input_{p.stem}{p.suffix}")
    return alt if alt.is_file() else p


def build_micobench(out: Path, ann_dir: Path, img_root: Path) -> None:
    rows = []
    tasks = ["object_centric", "human_centric", "hoi", "dere"]
    for task in tasks:
        f = ann_dir / f"mico_bench_{task}.jsonl"
        for line in f.open(encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            cid = d.get("idx", d.get("ed_idx"))
            inputs = d.get("input") or []
            prompt = (d.get("prompt") or "").strip()
            if not inputs or not prompt:
                continue
            rows.append({
                "sample_id": f"micobench_{task}_{cid:04d}",
                "task_id": f"{task}_{cid}",
                "task": task,
                "source_benchmark": "MICo-Bench",
                "editing_type": d.get("editing_type"),
                "prompt": prompt,
                "instruction": prompt,
                "refined_prompt": d.get("refined_prompt"),
                "num_references": len(inputs),
                "reference_output": str(img_root / d["reference"]) if d.get("reference") else None,
                "references": [
                    {"ref_id": i + 1, "role": f"image{i+1}", "role_name": f"image{i+1}",
                     "crop_path": str(_resolve_micobench_ref(img_root, rel)),
                     "source_path": str(_resolve_micobench_ref(img_root, rel))}
                    for i, rel in enumerate(inputs)
                ],
            })
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    kd: dict[int, int] = {}
    for r in rows:
        kd[r["num_references"]] = kd.get(r["num_references"], 0) + 1
    print(f"MICo-Bench: {len(rows)} cases -> {out}")
    print(f"  K distribution: {dict(sorted(kd.items()))}")
    print(f"  K<=5: {sum(v for k,v in kd.items() if k<=5)} | K>5: {sum(v for k,v in kd.items() if k>5)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["micon", "multibanana", "micobench", "both"], default="both")
    ap.add_argument("--micobench-out", type=Path,
                    default=ROOT / "experiments/external_ood/micobench_v1/samples/micobench_897.jsonl")
    ap.add_argument("--micobench-ann", type=Path,
                    default=Path("benchmark/MICo-Bench/annotations"))
    ap.add_argument("--micobench-images", type=Path,
                    default=Path("benchmark/MICo-Bench/images"))
    ap.add_argument("--micon-out", type=Path,
                    default=ROOT / "experiments/external_ood/micon_bench_v1/samples/micon_bench_1043.jsonl")
    ap.add_argument("--mb-out", type=Path,
                    default=MB_EXP / "samples/multibanana_k2to5.jsonl")
    args = ap.parse_args()
    if args.which in ("micon", "both"):
        build_micon(args.micon_out)
    if args.which == "micobench":
        build_micobench(args.micobench_out, args.micobench_ann, args.micobench_images)
    if args.which in ("multibanana", "both"):
        build_multibanana(args.mb_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
