# Data Description

## MR-RouteBench

`data/mr_routebench/queries.jsonl.gz` contains one row per query:

```json
{
  "query_id": "...",
  "task_structure": "...",
  "reference_count": 4,
  "instruction": "...",
  "references": [
    {"reference_id": 1, "description": "...", "source_file": "input_00.jpg"}
  ]
}
```

The source images derive from MICo and are not duplicated in this archive.
For feature extraction, add an `image_path` field to each reference after
obtaining the source data under its license.

`data/mr_routebench/split.jsonl` assigns every query to exactly one of:
`probe`, `train`, `validation`, or `test`. The four roles are disjoint.

`data/mr_routebench/outcomes.jsonl.gz` contains 90,000 rows with this schema:

```json
{
  "query_id": "...",
  "generator_id": "...",
  "reference_fidelity_30b": 0.8,
  "reference_fidelity_8b": 0.6,
  "reference_fidelity": 0.75,
  "prompt_following": 0.72,
  "atomic_satisfaction": 0.75,
  "atomic_satisfaction_smoothed": 0.775,
  "semantic_alignment": 0.747,
  "perceptual_quality": 0.69,
  "quality": 0.386
}
```

All factors are normalized to `[0,1]`. The public field names follow the paper:

- `R = reference_fidelity`
- `S = semantic_alignment`
- `V = perceptual_quality`
- `Final-Q = quality = R * S * V`

The auxiliary judge components are retained so the aggregation can be audited.
No generated-image path, server path, API response, or generator credential is
included.

`data/mr_routebench/cost_matrix.jsonl.gz` contains one row for every
query-generator pair. `paper_cost_usd` follows the paper's independent A100
cost audit and declared GPU-hour rate. For self-hosted generators,
`generation_runtime_seconds` additionally retains the source-generation timing;
the API generator has no local runtime. These fields are kept separate because
historical source-generation hardware and the normalized cross-benchmark cost
protocol answer different questions.

## CostBench-300

`data/costbench300/records.csv.gz` contains 2,700 sanitized timing records:

```text
generator_id,query_id,reference_count,runtime_seconds,
model_load_time_seconds,peak_gpu_memory_mb,success
```

CostBench-300 contains 300 queries balanced over 12 task structures. For every
generator, each reference count in `{2,3,4,5}` contributes 75 successful timed
queries. Timing uses serial batch-one inference on one NVIDIA A100-SXM4-40GB.
`runtime_seconds` excludes model loading and external prompt rewriting.

`data/costbench300/summary.csv` provides the mean, median, p90, per-reference-
count medians, and peak memory. `protocol.json` records the benchmark
composition and timing definition. Output paths, local
checkpoint paths, device indices, and machine-specific configuration strings
have been removed from the released records.

## Local Feature Manifest

`scripts/extract_features.py` expects one JSONL row per query with `query_id`,
`instruction`, and ordered `references`. Every reference must include a local
`image_path`. This local file is not part of the code release.

## Validation

Run `python scripts/verify_release.py` to validate split cardinalities, matrix
completeness, Final-Q reconstruction, and CostBench-300 coverage.
