# Benchmark

`queries.json` provides a minimal public preview of the query schema used by
MRG-Router. The larger anonymized benchmark metadata is released under
`data/mr_routebench/`.

Each query contains:

- `sample_id`: stable query identifier.
- `prompt` / `instruction`: text instruction for generation or editing.
- `num_references`: number of reference images.
- `references`: ordered reference descriptors with `ref_id`, role metadata, and
  optional image paths.

Evaluation scripts also support JSONL files where each line is one query object.
