# Scripts

The top-level scripts provide release verification, profile construction,
router training/evaluation, feature extraction, and open-pool generator
onboarding. The `evaluation/` subdirectory contains model runners and benchmark
conversion utilities used by MRG-Router experiments.

The scripts expect model checkpoints, benchmark images, and output directories
to be supplied with command-line arguments or environment variables. This keeps
the public release independent of any private storage layout.

Common variables:

```bash
export MRG_ROUTER_ROOT=/path/to/MRG-Router
export MODEL_ASSET_ROOT=/path/to/model/assets
export HF_HOME=/path/to/huggingface/cache
export MICON_BENCH_ROOT=/path/to/MICON-Bench
export MULTIBANANA_DATA_ROOT=/path/to/multibanana/data
```

Release checks:

```bash
PYTHONPATH=src python scripts/verify_release.py
pytest -q
```
