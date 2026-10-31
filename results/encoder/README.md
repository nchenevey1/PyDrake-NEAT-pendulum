# Encoder study: runs and results

Branch `claude/quantized-encoder-swing-up-r876l1` (assigned by the session instead of
`research/encoder-sensing`; branched off `claude/modest-carson-qx8ce5` at `ceac85d`).
Design and analysis: [docs/ENCODER_SENSING.md](../../docs/ENCODER_SENSING.md).
Tables: [summary.md](summary.md) (regenerate with `python experiments/encoder_report.py`).

## How runs are launched

Workflow `.github/workflows/encoder.yml`, cells in `experiments/encoder_stage2.json`.
Triggers: tag `enc-<set>-*`, `workflow_dispatch` (input `set`), or a push to this branch that
changes `experiments/encoder_launch.txt` (first word = set). **Tag pushes are rejected (HTTP 403)
from the cloud session**, so every launch below used the launch file; the "tag" column is the
launch label written in that file.

Download next session (artifacts are named `enc-<cell name>`):

```bash
gh run download <run id> -R nchenevey1/PyDrake-NEAT-pendulum -D results/encoder/runs/<run id>
```

## Run log

| Launch label (tag) | Set | Commit | Run ID | What | Status |
|---|---|---|---|---|---|
| enc-smoke-1 | smoke | 30c150c | 41158557545 | fd 4096 mlp:4, 2 seeds, 3M steps | success (pipeline check) |
| enc-s1-1 | s1 | 0228732 | 41158684527 | stage 1, realistic setting seed 0 + ideal setting seed 1 (8 shards) | launched |

## Local runs (this container, 2 cores)

| What | Command | Output |
|---|---|---|
| Stage 1, ideal setting, seed 0 | `python experiments/encoder_stage1.py --settings ideal --shard i/2` | `stage1_s0.jsonl`, `stage1_s1.jsonl` |
| Classic + fd / + model-based α-β observer | `python experiments/encoder_classic.py` | `classic.json` |
| Zero-shot: the 45-param full-state policy through the encoders | inline, see docs | docs/ENCODER_SENSING.md |
