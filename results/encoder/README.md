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

Download next session (artifacts are named `enc-<cell name>`). From the cloud container the
artifact and log blob host is blocked (HTTP 403), so `gh run download` fails there; the job logs
are readable through the GitHub MCP tool, and every job prints its results as `ENC>` / `ENC1>`
lines. Elsewhere:

```bash
gh run download <run id> -R nchenevey1/PyDrake-NEAT-pendulum -D results/encoder/runs/<run id>
```

## Run log

| Launch label (tag) | Set | Commit | Run ID | What | Status |
|---|---|---|---|---|---|
| enc-smoke-1 | smoke | 30c150c | 41158557545 | fd 4096 mlp:4, 2 seeds, 3M steps | success (pipeline check) |
| enc-s1-1 | s1 | 0228732 | 41158684527 | stage 1, realistic seed 0 + ideal seed 1 | success; results only in artifacts (logs did not print them) |
| enc-s1b-1 | s1b | 7e4029c | 41158747547 | stage 1, ideal seed 0 | success; results only in artifacts |
| enc-s1v2-1 | s1v2 | a37d422 | 41159760695 | stage 1 again, results printed to logs: realistic seed 0, ideal seeds 0 and 1 | success; seed-0 cells transcribed to `actions/stage1_41159760695.csv` (9 realistic cells cut off by the log tail were rerun locally); seed-1 cells are in the logs/artifacts, not transcribed |
| enc-main-1 | main | bf1f45c | **41161570797** | **stage 2**: 14 cells × 3 seeds, ≤ 4B steps | success (14/14 jobs, 02:47–04:47 UTC); all 42 reports transcribed from the job logs to `actions/stage2_41161570797.jsonl`; weights only in artifacts `enc-<cell>` |

## Local runs (this container, 2 cores)

| What | Command | Output |
|---|---|---|
| Stage 1, ideal setting, seed 0 (stopped after 22 of 64 cells; finished on Actions) | `python experiments/encoder_stage1.py --settings ideal --shard i/2` | `local/stage1_ideal_seed0_part*.jsonl` |
| Stage 1, cart resolution 1 mm / 0.01 mm | `--cells fd:mlp:4 hist3:mlp:4 pos:rnn:4 --cprs 1024 4096 --x-res 0.001 0.00001` | `local/stage1_xres_*.jsonl` |
| Stage 1, 9 realistic cells missing from the transcribed logs | see `local/stage1_real_missing_*.jsonl` | |
| Classic + fd / + model-based α-β observer | `python experiments/encoder_classic.py` | `classic.json`, `local/classic.log` |
| Zero-shot: the 45-param full-state policy through the encoders | inline, see docs | docs/ENCODER_SENSING.md |
| Failure window: full-state policy given encoder velocities in one 0.25 s window | `python experiments/encoder_window.py results/experiments/revcurr_mlp4_s0_v12.json 1024 256` | docs (stage 2) |
| Rerun of hist3 1024 mlp:2 seed 1 (did not reproduce the Actions solve: 0.4 %) | `python experiments/encoder_curriculum.py --mode hist --k 3 --cpr 1024 --arch mlp:2 --seed 1 --budget 4` | `local/repro_hist3_c1024_mlp2_s1.log` |
