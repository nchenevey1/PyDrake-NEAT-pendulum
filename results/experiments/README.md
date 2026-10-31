# Diagnostic experiments (review, 2026-10-28)

Run from the repository root, e.g. `python experiments/reverse_curriculum.py mlp:4 0 2.0`.

| script | question | result |
|---|---|---|
| `experiments/balance_only.py linear 0.05` | Is balancing (start near upright, target physics) hard for ES? | No: 10-param linear policy reaches 100 % test success in 50 CMA-ES generations (3M steps, ~1 s); mlp:4 99 % in 350 gens. |
| `distill.py --mode dagger/dart` | Can small reactive MLPs imitate the classic expert? | No (0 % val, 8-64 hidden). DAgger labels from a time-indexed expert are inconsistent for a reactive student; DART/BC underfit the short aggressive swing-up. |
| `experiments/clock_distill.py H clock/reactive` | Same with a clock input and phase-balanced data | No (0 % val, 16/32 hidden; never reaches upright). |
| `experiments/clock_dagger.py H` | DAgger with clock input | Reaches upright (~1.6-2.2 s) but never holds; degrades with more iterations: TVLQR is only locally valid, so off-tube expert labels are poor. Needs a replanning (MPC) expert. |
| `experiments/reverse_curriculum.py mlp:4 0 2.0` | Start-state (reverse) curriculum along the expert's swing-up at target physics | mlp:4 mastered starts from 2.1 s down to 0.45 s into the swing-up with 100 % full success per level; first non-zero success of any neural method on the real task (TEST 0.4 %, 1/256) within a 2B-step budget. |

| `experiments/deadline_curriculum.py mlp:4 0 2.0` | Deadline curriculum: start from rest, swing-up deadline 10 s → 2.5 s, target physics | Hard to enter, fast once in: mlp:4 never passed the 10 s level (2B steps); mlp:8 passed it after ~2,800 gens (1.2B steps), then reached 2.6 s in ~800 gens and stalled 0.1 s short of the task. |
| `experiments/reverse_curriculum.py mlp:4 1 2.0`, `rnn:4 0 2.0` | Does the start-state curriculum reproduce? | Yes: seed 1 reached starts 0.60 s into the swing-up and **9.0 % test success** (seed 0: 0.4 %); the RNN reached 1.30 s with no real-task success in 2B steps. |
| `experiments/reverse_curriculum.py mlp:4 0 2.0 12` (4th arg = velocity observation scale) | Was the legacy velocity scaling (/50, /90) blocking learning? | Yes. With scale 12 every run got further back; mlp:4 seed 0 solved the real task: **100 % test** (60 s 100 %, Drake 16/16, int8 91 %, 43 params after pruning). Seeds 1, 2 and mlp:8 stalled at 0.55 s (1.2 %, 0 %, 1.6 %). |
| `experiments/diagnose.py [policy.json]` | Where and why does a learned policy fail? | Gain ceiling, failure phase, start-noise funnel, chaos, LQR-catch, score noise, weight-noise sensitivity. See [docs/DIAGNOSIS.md](../../docs/DIAGNOSIS.md). |

Earlier logs are in this folder; logs written after the switch to `.gitignore` are kept locally only.
Design conclusions: [docs/CURRICULUM.md](../../docs/CURRICULUM.md).

## Reliability of the start-state curriculum (S2 + adaptive step), launched via GitHub Actions

Code: `configs/cmaes_mlp4_revcurr.yaml`, `configs/cmaes_mlp8_revcurr.yaml` (trainer mode
`curriculum.mode: start_state`, adaptive back-off step 5 → 2 → 1, velocity scale 12,
256 validation starts, stop at ≥ 99 % validation). Plans: `sweeps/reliability-mlp4.yaml`,
`sweeps/reliability-mlp8.yaml` (seeds 0–4, 4B env steps each). Goal: ≥ 90 % test in ≥ 3 of 5 seeds.

| Tag | Plan | Commit | Actions run | Status |
|---|---|---|---|---|
| `sweep-smoke-1` | smoke (2 seeds, 20M steps) | branch `claude/charming-noether-2plokb` (code at `f218abf`) | – | **not launched**: the session could not push tags or dispatch workflows |
| `sweep-reliability-mlp4-1` | reliability-mlp4 | same | – | not launched |
| `sweep-reliability-mlp8-1` | reliability-mlp8 | same | – | not launched |

Local smoke tests only (20–30M steps, 2 seeds; no results, pipeline check): curriculum moves
from 2.10 s to 1.75–1.90 s, report/verify files written. When the runs finish, add one row per
seed (level reached, test 20 s / 60 s, Drake, int8, weight noise σ = 0.01) and keep the rows above.
