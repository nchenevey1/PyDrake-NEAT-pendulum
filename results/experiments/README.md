# Diagnostic experiments (review, 2026-10-28)

Run from the repository root, e.g. `python experiments/reverse_curriculum.py mlp:4 0 2.0`.

| script | question | result |
|---|---|---|
| `experiments/balance_only.py linear 0.05` | Is balancing (start near upright, target physics) hard for ES? | No: 10-param linear policy reaches 100 % test success in 50 CMA-ES generations (3M steps, ~1 s); mlp:4 99 % in 350 gens. |
| `distill.py --mode dagger/dart` | Can small reactive MLPs imitate the classic expert? | No (0 % val, 8-64 hidden). DAgger labels from a time-indexed expert are inconsistent for a reactive student; DART/BC underfit the short aggressive swing-up. |
| `experiments/clock_distill.py H clock/reactive` | Same with a clock input and phase-balanced data | No (0 % val, 16/32 hidden; never reaches upright). |
| `experiments/clock_dagger.py H` | DAgger with clock input | Reaches upright (~1.6-2.2 s) but never holds; degrades with more iterations: TVLQR is only locally valid, so off-tube expert labels are poor. Needs a replanning (MPC) expert. |
| `experiments/reverse_curriculum.py mlp:4 0 2.0` | Start-state (reverse) curriculum along the expert's swing-up at target physics | mlp:4 mastered starts from 2.1 s down to 0.45 s into the swing-up with 100 % full success per level; first non-zero success of any neural method on the real task (TEST 0.4 %, 1/256) within a 2B-step budget. |

| `experiments/deadline_curriculum.py mlp:4 0 2.0` | Deadline curriculum: start from rest, swing-up deadline 10 s → 2.5 s, target physics | No: zero full successes even at the 10 s deadline after 4,000 generations (mlp:4) and 1,000 (mlp:8). |
| `experiments/reverse_curriculum.py mlp:4 1 2.0`, `rnn:4 0 2.0` | Does the start-state curriculum reproduce? | Yes: seed 1 reached starts 0.60 s into the swing-up and **9.0 % test success** (seed 0: 0.4 %); the RNN also advances. |

Earlier logs are in this folder; logs written after the switch to `.gitignore` are kept locally only.
Design conclusions: [docs/CURRICULUM.md](../../docs/CURRICULUM.md).
