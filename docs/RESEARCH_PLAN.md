# Research plan

## Question

**What is the most parameter-efficient neural controller that reliably swings up and balances
the cart double pendulum under the real task constraints, and which sparse training signal
finds it best?**

| Term | Meaning here |
|---|---|
| Reliable | ≥ 90 % success on 256 held-out test starts (20 s), reported also at 60 s and with wider starts |
| Parameter-efficient ("dense") | Fewest deployable parameters (and bits) for a given reliability, i.e. the size × success Pareto front |
| Sparse training signal | The episode score depends only on the longest upright streak; no shaping terms. What varies is *where and when episodes start* (the curriculum) and the advancement rule |
| Finds it best | Highest test success per training env step, per wall-clock hour, and smallest final size |

The task itself is fixed (see [TASK_AND_EVALUATION.md](TASK_AND_EVALUATION.md)). A classic
model-based controller solves it at 100 %, so the task is feasible as defined.

## What we know so far

| Finding | Evidence |
|---|---|
| The task is feasible | Classic trajopt + TVLQR + LQR: 100 % test success (20 s and 60 s) |
| Balance alone is easy | CMA-ES, start near upright: 10-param linear policy 100 % in 50 generations (~1 s) |
| The physics curriculum fails | 6 methods × 5B steps (NEAT: 4 h wall cap at 2.9B): 0 % test success; best policies never balance even at their own training physics |
| Imitating the classic controller fails | BC, DART, DAgger, with and without a clock input, 8–64 hidden units (89–5,000 params): 0 % |
| A deadline-only curriculum fails | From rest, deadline 10 s → 2.5 s: no full success even at 10 s after 3,000+ generations |
| **A start-state curriculum works** | Real physics, starts moved backwards along a reference swing-up: 100 % mastery at each level; first non-zero success of any neural method on the real task (45 params: 1/256 test after 2B steps); the curriculum progress reproduces with a second seed and with a recurrent network |

Details and reasoning: [CURRICULUM.md](CURRICULUM.md). Raw logs and scripts:
[`results/experiments/README.md`](../results/experiments/README.md).

## Experiment matrix

Every cell uses the same task, sparse score, budget (env steps) and evaluation.

**Training signals (curricula):**

| ID | Signal | Status |
|---|---|---|
| S0 | None: train on the real task from rest | baseline (configs `*_nocurr`) |
| S1 | Physics curriculum (original) | baseline, done: 0 % |
| S2 | Start-state curriculum along a reference swing-up | **main candidate** |
| S3 | Start-state curriculum without a reference (starts from backward random walks from upright) | to build: tests whether a demonstration is needed |
| S4 | Deadline curriculum (from rest, deadline shrinks) | tested: fails, kept as a negative result |

**Policies (size ladder):** linear (10 params) · mlp:2 (23) · mlp:4 (45) · mlp:8 (89) · rnn:4 (61) · NEAT (evolved topology)

**Optimizers:** CMA-ES (fixed topology), NEAT (topology + weights). OpenAI-ES and MAP-Elites
are not carried forward: they underperformed CMA-ES on the same signal.

**Seeds:** 3 per cell. **Budget:** 10B training env steps per run.

## Finding the smallest controller

1. **Competence first.** No size penalty while searching (it collapsed NEAT to ~5 params before
   it could do anything). Size enters only as a tie-breaker in model selection.
2. **Then compress** each successful controller:
   - prune connections while validation success stays within 2 %;
   - fine-tune with CMA-ES on the real task, from rest, with the same sparse score;
   - quantize (int8, int4) and fine-tune on the quantized grid.
3. **Report the Pareto front:** parameters (and bits) vs test success, with the classic
   controller (3,132 stored numbers) as a reference point.

## Success criteria

* Primary: ≥ 90 % test success (20 s) with ≤ 100 parameters.
* Secondary: ≥ 80 % at 60 s; int8 within 5 % of float; robustness to ±10 % mass error.
* Research result: a ranking of S0–S4 by success per env step, with 3 seeds and a statement of
  which signals fail and why.

## Next steps

- [ ] Add the start-state curriculum (S2) and the mastery advancement rule to `pendulum/curriculum.py` as a config mode
- [ ] Implement S3 (no reference trajectory)
- [ ] Run the matrix (S0, S2, S3 × size ladder × CMA-ES/NEAT × 3 seeds)
- [ ] Compression pipeline (prune → fine-tune → quantize) and Pareto plot in `compare.py`
- [ ] Robustness evaluation: mass ±10 %, sensor noise, one-step actuation delay
- [ ] Optional: formal verification of the balance region for the smallest networks
