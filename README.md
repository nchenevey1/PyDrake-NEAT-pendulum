# Minimal neural controllers for cart + double pendulum swing-up

**Question:** What is the most parameter-efficient neural controller that reliably swings up
and balances a cart double pendulum, and which sparse training signal finds it best?

The controller must swing both poles up from hanging rest within 2.5 s and keep them upright
for the rest of a 20 s episode, under full gravity, no drag and a 100 N force limit. It should
be small enough for a microcontroller.

Inspired by [johnBuffer/Pendulum-NEAT](https://github.com/johnBuffer/Pendulum-NEAT/tree/main).

## Status

| Approach | Params | Test success (real task) |
|---|---|---|
| Classic: trajectory optimization + LQR (reference) | 3,132 stored numbers | **100 %** |
| Evolution + original physics curriculum (NEAT, CMA-ES, OpenAI-ES, MAP-Elites, hybrid) | 10–89 | 0 % |
| Imitation of the classic controller (BC, DART, DAgger) | 89–5,000 | 0 % |
| Evolution + deadline curriculum | 45–89 | 0 % (89 params got to a 2.6 s deadline, 0.1 s short) |
| Evolution + start-state curriculum (CMA-ES), original observation scales | 45 | up to 9 % after 2B steps (2 seeds: 9 %, 0.4 %) |
| **Same, velocity observations rescaled to the real range** | **45** | **100 %** in 1 of 4 runs (1.1B steps, 9 min); others 0–1.6 % |

A 45-parameter network can solve the task (100 % test, 60 s, Drake-verified, 43 params after
pruning). The open problem is finding it *reliably*. Why earlier runs failed and what changed:
[docs/DIAGNOSIS.md](docs/DIAGNOSIS.md). Plan: [research plan](docs/RESEARCH_PLAN.md).

## Documentation

| Document | Contents |
|---|---|
| [docs/DIAGNOSIS.md](docs/DIAGNOSIS.md) | goal and research relevance, why learning failed (measured), what to change |
| [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md) | question, findings so far, experiment matrix, success criteria, next steps |
| [docs/CURRICULUM.md](docs/CURRICULUM.md) | curriculum and advancement design, swing-up deadline decisions |
| [docs/TASK_AND_EVALUATION.md](docs/TASK_AND_EVALUATION.md) | task constants, sparse score, evaluation protocol, classic reference |
| [docs/IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md) | code layout, methods, tests, bugs fixed |
| [results/experiments/README.md](results/experiments/README.md) | diagnostic experiments and their results |
| [configs/README.md](configs/README.md) | experiment configurations |

## Quick start

```bash
pip install -r requirements.txt     # numpy, numba, cma, pyyaml, matplotlib (+ drake for the classic reference)
python -m pytest -q tests           # correctness checks
```

Train one configuration, or a whole resumable sweep, then compare:

```bash
python run.py configs/cmaes_mlp4.yaml --seed 0
```

```bash
python sweep.py --jobs 4 --seeds 0 1 -- configs/*.yaml
```

```bash
python compare.py
```

Start-state curriculum (S2, adaptive step, resumable, writes `report.json`):

```bash
python run.py configs/cmaes_mlp4_revcurr.yaml --seed 0
```

Run a sweep plan on GitHub Actions (`sweeps/<plan>.yaml`; see
[IMPLEMENTATION_NOTES](docs/IMPLEMENTATION_NOTES.md#running-sweeps-on-github-actions)):

```bash
git tag sweep-reliability-mlp4-1 && git push origin sweep-reliability-mlp4-1
```

Evaluate, view or export a trained policy:

```bash
python evaluate.py runs/cmaes_mlp4_s0/best_val.json --drake 16
```

```bash
python visualize.py runs/cmaes_mlp4_s0/best_val.json --out swingup.gif
```

```bash
python -m pendulum.export runs/cmaes_mlp4_s0/best_val.json policy.c
```

Classic reference controller:

```bash
python classic_baseline.py
```
