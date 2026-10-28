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
| Evolution + deadline curriculum | 45–89 | 0 % |
| **Evolution + start-state curriculum** (CMA-ES) | 45 | **first successes**; under active development |

The task is feasible; the open problem is finding a *small* network that solves it. The
start-state curriculum is the main line of work. See the [research plan](docs/RESEARCH_PLAN.md).

## Documentation

| Document | Contents |
|---|---|
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

Start-state curriculum experiment (architecture, seed, budget in billions of env steps):

```bash
python experiments/reverse_curriculum.py mlp:4 0 2.0
```

Evaluate, view or export a trained policy:

```bash
python evaluate.py runs/cmaes_mlp4_s0/best_val.json
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
