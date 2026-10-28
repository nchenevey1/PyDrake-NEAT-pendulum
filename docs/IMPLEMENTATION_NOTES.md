# Implementation notes

## Code layout

```
pendulum/task.py        task definition (limits, tolerances, start distribution, physics)
pendulum/rollout.py     numba physics + network + episode kernel, parallel over (policy, start)
pendulum/policy.py      GraphPolicy (all methods), MLP/RNN builders, quantization, compaction
pendulum/fitness.py     sparse training scores
pendulum/curriculum.py  physics curriculum (baseline)
pendulum/evaluation.py  real-task evaluation, model selection, greedy pruning
pendulum/trainer.py     training loop: log.jsonl, checkpoint/resume, validation, final report
pendulum/methods/       NEAT, CMA-ES, OpenAI-ES, MAP-Elites, NEAT->CMA
pendulum/classic.py     classic reference controller
pendulum/drake_env.py   PyDrake reference simulator (tests, visual cross-check)
pendulum/export.py      C export of a policy (edge deployment)
experiments/            diagnostic scripts (curricula, distillation, balance-only)
configs/                experiment definitions (see configs/README.md)
legacy/                 original implementation, kept for reference
tests/                  equivalence and determinism tests (pytest)
```

## Methods

All methods use one `ask()/tell()` interface and produce a `GraphPolicy`: a small tanh graph,
optionally with recurrent edges, saved as JSON. Rollout, evaluation and C export are shared.

| Method | What is optimized |
|---|---|
| NEAT | topology + weights (port of the original algorithm) |
| CMA-ES | weights of a fixed network (pycma), local restarts, std capped at ¼ of the weight box |
| OpenAI-ES | weights; antithetic sampling, centered ranks, Adam, L2 |
| MAP-Elites | weights; diversity over peak tip height × mean \|x\| |
| NEAT→CMA | NEAT for N generations, then CMA-ES on the best topology |

All methods keep weights in ±8 and biases in ±4. NEAT already used these limits; they prevent
drift into saturated bang-bang networks and keep weights quantizable.

## Verified properties (`python -m pytest -q tests`)

* The numba dynamics equal Drake's discrete MultibodyPlant: one-step error ~1e-15, and 2000-step
  closed-loop runs agree.
* Compiled NEAT genomes reproduce the original PyTorch `CartNN` outputs exactly.
* The episode kernel matches a plain-Python reference loop.
* Resuming from a checkpoint reproduces an uninterrupted run exactly, for every method.
* The exported C controller matches Python (float32 tolerance).
* The classic controller solves the real task.

## Speed

About 3.2M env steps/s per core, versus about 4k for the original Drake + PyTorch loop (~800×).
A 1200-genome NEAT generation takes ~0.4 s; most of that is NEAT bookkeeping in Python.

## Bugs found in the original code

| Bug | Effect | Fix |
|---|---|---|
| `AdvanceTo(t + DT)` with accumulated float time | Drake sometimes ran two physics updates in one control step and none in the next | advance to the middle of each control period |
| NEAT innovation counter not checkpointed | after resume, crossover mixed unrelated genes | per-run innovation tracker, saved in checkpoints |
| `reset()` ignored its seed | training always used the exact rest state (memorization) | random starts every generation, held-out starts for evaluation |
| best model chosen by training score at any difficulty | the "best" model was usually a low-gravity, high-drag controller | selection on the real task only |
| `tanh(tanh(z))` output | force capped at 76 % of the limit | single tanh (`legacy_double_tanh: true` restores it) |
| fitness divides by the sum of action changes | saturates with streak length: short smooth balancing can outrank long balancing | `streak_smooth` (mean instead of sum) |
| whole first NEAT generation was clones of one genome | wasted first generation | independent initial weights |

## Fixes made during the experiments

| Problem observed | Fix |
|---|---|
| NEAT shrank to ~5 parameters (size penalty on deployable params rewarded silent gene deletion) | original genome-size penalty restored; the variant lives on as `neat_prune` |
| CMA-ES restarts from random weights threw away curriculum progress | local restart from the best evaluated candidate |
| CMA-ES step size diverged under the moving curriculum objective | per-coordinate std capped at ¼ of the weight box |
| Methods got very different sample budgets under generation caps | equal env-step budget for every method |

Old checkpoints can be evaluated with `python evaluate.py path/to/best_model.pth --legacy`
(requires torch).
