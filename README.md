# Minimal neural controllers for cart + double pendulum swing-up

Project inspired by: https://github.com/johnBuffer/Pendulum-NEAT/tree/main

Goal: find the **smallest** neural controller that reliably swings a cart-mounted double
pendulum up from rest and keeps it balanced under full gravity and no drag, suitable for
edge deployment. Several evolutionary methods (custom NEAT, CMA-ES, OpenAI-ES, MAP-Elites,
NEAT→CMA hybrid) share one fast simulator, one sparse training score, one physics curriculum
and one evaluation protocol on the true objective, so that they can be compared on model
size, sample efficiency, wall-clock time and success rate.

#### Initial learned swing up under low gravity and high drag (legacy run):
https://github.com/user-attachments/assets/5bb90e0b-ec8d-48d6-ac88-f8fd59f891be

## Quick start

```bash
pip install -r requirements.txt          # numpy, numba, cma, pyyaml, matplotlib (+ drake, optional)
python -m pytest -q tests                # simulator == Drake, NEAT == legacy CartNN, resume is exact
python run.py configs/cmaes_mlp4.yaml --seed 0          # one run -> runs/cmaes_mlp4_s0/
python sweep.py --jobs 4 --seeds 0 1 -- configs/*.yaml  # all methods, resumable
python compare.py                                       # results/comparison.md + learning_curves.png
python evaluate.py runs/cmaes_mlp4_s0/best_val.json     # test-set report for one policy
python visualize.py runs/cmaes_mlp4_s0/best_val.json --out swingup.gif [--drake]
```

`run.py ... --resume` continues an interrupted run bit-exactly (and can extend the budget with
`--set budget.generations=N`). Any config value can be overridden with `--set key.sub=value`.

## The problem (fixed for every method)

Defined in `pendulum/task.py`; values are those of the original `config.py`.

* **Plant**: `assets/double_pendulum_free.urdf`: 2 kg cart on a track, two 1 m links with
  0.5 kg point masses at their ends. Discrete dynamics with `dt = 0.01 s` (semi-implicit Euler,
  as Drake's discrete `MultibodyPlant`).
* **Action**: one scalar in [-1, 1] times `max_force` on the cart.
* **Observation** (9-D, unchanged): `[x/2.8, xd/50, sin th1, cos th1, w1/90, sin th2, cos th2, w2/90, cos(th1 - th2)]`
  (velocities clipped to [-1, 1]; `th2` is the *relative* elbow angle, so the last feature is
  not the pole alignment the old README described, but it is kept for compatibility).
* **Episode**: starts near hanging rest (each state uniformly perturbed by ±0.05), lasts 20 s. It
  ends early if the cart leaves `|x| <= 2.8 m`, or if after `2.5 s` the poles are not upright
  (`|th1 - pi| < 0.2` and `|th2| < 0.2`).
* **Target physics**: `g = 9.81`, `drag = 0`, `max_force = 100 N`. This is also the last point of the curriculum.

### True objective and success metric

A controller **succeeds** on an initial condition iff the episode runs the full 20 s under the
target physics: upright within 2.5 s, then upright on every step, cart on the track.
Reported metrics (`pendulum/evaluation.py`):

* `success_rate` on **256 held-out test initial conditions** (separate from the 64 validation ICs
  used for model selection during training, and from the per-generation training ICs);
* the same with a **60 s** horizon (is the balance actually sustained?), with **wider ICs**
  (±0.2) and from the nominal rest state;
* after **int8 / int6 / int4** weight quantization, and after **greedy magnitude-free pruning**
  (connections are removed one at a time while validation success stays within 2 %).
* model size = deployable parameters (non-zero weights + biases of non-input nodes).

### Training signal (sparse)

The score is driven by the longest *consecutive* upright streak (`pendulum/fitness.py`), so there is
no dense shaping. Spinning through the top earns only a few steps, and because the episode
ends after 2.5 s without balance, spinning cannot accumulate reward.

* `streak_smooth` (default): `T_streak / (1 + mean |Δa| in the streak)`;
* `legacy`: `T_streak / (1 + Σ |Δa| in the streak)` (the original formula);
* `streak`: `T_streak`.

The original divides by the *sum* of action changes, which grows with the streak length: for a
controller with per-step action change `r`, the score is `T / (1 + rT/dt)`, which saturates at
`dt/r`. It can rank 0.3 s of smooth balancing (0.23) above 17.5 s of slightly jittery balancing
(0.20). `streak_smooth` keeps the smoothness tie-breaker but bounds the divisor to [1, 3], so a
longer streak always wins. `configs/neat_fitlegacy.yaml` is the ablation that checks this choice
against the true objective. Selection fitness is `score / (1 + 0.005 · params)`.

Each candidate is scored on 5 random ICs per generation; all candidates in a generation get the same
ICs (common random numbers), and the ICs change every generation.

### Physics curriculum

`pendulum/curriculum.py` uses the same schedule and advancement rule as the original: gravity
1 → 9.81 at drag 3 and 1000 N, then drag 3 → 0 and force 1000 → 100 N. Difficulty grows by 0.005
whenever the best candidate is upright ≥ 20 % of the episode. Difficulty 1 is exactly the target
physics, so the final stage of training has no domain shift. Model selection only ever uses the
target physics. `*_nocurr` configs are ablations that train directly on the target.

## Methods (`pendulum/methods/`)

All methods implement the same `ask()/tell()` interface and produce `GraphPolicy` objects
(`pendulum/policy.py`): small tanh graphs with optional recurrent (one-step-delayed) edges,
serialized as JSON, so the same rollout kernel, evaluation and export apply to all of them.

| method | file | what is optimized |
|---|---|---|
| custom NEAT | `neat.py` | topology + weights (port of the original algorithm, see below) |
| CMA-ES | `es.py` | weights of a fixed MLP / RNN (`linear`, `mlp:4`, `mlp:8`, `rnn:4`, …) via pycma, with restarts |
| OpenAI-ES | `es.py` | same, antithetic sampling + centered ranks + Adam + L2 |
| MAP-Elites | `qd.py` | quality-diversity over fixed-architecture weights; descriptor = peak tip height × mean \|x\| |
| NEAT→CMA | `hybrid.py` | NEAT architecture search for N generations, then CMA-ES on the best topology's weights |

The NEAT port keeps the original hyper-parameters and mechanics: dynamic speciation threshold
towards 100 species, fitness sharing, per-species elitism, tournament selection, crossover,
species stagnation, sigma shock, and recurrent connections. It adds:

* per-run innovation tracking (fixes a resume bug, see below);
* independent initial weights;
* an optional connection-deletion mutation as pressure toward small networks;
* pruning of nodes that cannot reach the output before evaluation and size accounting.

## Code layout

```
pendulum/task.py        problem definition (limits, tolerances, IC distribution, target physics)
pendulum/rollout.py     numba physics + network + episode kernel, parallel over (policy, IC)
pendulum/policy.py      GraphPolicy, MLP/RNN builders, packing, quantization, compaction
pendulum/fitness.py     sparse training scores
pendulum/curriculum.py  physics curriculum
pendulum/evaluation.py  true-objective evaluation, model selection key, greedy pruning
pendulum/trainer.py     generic loop: logging (log.jsonl), checkpoint/resume, validation, final report
pendulum/methods/       NEAT, CMA-ES, OpenAI-ES, MAP-Elites, NEAT->CMA
pendulum/drake_env.py   PyDrake reference simulator (tests, visualization cross-check)
configs/                experiment definitions (see configs/README.md)
legacy/                 original implementation, kept for reference
tests/                  equivalence and determinism tests
```

## Changes relative to the original code, and bugs found

* **Speed: ~800× per core.** The legacy loop ran about 4k env steps/s per core (Python Drake
  `LeafSystem` + per-node PyTorch ops). The numba kernel (closed-form dynamics, network and
  episode logic in one compiled loop, parallel over episodes) runs about 3.2M steps/s per core. A
  1200-genome × 5-IC NEAT generation takes about 0.4 s, most of it Python NEAT bookkeeping.
  Species assignment is also compiled.
* **Dynamics are identical to Drake.** One-step agreement is about 1e-15, and 2000-step
  closed-loop trajectories agree (`tests/test_dynamics.py`). The URDF's ±3 m slider limit is not
  modeled; episodes end at |x| > 2.8 m before it can act.
* **Bug: legacy Drake timing.** `AdvanceTo(t + DT)` accumulated floating-point time, so Drake
  sometimes did two physics updates in one control step and none in the next (for example steps
  5 and 6). `pendulum/drake_env.py` advances to the middle of each control period instead.
* **Bug: NEAT innovation numbers restarted on resume.** The counter was a class attribute that
  was not checkpointed, so after `--resume` new genes reused existing innovation numbers and
  crossover mixed unrelated connections.
* **Training used a single deterministic initial state.** `reset()` ignored its seed and always
  started at exact rest, so controllers could memorize one trajectory. Training now uses 5 random
  ICs per generation (the ±0.05 perturbation from the commented-out code), and evaluation uses
  held-out ICs.
* **Model selection under domain shift.** `best_model.pth` was the highest training score at any
  curriculum difficulty, so it was usually a low-gravity, high-drag controller. Selection now uses
  success on held-out ICs under the target physics.
* **Double tanh on the output** (`tanh(tanh(z))`) capped the force at 76 % of `max_force`. New
  policies use a single tanh. `legacy_double_tanh: true` / `from_legacy_genome(...).compile(double_tanh=True)`
  reproduce the old network exactly (`tests/test_neat.py`).
* **Fitness saturation**: see *Training signal* above.
* The whole initial NEAT population was 1200 clones of one genome; each now gets its own weights.

Old checkpoints can be evaluated on the true objective with
`python evaluate.py path/to/best_model.pth --legacy` (requires torch).

## Results

See `results/comparison.md` (generated by `compare.py`) and the section below.
