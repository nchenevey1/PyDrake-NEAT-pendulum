# Task, training score and evaluation

## Task (fixed for every method)

Defined in `pendulum/task.py`.

| | Value |
|---|---|
| Plant | 2 kg cart, two 1 m links with 0.5 kg end masses (`assets/double_pendulum_free.urdf`) |
| Physics | g = 9.81 m/s², no drag, cart force ≤ 100 N |
| Time step | 0.01 s (same discrete integrator as Drake) |
| Start | hanging at rest, each state perturbed uniformly by ±0.05 |
| Swing-up deadline | 2.5 s |
| Upright | both poles within 0.2 rad of vertical |
| Track | \|x\| ≤ 2.8 m |
| Episode | 20 s |
| Observation | 9 values: x, ẋ, sin/cos θ1, ω1, sin/cos θ2, ω2, cos(θ1 − θ2), scaled to ~[−1, 1] |
| Action | one value in [−1, 1], times 100 N |

An episode ends early if the cart leaves the track, or if the poles are not upright at any
step after 2.5 s.

## Success

A start **succeeds** if the episode runs the full 20 s: upright by 2.5 s, upright on every later
step, cart on the track.

## Training score (sparse)

`pendulum/fitness.py`: the longest consecutive upright streak, with a smoothness tie-breaker.

| Name | Formula | Use |
|---|---|---|
| `streak_smooth` | streak time / (1 + mean \|Δaction\| in the streak) | default |
| `legacy` | streak time / (1 + sum \|Δaction\| in the streak) | original; saturates with streak length, so not recommended |
| `streak` | streak time | ablation |

There are no shaping terms. Spinning through the top earns only a few steps, and the 2.5 s
rule ends the episode.

## Evaluation protocol

| Set | Size | Used for |
|---|---|---|
| Training starts | 5–8 new random starts per generation, shared by all candidates | the score |
| Validation | 64 fixed starts | model selection during training |
| Test | 256 fixed starts, never seen in training or selection | all reported numbers |

The final report for a controller (`evaluate.py`) contains:
* test success at 20 s and 60 s;
* success with wide starts (±0.2) and from exact rest;
* success after int8, int6 and int4 quantization;
* size after greedy pruning, and its success;
* parameter count (non-zero weights + biases).

## Classic reference

`python classic_baseline.py` (code: `pendulum/classic.py`):
* swing-up: SNOPT trajectory optimization on the exact dynamics;
* tracking: time-varying LQR;
* balance: a 6-gain LQR.

It uses the same observation plus a step counter.

| Metric | Result |
|---|---|
| Test success, 20 s / 60 s | **100 % / 100 %** |
| Wide starts (±0.2) | 73 % |
| Stored numbers | 3,132 |
| Replayed in Drake | 16/16 succeed |
| Robust to action noise | 100 % with 8 N random noise or a 4 N constant bias |
