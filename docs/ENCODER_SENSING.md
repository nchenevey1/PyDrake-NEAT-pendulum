# Encoder-only sensing: how small can the controller be?

**Question.** When the controller sees only quantized encoder positions (no velocity sensors),
plus realistic actuation delay, how many parameters and bits does it need to swing up and
balance? Which way of getting velocity information works best per parameter?

Branch: the brief asked for `research/encoder-sensing` off `claude/modest-carson-qx8ce5`; the
session assigned `claude/quantized-encoder-swing-up-r876l1`, so the work lives there (branched
off `claude/modest-carson-qx8ce5` at `ceac85d`). Velocity observation scale **12** throughout.

## Sensor model (`pendulum/sensors.py`, off by default)

| Channel | Model |
|---|---|
| Pole angles θ1 (shoulder, absolute) and θ2 (elbow, relative) | incremental encoders, `CPR` ∈ {256, 1024, 4096, 16384} counts/rev, reading = floor(θ / q)·q, q = 2π / CPR, zeroed at hanging rest; no wrap (counts accumulate) |
| Cart position | linear encoder, resolution ∈ {1 mm, 0.1 mm, 0.01 mm}, floor |
| Actuation delay (option) | the force applied in step t is the command computed in step t − 1 |
| Force command (option) | 8 bits: action rounded to k/127, k ∈ [−127, 127] |

Nothing in the shared core changes: the sensed rollout is a separate numba kernel that reuses
`rollout.physics_step` and `rollout._forward` and returns the same statistics.

## Velocity resolution of finite differencing (written before any run)

Finite differencing over one control step (dt = 0.01 s) gives a velocity resolution of
q / dt. With floor quantization the error of a one-step difference is the difference of two
uniform errors: triangular, standard deviation q / (dt·√6).

| Pole CPR | q (rad) | velocity step q/dt (rad/s) | fd noise sd (rad/s) | step in obs units (÷12) | levels across the pumping phase (0–0.5 s, ≤ 2.9 rad/s) | levels to the swing-up peak (8.9 rad/s) |
|---|---|---|---|---|---|---|
| 256 | 0.0245 | **2.45** | 1.00 | 0.20 | **1.2** | 3.6 |
| 1024 | 0.0061 | 0.61 | 0.25 | 0.051 | 4.7 | 15 |
| 4096 | 0.0015 | 0.15 | 0.063 | 0.013 | 19 | 58 |
| 16384 | 0.00038 | 0.038 | 0.016 | 0.0032 | 75 | 233 |

| Cart resolution | velocity step (m/s) | fd noise sd (m/s) | levels across the pumping phase (≤ 2.2 m/s) | to peak (10 m/s) |
|---|---|---|---|---|
| 1 mm | 0.1 | 0.041 | 22 | 100 |
| 0.1 mm | 0.01 | 0.0041 | 220 | 1,000 |
| 0.01 mm | 0.001 | 0.00041 | 2,200 | 10,000 |

Reference velocities ([DIAGNOSIS.md](DIAGNOSIS.md), measured on the stored reference): cart
up to 10.0 m/s, poles up to 8.9 / 8.5 rad/s (0.5–1.0 s); in the energy-pumping phase
(0–0.5 s) angles stay below 0.3 rad and velocities below 2.2 m/s and 2.4 / 2.9 rad/s.

What this predicts:

* **Swing-up at 256 CPR**: during pumping a pole moves about one count per step, so the
  differenced velocity is essentially a 1-bit "moved / did not move" signal. DIAGNOSIS showed
  that aliasing of exactly these velocities was the main blocker for the full-state network.
  Expect failure without memory (history or recurrence) at 256; 1024 gives ~5 levels.
* **Balance through the classic LQR**: its velocity gains are 590 and 528 N per rad/s. With
  one-step differencing the force noise is ≈ 790 N (256), 200 N (1024), 50 N (4096), 12 N
  (16384) RMS, against a 100 N limit and a measured tolerance of ~8 N random noise.
  Prediction: the classic controller with a plain differencer balances only at 16384, perhaps
  4096; it needs a filter (observer) below that. A learned controller can choose lower gains
  or average several frames, which is the room the neural variants have.
* Position quantization alone is benign: the LQR's θ2 gain (2,155 N/rad) times q is 53 N at
  256 and ≤ 13 N from 1024 up.
* The cart encoder is never the bottleneck at 0.1 mm or finer (velocity noise 0.004 m/s ×
  136 N per m/s ≈ 0.6 N). 1 mm gives ~6 N, comparable to the 4096-CPR pole channel.

## Controllers (size ladder mlp:2, mlp:4, mlp:8, rnn:4)

| # | Name (`--mode`) | Inputs | mlp:2 / mlp:4 / mlp:8 params | State memory (words) |
|---|---|---|---|---|
| 1 | `full` | exact state, 9 values (existing upper bound) | 23 / 45 / 89 | 0 |
| 2 | `fd` | quantized positions + one-step finite differences, same 9-value layout | 23 / 45 / 89 | 3 (previous frame) |
| 3 | `hist` k = 3, 4 | current quantized frame (6 values) + the k − 1 most recent one-step differences (3 each) | k=3: 29 / 57 / 113; k=4: 35 / 69 / 137 | 3(k − 1) |
| 3′ | `raw` k = 2 | the last k frames as plain features (5 per frame), no differencing | 25 / 49 / 97 | 3(k − 1) |
| 4 | `pos` + rnn | current quantized frame only (6 values) into an Elman RNN | rnn:2 / rnn:4 / rnn:8 = 21 / 49 / 129 | hidden units |
| — | `pos` + mlp | current frame only, no memory (control: should fail) | 17 / 33 / 65 | 0 |
| 5 | classic + fd / + α-β observer | 3,132 numbers + 0 / 2 filter constants | — | 3 / 6 |

`hist` with k = 2 is exactly `fd` (the current frame plus one difference), so method 3 at
k = 2 is reported in the `fd` row. The differences are a fixed, lossless linear re-encoding of
the last k positions; without it (`raw`) the network must form a difference of two nearly equal
inputs with weights inside the ±8 box, which reproduces the gain ceiling of DIAGNOSIS (a
weight of 8 on a 0.01-rad difference is a gain of 0.08 per rad/s). `raw` measures that cost.

Training-only detail: curriculum and balance starts that are not at rest come with a warm-up
history (the 10 preceding states of the reference, or a constant-velocity back-extrapolation
for balance starts), which fills the encoder history and the RNN state before the episode
begins. Real-task episodes start from rest with an empty history (repeated current frame, zero
RNN state, zero pending command), exactly as deployed.

## Protocol

Same task, sparse score (`streak_smooth`), start-state curriculum, CMA-ES and 256 test starts
as the full-state result. No dense rewards.

* **Stage 1 (balance only, local).** Start upright ± 0.05 on all states, 20 s, must stay
  upright from step 1. CMA-ES (pop 32, 5 starts per generation, 10 s training episodes) for up
  to 600 generations. Cell passes at ≥ 90 % on 256 test starts. Two actuation settings:
  *ideal* (no delay, float force) and *realistic* (one-step delay, 8-bit force).
* **Stage 2 (swing-up, GitHub Actions).** Promising cells, 3 seeds, up to 4B env steps each,
  `experiments/encoder_curriculum.py`. Reported: test success at 20 s and 60 s, int8, Drake
  replay (16 starts, Python reference of the sensor model), and success when the delay setting
  is switched.

## Early findings (before any training)

**1. Semi-implicit Euler makes one-step differencing exact.** The plant (and Drake's discrete
plant) updates q⁺ = q + dt·v⁺, so (q_t − q_{t−1}) / dt is the *current* velocity exactly; the
only error is quantization. A constant-velocity α-β filter, in contrast, lags by
≈ A·dt·(α − β/2)/β ≈ 0.7 rad/s at the swing-up's ~100 rad/s² and failed at every CPR. On real
hardware (continuous plant, sampled encoder) the difference is a half-step-lagged average.

**2. The 45-param full-state policy works zero-shot on encoder velocities** (no retraining;
`results/experiments/revcurr_mlp4_s0_v12.json` through `fd`, 256 test starts, 20 s):

| Pole CPR | cart 1 mm | 0.1 mm | 0.01 mm | with one-step delay |
|---|---|---|---|---|
| exact state | – | 100 % | – | **0 %** |
| 256 | 0 % | 0 % | 0 % | 0 % |
| 1024 | 64 % | 61 % | 62 % | 0 % |
| 4096 | 99.6 % | **100 %** | 99.6 % | 0 % |
| 16384 | 99.2 % | 100 % | 99.6 % | 0 % |

8-bit force command: 100 %. So velocity *sensing* costs nothing down to 4096 CPR; the binding
constraint is **actuation delay**, which breaks even the exact-state policy.

**3. Classic reference through the encoders** (`experiments/encoder_classic.py`, swing-up test
20 s, 256 starts):

| | 256 | 1024 | 4096 | 16384 |
|---|---|---|---|---|
| classic + fd (3,132 numbers) | 0 % | 0 % | 10 % (balance alone 100 %) | 100 % |
| classic + model-based α-β observer (3,139 numbers) | 0 % | **99.6 %** | 100 % | 100 % |

As predicted from the gains (fd force noise ≈ 50 N RMS at 4096, 12 N at 16384), the
high-gain TVLQR needs 16384 CPR with a plain differencer; a model-based observer (+7 numbers)
moves the threshold to 1024. Realistic setting (delay + 8-bit): see `results/encoder/summary.md`.

## Stage 1 results (balance only, map done)

Full tables: [`results/encoder/summary.md`](../results/encoder/summary.md); figure:
[`fig_success_vs_bits_stage1.png`](../results/encoder/fig_success_vs_bits_stage1.png). One seed
per cell (seed 0; 600 CMA-ES generations), cart encoder 0.1 mm. Where it ran: Actions run
41159760695 (cells transcribed from the job logs into `results/encoder/actions/`), plus local
runs (`results/encoder/local/`).

**Coarsest CPR with ≥ 90 % at 20 s** (smallest passing network in brackets; † = held loosely,
max angle error over the last 5 s > 0.05 rad, i.e. a limit cycle inside the ±0.2 rad band):

| Velocity information | No delay, float force | One-step delay, 8-bit force |
|---|---|---|
| full state (bound) | any (23 params) | any (23) |
| `fd`, reactive MLP | 1024† (23); tight from 4096 | 1024 (23, tight: 96/96 %) |
| `hist` k = 3 | 1024† (29) | **256†** (29); tight from 1024 (29) |
| `hist` k = 4 | **256†** (35) | **256†** (35); tight from 1024 (35) |
| `fd` + rnn:4 | **256†** (61) | 4096 (61) |
| `raw` k = 2 (no differencing) | 1024† (49) | **none** |
| `pos` + RNN | 1024† (rnn:4, 49); 4096† (rnn:2, 21) | 1024† (rnn:2, 21) |
| `pos` + MLP (no velocity, no memory) | 4096† (33) | **none** (0 % everywhere) |
| classic + fd (3,132 numbers) | 4096 (balance only) | 4096 |
| classic + model-based α-β (3,139) | 1024 | 1024 |

What the map says:

1. **The ±0.2 rad task band admits balancing without velocity.** A 33-param reactive MLP on
   positions alone holds 85–93 % for 20 s as a ~0.1 rad limit cycle, but drifts off at 60 s
   and dies completely under a one-step delay. So "balances" is reported together with 60 s
   success and hold error; the task's own success criterion does not separate these.
2. **1024 CPR is the practical floor for balance**; below it only controllers with memory (3–4
   frames or a recurrent state) pass, and they hold loosely. Tight holds need 4096 with a plain
   differencer, or 1024 with a 3–4 frame history.
3. **Per parameter, short history wins once delay is included**: `hist` k = 3/4 with mlp:2
   (29/35 params, 6/9 words of memory) is the only family that passes at 256 under delay and
   holds tightly from 1024. Without delay the plain differencer with mlp:2 (23 params) is the
   cheapest at ≥ 1024.
4. **Differencing must be in the input encoding.** Raw frames (`raw2`) work without delay but
   fail everywhere under delay: the network cannot form a clean difference inside the ±8 box.
5. **Delay must be trained for.** Policies trained without delay fail when it is added (fd at
   ≤ 1024: 0 %), mirroring the zero-shot result; policies trained with delay usually also work
   without it.
6. **The cart encoder is not a bottleneck**: 1 mm, 0.1 mm and 0.01 mm give the same picture
   (differences track optimiser noise).
7. **Bigger networks often scored 0 %** (mlp:8, rnn:8, 89–137 params): with pop 32 and 600
   generations CMA-ES did not converge. This is a budget artefact, not a capacity limit.
8. **int8 is free for the small networks** (within a few % of float in most passing cells),
   so the bit cost is params × 8: the smallest robust encoder controller is
   hist3 mlp:2 = 29 × 8 = 232 bits (+ 6 × 14-bit history words), against 3,139 × 32 bits for
   the classic controller with an observer.

## Stage 2
## Stage 2

Launched: Actions run **41161570797** (commit `bf1f45c`, launch label `enc-main-1`), 14 cells × 3
seeds, up to 4B env steps each, early stop at 99 % validation. Cells (chosen from the stage-1 map):

| Setting | Cells |
|---|---|
| no delay | full mlp:4 (bound) · fd 4096 mlp:2 · fd 4096 mlp:4 · fd 1024 mlp:4 · hist3 1024 mlp:2 · hist3 1024 mlp:4 · hist4 256 mlp:2 · fd 256 rnn:4 · pos 1024 rnn:4 |
| delay + 8-bit | full mlp:4 · fd 4096 mlp:4 · hist3 4096 mlp:4 · hist3 1024 mlp:2 · pos 1024 rnn:2 |

Results: not in yet. Runs, launch labels and commits:
[`results/encoder/README.md`](../results/encoder/README.md). Each job prints its reports
(without weights) to the job log as `ENC>` lines; `python experiments/encoder_compact.py --parse
<log>` turns a saved log back into JSON lines if the artifacts cannot be downloaded.
