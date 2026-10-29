# Why learning fails, and what to change

## 1. Goal

**Engineering goal.** A neural controller with ≤ 100 parameters (int8-deployable) that solves
the fixed task (swing up from hanging rest within 2.5 s, balance to 20 s, 100 N, g = 9.81,
no drag) on ≥ 90 % of held-out starts.

**Research goal.** Explain *what makes this task learnable* by black-box search from a sparse
success signal, and use that to find the smallest controller. The task is a hard instance of a
general problem: a chaotic, underactuated system where success is a thin target in state space
and the only signal is "did it balance".

**Why this is useful research**

| Gap | What we add |
|---|---|
| Neuroevolution's standard benchmark (double-pole balancing) is *balance only*. Swing-up of a serial double pendulum on a cart with tiny networks and a sparse signal is largely unreported. | A harder, fast, reproducible benchmark with a classic upper bound (100 %) and a fixed protocol. |
| Reverse/start-state curricula (Florensa 2017, Salimans & Chen 2018, Go-Explore) are known to work, but little is said about *where* they stall and why. | A mechanistic account of stalls (input conditioning, state aliasing, chaos), with measurements. |
| Tiny and verifiable controllers (WANN, neural circuit policies, Lyapunov-certified nets) rarely tackle swing-up. | The size × reliability Pareto front for this task, compared with a 3,132-number classic controller. |

What would *not* be a contribution: a generic "NEAT vs CMA-ES" table, or a win obtained by
tuning hyperparameters to one seed.

**Measures** (unchanged): test success on 256 held-out starts at a fixed env-step budget;
parameters and bits. **Diagnostic measures** (never used as training signal): how far back the
start-state curriculum gets, success vs start-noise width, and success under small weight
perturbations.

## 2. What the diagnostics show

Run `python experiments/diagnose.py [policy.json]`. Policy analysed: the best learned controller
(45-param MLP, start-state curriculum, 9 % test success).

| # | Finding | Measurement |
|---|---|---|
| 1 | **Gain ceiling.** The observation divides velocities by 50 m/s and 90 rad/s; the real range is ±10. With weights in ±8 the network cannot express the feedback gains a working controller uses. | Classic gains in action-per-observation-unit: balance LQR on pole velocities **≈ 500**, swing-up TVLQR median **120–220**. Reachable: 8 (linear), ≈ 64 per hidden unit at the box edge. |
| 2 | **The early swing-up is where it fails.** The reference pumps energy for the first 0.5 s with angles < 0.3 rad but velocities up to 2.8 rad/s, changing force direction ~5 times. Seen through a /90 scale, those velocities are ±0.03: states that need opposite forces look almost identical (state aliasing). | 86 % of test episodes **leave the track at a median 0.93 s**, with the tip never above −0.65 m. The curriculum stalled at starts 0.45–0.60 s into the swing-up in both seeds. |
| 3 | **No real feedback.** Neither learned policy has a funnel around its swing-up: one succeeds only by luck on some wide starts; the other works only very close to its memorised trajectory. | Success vs start noise ±0 / ±0.002 / ±0.01 / ±0.05: seed 1 (9 % policy) 0 / 0 / 0 / 8 %; seed 0 0 / 7 / 1 / 1 %. Classic: 100 % at every width. Seed 0 also leaves the track in 77 % of episodes (median 1.77 s). |
| 4 | **The solution is a needle in weight space.** | Perturbing the weights by σ = 0.01 (box is ±8) drops success from 9 % to 0.2 % median; σ = 0.03 → 0 %. |
| 5 | **The system is chaotic, so feedback is required.** | Open-loop replay of the reference forces: a 0.001 start error grows ~50× in 2 s, mostly in 1.0–2.0 s. |
| 6 | **The catch is not the bottleneck.** | Handing over to the classic balance LQR once both angles are within 0.2 or 0.4 rad: 1.6 % success. The swing-up never delivers the poles near upright. |
| 7 | **The sparse score is very noisy on 8 starts.** | Real-task score of the 9 % policy on 8 random starts: mean 1.4, sd 1.7. |
| 8 | **Runs were short.** | 2B env steps = 17 single-core minutes. No run of a working signal has had a long budget yet. |

Earlier failures, for completeness:
* **Physics curriculum:** advanced without mastery, trained on a different problem (low g, high
  drag), never recovered (see [CURRICULUM.md](CURRICULUM.md)).
* **Imitation of the classic controller:** its feedback (TVLQR) is valid only near its own
  trajectory, so labels at the student's off-trajectory states are wrong.
* **Deadline curriculum:** its first level already needs swing-up and balance together.

**One-line diagnosis:** the training signal (start-state curriculum + mastery) is sound. The
networks were blind to the variable that matters most in the energy-pumping phase (velocity) and
could not express the gains needed. They memorised an open-loop trajectory, which chaos and the
track limit punish. Section 3 tests this by changing only the velocity scale.

## 3. Test of the main hypothesis

Same start-state curriculum, same 2B budget, only the velocity observation scale changed from
(50, 90) to 12 for both cart and poles (`python experiments/reverse_curriculum.py mlp:4 0 2.0 12`).

| Run (CMA-ES, 2B steps) | Old scales: closest start reached / test | Velocity scale 12: closest start / test |
|---|---|---|
| mlp:4 (45 params) seed 0 | 0.45 s / 0.4 % | **0 s (real task) / 100 %**, after 1.08B steps (9 min) |
| mlp:4 seed 1 | 0.60 s / 9.0 % | 0.30 s / 1.2 % |
| mlp:4 seed 2 | (not run) | 0.55 s / 0 % |
| mlp:8 (89 params) seed 0 | 1.30 s / 0 % | 0.55 s / 1.6 % |

**Every rescaled run got further back than any old-scale run of the same size, and faster.**
One of four runs solved the task. The other three stalled at the same point, 0.50–0.55 s into
the reference, where the reference reverses force from −81 N to +61 N.

The solved controller (`results/experiments/revcurr_mlp4_s0_v12.json`, 45 params), independently
re-checked:

| Check | Result |
|---|---|
| Test 20 s / 60 s / from exact rest | **100 % / 100 % / 100 %** |
| Replayed in Drake (16 starts) | 16/16 |
| int8 / int6 / int4 | 91 % / 55 % / 0 % |
| Greedy pruning | 43 params, 99 % (int8 99 %) |
| Wide starts ±0.2 | 29 % (classic 73 %) |
| Weight noise σ = 0.01 / 0.03 | 99 % / 76 % median (old-scale 9 % policy: 0.2 % / 0 %) |
| Model error: g + 5 % / force limit 90 N / drag 0.1 | 92 % / 70 % / 0 % |
| Behaviour | follows the reference's pumping pattern to ~1.5 s, then catches earlier (upright at 1.9 s) |

What this does and does not show:
* The input-conditioning diagnosis is confirmed. The optimum is now wide, and the policy has a
  real feedback funnel (100 % at every start-noise width).
* The method is **not yet reliable**: 1 of 4 runs. The next bottleneck is the 0.55 s level.
* The policy is robust to small parameter changes but not to drag. Robustness is a separate goal
  (see section 4).

## 4. What to change (ordered by evidence and cost)

| Cause | Change | Keeps the sparse signal? |
|---|---|---|
| Gain ceiling, aliasing (1, 2) | **Observation scales from the real state range** (fixed constants, so deployment is unchanged). Same for every method. **Confirmed (section 3)**: make it the default and store the scales with each policy. | yes |
| Common stall at 0.55 s (section 3) | **Adaptive step**: after a back-off, halve the start-time step (5 → 2 → 1 control steps), so the force reversal is crossed gradually. Then longer runs. | yes |
| Aliasing in the pumping phase (2) | **Better reference swing-ups:** smoother (penalise force changes), leave track margin (\|x\| ≤ 1.5 m instead of 2.15 m), both directions, several optimiser seeds. Then the **no-reference** variant (S3) to see if a demonstration is needed. | yes |
| Luck instead of feedback (3, 4, 5) | **Mastery on more starts at the final levels** (32–64), and keep 25 % replay of earlier levels to the end. Model selection on 256 validation starts. | yes |
| Score noise (7) | Common random starts already; add more starts per candidate only at hard levels (adaptive K). | yes |
| Timing (deadline curriculum result) | **Deadline slack** (S5): +1 s while moving the start back, then shrink to 0. | yes |
| Short runs (8) | **Long runs:** 20B+ env steps (~3 h/core), 3 seeds, once the above is in. | yes |
| Robustness (drag 0.1 → 0 %) | Only after reliability: evaluate (and later train) with ±10 % mass, small drag and sensor noise. | yes |
| Size | Only after competence: prune, fine-tune, quantise (see [RESEARCH_PLAN.md](RESEARCH_PLAN.md)). | yes |

Not recommended now:
* Dense rewards or tracking losses (excluded by the brief; they also invite spinning exploits).
* Widening the weight box instead of rescaling inputs: same gains, but worse for quantisation.
* More imitation learning: needs a replanning (MPC) expert; a larger, separate project.
