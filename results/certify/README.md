# Certified balance region of the 45-parameter neural controller

**Question.** From which states is the 45-parameter tanh MLP
(`results/experiments/revcurr_mlp4_s0_v12.json`, velocity scale 12, 100 % test success)
*mathematically guaranteed* to keep both poles upright (|angle error| < 0.2 rad) and the cart on
the track (|x| ≤ 2.8 m) forever, under the exact discrete-time closed loop of
`rollout.physics_step` (dt = 0.01 s, u = 100 N × policy output, real observation pipeline)?

**Status: steps 1–3 done for the 45-param controller (certified ρ = 1.6152).** Per the stop rule, the
comparisons (pruned, int8/int6, LQR), the 10⁶-sample check, the empirical region and the
certified catch (steps 5–7) are prepared but not run. The maximal-level run is launched on Actions.

## Results

| | Value |
|---|---|
| Closed-loop equilibria (upright, zero velocity, u = 0) | x = **+0.9958 m** (stable), x = −2.1168 m (unstable, spectral radius 1.0059) |
| Policy output at the upright state with x = 0 | +0.375 (37.5 N): the controller balances with the cart parked ~1 m off-center |
| Jacobian at x* = (0.9958, π, 0, 0, 0, 0): spectral radius | **0.99743** (slow complex pair 0.99739 ± 0.0085i: cart-position mode, time constant ≈ 3.9 s) |
| Lyapunov function | V = (x − x*)ᵀ P (x − x*), P from (A/β)ᵀP(A/β) − P = −Q, β = 0.999, Q = diag(1/[1.8, 0.2, 0.2, 1, 1, 1]²) |
| Level allowed by the safe set (ρ_safe) | 33.96 (Ω_ρ ⊂ upright ∩ track) |
| **Certified level ρ (rigorous)** | **ρ = 1.6152** (Actions run 41163866994: 10 shards, 78 M boxes, complete; `mlp45/certificate_actions.json`). Locally certified and rechecked from scratch: ρ = 1.0 (3.2 M boxes, 118 s) |
| Empirical upper bound for this V (falsification) | ρ ≤ 1.764: an explicit state with \|z\|² = 1.764 where V does not decrease |
| Certified set extent (max \|x − x*\| over Ω_ρ, ρ = 1.6152) | x ±0.105 m, θ1 ±0.041 rad, θ2 ±0.044 rad, ẋ ±0.67 m/s, ω1 ±0.65 rad/s, ω2 ±0.65 rad/s |
| Certified volume (state units m·rad²·(m/s)·(rad/s)²) | 1.08 × 10⁻⁶ (ρ = 1.6152); 2.57 × 10⁻⁷ at ρ = 1 |
| Sanity: 20 000 samples of Ω_ρ (half in the outer 1 % shell), 60 s, repository episode kernel | **0 failures** |
| Gap to the empirical bound | certified 1.6152 vs counterexample at 1.7643: **92 %** of the best level for this V (volume ratio (1.615/1.764)³ = 77 %) |
| Sanity at ρ = 1.6152 | local: 40 000 samples (half in the outer 1 % shell), 60 s: **0 failures**; Actions 10⁶ × 60 s: see run log |

What limits the region: not the safe set (ρ_safe = 34) but the one-step decrease. The slowest
mode contracts by only 0.9974 per step, so V can shrink by at most ~0.5 % per step, and the
nonlinearity (mostly the network's curvature acting through the force channel) cancels that
margin at |z|² ≈ 1.76. The certified ellipsoid is thin in the angles (±0.03 rad) compared to
the upright tolerance (±0.2 rad).

## The certificate (what is proved, and how)

z = W (x − x*), V = |z|², with W a float matrix. Three facts are checked:

1. **Decrease** — V(F(x)) − V(x) < 0 for all ε² < |z|² ≤ ρ: branch and bound over boxes in
   z-space (`pendulum/certify/verify.py`).
2. **Local contraction** — on |z| ≤ ε (ε ≈ 2 × 10⁻³) F is a contraction in |·| that maps the
   ball into itself: interval Jacobian, ‖D y‖₂ ≤ κ < 1, |y(0)| + κε ≤ ε; κ = 0.999995.
3. **Safety** — Ω_ρ lies in the upright set and on the track (interval bound of W⁻¹).

Together: every trajectory that starts in Ω_ρ stays in Ω_ρ (so it is upright and on the track at
every step, forever) and converges to the unique fixed point in the ε-ball.

Rigor details:
* F is the real-number map with the float64 constants the simulator uses. The closed loop is
  traced once into a straight-line program (`pendulum/certify/model.py`) of +, −, ×, ÷, sin,
  cos, tanh, clip. Evaluation is interval arithmetic in numba with outward rounding of every
  operation (≥ 1 ulp, tested against `np.nextafter`) and ~18 ulps for libm sin/cos/tanh.
* W⁻¹ is enclosed exactly with rational arithmetic; ‖Â‖₂ bounds are proved with an exact
  (rational) positive-definiteness test.
* The physics is traced in an algebraically simplified but identical form (sin² + cos² = 1,
  angle-difference identities), checked against `physics_step` to 1e-12 in tests.
* Velocity clipping in the observation and the LQR saturation are Lipschitz: slope [0, 1].

Why it is tight enough to work (each was a ~10–300× loss before):
* **Exact quadratic part.** dV = zᵀNz + (higher order) with N = ÂᵀÂ − I diagonal in the chosen
  coordinates, so the dominant term is bounded exactly over a box.
* **Structure of the step.** F(x) = Φx + Γ a(x) and a = m(q)·u + p(x). The remainder is split into
  the scalar control channel u and the rest, so W Γ m (the input direction, |W b| ≈ 0.01) is
  never multiplied by an interval vector.
* **Mean-value refinement of every intermediate** range (center value + slope × half-width, in
  lockstep), and the cubic term φ = 2(b_m·z)s_u + 2(Bz)ᵀt is a single program output.
* **Shifted Lyapunov equation** (β = 0.999 instead of 1): the plain equation leaves a direction
  with a decrease rate of 1.4 × 10⁻⁴; β = 0.999 guarantees ≥ 2.1 × 10⁻³, at almost no cost in
  volume (sampled bound 2.07 vs 2.00).

## Reproduce

```bash
python experiments/certify_balance.py setup mlp45          # equilibrium, V, rho_safe, sampled bound
```

```bash
python experiments/certify_balance.py run mlp45 --rho0 1.0 --grid 2 --out /tmp/s0.json
```

```bash
python experiments/certify_balance.py combine mlp45 /tmp/s0.json
```

```bash
python experiments/certify_balance.py recheck mlp45        # recompute the stored certificate from scratch
```

```bash
python experiments/certify_balance.py sanity mlp45 --n 100000 --seconds 60
```

```bash
python -m pytest -q tests/test_certify.py                  # 11 soundness tests, ~10 s
```

Tests: outward rounding ≥ nextafter; scalar enclosures (incl. sin/cos extrema); the traced
program equals `physics_step` and the episode kernel; range and slope enclosures contain
sampled values and finite differences (MLP and saturated LQR); exact inverse; the box bound
upper-bounds the true dV on random boxes; an end-to-end certificate at ρ = 0.05 checked by
simulation; and the B&B lowers the level below a known counterexample.

Files: `mlp45/setup.json` (x*, W, Â, equilibria, sampled bound), `mlp45/certificate.json`
(level, box statistics), `mlp45/local_rho1_shard.json`, `mlp45/sanity_local.json`.

## Runs on GitHub Actions

`.github/workflows/certify.yml` shards the initial boxes by job index (`max-parallel: 5`) and
runs combine + a 10⁶-sample, 60 s check. Agent sessions cannot push tags, so besides `cert-*`
tags and manual dispatch, a push to a certify branch that changes `results/certify/job.json`
launches the job described there. Results are uploaded as artifacts and printed in the logs.

| Launch | Run | Commit | Job | Status |
|---|---|---|---|---|
| smoke (`"smoke": true`) | [41163608223](https://github.com/nchenevey1/PyDrake-NEAT-pendulum/actions/runs/41163608223) | 748d0d8 | mlp45, ρ₀ = 0.05, 2 shards, 10⁴ samples | **success**: ρ = 0.05 complete (31k boxes), sanity 0/10⁴ failures |
| job.json (no tag: agent sessions cannot push tags) | [41163866994](https://github.com/nchenevey1/PyDrake-NEAT-pendulum/actions/runs/41163866994) | 241a81b | mlp45, ρ₀ = 1.764 (sampled bound), 10 shards, grid 4, give-up width 1 % of \|z\|, then 10⁶ × 60 s sanity | **success**: all 10 shards complete (2–21 min each), combine ok, **ρ = 1.6152** (limiting shard 8; shard 4: 1.6172); 2154 proven counterexample centers, 220 boxes given up at 1 % width |

## What failed or is inconclusive

* Plain Lyapunov P (β = 1): the ε-ball contraction failed for every ε (decrease rate 1.4e-4).
* Naive interval Jacobian of F in Lyapunov coordinates: ~1500× wider than the true Jacobian
  variation; certification was hopeless until the structure above was used.
* The certified level 1.6152 is 8 % below the empirical bound 1.7643. Part of the gap is the 1 %
  give-up width (220 boxes, which set the level to their min |z|^2), part is real counterexamples
  close to the bound (2154 proven at box centers, nearest |c|^2 = 1.64). The limiting point
  (z ≈ (−0.03, −0.38, −0.34, −0.69, −0.35, −0.89)) is in the same direction as the sampled
  counterexample. A full `recheck` at 1.6152 needs ~1.6 core-hours (Actions, not local).
* The certificate is for the real-number map with float64 constants. The float64 simulator
  differs from it by rounding (~1e-16 per step); this is not separately bounded.
* No CROWN/auto_LiRPA tightening and no SOS comparison (optional steps) were attempted.

## Next steps (prepared, not run)

1. Re-run the certificate at ρ₀ = 1.6152 on Actions as an independent recheck (job.json with `"rho0": 1.6151688736209255`).
2. Optimise Q (and β) for certified volume; the sampled bound is a cheap proxy.
3. Same pipeline for `mlp45_pruned` (regenerated with `evaluation.greedy_prune`), `mlp45_int8`,
   `mlp45_int6`, `lqr`: `setup NAME`, then set `results/certify/job.json`.
4. Empirical region (sampling the safe box, 60 s) and certified/empirical volume.
5. Certified catch: first entry of the 256 test swing-ups into Ω_ρ.

## Related work

Novelty check (web search, 2026-10-31). Certified regions of attraction (ROA) for neural
controllers are an active area. This study's pieces exist separately:

* **Neural Lyapunov control / verified ROA.** Chang, Roohi & Gao (NeurIPS 2019) learn controllers
  and Lyapunov functions with SMT (dReal) verification in continuous time. Dai et al.
  ["Lyapunov-stable neural-network control"](https://arxiv.org/abs/2109.14152) (RSS 2021)
  use MIP on ReLU networks with an inner ROA for pendulum and quadrotor.
  [Wu et al., "Neural Lyapunov Control for Discrete-Time Systems"](https://arxiv.org/abs/2305.06547)
  (NeurIPS 2023) is discrete time with MILP verification, including cartpole.
  [Yang, Dai, Shi, Hsieh, Tedrake & Zhang](https://arxiv.org/abs/2404.07956) (ICML 2024) is
  discrete time with α,β-CROWN branch and bound and a larger verifiable ROA, on pendulum and
  2D quadrotor; it became a VNN-COMP 2024 benchmark
  ([summary](https://arxiv.org/abs/2412.19985)).
* **ROA analysis of a given NN controller.** [Yin, Seiler & Arcak](https://arxiv.org/abs/2006.07579)
  (IEEE TAC 2022) compute an ellipsoidal inner ROA estimate with quadratic constraints and SDP, on
  an inverted pendulum. Interval bound propagation for closed-loop contraction of learned
  controllers is also studied ([arXiv 2512.02262](https://arxiv.org/abs/2512.02262)).
* **ROA of underactuated systems.** SOS-based ROA (Tedrake's LQR-trees, Tobenkin et al.) for
  polynomial or Taylor-approximated dynamics, mostly continuous time.

What this study adds, as far as the search found:
* a rigorous ROA for a learned controller of a **cart + double pendulum** (6-D, strongly
  underactuated) under its **exact discrete-time** step, including the policy's real
  observation pipeline (sin/cos, clipping);
* a tiny **tanh** network found by evolution, not trained to be verifiable;
* (planned) how **pruning and int8/int6 quantization** change the certified region. The search
  found no work certifying ROA changes under quantization of a neural controller.

The verification ideas used here are standard: interval arithmetic, mean-value forms, branch
and bound, Lyapunov sublevel sets, contraction near the equilibrium. The contribution is the
application and the measurements, not the method. Two observations may be of independent
interest: the learned controller balances at an offset equilibrium (x* ≈ 1 m), and the certified
level is limited by a 0.9974 slow mode, not by the upright tolerance.
