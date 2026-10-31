"""Gradient diagnostics before any training (results/diffsim/README.md, step 3).

  python diffsim/diagnostics.py chaos      open- and closed-loop Jacobian growth (finite-time Lyapunov)
  python diffsim/diagnostics.py gradnorm   |grad_theta J_H| vs horizon H = 0.1..2.5 s
  python diffsim/diagnostics.py cosine     first-order vs zeroth-order (randomized smoothing) gradients
  python diffsim/diagnostics.py timing     wall time of forward vs forward+backward rollouts

Writes results/diffsim/diag/<section>.json. Plots: diffsim/plot_diagnostics.py.

Objectives (pendulum/jax_objectives.py; diagnostics only, success is never measured with them):
  term  J = c(s_H)                       terminal state cost at the horizon
  run   J = mean_{k<=H} c(s_k)           running state cost (dense reference)
  soft  J = mean_{k<=H} soft_upright     sigmoid upright indicator, temp 0.005 (sparse-score relaxation)
Starts: hanging rest +-0.05 (the task's start distribution; a diagnostic seed, not val/test).
"""
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from pendulum import jax_sim as J  # noqa: E402
from pendulum.classic import ClassicController  # noqa: E402
from pendulum.jax_objectives import soft_upright, state_cost  # noqa: E402
from pendulum.policy import GraphPolicy, build_arch  # noqa: E402
from pendulum.task import DEFAULT_TASK, TARGET_PHYSICS as P  # noqa: E402

T = replace(DEFAULT_TASK, max_v_cart=12.0, max_w_pole=12.0)   # velocity scale 12 throughout
OUT = ROOT / "results" / "diffsim" / "diag"
DIAG_SEED = 7_000_003
N_MAX = 250                       # 2.5 s
SOFT_TEMP = 0.005
EXP = ROOT / "results" / "experiments"


def policies(random_seeds=(0, 1, 2)):
    """name -> GraphPolicy. Trained: the solved controller and two stalled curriculum runs."""
    pols = {
        "solved mlp:4 (100%)": GraphPolicy.load(EXP / "revcurr_mlp4_s0_v12.json"),
        "stalled mlp:4 (0.30 s, 1.2%)": GraphPolicy.load(EXP / "revcurr_mlp4_s1_v12.json"),
        "stalled mlp:8 (0.55 s, 1.6%)": GraphPolicy.load(EXP / "revcurr_mlp8_s0_v12.json"),
    }
    for a in ("mlp:2", "mlp:4", "mlp:8"):
        for s in random_seeds:
            # same initialisation as the CMA-ES template (build_arch with the run's rng)
            pols[f"random {a} s{s}"] = build_arch(a, 9, np.random.default_rng(s))
    return pols


def starts(n, seed=DIAG_SEED):
    return jnp.asarray(np.random.default_rng(seed).uniform(-T.ic_noise, T.ic_noise, (n, 6)))


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.json", "w") as f:
        json.dump(obj, f, indent=1)
    print(f"wrote {OUT / name}.json", flush=True)


def fit_rate(h_s, y, lo=0.5, hi=2.5):
    """Least-squares slope of log(y) against time on [lo, hi] (growth rate in 1/s)."""
    h_s = np.asarray(h_s); y = np.asarray(y)
    m = (h_s >= lo - 1e-9) & (h_s <= hi + 1e-9) & (y > 0)
    return float(np.polyfit(h_s[m], np.log(y[m]), 1)[0])


# --------------------------------------------------------------------------- chaos
def chaos():
    """Largest singular value of d s_k / d s_0 along trajectories.
    Open loop: the classic reference forces replayed from perturbed rest (the setting of the
    50x measurement in docs/DIAGNOSIS.md). Closed loop: each policy's own feedback."""
    ex = ClassicController.load(ROOT / "results/classic/controller.npz")
    S0 = starts(32)
    t = (np.arange(N_MAX) + 1) * T.dt
    res = {"t": t.tolist()}

    # nonlinear check of the documented measurement: 1e-3 start error, position error after 2 s
    u = jnp.asarray(ex.u_nom[:N_MAX])          # the reference swing-up is 2.4 s long
    ref = J.open_loop(jnp.asarray(ex.S_nom[0]), u, P, T)
    rng = np.random.default_rng(0)
    d0 = jnp.asarray(rng.uniform(-1e-3, 1e-3, (32, 6)))
    pert = jax.vmap(lambda d: J.open_loop(jnp.asarray(ex.S_nom[0]) + d, u, P, T))(d0)
    pos_err = np.median(np.linalg.norm(np.asarray(pert[:, :, :3] - ref[None, :, :3]), axis=-1), 0)
    res["open_loop_nonlinear_pos_err_1e-3"] = pos_err.tolist()

    jac_ol = jax.jit(jax.vmap(jax.jacfwd(lambda s0: J.open_loop(s0, u, P, T))))
    Phi = np.asarray(jac_ol(jnp.asarray(ex.S_nom[0]) + S0))          # (K, N, 6, 6)
    smax = np.linalg.svd(Phi, compute_uv=False)[..., 0]
    t_ol = (np.arange(len(u)) + 1) * T.dt
    res["t_open_loop"] = t_ol.tolist()
    res["open_loop"] = {"sigma_max_median": np.median(smax, 0).tolist(),
                        "rate_per_s": fit_rate(t_ol, np.median(smax, 0), hi=2.0)}
    print(f"open loop: sigma_max(2 s) median {np.median(smax, 0)[199]:.1f}, "
          f"nonlinear pos err ratio at 2 s {pos_err[199] / 1e-3:.1f}, rate {res['open_loop']['rate_per_s']:.2f}/s")

    res["closed_loop"] = {}
    for name, p in policies().items():
        st = J.GraphStructure.of(p)
        th = jnp.asarray(p.get_flat())
        jac = jax.jit(jax.vmap(jax.jacfwd(lambda s0: J.rollout(st, th, s0, N_MAX, P, T)[0])))
        Phi = np.asarray(jac(S0))
        smax = np.median(np.linalg.svd(Phi, compute_uv=False)[..., 0], 0)
        r = fit_rate(t, smax)
        res["closed_loop"][name] = {"sigma_max_median": smax.tolist(), "rate_per_s": r}
        print(f"closed loop {name:32s}: sigma_max at 1/2/2.5 s {smax[99]:.3g} {smax[199]:.3g} {smax[249]:.3g}, "
              f"rate {r:.2f}/s", flush=True)
    save("chaos", res)


# --------------------------------------------------------------------------- gradient norm vs horizon
H_STEPS = np.arange(10, N_MAX + 1, 10)       # 0.1 .. 2.5 s


def objectives_all_horizons(st, th, s0):
    """(3, nH): term, run, soft objective for every horizon in H_STEPS, from one 2.5 s rollout."""
    S, _ = J.rollout(st, th, s0, N_MAX, P, T)
    c = state_cost(S, T)
    u = soft_upright(S, SOFT_TEMP, T)
    idx = jnp.asarray(H_STEPS)
    return jnp.stack([c[idx - 1], jnp.cumsum(c)[idx - 1] / idx, -jnp.cumsum(u)[idx - 1] / idx])


def gradnorm(K=64):
    S0 = starts(K)
    res = {"h_s": (H_STEPS * T.dt).tolist(), "objectives": ["term", "run", "soft"], "policies": {}}
    for name, p in policies().items():
        st = J.GraphStructure.of(p)
        th = jnp.asarray(p.get_flat())
        f = jax.jit(jax.vmap(jax.jacrev(lambda t, s0: objectives_all_horizons(st, t, s0)), (None, 0)))
        G = np.asarray(f(th, S0))                     # (K, 3, nH, P)
        out = {}
        for oi, on in enumerate(res["objectives"]):
            g = G[:, oi]                                  # (K, nH, P)
            per = np.linalg.norm(g, axis=-1)              # per-start norms
            mean = np.linalg.norm(g.mean(0), axis=-1)     # norm of the batch gradient
            snr = mean ** 2 / np.maximum(np.mean(per ** 2, 0), 1e-300)   # 1 = all starts agree, ~1/K = noise
            nz = np.mean(per > 1e-12, 0)
            out[on] = {"per_start_median": np.median(per, 0).tolist(),
                       "per_start_p10": np.percentile(per, 10, 0).tolist(),
                       "per_start_p90": np.percentile(per, 90, 0).tolist(),
                       "batch_norm": mean.tolist(), "snr": snr.tolist(), "frac_nonzero": nz.tolist(),
                       "rate_per_s": fit_rate(H_STEPS * T.dt, np.median(per, 0))}
        res["policies"][name] = out
        o = out["run"]
        print(f"{name:32s} run: |g| median at 0.5/1/2/2.5 s "
              + " ".join(f"{o['per_start_median'][i]:.3g}" for i in (4, 9, 19, 24))
              + f" rate {o['rate_per_s']:.2f}/s | term rate {out['term']['rate_per_s']:.2f}/s"
              + f" | soft nonzero@2.5 {out['soft']['frac_nonzero'][-1]:.2f}", flush=True)
    save("gradnorm", res)


# --------------------------------------------------------------------------- FO vs ZO
COS_H = [25, 50, 100, 150, 200, 250]
SIGMAS = [0.01, 0.1]


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def _rel_var(samples):
    """Per-sample relative variance tr Cov / |mean|^2 (samples: (N, P))."""
    m = samples.mean(0)
    return float(np.sum(samples.var(0)) / max(m @ m, 1e-300))


def cosine(K=32, M=64, names=None, seed=0):
    """For each policy, horizon H and smoothing sigma, on K starts x M perturbations eps_m ~ N(0, I):
      FO  sample  grad J_i(theta + sigma eps_m)                    (first-order, smoothed objective)
      FO0 sample  grad J_i(theta)                                  (plain BPTT gradient)
      ZO  sample  (J_i(theta + sigma eps) - J_i(theta - sigma eps)) / (2 sigma) * eps   (antithetic)
    Reports cosines between the means, split-half self-cosines (estimator noise), per-sample relative
    variance, and the variance of per-start means across starts."""
    S0 = starts(K, DIAG_SEED + 1)
    pols = policies(random_seeds=(0,))
    if names:
        pols = {k: v for k, v in pols.items() if any(n in k for n in names)}
    rng = np.random.default_rng(seed)
    res = {"K": K, "M": M, "h_s": [h * T.dt for h in COS_H], "sigmas": SIGMAS, "policies": {}}
    for name, p in pols.items():
        st = J.GraphStructure.of(p)
        th = jnp.asarray(p.get_flat()); nP = len(th)
        res["policies"][name] = {}
        for n in COS_H:
            def single(th, s0, kind):
                S, _ = J.rollout(st, th, s0, n, P, T)
                c = state_cost(S, T)
                return c[-1] if kind == "term" else jnp.mean(c)
            for kind in ("term", "run"):
                # one independent perturbation per (m, i) sample: sharing eps_m across starts would
                # leave only M directions in P dimensions and bias the ZO mean by ~sqrt(P / M)
                val = jax.jit(jax.vmap(lambda t, s: single(t, s, kind)))
                grad = jax.jit(jax.vmap(jax.grad(lambda t, s: single(t, s, kind))))
                S0r = jnp.tile(S0, (M, 1))                                 # sample (m, i) -> row m*K + i
                g0 = np.asarray(grad(jnp.tile(th[None], (K, 1)), S0))      # (K, P)
                for sig in SIGMAS:
                    eps = rng.standard_normal((M * K, nP))
                    TH = th[None] + sig * jnp.asarray(eps)
                    fo = np.asarray(grad(TH, S0r))                         # (M*K, P)
                    jp = np.asarray(val(TH, S0r)); jm = np.asarray(val(th[None] - sig * jnp.asarray(eps), S0r))
                    zo = ((jp - jm) / (2 * sig))[:, None] * eps
                    gfo = fo.reshape(M, K, nP); gzo = zo.reshape(M, K, nP)
                    half = rng.permutation(M * K) < (M * K) // 2
                    r = {"cos_FO_ZO": _cos(fo.mean(0), zo.mean(0)),
                         "cos_FO0_ZO": _cos(g0.mean(0), zo.mean(0)),
                         "cos_FO0_FO": _cos(g0.mean(0), fo.mean(0)),
                         "self_cos_FO": _cos(fo[half].mean(0), fo[~half].mean(0)),
                         "self_cos_ZO": _cos(zo[half].mean(0), zo[~half].mean(0)),
                         "relvar_FO": _rel_var(fo), "relvar_ZO": _rel_var(zo), "relvar_FO0": _rel_var(g0),
                         "start_relvar_FO": _rel_var(gfo.mean(0)), "start_relvar_ZO": _rel_var(gzo.mean(0)),
                         "norm_FO": float(np.linalg.norm(fo.mean(0))), "norm_ZO": float(np.linalg.norm(zo.mean(0))),
                         "norm_FO0": float(np.linalg.norm(g0.mean(0)))}
                    res["policies"][name][f"{kind}|{n * T.dt:.2f}|{sig}"] = r
                    print(f"{name:30s} {kind} H={n * T.dt:.2f} s sig={sig}: cos(FO,ZO) {r['cos_FO_ZO']:+.2f} "
                          f"cos(FO0,ZO) {r['cos_FO0_ZO']:+.2f} self FO {r['self_cos_FO']:+.2f} ZO {r['self_cos_ZO']:+.2f} "
                          f"| relvar FO {r['relvar_FO']:.3g} ZO {r['relvar_ZO']:.3g} | |FO| {r['norm_FO']:.3g} "
                          f"|ZO| {r['norm_ZO']:.3g}", flush=True)
        save("cosine" + ("_" + "_".join(names) if names else ""), res)


# --------------------------------------------------------------------------- timing
def timing(K=64, reps=5):
    """Wall-time of a batch of K 2.5 s rollouts: forward only vs value_and_grad (one parameter vector)."""
    S0 = starts(K)
    res = {"K": K, "steps": N_MAX, "cpu": jax.devices()[0].device_kind, "archs": {}}
    for a in ("mlp:2", "mlp:4", "mlp:8"):
        p = build_arch(a, 9, np.random.default_rng(0)); st = J.GraphStructure.of(p)
        th = jnp.asarray(p.get_flat())

        def cost(th):
            S, _ = J.rollout_batch(st, th, S0, N_MAX, P, T)
            return jnp.mean(state_cost(S, T))
        r = {}
        for nm, fn in (("fwd", jax.jit(cost)), ("fwd_bwd", jax.jit(jax.value_and_grad(cost)))):
            jax.block_until_ready(fn(th)); t0 = time.time()
            for _ in range(reps):
                jax.block_until_ready(fn(th))
            r[nm] = (time.time() - t0) / reps
        r["ratio"] = r["fwd_bwd"] / r["fwd"]
        r["fwd_steps_per_s"] = K * N_MAX / r["fwd"]
        res["archs"][a] = r
        print(a, {k: round(v, 4) for k, v in r.items()}, flush=True)
    save("timing", res)


if __name__ == "__main__":
    sec = sys.argv[1]
    args = sys.argv[2:]
    {"chaos": chaos, "gradnorm": gradnorm, "timing": timing,
     "cosine": lambda: cosine(names=args or None)}[sec]()
