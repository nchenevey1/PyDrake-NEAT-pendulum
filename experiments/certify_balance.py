"""
Certified balance region of the small neural controllers (see results/certify/README.md).

  python experiments/certify_balance.py setup  NAME            equilibrium, Jacobian, V, safe level,
                                                               sampled upper bound -> setup.json
  python experiments/certify_balance.py run    NAME --rho0 R [--shard i n] [--grid g] [--out f]
                                                               branch and bound (one shard)
  python experiments/certify_balance.py combine NAME f1 f2 ... certificate.json (min over shards)
  python experiments/certify_balance.py recheck NAME           recompute the certificate from scratch
  python experiments/certify_balance.py sanity NAME [--n N] [--seconds T]
                                                               simulate samples of the certified set

NAME: mlp45 (the 45-param controller), mlp45_int8, mlp45_int6, mlp45_pruned, lqr.
Outputs go to results/certify/NAME/.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pendulum.certify.model import (ControllerData, find_equilibria, from_lqr, from_policy, step_np)
from pendulum.certify import verify as VF
from pendulum.policy import GraphPolicy
from pendulum.task import DEFAULT_TASK, TARGET_PHYSICS

MLP = ROOT / "results/experiments/revcurr_mlp4_s0_v12.json"
OUT = ROOT / "results/certify"
LIMITS = (DEFAULT_TASK.limit_x, DEFAULT_TASK.upright_tol)
# V recipe (same for every controller): shifted Lyapunov equation, Q = diag(1/scale^2)
SCALES = np.array([1.8, 0.2, 0.2, 1.0, 1.0, 1.0])
BETA = 0.999


def policy_for(name: str) -> GraphPolicy:
    pol = GraphPolicy.load(MLP)
    if name == "mlp45":
        return pol
    if name == "mlp45_int8":
        return pol.quantized(8)
    if name == "mlp45_int6":
        return pol.quantized(6)
    if name == "mlp45_pruned":
        path = OUT / "mlp45_pruned" / "policy.json"
        if not path.exists():
            from pendulum.evaluation import greedy_prune, val_ics
            task = replace(DEFAULT_TASK, **pol.meta["obs_scale"])
            # same procedure as evaluation.full_report (validation starts, 2 % tolerance), with
            # the policy's own observation scales
            pr = greedy_prune(pol, val_ics(128, task), tolerance=0.02, task=task)
            path.parent.mkdir(parents=True, exist_ok=True)
            pr.save(path)
        return GraphPolicy.load(path)
    raise KeyError(name)


def controller(name: str) -> ControllerData:
    if name == "lqr":
        d = np.load(ROOT / "results/classic/controller.npz")
        return from_lqr(d["K_bal"], name="lqr")
    return from_policy(policy_for(name), name=name)


def setup_path(name):
    return OUT / name / "setup.json"


def load_setup(name):
    d = json.loads(setup_path(name).read_text())
    return d, VF.Setup.from_dict(d["setup"])


def setup_hash(d):
    return hashlib.sha256(json.dumps(d["setup"], sort_keys=True).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------- falsification

def sampled_upper(cd, st, rho_max, n_dir=20000, seed=0):
    """Smallest |z|^2 (<= rho_max) found with dV >= 0 along random rays (float model), refined by
    bisection along the ray. An empirical upper bound on the certifiable level for this V."""
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n_dir, 6)); U /= np.linalg.norm(U, axis=1, keepdims=True)
    Minv = np.linalg.inv(st.W)
    E = U @ Minv.T
    def dV(Z, s):
        X = st.xs + s[:, None] * E[Z]
        Y = (step_np(X, cd) - st.xs) @ st.W.T
        return (Y ** 2).sum(1) - s ** 2
    R = math.sqrt(rho_max)
    radii = np.linspace(0, R, 201)[1:]
    first = np.full(n_dir, np.inf)
    idx = np.arange(n_dir)
    for r in radii:
        live = idx[np.isinf(first)]
        if len(live) == 0:
            break
        bad = dV(live, np.full(len(live), r)) >= 0
        first[live[bad]] = r
    if not np.any(np.isfinite(first)):
        return rho_max, None

    def crossing(u):
        """Smallest radius (bisection after a scan) with dV >= 0 along direction u."""
        u = u / np.linalg.norm(u)
        e = Minv @ u
        f = lambda s: float(((((step_np(st.xs + s * e, cd)[0] - st.xs) @ st.W.T) ** 2).sum()) - s * s)
        prev = 0.0
        for r in radii:
            if f(r) >= 0:
                lo, hi = prev, r
                for _ in range(50):
                    m = 0.5 * (lo + hi)
                    lo, hi = (lo, m) if f(m) >= 0 else (m, hi)
                return hi
            prev = r
        return R
    # local refinement of the best rays: minimise the crossing radius over the direction
    from scipy.optimize import minimize
    best_r, best_u = np.inf, None
    for k in np.argsort(first)[:5]:
        r = minimize(lambda v: crossing(v), U[k], method="Nelder-Mead",
                     options={"maxfev": 400, "xatol": 1e-4, "fatol": 1e-6})
        rk = crossing(r.x)
        if rk < best_r:
            best_r, best_u = rk, r.x / np.linalg.norm(r.x)
    return best_r ** 2, (best_r * best_u).tolist()


# ----------------------------------------------------------------------------- commands

def cmd_setup(a):
    cd = controller(a.name)
    eqs = find_equilibria(cd)
    info = {"name": a.name, "n_params": cd.n_params, "equilibria": []}
    stable = None
    for e in eqs:
        J = VF.jacobian_ad(cd, e)
        sr = float(np.max(np.abs(np.linalg.eigvals(J))))
        info["equilibria"].append({"x": float(e[0]), "spectral_radius": sr,
                                   "eig": [str(np.round(v, 6)) for v in np.linalg.eigvals(J)]})
        if sr < 1 and (stable is None or abs(e[0]) < abs(stable[0][0])):
            stable = (e, J, sr)
    print(json.dumps(info, indent=1))
    if stable is None:
        print("no locally stable equilibrium: nothing to certify")
        (OUT / a.name).mkdir(parents=True, exist_ok=True)
        setup_path(a.name).write_text(json.dumps(info, indent=1))
        return
    xs, J, sr = stable
    beta = max(BETA, 0.5 * (1 + sr))
    st = VF.lyapunov_setup(cd, xs, np.diag(1 / SCALES ** 2), beta=beta, A=J)
    pp = VF.prepare(cd, st)
    rs = VF.rho_safe(pp, LIMITS)
    ub, zc = sampled_upper(cd, st, rs)
    info.update({"xs": xs.tolist(), "spectral_radius": sr, "beta": beta,
                 "Q_scales": SCALES.tolist(), "rho_safe": rs, "rho_sampled_upper": ub,
                 "sampled_counterexample_z": zc,
                 "N_diag": np.diag(pp.data[7]).tolist(), "setup": st.to_dict()})
    info["setup_hash"] = setup_hash(info)
    (OUT / a.name).mkdir(parents=True, exist_ok=True)
    setup_path(a.name).write_text(json.dumps(info, indent=1))
    print(f"{a.name}: x*={xs[0]:.6f} sr={sr:.6f} rho_safe={rs:.4g} sampled upper={ub:.4g}")


def cmd_run(a):
    info, st = load_setup(a.name)
    cd = controller(a.name)
    res = VF.certify(cd, st, LIMITS, rho0=a.rho0, grid=a.grid, minw_rel=a.minw,
                     max_boxes=a.max_boxes, shard=(a.shard[0], a.shard[1]))
    d = res.to_dict()
    d.update({"name": a.name, "setup_hash": info["setup_hash"], "grid": a.grid, "minw": a.minw})
    out = Path(a.out) if a.out else OUT / a.name / f"shard_{a.shard[0]}_of_{a.shard[1]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, indent=1))


def cmd_combine(a):
    info, st = load_setup(a.name)
    rs = [json.loads(Path(f).read_text()) for f in a.files]
    n = rs[0]["shard"][1]
    shards = sorted(r["shard"][0] for r in rs)
    assert shards == list(range(n)), f"missing shards: {sorted(set(range(n)) - set(shards))}"
    assert all(r["setup_hash"] == info["setup_hash"] for r in rs), "setup mismatch"
    assert len({r["rho0"] for r in rs}) == 1 and len({r["eps"] for r in rs}) == 1
    rho = VF.combine(rs)
    tot = {k: sum(r["stats"][k] for r in rs) for k in rs[0]["stats"]}
    W = np.array(st.W)
    cert = {"name": a.name, "rho": rho, "rho0": rs[0]["rho0"], "eps": rs[0]["eps"],
            "complete": all(r["complete"] for r in rs), "shards": n, "stats": tot,
            "cpu_seconds": sum(r["seconds"] for r in rs), "setup_hash": info["setup_hash"],
            "volume": VF.ellipsoid_volume(W, rho) if rho > 0 else 0.0,
            "limiting_z": min(rs, key=lambda r: r["rho"])["worst_z"]}
    (OUT / a.name / "certificate.json").write_text(json.dumps(cert, indent=1))
    print(json.dumps(cert, indent=1))


def cmd_recheck(a):
    """Recompute the stored certificate from scratch (fresh trace, inverse, ball, B&B at the
    stored level)."""
    info, st = load_setup(a.name)
    cert = json.loads((OUT / a.name / "certificate.json").read_text())
    cd = controller(a.name)
    res = VF.certify(cd, st, LIMITS, rho0=cert["rho"], grid=a.grid, minw_rel=a.minw,
                     max_boxes=a.max_boxes)
    ok = res.complete and res.rho >= cert["rho"]
    print(f"recheck {a.name}: stored rho={cert['rho']:.6g} -> {'CONFIRMED' if ok else 'NOT CONFIRMED'}")
    sys.exit(0 if ok else 1)


def sample_ellipsoid(st, rho, n, rng, shell=0.5):
    """n points of {|W (x - xs)|^2 <= rho}: a fraction `shell` in the outer 1 % shell, the rest
    uniform in volume."""
    U = rng.normal(size=(n, 6)); U /= np.linalg.norm(U, axis=1, keepdims=True)
    r = rng.uniform(size=n) ** (1 / 6)
    k = int(shell * n)
    r[:k] = rng.uniform(0.99, 1.0, size=k)
    Z = U * (r * math.sqrt(rho))[:, None]
    return st.xs + Z @ np.linalg.inv(st.W).T


def cmd_sanity(a):
    from pendulum.policy import pack
    from pendulum.rollout import rollout, S_STEPS, S_REASON
    info, st = load_setup(a.name)
    cert = json.loads((OUT / a.name / "certificate.json").read_text())
    rho = cert["rho"] * (1 - 1e-9)
    rng = np.random.default_rng(a.seed)
    X = sample_ellipsoid(st, rho, a.n, rng)
    t0 = time.time()
    if a.name == "lqr":
        from pendulum.certify.model import simulate_lqr_batch
        cd = controller("lqr")
        task = replace(DEFAULT_TASK, duration=a.seconds, swing_up_time=0.0)
        okb = simulate_lqr_batch(cd.K, X, task.max_steps, cd, task.upright_tol)
        stats = np.zeros((len(X), 9)); stats[:, S_STEPS] = np.where(okb, task.max_steps, 0)
        stats[:, S_REASON] = np.where(okb, 0, 2)
    else:
        pol = policy_for(a.name)
        task = replace(DEFAULT_TASK, duration=a.seconds, swing_up_time=0.0,
                       **pol.meta.get("obs_scale", {}))
        stats, _ = rollout(pack([pol]), X, TARGET_PHYSICS, task)
        stats = stats[0]
    ok = (stats[:, S_STEPS] >= task.max_steps) & (stats[:, S_REASON] == 0)
    out = {"name": a.name, "rho": cert["rho"], "n": a.n, "seconds": a.seconds,
           "failures": int((~ok).sum()), "wall": time.time() - t0, "seed": a.seed}
    print(json.dumps(out))
    p = Path(a.out) if a.out else OUT / a.name / f"sanity_{a.seed}.json"
    p.write_text(json.dumps(out, indent=1))
    if not ok.all():
        sys.exit(1)


def _cert_rho(name):
    p = OUT / name / "certificate_actions.json"
    if not p.exists():
        p = OUT / name / "certificate.json"
    return json.loads(p.read_text())["rho"]


def _survives(name, X, seconds):
    """Bool per start: upright and on the track at every step for `seconds` (repository kernels)."""
    from pendulum.policy import pack
    from pendulum.rollout import rollout, S_STEPS, S_REASON
    if name == "lqr":
        from pendulum.certify.model import simulate_lqr_batch
        cd = controller("lqr")
        task = replace(DEFAULT_TASK, duration=seconds, swing_up_time=0.0)
        return simulate_lqr_batch(cd.K, X, task.max_steps, cd, task.upright_tol)
    pol = policy_for(name)
    task = replace(DEFAULT_TASK, duration=seconds, swing_up_time=0.0, **pol.meta.get("obs_scale", {}))
    st, _ = rollout(pack([pol]), X, TARGET_PHYSICS, task)
    return (st[0, :, S_STEPS] >= task.max_steps) & (st[0, :, S_REASON] == 0)


def cmd_empirical(a):
    """Monte-Carlo volume of the empirical balance region: states in the safe set (upright, on the
    track) with |velocities| inside a box, from which the closed loop stays safe for `seconds`."""
    info, st = load_setup(a.name)
    rng = np.random.default_rng(a.seed)
    lo = np.array([-2.8, np.pi - 0.2, -0.2, -a.vcart, -a.wpole, -a.wpole])
    hi = np.array([2.8, np.pi + 0.2, 0.2, a.vcart, a.wpole, a.wpole])
    X = rng.uniform(lo, hi, size=(a.n, 6))
    ok = _survives(a.name, X, a.seconds)
    box = float(np.prod(hi - lo))
    p = ok.mean()
    # does the box contain the region? success rate in the outer 10 % of each velocity range
    edge = {k: float(ok[np.abs(X[:, i]) > 0.9 * hi[i]].mean()) for k, i in (("xd", 3), ("w1", 4), ("w2", 5))}
    rho = _cert_rho(a.name)
    vc = VF.ellipsoid_volume(np.array(st.W), rho)
    out = {"name": a.name, "n": a.n, "seconds": a.seconds, "box_lo": lo.tolist(), "box_hi": hi.tolist(),
           "box_volume": box, "fraction": float(p), "volume": p * box,
           "volume_rel_se": float(np.sqrt((1 - p) / max(a.n * p, 1))),
           "edge_success": edge, "rho_cert": rho, "certified_volume": vc,
           "certified_over_empirical": vc / (p * box) if p > 0 else None}
    print(json.dumps(out, indent=1))
    (OUT / a.name / "empirical.json").write_text(json.dumps(out, indent=1))


def cmd_catch(a):
    """First entry of the 256 test swing-ups into the certified set. An episode is *certified to
    succeed* if it enters Omega_rho at step k and the simulated prefix (steps <= k) obeys the episode
    rules (on the track; upright at every step after the swing-up deadline)."""
    from pendulum.evaluation import test_ics
    from pendulum.rollout import simulate
    info, st = load_setup(a.name)
    rho = _cert_rho(a.name) * (1 - 1e-6)
    W = np.array(st.W); xs = np.array(st.xs)
    if a.name == "lqr":
        raise SystemExit("catch: the LQR has no swing-up of its own (use the classic controller)")
    pol = policy_for(a.name)
    task = replace(DEFAULT_TASK, duration=a.seconds, **pol.meta.get("obs_scale", {}))
    ics = test_ics(256, task)
    times, cert, succ = [], [], []
    for ic in ics:
        stats, traj = simulate(pol, ic, TARGET_PHYSICS, task)
        S = traj[:, :6]
        succ.append(len(S) - 1 >= task.max_steps)
        V = (((S - xs) @ W.T) ** 2).sum(1)
        inside = np.nonzero(V <= rho)[0]
        if len(inside) == 0:
            cert.append(False); continue
        k = int(inside[0])
        e1 = (S[1:k + 1, 1] - np.pi + np.pi) % (2 * np.pi) - np.pi
        e2 = (S[1:k + 1, 2] + np.pi) % (2 * np.pi) - np.pi
        up = (np.abs(e1) < task.upright_tol) & (np.abs(e2) < task.upright_tol)
        steps = np.arange(1, k + 1)
        ok = np.all(np.abs(S[1:k + 1, 0]) <= task.limit_x) and np.all(up | (steps <= task.swing_up_steps))
        cert.append(bool(ok))
        if ok:
            times.append(k * task.dt)
    out = {"name": a.name, "rho": rho, "n": len(ics), "simulated_success": float(np.mean(succ)),
           "certified_success": float(np.mean(cert)),
           "catch_time_median_s": float(np.median(times)) if times else None,
           "catch_time_p10_p90_s": [float(np.percentile(times, 10)), float(np.percentile(times, 90))] if times else None,
           "catch_time_max_s": float(np.max(times)) if times else None}
    print(json.dumps(out, indent=1))
    (OUT / a.name / "catch.json").write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("setup"); p.add_argument("name"); p.set_defaults(f=cmd_setup)
    p = sub.add_parser("run"); p.add_argument("name"); p.add_argument("--rho0", type=float, default=None)
    p.add_argument("--shard", type=int, nargs=2, default=[0, 1]); p.add_argument("--grid", type=int, default=4)
    p.add_argument("--minw", type=float, default=1e-3); p.add_argument("--max-boxes", type=int, default=10 ** 10)
    p.add_argument("--out", default=None); p.set_defaults(f=cmd_run)
    p = sub.add_parser("combine"); p.add_argument("name"); p.add_argument("files", nargs="+")
    p.set_defaults(f=cmd_combine)
    p = sub.add_parser("recheck"); p.add_argument("name"); p.add_argument("--grid", type=int, default=4)
    p.add_argument("--minw", type=float, default=1e-3); p.add_argument("--max-boxes", type=int, default=10 ** 10)
    p.set_defaults(f=cmd_recheck)
    p = sub.add_parser("sanity"); p.add_argument("name"); p.add_argument("--n", type=int, default=100000)
    p.add_argument("--seconds", type=float, default=60.0); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None); p.set_defaults(f=cmd_sanity)
    p = sub.add_parser("empirical"); p.add_argument("name"); p.add_argument("--n", type=int, default=1000000)
    p.add_argument("--seconds", type=float, default=60.0); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vcart", type=float, default=7.0); p.add_argument("--wpole", type=float, default=6.0)
    p.set_defaults(f=cmd_empirical)
    p = sub.add_parser("catch"); p.add_argument("name"); p.add_argument("--seconds", type=float, default=20.0)
    p.set_defaults(f=cmd_catch)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
