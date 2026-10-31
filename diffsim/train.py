"""Training methods for the differentiable-simulation study (results/diffsim/README.md, steps 4-6).

  python diffsim/train.py METHOD ARCH SEED BUDGET_B [--hp JSON] [--out DIR] [--no-test]

Methods (all start from the same CMA-ES initialisation, build_arch(arch, 9, default_rng(seed))):
  cmaes        start-state curriculum + CMA-ES on the sparse score (experiments/reverse_curriculum.py)
  bptt_curr    same curriculum; Adam on the analytic gradient of surrogate (b), a terminal-state
               cost at the end of a segment of L steps starting at the curriculum level, plus the
               task's own rule (stay within the upright tolerance after the level's deadline) as a
               soft constraint
  zo_curr      same curriculum and surrogate (b); antithetic randomized-smoothing gradient + Adam
  bptt_full    full BPTT (4 s rollouts from the task's starts) on surrogate (a), the smoothed sparse
               score (sigmoid upright indicator and streak), temperature annealed toward the true score
  bptt_trunc   as bptt_full, gradient truncated every W steps (stop_gradient on the carried state)
  bptt_dense   DENSE-REWARD REFERENCE ONLY: full BPTT on surrogate (c), energy + upright cost
  hybrid       cmaes for a fraction `switch` of the budget, then bptt_curr from the CMA-ES mean,
               continuing at the curriculum level CMA-ES reached

Budget: env-step equivalents. A forward step counts 1 (numba or JAX); a step that is also
backpropagated counts `ratio`, the measured wall-time ratio (forward+backward) / forward of this
architecture on this machine (measured at start-up). Mastery checks count; validation (model
selection) and test do not, as in experiments/reverse_curriculum.py.

Success is only ever measured by the numba kernel with the task's exact sparse criterion. Model
selection: 64 validation starts. Test: 256 test starts at 20 s and 60 s (velocity scale 12).
"""
import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numba  # noqa: E402
import numpy as np  # noqa: E402

numba.set_num_threads(1)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

from pendulum import jax_sim as J  # noqa: E402
from pendulum.classic import ClassicController  # noqa: E402
from pendulum.evaluation import evaluate_policies, test_ics, val_ics  # noqa: E402
from pendulum.fitness import policy_scores  # noqa: E402
from pendulum.jax_objectives import soft_upright, state_cost  # noqa: E402
from pendulum.methods.es import CMAES  # noqa: E402
from pendulum.policy import build_arch, pack  # noqa: E402
from pendulum.rollout import L1, L2, M1, M2, S_STEPS, rollout  # noqa: E402
from pendulum.task import DEFAULT_TASK, TARGET_PHYSICS as P  # noqa: E402

T = replace(DEFAULT_TASK, max_v_cart=12.0, max_w_pole=12.0)   # velocity scale 12 throughout
N_UP = 210            # the classic nominal is upright at rest from step ~210
STEP = 5              # curriculum step (control steps)
GOAL = np.array([0.0, np.pi, 0.0, 0.0, 0.0, 0.0])
VAL_EVERY = 20e6      # budget units between validations (5e6 once the real task is reached)
PATIENCE = 50e6       # gradient methods: back off a level after this many units without any full success
CHECK_EVERY = 2e6     # gradient methods: mastery check interval (budget units)

DEFAULT_HP = {
    "cmaes": {"popsize": 64, "sigma0": 0.5},
    "bptt_curr": {"lr": 1e-2, "L": 50, "B": 16, "clip": 1.0},
    "zo_curr": {"lr": 1e-2, "L": 50, "pairs": 32, "sigma": 0.03, "clip": 1.0},
    "bptt_full": {"lr": 1e-2, "B": 16, "H": 400, "temp0": 0.3, "temp1": 0.002, "clip": 1.0},
    "bptt_trunc": {"lr": 1e-2, "B": 16, "H": 400, "W": 50, "temp0": 0.3, "temp1": 0.002, "clip": 1.0},
    "bptt_dense": {"lr": 1e-2, "B": 16, "H": 400, "w_energy": 1.0, "clip": 1.0},
    "hybrid": {"popsize": 64, "sigma0": 0.5, "switch": 0.5, "lr": 3e-3, "L": 50, "B": 16, "clip": 1.0},
}


# =============================================================================== bookkeeping
class Budget:
    def __init__(self, total, ratio):
        self.total = total; self.ratio = ratio
        self.used = 0.0; self.fwd = 0; self.bwd = 0; self.t0 = time.time()

    def forward(self, n):
        self.used += n; self.fwd += int(n)

    def gradient(self, n):
        """n env steps run forward and backpropagated."""
        self.used += n * self.ratio; self.bwd += int(n)

    @property
    def done(self):
        return self.used >= self.total

    def wall(self):
        return time.time() - self.t0


class Curriculum:
    """Start-state curriculum along the classic expert's nominal swing-up (reverse_curriculum.py)."""

    def __init__(self, rng):
        self.ex = ClassicController.load(ROOT / "results/classic/controller.npz")
        self.rng = rng
        self.k = N_UP
        self.k_min = N_UP
        self.since_success = 0.0

    @staticmethod
    def level_task(kk):
        return replace(T, swing_up_time=max(0.0, T.swing_up_time - kk * T.dt), duration=T.duration - kk * T.dt)

    def starts(self, kk, n):
        return self.ex.S_nom[kk] + self.rng.uniform(-T.ic_noise, T.ic_noise, (n, 6))

    def advance(self):
        self.k = max(0, self.k - STEP); self.k_min = min(self.k_min, self.k); self.since_success = 0.0

    def back_off(self):
        self.k = min(N_UP, self.k + STEP); self.since_success = 0.0

    def mastery(self, policy, budget):
        """16 fresh starts at the current level; advance if >= 80 % fully succeed. Returns the rate."""
        tk = self.level_task(self.k)
        st, n = rollout(pack([policy]), self.starts(self.k, 16), P, tk)
        budget.forward(n)
        full = float(np.mean(st[0, :, S_STEPS] >= tk.max_steps))
        if self.k > 0 and full >= 0.8:
            self.advance()
        return full


class Selector:
    """Model selection on 64 validation starts (exact sparse criterion); early stop at >= 99 %."""

    def __init__(self, budget, log):
        self.budget = budget; self.log = log
        self.best = (-1.0, -1.0); self.best_policy = None; self.best_at = 0.0
        self.first_success = None; self.next_val = 0.0
        self.vics = val_ics(64)

    @property
    def solved(self):
        return self.best[0] >= 0.99

    def maybe(self, policies, info, force=False):
        level = info.get("level")
        if not force and self.budget.used < self.next_val:
            return
        self.next_val = self.budget.used + (5e6 if level == 0 else VAL_EVERY)
        res = evaluate_policies(policies, self.vics, P, T)
        for p, r in zip(policies, res):
            key = (r["success_rate"], r["mean_streak_s"])
            if key > self.best:
                self.best = key; self.best_policy = p; self.best_at = self.budget.used
            if r["success_rate"] > 0 and self.first_success is None:
                self.first_success = self.budget.used
        rec = {"units": self.budget.used, "fwd": self.budget.fwd, "bwd": self.budget.bwd,
               "wall": round(self.budget.wall(), 1), "val": max(r["success_rate"] for r in res),
               "val_streak": max(r["mean_streak_s"] for r in res), "best_val": self.best[0], **info}
        self.log.append(rec)
        print(json.dumps(rec), flush=True)


# =============================================================================== CMA-ES
def run_cmaes(arch, seed, hp, budget, cur, sel, stop_at=None):
    """experiments/reverse_curriculum.py with the shared budget/selection bookkeeping.
    Returns the CMA-ES object (for the hybrid)."""
    rng = cur.rng
    np.random.seed(seed)
    m = CMAES({"arch": arch, "popsize": hp["popsize"], "sigma0": hp["sigma0"]}, 9, np.random.default_rng(seed))
    K = 8; stuck = 0; g = 0
    stop_at = budget.total if stop_at is None else stop_at
    while budget.used < stop_at and not sel.solved:
        g += 1
        k = cur.k
        k_replay = int(rng.integers(k, N_UP + 1))
        pols = m.ask(); pk = pack(pols)
        sc = np.zeros(len(pols)); st_cur = None
        for kk in (k, k_replay):
            tk = cur.level_task(kk)
            st, n = rollout(pk, cur.starts(kk, K // 2), P, tk); budget.forward(n)
            sc += 0.5 * policy_scores(st, "streak_smooth", tk) / tk.duration
            if kk == k:
                st_cur = st
        m.tell(sc, sc, st_cur)
        ib = int(np.argmax(sc[:-1]))
        full_best = np.mean(st_cur[ib, :, S_STEPS] >= cur.level_task(k).max_steps)
        stuck = 0 if full_best > 0 else stuck + 1
        if k > 0 and g % 5 == 0:
            cur.mastery(m.champions()[0], budget)
            if cur.k != k:
                stuck = 0
        if stuck >= 300 and cur.k < N_UP:
            cur.back_off(); stuck = 0
        sel.maybe(m.champions(), {"level": cur.k, "level_min": cur.k_min, "gen": g, "phase": "cmaes"})
    return m


# =============================================================================== JAX surrogates
def seg_cost(s, target):
    """Surrogate (b): terminal error to the target state. Angles enter through the chord length
    2 sin(d/2) (smooth, periodic); velocities in rad/s and m/s."""
    e_x = s[0] - target[0]
    e1 = 2.0 * jnp.sin((s[1] - target[1]) / 2.0); e2 = 2.0 * jnp.sin((s[2] - target[2]) / 2.0)
    ev = s[3:] - target[3:]
    return e1 ** 2 + e2 ** 2 + 0.25 * e_x ** 2 + 0.02 * jnp.sum(ev ** 2)


def track_penalty(S):
    return jnp.mean(jax.nn.relu(jnp.abs(S[..., 0]) - 2.6) ** 2)


CHORD_TOL = (2.0 * np.sin(0.15 / 2.0)) ** 2     # inside the task's 0.2 rad tolerance, with margin


def deadline_penalty(S, dl):
    """The task's termination rule as a soft constraint: after the level's swing-up deadline
    (step dl of the segment) both angle errors must stay within tolerance. Squared chord excess."""
    c1 = (2.0 * jnp.sin((S[:, 1] - jnp.pi) / 2.0)) ** 2; c2 = (2.0 * jnp.sin(S[:, 2] / 2.0)) ** 2
    after = (jnp.arange(S.shape[0]) + 1) > dl
    return jnp.mean(after * (jax.nn.relu(c1 - CHORD_TOL) + jax.nn.relu(c2 - CHORD_TOL)))


def make_balance_loss(st, H):
    """Hold term of surrogate (b): from the first curriculum level (upright at rest +- noise), mean error to
    upright over the first h of H steps; h starts at 50 and grows x1.5 whenever the hold cost is < 0.05. A 0.5 s segment cannot see a slow unstable mode of the balance loop; near upright a
    stabilising controller has well-behaved gradients (diagnostics, section 3)."""
    goal = jnp.asarray(GOAL)

    def loss1(theta, s0, h):
        """Mean error over the first h <= H steps (h grows during training: a hold-horizon curriculum)."""
        S, _ = J.rollout(st, theta, s0, H, P, T)
        w = (jnp.arange(H) < h) / h
        return jnp.sum(w * jax.vmap(lambda s: seg_cost(s, goal))(S)) + track_penalty(S[:50])
    return loss1


def make_seg_loss(st, L, w_deadline):
    """Surrogate (b): terminal cost at the segment end + the deadline constraint + track limit."""
    def loss1(theta, s0, target, dl):
        S, _ = J.rollout(st, theta, s0, L, P, T)
        return seg_cost(S[-1], target) + w_deadline * deadline_penalty(S, dl) + track_penalty(S)
    return loss1


def soft_score(st, H, W, swing):
    """Surrogate (a): -mean over steps after the swing-up deadline of the soft upright streak,
    streak_k = u_k (streak_{k-1} + dt), u = sigmoid upright indicator x sigmoid on-track indicator.
    As temp -> 0 the streak is the task's exact consecutive-upright counter. W: truncation window."""
    def loss1(theta, s0, temp):
        wb = J.unflatten(st, theta)

        def body(carry, k):
            s, streak = carry
            if W:
                cut = (k % W) == 0
                s = jnp.where(cut, jax.lax.stop_gradient(s), s)
                streak = jnp.where(cut, jax.lax.stop_gradient(streak), streak)
            a = J.forward(st, wb, J.observe(s, T))
            s2 = J.physics_step(s, a * P.max_force, P.g, P.drag, T.dt)
            u = soft_upright(s2, temp, T) * jax.nn.sigmoid((T.limit_x - jnp.abs(s2[0])) / 0.05)
            streak = u * (streak + T.dt)
            return (s2, streak), streak
        _, streaks = jax.lax.scan(body, (s0, jnp.zeros(())), jnp.arange(H))
        return -jnp.mean(streaks[swing:])
    return loss1


E_UP = P.g * (M1 * L1 + M2 * (L1 + L2))


def energy(s):
    """Mechanical energy of the two masses (potential relative to the pivot height, kinetic incl. cart motion)."""
    x, t1, t2, xd, w1, w2 = s[0], s[1], s[2], s[3], s[4], s[5]
    v1x = xd - L1 * jnp.cos(t1) * w1; v1z = L1 * jnp.sin(t1) * w1
    v2x = v1x - L2 * jnp.cos(t1 + t2) * (w1 + w2); v2z = v1z + L2 * jnp.sin(t1 + t2) * (w1 + w2)
    h1 = -L1 * jnp.cos(t1); h2 = h1 - L2 * jnp.cos(t1 + t2)
    return 0.5 * M1 * (v1x ** 2 + v1z ** 2) + 0.5 * M2 * (v2x ** 2 + v2z ** 2) + P.g * (M1 * h1 + M2 * h2)


def dense_loss(st, H, w_energy):
    """Surrogate (c), DENSE REFERENCE: mean over the rollout of energy error^2 + upright state cost."""
    def loss1(theta, s0):
        S, _ = J.rollout(st, theta, s0, H, P, T)
        e = jax.vmap(energy)(S)
        return jnp.mean(w_energy * ((e - E_UP) / E_UP) ** 2 + state_cost(S, T)) + track_penalty(S)
    return loss1


def measure_ratio(st, n_par, reps=20):
    """Wall-time ratio (forward+backward)/forward for 16 starts x 100 steps of this architecture."""
    S0 = jnp.asarray(np.random.default_rng(0).uniform(-0.05, 0.05, (16, 6)))
    th = jnp.zeros(n_par) + 0.1

    def f(theta):
        S, _ = J.rollout_batch(st, theta, S0, 100, P, T)
        return jnp.mean(state_cost(S, T))
    out = {}
    for name, fn in (("fwd", jax.jit(f)), ("bwd", jax.jit(jax.value_and_grad(f)))):
        jax.block_until_ready(fn(th)); ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); jax.block_until_ready(fn(th)); ts.append(time.perf_counter() - t0)
        out[name] = float(np.median(ts))
    return out["bwd"] / out["fwd"]


# =============================================================================== gradient methods
def run_gradient(method, template, theta0, hp, budget, cur, sel, rng):
    st = J.GraphStructure.of(template)
    bound = jnp.asarray(template.flat_bounds())
    opt = optax.chain(optax.clip_by_global_norm(hp["clip"]), optax.adam(hp["lr"]))
    theta = jnp.clip(jnp.asarray(theta0), -bound, bound)
    ostate = opt.init(theta)
    to_pol = lambda th: template.with_flat(np.asarray(th))  # noqa: E731
    curriculum = method in ("bptt_curr", "zo_curr", "hybrid")
    swing = T.swing_up_steps

    if curriculum:
        L = hp["L"]
        loss1 = make_seg_loss(st, L, hp.get("w_deadline", 10.0))
        n_bal = hp.get("B_bal", 8); H_bal = hp.get("H_bal", 300)
        bal1 = make_balance_loss(st, H_bal)
        bal_eval = jax.jit(lambda th, Sb, h: jnp.mean(jax.vmap(bal1, (None, 0, None))(th, Sb, h)))
        h_bal = 50.0
        if method == "zo_curr":
            n_bal = n_b                    # each antithetic pair evaluates one segment + one balance rollout
            n_b = hp["pairs"]; sig = hp["sigma"]

            @jax.jit
            def grad_fn(theta, S0, tg, dl, Sb, h, key):
                # one perturbation per (segment start, balance start) pair: the same objective as bptt_curr
                eps = jax.random.normal(key, (n_b, theta.shape[0]))
                Sb = jnp.tile(Sb, (n_b // Sb.shape[0] + 1, 1))[:n_b]
                f = lambda th, s0, t, d, sb: loss1(th, s0, t, d) + bal1(th, sb, h)  # noqa: E731
                fv = jax.vmap(f)
                jp = fv(theta[None] + sig * eps, S0, tg, dl, Sb); jm = fv(theta[None] - sig * eps, S0, tg, dl, Sb)
                return jnp.mean(0.5 * (jp + jm)), jnp.mean(((jp - jm) / (2 * sig))[:, None] * eps, 0)
        else:
            n_b = hp["B"]
            grad_fn = jax.jit(jax.value_and_grad(
                lambda th, S0, tg, dl, Sb, h: jnp.mean(jax.vmap(loss1, (None, 0, 0, 0))(th, S0, tg, dl))
                + jnp.mean(jax.vmap(bal1, (None, 0, None))(th, Sb, h))))
        cost_per_it = n_b * L + n_bal * H_bal
    else:
        H = hp["H"]; n_b = hp["B"]
        if method == "bptt_dense":
            l1 = dense_loss(st, H, hp["w_energy"])
            grad_fn = jax.jit(jax.value_and_grad(lambda th, S0, temp: jnp.mean(jax.vmap(l1, (None, 0))(th, S0))))
        else:
            l1 = soft_score(st, H, hp.get("W") if method == "bptt_trunc" else None, swing)
            grad_fn = jax.jit(jax.value_and_grad(lambda th, S0, temp: jnp.mean(jax.vmap(l1, (None, 0, None))(th, S0, temp))))
        cost_per_it = n_b * H

    @jax.jit
    def update(theta, ostate, g):
        upd, ostate = opt.update(g, ostate, theta)
        return jnp.clip(optax.apply_updates(theta, upd), -bound, bound), ostate

    key = jax.random.PRNGKey(int(rng.integers(2**31)))
    it = 0; next_check = budget.used + CHECK_EVERY; start_used = budget.used; loss_ema = None
    while not budget.done and not sel.solved:
        it += 1
        if curriculum:
            k = cur.k
            kk = np.where(np.arange(n_b) < n_b // 2, k, rng.integers(k, N_UP + 1, n_b))
            S0 = cur.ex.S_nom[kk] + rng.uniform(-T.ic_noise, T.ic_noise, (n_b, 6))
            tidx = kk + L
            tg = np.where((tidx >= N_UP)[:, None], GOAL[None], cur.ex.S_nom[np.minimum(tidx, len(cur.ex.S_nom) - 1)])
            dl = jnp.asarray(np.maximum(0, swing - kk))     # segment step after which the poles must stay upright
            Sb = jnp.asarray(cur.starts(N_UP, n_bal))
            if method == "zo_curr":
                key, sk = jax.random.split(key)
                loss, g = grad_fn(theta, jnp.asarray(S0), jnp.asarray(tg), dl, Sb, h_bal, sk)
                budget.forward(2 * cost_per_it)
            else:
                loss, g = grad_fn(theta, jnp.asarray(S0), jnp.asarray(tg), dl, Sb, h_bal)
                budget.gradient(cost_per_it)
        else:
            frac = (budget.used - start_used) / max(budget.total - start_used, 1.0)
            temp = hp.get("temp0", 1.0) * (hp.get("temp1", 1.0) / hp.get("temp0", 1.0)) ** frac
            S0 = rng.uniform(-T.ic_noise, T.ic_noise, (n_b, 6))
            loss, g = grad_fn(theta, jnp.asarray(S0), temp)
            budget.gradient(cost_per_it)
        if bool(jnp.all(jnp.isfinite(g))):
            theta, ostate = update(theta, ostate, g)
        lf = float(loss)
        loss_ema = lf if loss_ema is None else 0.99 * loss_ema + 0.01 * lf
        info = {"phase": method, "it": it, "loss": round(loss_ema, 5)}
        if curriculum:
            if budget.used >= next_check:
                next_check = budget.used + CHECK_EVERY
                k_before = cur.k
                full = cur.mastery(to_pol(theta), budget)
                if h_bal < H_bal:
                    Sb = jnp.asarray(cur.starts(N_UP, 16))
                    budget.forward(16 * H_bal)
                    if float(bal_eval(theta, Sb, h_bal)) < 0.05:
                        h_bal = min(float(H_bal), 1.5 * h_bal)
                cur.since_success = 0.0 if (full > 0 or cur.k != k_before) else cur.since_success + CHECK_EVERY
                if cur.since_success >= PATIENCE and cur.k < N_UP:
                    cur.back_off()
            info.update(level=cur.k, level_min=cur.k_min, h_bal=h_bal)
        else:
            info["temp"] = round(float(temp), 5) if method != "bptt_dense" else None
        sel.maybe([to_pol(theta)], info)
    sel.maybe([to_pol(theta)], {**info, "final": True}, force=True)
    sel.final_policy = to_pol(theta)
    return theta


# =============================================================================== Drake check
def drake_check(policy, n=16):
    """Closed-loop replay in PyDrake on the first n test starts; same episode rules (20 s)."""
    from pendulum.drake_env import DrakeEnv
    from pendulum.task import observe
    env = DrakeEnv(P, T)
    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi  # noqa: E731
    ok = 0
    for ic in test_ics(n):
        s = env.reset(ic); step = policy.make_stepper(); good = True
        for k in range(1, T.max_steps + 1):
            a = float(np.clip(np.ravel(step(observe(s, T)))[0], -1, 1))
            s = env.step(a)
            up = abs(wrap(s[1] - np.pi)) < T.upright_tol and abs(wrap(s[2])) < T.upright_tol
            if abs(s[0]) > T.limit_x or (k > T.swing_up_steps and not up):
                good = False; break
        ok += good
    return ok, n


# =============================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method"); ap.add_argument("arch"); ap.add_argument("seed", type=int)
    ap.add_argument("budget_b", type=float)
    ap.add_argument("--hp", default="{}", help="JSON overrides of DEFAULT_HP[method]")
    ap.add_argument("--out", default="runs/diffsim")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-test", action="store_true", help="tuning runs: validation only")
    args = ap.parse_args()
    hp = {**DEFAULT_HP[args.method], **json.loads(args.hp)}
    name = f"{args.method}_{args.arch.replace(':', '')}_s{args.seed}" + (f"_{args.tag}" if args.tag else "")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    template = build_arch(args.arch, 9, np.random.default_rng(args.seed))
    needs_grad = args.method not in ("cmaes", "zo_curr")
    ratio = measure_ratio(J.GraphStructure.of(template), template.n_flat) if needs_grad else None
    budget = Budget(args.budget_b * 1e9, ratio or 1.0)
    budget.t0 = time.time()
    log = []
    cur = Curriculum(rng)
    sel = Selector(budget, log)
    print(f"{name}: hp {hp} ratio {ratio}", flush=True)

    sel.final_policy = None
    if args.method == "cmaes":
        sel.final_policy = run_cmaes(args.arch, args.seed, hp, budget, cur, sel).champions()[0]
    elif args.method == "hybrid":
        m = run_cmaes(args.arch, args.seed, hp, budget, cur, sel, stop_at=hp["switch"] * budget.total)
        sel.final_policy = m.champions()[0]
        if not sel.solved:
            run_gradient("hybrid", template, np.asarray(m.es.mean),
                         hp, budget, cur, sel, rng)
    else:
        run_gradient(args.method, template, template.get_flat(), hp, budget, cur, sel, rng)

    res = {"name": name, "method": args.method, "arch": args.arch, "seed": args.seed, "hp": hp,
           "ratio": ratio, "budget": budget.total, "used": budget.used, "fwd_steps": budget.fwd,
           "bwd_steps": budget.bwd, "wall_s": round(budget.wall(), 1),
           "level_min": cur.k_min if args.method in ("cmaes", "bptt_curr", "zo_curr", "hybrid") else None,
           "level_min_s": cur.k_min * T.dt if args.method in ("cmaes", "bptt_curr", "zo_curr", "hybrid") else None,
           "best_val": sel.best[0], "best_val_streak": sel.best[1], "units_at_best": sel.best_at,
           "units_first_val_success": sel.first_success, "params": template.n_params,
           "machine": {"cpus": os.cpu_count(), "affinity": len(os.sched_getaffinity(0))}}
    if sel.final_policy is not None:
        sel.final_policy.save(out / f"{name}.final.json")
    pol = sel.best_policy
    if pol is not None:
        pol.meta["obs_scale"] = {"max_v_cart": T.max_v_cart, "max_w_pole": T.max_w_pole}
        pol.meta["diffsim"] = {"method": args.method, "seed": args.seed}
        pol.save(out / f"{name}.policy.json")
        if not args.no_test:
            ics = test_ics(256)
            t20 = evaluate_policies([pol], ics, P, T)[0]
            t60 = evaluate_policies([pol], ics, P, replace(T, duration=60.0))[0]
            res["test_20s"] = t20["success_rate"]; res["test_60s"] = t60["success_rate"]
            res["test_20s_mean_streak"] = t20["mean_streak_s"]
            # success per billion env-step equivalents spent until the selected policy was found
            res["success_per_Bstep"] = t20["success_rate"] / max(sel.best_at / 1e9, 1e-9)
    with open(out / f"{name}.json", "w") as f:
        json.dump(res, f, indent=1)
    with open(out / f"{name}.log.jsonl", "w") as f:
        f.write("\n".join(json.dumps(r) for r in log))
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
