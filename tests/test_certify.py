"""Soundness tests for the balance certificate (pendulum/certify)."""
import math

import numpy as np
import pytest

from pendulum.certify import interval as I
from pendulum.certify import verify as VF
from pendulum.certify.model import (find_equilibria, from_lqr, from_policy, phi_gamma, step_np,
                                    step_ref, trace_closed_loop)
from pendulum.policy import GraphPolicy

from conftest import ROOT

MLP = ROOT / "results/experiments/revcurr_mlp4_s0_v12.json"
K_LQR = np.load(ROOT / "results/classic/controller.npz")["K_bal"]


@pytest.fixture(scope="module")
def mlp():
    return from_policy(GraphPolicy.load(MLP), name="mlp45")


@pytest.fixture(scope="module")
def mlp_setup(mlp):
    xs = find_equilibria(mlp)[-1]
    st = VF.lyapunov_setup(mlp, xs, np.diag(1 / np.array([1.8, .2, .2, 1, 1, 1]) ** 2), beta=0.999)
    return st, VF.prepare(mlp, st)


def _prog_step(cd, x):
    p = trace_closed_loop(cd)
    ops, cst, out = p.arrays()
    V = np.zeros(len(ops))
    I.run_prog_float(ops, cst, x, V)
    v = V[out]
    a = v[1:4] * v[0] + v[4:7]
    G = np.vstack([cd.dt ** 2 * np.eye(3), cd.dt * np.eye(3)])
    return phi_gamma(cd.dt) @ x + G @ a


def test_outward_rounding_at_least_nextafter():
    rng = np.random.default_rng(0)
    xs = np.concatenate([rng.normal(size=1000) * 10.0 ** rng.integers(-300, 300, 1000),
                         [0.0, 1.0, -1.0, 2.0 ** -1022, 5e-324, 1.5, 2.0 ** 52]])
    for x in xs:
        assert I.dn(x) <= np.nextafter(x, -np.inf)
        assert I.up(x) >= np.nextafter(x, np.inf)


def test_scalar_enclosures():
    rng = np.random.default_rng(1)
    for _ in range(2000):
        a = np.sort(rng.normal(size=2) * 3); b = np.sort(rng.normal(size=2) * 3)
        pa = rng.uniform(a[0], a[1], 20); pb = rng.uniform(b[0], b[1], 20)
        for f, F in [(I.imul, np.multiply), (I.iadd, np.add), (I.isub, np.subtract)]:
            lo, hi = f(a[0], a[1], b[0], b[1])
            v = F(pa, pb)
            assert np.all(lo <= v) and np.all(v <= hi)
        if b[0] > 0 or b[1] < 0:
            lo, hi = I.idiv(a[0], a[1], b[0], b[1])
            assert np.all(lo <= pa / pb) and np.all(pa / pb <= hi)
        for f, F in [(I.isin, np.sin), (I.icos, np.cos), (I.itanh, np.tanh)]:
            lo, hi = f(a[0], a[1])
            grid = np.linspace(a[0], a[1], 2001)
            assert np.all(lo <= F(grid)) and np.all(F(grid) <= hi)
        lo, hi = I.isqr(a[0], a[1])
        assert np.all(lo <= pa ** 2) and np.all(pa ** 2 <= hi)
    # extrema inside the interval
    assert I.icos(3.0, 3.3)[0] == -1.0 and I.isin(1.5, 1.7)[1] == 1.0


@pytest.mark.parametrize("kind", ["mlp", "lqr"])
def test_program_matches_physics_step(mlp, kind):
    cd = mlp if kind == "mlp" else from_lqr(K_LQR)
    rng = np.random.default_rng(2)
    for _ in range(200):
        x = np.array([0, math.pi, 0, 0, 0, 0]) + rng.normal(size=6) * [1, .3, .3, 1, 2, 2]
        ref = step_ref(x, cd)                     # rollout.physics_step
        assert np.max(np.abs(step_np(x, cd)[0] - ref)) < 1e-12
        assert np.max(np.abs(_prog_step(cd, x) - ref)) < 1e-12


def test_episode_kernel_matches_float_model(mlp):
    """The certified map is the map the repository's episode kernel iterates."""
    from dataclasses import replace
    from pendulum.rollout import simulate
    from pendulum.task import DEFAULT_TASK, TARGET_PHYSICS
    pol = GraphPolicy.load(MLP)
    task = replace(DEFAULT_TASK, **pol.meta["obs_scale"], duration=1.0, swing_up_time=0.0)
    x0 = find_equilibria(mlp)[-1] + np.array([0.05, 0.02, -0.01, 0.1, 0.0, 0.05])
    _, traj = simulate(pol, x0, TARGET_PHYSICS, task)
    x = x0.copy()
    for k in range(1, len(traj)):
        x = step_np(x, mlp)[0]
        assert np.max(np.abs(x - traj[k, :6])) < 1e-9


@pytest.mark.parametrize("kind", ["mlp", "lqr"])
def test_program_range_and_slope_enclosure(mlp, kind):
    """Ranges and slopes (incl. the mean-value refined version) contain sampled values and
    finite-difference derivatives; the LQR case exercises the saturation (clip) slopes."""
    cd = mlp if kind == "mlp" else from_lqr(K_LQR)
    p = trace_closed_loop(cd)
    ops, cst, out = p.arrays()
    n = len(ops)
    V = np.zeros((n, 2)); VC = np.zeros((n, 2)); G = np.zeros((n, 6, 2)); Vf = np.zeros(n)
    rng = np.random.default_rng(3)
    S = np.linalg.qr(rng.normal(size=(6, 6)))[0] * 0.05
    E = np.eye(6)
    for t in range(60):
        c = np.array([1.0, math.pi, 0, 0, 0, 0]) + rng.normal(size=6) * [.3, .05, .05, .2, .3, .3]
        if kind == "lqr":
            c[1] += 0.1 * (t % 3 - 1)            # some boxes saturate, some straddle
        h = np.abs(rng.normal(size=6)) * 0.01
        lo = np.array([I.dn(v) for v in c - S.__abs__() @ h]); hi = c + np.abs(S) @ h
        I.run_prog_mv(ops, cst, lo, hi, c, c, h, V, VC, G, S, S)
        for _ in range(30):
            tvec = rng.uniform(-1, 1, 6) * h
            x = c + S @ tvec
            I.run_prog_float(ops, cst, x, Vf)
            assert np.all(V[out, 0] <= Vf[out]) and np.all(Vf[out] <= V[out, 1])
            # directional finite differences of the outputs w.r.t. t
            for j in range(6):
                d = 1e-7 * h[j] / max(h[j], 1e-12)
                xp = x + S[:, j] * d; xm = x - S[:, j] * d
                Vp = np.zeros(n); Vm = np.zeros(n)
                I.run_prog_float(ops, cst, xp, Vp); I.run_prog_float(ops, cst, xm, Vm)
                fd = (Vp[out] - Vm[out]) / (2 * d)
                tol = 1e-5 * (1 + np.abs(fd))
                assert np.all(G[out, j, 0] - tol <= fd) and np.all(fd <= G[out, j, 1] + tol)


def test_exact_inverse_and_pd():
    rng = np.random.default_rng(4)
    W = rng.normal(size=(6, 6)) * 10.0 ** rng.integers(-2, 3, (6, 6))
    lo, hi = VF.exact_inverse_enclosure(W)
    inv = np.linalg.inv(W)
    assert np.all(lo <= hi) and np.all(np.abs(inv - 0.5 * (lo + hi)) <= 1e-8 * (1 + np.abs(inv)))
    assert VF.exact_spectral_norm_bound(np.eye(3) * 0.5, 0.5000001)
    assert not VF.exact_spectral_norm_bound(np.eye(3) * 0.5, 0.4999999)


def test_eval_box_bounds_sampled_decrease(mlp, mlp_setup):
    """The box bound is an upper bound of the true one-step change of V (float model)."""
    st, pp = mlp_setup
    n = len(pp.ops)
    V = np.zeros((n, 2)); VC = np.zeros((n, 2)); G = np.zeros((n, 6, 2)); res = np.zeros(8)
    Minv = np.linalg.inv(st.W)
    rng = np.random.default_rng(5)
    for _ in range(100):
        u = rng.normal(size=6)
        c = u / np.linalg.norm(u) * rng.uniform(0.05, 2.0)
        h = rng.uniform(0.005, 0.1)
        VF.eval_box(pp.data, c - h, c + h, V, VC, G, res)
        Z = c + rng.uniform(-h, h, (200, 6))
        X = st.xs + Z @ Minv.T
        Y = (step_np(X, mlp) - st.xs) @ st.W.T
        dV = (Y ** 2).sum(1) - (Z ** 2).sum(1)
        assert np.all(dV <= res[0] + 1e-12 * (1 + np.abs(dV)))
        dVc = ((((step_np(st.xs + Minv @ c, mlp)[0] - st.xs) @ st.W.T) ** 2).sum() - c @ c)
        assert res[1] <= dVc + 1e-12


def test_certificate_small_level(mlp, mlp_setup):
    """End-to-end: a small level is certified; then the claim is checked by simulation."""
    st, _ = mlp_setup
    res = VF.certify(mlp, st, (2.8, 0.2), rho0=0.05, grid=2, verbose=False)
    assert res.complete and res.rho == pytest.approx(0.05)
    rng = np.random.default_rng(6)
    U = rng.normal(size=(500, 6)); U /= np.linalg.norm(U, axis=1, keepdims=True)
    X = st.xs + (U * math.sqrt(res.rho) * 0.999) @ np.linalg.inv(st.W).T
    V0 = res.rho * 0.999 ** 2
    for k in range(300):
        X = step_np(X, mlp)
        Vk = (((X - st.xs) @ st.W.T) ** 2).sum(1)
        assert np.all(Vk <= V0 * (1 + 1e-9))


def test_certificate_rejects_too_large_level(mlp, mlp_setup):
    """Around a point where V does not decrease, the branch and bound must lower the level below
    that point's |z|^2 (here on one small initial box containing the point)."""
    st, pp = mlp_setup
    Minv = np.linalg.inv(st.W)
    rng = np.random.default_rng(7)
    found = None
    while found is None:
        u = rng.normal(size=6); u /= np.linalg.norm(u)
        for s in np.linspace(0.5, 3.0, 26):
            z = s * u
            y = (step_np(st.xs + Minv @ z, mlp)[0] - st.xs) @ st.W.T
            if y @ y - z @ z > 1e-3:
                found = z
                break
    r2 = float(found @ found)
    eps = VF.largest_ball(pp, 0.5)
    BL = (found - 0.02)[None, :].copy(); BH = (found + 0.02)[None, :].copy()
    rhos, stats, worst = VF.bnb_many(pp.data, BL, BH, r2 * 1.1, eps * eps, 1e-3, 1e-6, 2_000_000)
    assert 0 <= rhos[0] < r2
    assert stats[0, 5] > 0          # proven counterexample(s) at box centers


def test_lqr_batch_sim_matches_classic_rollout():
    from dataclasses import replace
    from pendulum.classic import ClassicController, rollout_controller
    from pendulum.certify.model import simulate_lqr_batch
    from pendulum.rollout import S_STEPS
    from pendulum.task import DEFAULT_TASK, TARGET_PHYSICS
    ctrl = ClassicController.load(ROOT / "results/classic/controller.npz")
    ctrl.u_nom = ctrl.u_nom[:0]
    task = replace(DEFAULT_TASK, duration=3.0, swing_up_time=0.0)
    rng = np.random.default_rng(8)
    X = np.array([0, math.pi, 0, 0, 0, 0]) + rng.normal(size=(40, 6)) * [.3, .08, .08, .5, .8, .8]
    ref = rollout_controller(ctrl, X, TARGET_PHYSICS, task)[:, S_STEPS] >= task.max_steps
    ok = simulate_lqr_batch(ctrl.K_bal, X, task.max_steps, from_lqr(K_LQR), task.upright_tol)
    assert ref.any() and (~ref).any()
    assert np.array_equal(ok, ref)
