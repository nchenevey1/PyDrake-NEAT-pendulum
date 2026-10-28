"""
Classic model-based baseline: trajectory optimization + time-varying LQR.

* Swing-up: direct transcription of the exact discrete dynamics of
  `rollout.physics_step` (solved with SNOPT via pydrake) gives a nominal force sequence
  from hanging rest to exactly upright at rest before the swing-up deadline, with
  |force| <= `u_frac` * max_force and |x| <= 2.3 m.
* Tracking: time-varying LQR along the nominal trajectory.
* Balance: infinite-horizon discrete LQR about the upright equilibrium.

The controller sees only the task's 9-D observation (it reconstructs the state from
it) and keeps an internal step counter (the time-varying part), so it is evaluated by
exactly the same episode rules and metrics as the neural policies. Its "size" is the
stored nominal trajectory + gains, reported separately from neural parameter counts.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .rollout import (N_STATS, S_BEST_STREAK, S_EFFORT, S_FINAL_X, S_FIRST_UP, S_JITTER,
                      S_MAX_HEIGHT, S_MEAN_ABS_X, S_REASON, S_STEPS, S_STREAK_JITTER, S_UPRIGHT,
                      physics_step)
from .task import DEFAULT_TASK, TARGET_PHYSICS, PhysicsParams, TaskConfig, observe

GOAL = np.array([0.0, np.pi, 0.0, 0.0, 0.0, 0.0])


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def state_error(s, ref):
    e = s - ref
    e[1] = _wrap(e[1]); e[2] = _wrap(e[2])
    return e


def f(s, u, phys: PhysicsParams, dt):
    """One step with cart force u [N]."""
    s2 = np.array(s, dtype=np.float64)
    physics_step(s2, float(u), phys.g, phys.drag, dt)
    return s2


def linearize(s, u, phys, dt, eps=1e-6):
    A = np.zeros((6, 6)); B = np.zeros((6, 1))
    for i in range(6):
        d = np.zeros(6); d[i] = eps
        A[:, i] = (f(s + d, u, phys, dt) - f(s - d, u, phys, dt)) / (2 * eps)
    B[:, 0] = (f(s, u + eps, phys, dt) - f(s, u - eps, phys, dt)) / (2 * eps)
    return A, B


def dlqr(A, B, Q, R, iters=20000, tol=1e-10):
    P = Q.copy()
    for _ in range(iters):
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        Pn = Q + A.T @ P @ (A - B @ K)
        if np.max(np.abs(Pn - P)) < tol * max(1.0, np.max(np.abs(P))):
            P = Pn
            break
        P = Pn
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


# Cost weights (state error in [m, rad, rad, m/s, rad/s, rad/s]; force in N)
Q_BAL = np.diag([10.0, 100.0, 100.0, 1.0, 1.0, 1.0])
R_BAL = np.array([[1e-3]])
Q_TRACK = np.diag([10.0, 50.0, 50.0, 1.0, 1.0, 1.0])
R_TRACK = np.array([[1e-3]])


def _step_generic(s, u, g, drag, dt):
    """Same semi-implicit Euler update as rollout.physics_step, written with generic
    arithmetic so that it also works on pydrake AutoDiffXd (for the NLP gradients)."""
    from pydrake.math import cos, sin
    from .rollout import L1, L2, M1, M2, MC
    x, t1, t2, xd, w1, w2 = s
    s1 = sin(t1); c1 = cos(t1); s12 = sin(t1 + t2); c12 = cos(t1 + t2)
    a1x = -L1 * c1; a1z = L1 * s1
    b1x = -L1 * c1 - L2 * c12; b1z = L1 * s1 + L2 * s12
    b2x = -L2 * c12; b2z = L2 * s12
    m00 = MC + M1 + M2; m01 = M1 * a1x + M2 * b1x; m02 = M2 * b2x
    m11 = M1 * (a1x * a1x + a1z * a1z) + M2 * (b1x * b1x + b1z * b1z)
    m12 = M2 * (b1x * b2x + b1z * b2z); m22 = M2 * (b2x * b2x + b2z * b2z)
    pd = w1 + w2
    p1x = L1 * s1 * w1 * w1; p1z = L1 * c1 * w1 * w1 + g
    p2x = p1x + L2 * s12 * pd * pd; p2z = L1 * c1 * w1 * w1 + L2 * c12 * pd * pd + g
    r0 = u - drag * xd - M1 * p1x - M2 * p2x
    r1 = -drag * w1 - M1 * (a1x * p1x + a1z * p1z) - M2 * (b1x * p2x + b1z * p2z)
    r2 = -drag * w2 - M2 * (b2x * p2x + b2z * p2z)
    c00 = m11 * m22 - m12 * m12; c01 = m02 * m12 - m01 * m22; c02 = m01 * m12 - m02 * m11
    c11 = m00 * m22 - m02 * m02; c12_ = m01 * m02 - m00 * m12; c22 = m00 * m11 - m01 * m01
    det = m00 * c00 + m01 * c01 + m02 * c02
    ax = (c00 * r0 + c01 * r1 + c02 * r2) / det
    a1 = (c01 * r0 + c11 * r1 + c12_ * r2) / det
    a2 = (c02 * r0 + c12_ * r1 + c22 * r2) / det
    xd = xd + dt * ax; w1 = w1 + dt * a1; w2 = w2 + dt * a2
    return np.array([x + dt * xd, t1 + dt * w1, t2 + dt * w2, xd, w1, w2])


def trajopt_swingup(phys: PhysicsParams, task: TaskConfig, T_up: float, T_hold: float,
                    u_frac: float = 0.85, x_max: float = 2.3, direction: float = 1.0,
                    seed: int = 0, verbose: bool = False):
    """Direct transcription of the exact discrete dynamics, solved with SNOPT.
    Hanging rest -> exactly upright at rest at T_up, |u| <= u_frac * max_force,
    |x| <= x_max; minimize sum u^2. Then hold the equilibrium (u = 0) for T_hold."""
    from pydrake.solvers import MathematicalProgram, SnoptSolver
    dt = task.dt
    N = int(round(T_up / dt))
    U = u_frac * phys.max_force
    prog = MathematicalProgram()
    X = prog.NewContinuousVariables(N + 1, 6, "x")
    Uv = prog.NewContinuousVariables(N, "u")
    goal = GOAL.copy(); goal[1] = direction * np.pi
    prog.AddBoundingBoxConstraint(np.zeros(6), np.zeros(6), X[0])
    prog.AddBoundingBoxConstraint(goal, goal, X[N])
    for k in range(N):
        prog.AddBoundingBoxConstraint(-U, U, Uv[k])
        prog.AddBoundingBoxConstraint(-x_max, x_max, X[k + 1, 0])
        prog.AddConstraint(
            lambda z: _step_generic(z[:6], z[6], phys.g, phys.drag, dt) - z[7:],
            np.zeros(6), np.zeros(6), np.concatenate([X[k], [Uv[k]], X[k + 1]]))
    prog.AddQuadraticCost(1e-4 * np.eye(N), np.zeros(N), Uv)
    rng = np.random.default_rng(seed)
    lam = np.linspace(0, 1, N + 1)[:, None]
    prog.SetInitialGuess(X, lam * goal[None] + 0.05 * rng.standard_normal((N + 1, 6)))
    prog.SetInitialGuess(Uv, 5.0 * rng.standard_normal(N))
    res = SnoptSolver().Solve(prog)
    Xs = res.GetSolution(X); Us = res.GetSolution(Uv)
    # Re-simulate the nominal with the true step (removes solver tolerance drift)
    S = np.zeros((N + 1, 6))
    for k in range(N):
        S[k + 1] = f(S[k], Us[k], phys, dt)
    n_hold = int(round(T_hold / dt))
    S = np.vstack([S, np.repeat(S[-1:], n_hold, axis=0)])
    u = np.concatenate([Us, np.zeros(n_hold)])
    err = np.abs(state_error(S[N], goal)).max()
    if verbose:
        print(f"  SNOPT success={res.is_success()} sum u^2={np.sum(Us ** 2):.0f} "
              f"final max|err| (re-sim)={err:.2e} max|x|={np.abs(S[:, 0]).max():.2f}")
    return S, u, (res.is_success(), err)


def tvlqr(S, u_nom, phys, task, P_final):
    N = len(u_nom)
    Ks = np.zeros((N, 6)); P = P_final
    for k in range(N - 1, -1, -1):
        A, B = linearize(S[k], u_nom[k], phys, task.dt)
        K = np.linalg.solve(R_TRACK + B.T @ P @ B, B.T @ P @ A)
        P = Q_TRACK + A.T @ P @ (A - B @ K)
        P = 0.5 * (P + P.T)
        Ks[k] = K[0]
    return Ks


@dataclass
class ClassicController:
    S_nom: np.ndarray        # (N+1, 6) nominal states
    u_nom: np.ndarray        # (N,) nominal forces [N]
    K_tv: np.ndarray         # (N, 6) TVLQR gains
    K_bal: np.ndarray        # (6,) balance LQR gains
    max_force: float
    task: TaskConfig = DEFAULT_TASK

    def reset(self):
        self.k = 0

    def state_from_obs(self, o):
        t = self.task
        return np.array([o[0] * t.limit_x, np.arctan2(o[2], o[3]), np.arctan2(o[5], o[6]),
                         o[1] * t.max_v_cart, o[4] * t.max_w_pole, o[7] * t.max_w_pole])

    def __call__(self, obs) -> float:
        s = self.state_from_obs(obs)
        k = self.k
        self.k += 1
        if k < len(self.u_nom):
            u = self.u_nom[k] - self.K_tv[k] @ state_error(s, self.S_nom[k])
        else:
            u = -self.K_bal @ state_error(s, GOAL)
        return float(np.clip(u / self.max_force, -1.0, 1.0))

    @property
    def n_numbers(self) -> int:
        return int(self.S_nom.size + self.u_nom.size + self.K_tv.size + self.K_bal.size)

    def save(self, path):
        np.savez(path, S_nom=self.S_nom, u_nom=self.u_nom, K_tv=self.K_tv, K_bal=self.K_bal,
                 max_force=self.max_force)

    @staticmethod
    def load(path):
        d = np.load(path)
        return ClassicController(d["S_nom"], d["u_nom"], d["K_tv"], d["K_bal"], float(d["max_force"]))


def design(phys: PhysicsParams = TARGET_PHYSICS, task: TaskConfig = DEFAULT_TASK,
           T_up: Optional[float] = None, T_hold: float = 0.3, n_tries: int = 6,
           verbose: bool = True) -> ClassicController:
    """Trajectory optimization (both swing directions, a few random initial guesses)
    + TVLQR tracking + LQR balance."""
    T_up = T_up if T_up is not None else task.swing_up_time - 0.4
    best = None
    for i in range(n_tries):
        direction = 1.0 if i % 2 == 0 else -1.0
        S, u, (ok, err) = trajopt_swingup(phys, task, T_up, T_hold, direction=direction,
                                          seed=i, verbose=verbose)
        score = (not ok, err, float(np.sum(u ** 2)))
        if best is None or score < best[0]:
            best = (score, S, u)
        if ok and err < 1e-6:
            break
    _, S, u = best
    A, B = linearize(GOAL, 0.0, phys, task.dt)
    K_bal, P = dlqr(A, B, Q_BAL, R_BAL)
    K_tv = tvlqr(S[:-1], u, phys, task, P)
    return ClassicController(S, u, K_tv, K_bal[0], phys.max_force, task)


def rollout_controller(ctrl, ics: np.ndarray, phys: PhysicsParams = TARGET_PHYSICS,
                       task: TaskConfig = DEFAULT_TASK) -> np.ndarray:
    """Python episode loop with exactly the kernel's semantics; returns stats[K, N_STATS]."""
    out = np.zeros((len(ics), N_STATS))
    for e, ic in enumerate(ics):
        ctrl.reset()
        s = np.array(ic, dtype=np.float64)
        a_prev = 0.0; streak = best = 0; sj = bj = 0.0; up_n = 0; first = -1
        jit = eff = 0.0; max_h = -1e9; sum_ax = 0.0; reason = 0; steps = 0
        for k in range(1, task.max_steps + 1):
            a = ctrl(observe(s, task))
            physics_step(s, a * phys.max_force, phys.g, phys.drag, task.dt)
            steps = k
            da = abs(a - a_prev); a_prev = a; jit += da; eff += abs(a); sum_ax += abs(s[0])
            max_h = max(max_h, -np.cos(s[1]) - np.cos(s[1] + s[2]))
            up = abs(_wrap(s[1] - np.pi)) < task.upright_tol and abs(_wrap(s[2])) < task.upright_tol
            if up:
                streak += 1; sj += da; up_n += 1
                if first < 0:
                    first = k
                if streak > best:
                    best, bj = streak, sj
            else:
                streak, sj = 0, 0.0
            if abs(s[0]) > task.limit_x:
                reason = 1; break
            if k > task.swing_up_steps and not up:
                reason = 2; break
        out[e, [S_STEPS, S_BEST_STREAK, S_STREAK_JITTER, S_UPRIGHT, S_FIRST_UP, S_JITTER, S_EFFORT,
                S_MAX_HEIGHT, S_REASON, S_FINAL_X, S_MEAN_ABS_X]] = [
            steps, best, bj, up_n, first, jit, eff, max_h, reason, s[0], sum_ax / max(steps, 1)]
    return out
