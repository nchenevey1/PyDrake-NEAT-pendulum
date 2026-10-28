"""
Fast batched rollouts (numba).

Physics: closed-form cart + double pendulum dynamics for the model in
`assets/double_pendulum_free.urdf` (cart 2 kg; two 1 m links whose 0.5 kg masses
sit at the link ends and whose inertia about the joint axis (y) is zero),
integrated exactly like Drake's discrete MultibodyPlant (time_step = dt):
semi-implicit Euler, v+ = v + dt * M^-1 (tau - C v + tau_g), q+ = q + dt v+,
with the actuation and linear drag sampled at the start of the step.
`tests/test_dynamics.py` checks this against Drake to ~1e-12.

Episode semantics (same as legacy `train.evaluate_agent` + `DoublePendulumEnv`):
  * obs from the current state, action = policy(obs) in [-1, 1], force = action * max_force
  * one physics step, then: upright = |th1 - pi| < tol and |th2| < tol (wrapped)
  * terminate if |x| > limit_x, or t > swing_up_time and not upright, or t >= duration
  * the upright flag of the terminating step still counts.
"""

import math

import numba as nb
import numpy as np

from .policy import PackedPolicies, ACT_TANH, ACT_RELU
from .task import PhysicsParams, TaskConfig, DEFAULT_TASK

MC, M1, M2, L1, L2 = 2.0, 0.5, 0.5, 1.0, 1.0

# Per-episode statistics returned by the kernel
S_STEPS = 0            # steps survived
S_BEST_STREAK = 1      # longest consecutive upright steps
S_STREAK_JITTER = 2    # sum |a_t - a_{t-1}| inside the best streak
S_UPRIGHT = 3          # total upright steps
S_FIRST_UP = 4         # first upright step (or -1)
S_JITTER = 5           # total sum |a_t - a_{t-1}|
S_EFFORT = 6           # sum |a_t|
S_MAX_HEIGHT = 7       # max tip height (m, relative to pivot; +2 = fully upright)
S_REASON = 8           # 0 = full duration, 1 = track limit, 2 = fell / failed swing-up, 3 = non-finite
S_FINAL_X = 9          # cart position at the end
S_MEAN_ABS_X = 10      # mean |x| over the episode
N_STATS = 11

STAT_NAMES = ["steps", "best_streak", "streak_jitter", "upright", "first_up", "jitter",
              "effort", "max_height", "reason", "final_x", "mean_abs_x"]


@nb.njit(cache=True, fastmath=False, inline="always")
def _wrap(a):
    return ((a + math.pi) % (2.0 * math.pi)) - math.pi


@nb.njit(cache=True, fastmath=False)
def physics_step(s, u, g, drag, dt):
    """Advance state s = [x, th1, th2, xd, w1, w2] in place by one step with cart force u."""
    x, t1, t2, xd, w1, w2 = s[0], s[1], s[2], s[3], s[4], s[5]
    s1 = math.sin(t1); c1 = math.cos(t1)
    s12 = math.sin(t1 + t2); c12 = math.cos(t1 + t2)
    # Jacobian columns of mass 1 (a*) and mass 2 (b*) positions (x-z plane)
    a1x = -L1 * c1; a1z = L1 * s1
    b1x = -L1 * c1 - L2 * c12; b1z = L1 * s1 + L2 * s12
    b2x = -L2 * c12; b2z = L2 * s12
    # Mass matrix
    m00 = MC + M1 + M2
    m01 = M1 * a1x + M2 * b1x
    m02 = M2 * b2x
    m11 = M1 * (a1x * a1x + a1z * a1z) + M2 * (b1x * b1x + b1z * b1z)
    m12 = M2 * (b1x * b2x + b1z * b2z)
    m22 = M2 * (b2x * b2x + b2z * b2z)
    # Velocity-product accelerations + gravity (z up)
    pd = w1 + w2
    p1x = L1 * s1 * w1 * w1
    p1z = L1 * c1 * w1 * w1 + g
    p2x = p1x + L2 * s12 * pd * pd
    p2z = L1 * c1 * w1 * w1 + L2 * c12 * pd * pd + g
    r0 = u - drag * xd - M1 * p1x - M2 * p2x
    r1 = -drag * w1 - M1 * (a1x * p1x + a1z * p1z) - M2 * (b1x * p2x + b1z * p2z)
    r2 = -drag * w2 - M2 * (b2x * p2x + b2z * p2z)
    # Solve symmetric 3x3 (Cramer)
    c00 = m11 * m22 - m12 * m12
    c01 = m02 * m12 - m01 * m22
    c02 = m01 * m12 - m02 * m11
    c11 = m00 * m22 - m02 * m02
    c12_ = m01 * m02 - m00 * m12
    c22 = m00 * m11 - m01 * m01
    det = m00 * c00 + m01 * c01 + m02 * c02
    ax = (c00 * r0 + c01 * r1 + c02 * r2) / det
    a1 = (c01 * r0 + c11 * r1 + c12_ * r2) / det
    a2 = (c02 * r0 + c12_ * r1 + c22 * r2) / det
    xd += dt * ax; w1 += dt * a1; w2 += dt * a2
    s[0] = x + dt * xd; s[1] = t1 + dt * w1; s[2] = t2 + dt * w2
    s[3] = xd; s[4] = w1; s[5] = w2


@nb.njit(cache=True, fastmath=False, inline="always")
def _observe(s, buf, tp):
    lx = tp[2]; vmax = tp[5]; wmax = tp[6]
    s1 = math.sin(s[1]); c1 = math.cos(s[1]); s2 = math.sin(s[2]); c2 = math.cos(s[2])
    buf[0] = s[0] / lx
    buf[1] = min(1.0, max(-1.0, s[3] / vmax))
    buf[2] = s1
    buf[3] = c1
    buf[4] = min(1.0, max(-1.0, s[4] / wmax))
    buf[5] = s2
    buf[6] = c2
    buf[7] = min(1.0, max(-1.0, s[5] / wmax))
    buf[8] = s1 * s2 + c1 * c2


@nb.njit(cache=True, fastmath=False, inline="always")
def _forward(p, n_in, node_ptr, bias, act, cptr, src, w, delayed, out_pos, action_mode, buf, prev):
    n0 = node_ptr[p]; n1 = node_ptr[p + 1]
    size = n_in + (n1 - n0)
    for i in range(size):
        prev[i] = buf[i]
    for k in range(n0, n1):
        j = n_in + (k - n0)
        acc = bias[k]
        for c in range(cptr[k], cptr[k + 1]):
            if delayed[c]:
                acc += w[c] * prev[src[c]]
            else:
                acc += w[c] * buf[src[c]]
        a = act[k]
        if a == ACT_TANH:
            buf[j] = math.tanh(acc)
        elif a == ACT_RELU:
            buf[j] = acc if acc > 0.0 else 0.0
        else:
            buf[j] = acc
    out = buf[out_pos[p]]
    if action_mode[p] == 1:
        out = math.tanh(out)
    if out > 1.0:
        out = 1.0
    elif out < -1.0:
        out = -1.0
    return out


@nb.njit(cache=True, fastmath=False)
def _episode(p, n_in, node_ptr, bias, act, cptr, src, w, delayed, out_pos, action_mode,
             ic, phys, tp, max_buf, out_stats, traj):
    g = phys[0]; drag = phys[1]; fmax = phys[2]
    dt = tp[0]; max_steps = int(tp[1]); lx = tp[2]; swing_steps = int(tp[3]); tol = tp[4]
    record = traj.shape[0] > 1
    s = ic.copy()
    buf = np.zeros(max_buf)
    prev = np.zeros(max_buf)
    a_prev = 0.0
    streak = 0; streak_j = 0.0; best = 0; best_j = 0.0
    upright_n = 0; first_up = -1; jit = 0.0; eff = 0.0
    max_h = -1e9; sum_ax = 0.0; reason = 0.0
    steps = 0
    for k in range(1, max_steps + 1):
        _observe(s, buf, tp)
        a = _forward(p, n_in, node_ptr, bias, act, cptr, src, w, delayed, out_pos, action_mode, buf, prev)
        if not math.isfinite(a):
            reason = 3.0
            break
        physics_step(s, a * fmax, g, drag, dt)
        steps = k
        if record:
            traj[k, 0] = s[0]; traj[k, 1] = s[1]; traj[k, 2] = s[2]
            traj[k, 3] = s[3]; traj[k, 4] = s[4]; traj[k, 5] = s[5]; traj[k, 6] = a
        da = abs(a - a_prev)
        a_prev = a
        jit += da
        eff += abs(a)
        sum_ax += abs(s[0])
        h = -L1 * math.cos(s[1]) - L2 * math.cos(s[1] + s[2])
        if h > max_h:
            max_h = h
        if not (math.isfinite(s[0]) and math.isfinite(s[1]) and math.isfinite(s[2])):
            reason = 3.0
            break
        up = abs(_wrap(s[1] - math.pi)) < tol and abs(_wrap(s[2])) < tol
        if up:
            streak += 1
            streak_j += da
            upright_n += 1
            if first_up < 0:
                first_up = k
            if streak > best:
                best = streak
                best_j = streak_j
        else:
            streak = 0
            streak_j = 0.0
        if abs(s[0]) > lx:
            reason = 1.0
            break
        if k > swing_steps and not up:
            reason = 2.0
            break
    out_stats[S_STEPS] = steps
    out_stats[S_BEST_STREAK] = best
    out_stats[S_STREAK_JITTER] = best_j
    out_stats[S_UPRIGHT] = upright_n
    out_stats[S_FIRST_UP] = first_up
    out_stats[S_JITTER] = jit
    out_stats[S_EFFORT] = eff
    out_stats[S_MAX_HEIGHT] = max_h
    out_stats[S_REASON] = reason
    out_stats[S_FINAL_X] = s[0]
    out_stats[S_MEAN_ABS_X] = sum_ax / max(steps, 1)
    return steps


@nb.njit(cache=True, parallel=True)
def _rollout_batch(n_in, node_ptr, bias, act, cptr, src, w, delayed, out_pos, action_mode,
                   ics, phys, tp, max_buf, stats):
    P = out_pos.shape[0]
    K = ics.shape[0]
    dummy = np.zeros((1, 7))
    total = 0
    for e in nb.prange(P * K):
        p = e // K
        k = e % K
        total += _episode(p, n_in, node_ptr, bias, act, cptr, src, w, delayed, out_pos, action_mode,
                          ics[k], phys, tp, max_buf, stats[p, k], dummy)
    return total


@nb.njit(cache=True)
def _rollout_one(n_in, node_ptr, bias, act, cptr, src, w, delayed, out_pos, action_mode,
                 ic, phys, tp, max_buf, stats, traj):
    return _episode(0, n_in, node_ptr, bias, act, cptr, src, w, delayed, out_pos, action_mode,
                    ic, phys, tp, max_buf, stats, traj)


def _max_buf(pk: PackedPolicies) -> int:
    return int(pk.n_in + np.max(np.diff(pk.node_ptr))) + 1


def rollout(pk: PackedPolicies, ics: np.ndarray, physics: PhysicsParams,
            task: TaskConfig = DEFAULT_TASK):
    """Run every policy on every initial condition (common random numbers).

    Returns (stats[P, K, N_STATS], total_env_steps).
    """
    ics = np.ascontiguousarray(ics, dtype=np.float64).reshape(-1, 6)
    stats = np.zeros((len(pk), len(ics), N_STATS))
    phys = np.array([physics.g, physics.drag, physics.max_force], dtype=np.float64)
    total = _rollout_batch(pk.n_in, pk.node_ptr, pk.bias, pk.act, pk.cptr, pk.src, pk.w,
                           pk.delayed, pk.out_pos, pk.action_mode, ics, phys,
                           task.kernel_params(), _max_buf(pk), stats)
    return stats, int(total)


def simulate(policy, ic, physics: PhysicsParams, task: TaskConfig = DEFAULT_TASK):
    """Single episode with full trajectory. Returns (stats, traj[steps+1, 7]) with columns
    [x, th1, th2, xd, w1, w2, action]; row 0 is the initial state."""
    from .policy import pack
    pk = pack([policy])
    traj = np.zeros((task.max_steps + 1, 7))
    traj[0, :6] = ic
    stats = np.zeros(N_STATS)
    phys = np.array([physics.g, physics.drag, physics.max_force], dtype=np.float64)
    steps = _rollout_one(pk.n_in, pk.node_ptr, pk.bias, pk.act, pk.cptr, pk.src, pk.w, pk.delayed,
                         pk.out_pos, pk.action_mode, np.asarray(ic, dtype=np.float64), phys,
                         task.kernel_params(), _max_buf(pk), stats, traj)
    return stats, traj[:steps + 1]
