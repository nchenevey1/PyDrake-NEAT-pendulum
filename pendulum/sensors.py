"""
Encoder-only sensing (optional; nothing else uses it unless asked).

Sensor model
  * pole angles th1 (absolute shoulder) and th2 (relative elbow): incremental encoders with
    `cpr` counts per revolution, reading floor(th / q) * q, q = 2 pi / cpr (no wrap: counts
    accumulate, zero at hanging rest). cpr = 0 means exact.
  * cart position: floor(x / x_res) * x_res. x_res = 0 means exact.
  * actuation delay (0 or 1 control steps): the force applied in step t is the command
    computed in step t - delay.
  * force command bits (0 = float): action rounded to k / (2^(bits-1) - 1).

Observation modes (velocity scale = task.max_v_cart / task.max_w_pole, 12 in this study)
  full   the task's 9-D observation of the exact state (upper bound; ignores the encoders)
  fd     same 9-D layout, quantized positions, velocities = one-step finite differences
  hist   current frame [x/Lx, s1, c1, s2, c2, cos(th1-th2)] + the k-1 most recent one-step
         differences [dx, dth1, dth2] / (dt * vscale), newest first; k = 2 is exactly `fd`
         up to input order
  raw    the last k frames as [x/Lx, s1, c1, s2, c2] each, newest first (no differencing)
  pos    current frame only (6 values; for recurrent policies)

Episodes that start mid-motion (curriculum levels, balance starts) may pass `warm` states:
they are sensed and fed through the policy before step 1 (outputs ignored, except that the
last one becomes the pending command under delay), which fills the encoder history and the
recurrent state. Without warm states the history is the repeated first frame.

`rollout_sensed` is a drop-in for `rollout.rollout` (same statistics). `SensedPolicy` and
`py_episode` are a plain-python reference used by the tests, the Drake replay and the
classic controller with estimated velocities.
"""

import math
from dataclasses import dataclass, asdict

import numba as nb
import numpy as np

from .rollout import (N_STATS, S_BEST_STREAK, S_EFFORT, S_FINAL_X, S_FIRST_UP, S_JITTER,
                      S_MAX_HEIGHT, S_MEAN_ABS_X, S_REASON, S_STEPS, S_STREAK_JITTER, S_UPRIGHT,
                      L1, L2, _forward, _max_buf, _wrap, physics_step)
from .task import DEFAULT_TASK, PhysicsParams, TaskConfig

MODES = {"full": 0, "fd": 1, "hist": 2, "raw": 3, "pos": 4}
KMAX = 8


@dataclass(frozen=True)
class SensorConfig:
    mode: str = "full"
    k: int = 2              # frames used by hist / raw
    cpr: int = 0            # pole encoder counts per revolution (0 = exact)
    x_res: float = 0.0      # cart position resolution [m] (0 = exact)
    delay: int = 0          # actuation delay [control steps], 0 or 1
    force_bits: int = 0     # 0 = float command

    def __post_init__(self):
        assert self.mode in MODES, self.mode
        assert 1 <= self.k <= KMAX and self.delay in (0, 1)

    @property
    def n_in(self) -> int:
        return {"full": 9, "fd": 9, "hist": 6 + 3 * (self.k - 1), "raw": 5 * self.k, "pos": 6}[self.mode]

    @property
    def frames(self) -> int:
        """Encoder frames the observation needs (current one included)."""
        return {"full": 1, "fd": 2, "hist": self.k, "raw": self.k, "pos": 1}[self.mode]

    @property
    def memory_words(self) -> int:
        """Stored past measurements (recurrent state is counted separately)."""
        return 3 * (self.frames - 1)

    def kernel_params(self) -> np.ndarray:
        q = 2 * math.pi / self.cpr if self.cpr else 0.0
        lev = float(2 ** (self.force_bits - 1) - 1) if self.force_bits else 0.0
        return np.array([MODES[self.mode], self.k, q, self.x_res, self.delay, lev], dtype=np.float64)

    def label(self) -> str:
        s = self.mode + (str(self.k) if self.mode in ("hist", "raw") else "")
        if self.mode != "full":
            s += f"_c{self.cpr}_x{self.x_res * 1e3:g}mm"
        if self.delay:
            s += f"_d{self.delay}"
        if self.force_bits:
            s += f"_f{self.force_bits}"
        return s

    def to_dict(self) -> dict:
        return asdict(self)


NO_SENSOR = SensorConfig()


# =============================================================================
# numba kernel
# =============================================================================

@nb.njit(cache=True, inline="always")
def _quant(v, step):
    if step > 0.0:
        return math.floor(v / step) * step
    return v


@nb.njit(cache=True)
def _push(s, hist, sp, fill):
    """Shift the frame history and store the new measurement in hist[0]."""
    m0 = _quant(s[0], sp[3]); m1 = _quant(s[1], sp[2]); m2 = _quant(s[2], sp[2])
    if fill:
        for j in range(KMAX):
            hist[j, 0] = m0; hist[j, 1] = m1; hist[j, 2] = m2
    else:
        for j in range(KMAX - 1, 0, -1):
            hist[j, 0] = hist[j - 1, 0]; hist[j, 1] = hist[j - 1, 1]; hist[j, 2] = hist[j - 1, 2]
        hist[0, 0] = m0; hist[0, 1] = m1; hist[0, 2] = m2


@nb.njit(cache=True, inline="always")
def _clip1(v):
    return min(1.0, max(-1.0, v))


@nb.njit(cache=True)
def _sensed_obs(s, hist, sp, tp, buf):
    mode = int(sp[0]); k = int(sp[1])
    dt = tp[0]; lx = tp[2]; vmax = tp[5]; wmax = tp[6]
    if mode == 0:
        t1 = s[1]; t2 = s[2]
    else:
        t1 = hist[0, 1]; t2 = hist[0, 2]
    s1 = math.sin(t1); c1 = math.cos(t1); s2 = math.sin(t2); c2 = math.cos(t2)
    if mode == 0 or mode == 1:
        if mode == 0:
            x = s[0]; vx = s[3]; v1 = s[4]; v2 = s[5]
        else:
            x = hist[0, 0]
            vx = (hist[0, 0] - hist[1, 0]) / dt
            v1 = (hist[0, 1] - hist[1, 1]) / dt
            v2 = (hist[0, 2] - hist[1, 2]) / dt
        buf[0] = x / lx; buf[1] = _clip1(vx / vmax); buf[2] = s1; buf[3] = c1
        buf[4] = _clip1(v1 / wmax); buf[5] = s2; buf[6] = c2; buf[7] = _clip1(v2 / wmax)
        buf[8] = s1 * s2 + c1 * c2
    elif mode == 2 or mode == 4:
        buf[0] = hist[0, 0] / lx; buf[1] = s1; buf[2] = c1; buf[3] = s2; buf[4] = c2
        buf[5] = s1 * s2 + c1 * c2
        if mode == 2:
            for j in range(k - 1):
                o = 6 + 3 * j
                buf[o] = _clip1((hist[j, 0] - hist[j + 1, 0]) / (dt * vmax))
                buf[o + 1] = _clip1((hist[j, 1] - hist[j + 1, 1]) / (dt * wmax))
                buf[o + 2] = _clip1((hist[j, 2] - hist[j + 1, 2]) / (dt * wmax))
    else:
        for j in range(k):
            o = 5 * j
            buf[o] = hist[j, 0] / lx
            buf[o + 1] = math.sin(hist[j, 1]); buf[o + 2] = math.cos(hist[j, 1])
            buf[o + 3] = math.sin(hist[j, 2]); buf[o + 4] = math.cos(hist[j, 2])


@nb.njit(cache=True, inline="always")
def _command(a, lev):
    if lev > 0.0:
        return round(a * lev) / lev
    return a


@nb.njit(cache=True, fastmath=False)
def _sensed_episode(p, n_in, node_ptr, bias, act, cptr, src, w, delayed, out_pos, action_mode,
                    ic, warm, n_warm, phys, tp, sp, max_buf, out_stats, traj):
    g = phys[0]; drag = phys[1]; fmax = phys[2]
    dt = tp[0]; max_steps = int(tp[1]); lx = tp[2]; swing_steps = int(tp[3]); tol = tp[4]
    delay = int(sp[4]); lev = sp[5]
    record = traj.shape[0] > 1
    s = ic.copy()
    hist = np.zeros((KMAX, 3))
    buf = np.zeros(max_buf)
    prev = np.zeros(max_buf)
    pending = 0.0
    for j in range(n_warm):
        _push(warm[j], hist, sp, j == 0)
        _sensed_obs(warm[j], hist, sp, tp, buf)
        pending = _command(_forward(p, n_in, node_ptr, bias, act, cptr, src, w, delayed, out_pos,
                                    action_mode, buf, prev), lev)
    if delay == 0:
        pending = 0.0
    a_prev = 0.0
    streak = 0; streak_j = 0.0; best = 0; best_j = 0.0
    upright_n = 0; first_up = -1; jit = 0.0; eff = 0.0
    max_h = -1e9; sum_ax = 0.0; reason = 0.0
    steps = 0
    for k in range(1, max_steps + 1):
        _push(s, hist, sp, k == 1 and n_warm == 0)
        _sensed_obs(s, hist, sp, tp, buf)
        cmd = _forward(p, n_in, node_ptr, bias, act, cptr, src, w, delayed, out_pos, action_mode, buf, prev)
        if not math.isfinite(cmd):
            reason = 3.0
            break
        cmd = _command(cmd, lev)
        if delay == 1:
            a = pending
            pending = cmd
        else:
            a = cmd
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
def _sensed_batch(n_in, node_ptr, bias, act, cptr, src, w, delayed, out_pos, action_mode,
                  ics, warm, n_warm, phys, tp, sp, max_buf, stats):
    P = out_pos.shape[0]
    K = ics.shape[0]
    dummy = np.zeros((1, 7))
    total = 0
    for e in nb.prange(P * K):
        p = e // K
        k = e % K
        total += _sensed_episode(p, n_in, node_ptr, bias, act, cptr, src, w, delayed, out_pos,
                                 action_mode, ics[k], warm[k], n_warm, phys, tp, sp, max_buf,
                                 stats[p, k], dummy)
    return total


@nb.njit(cache=True)
def _sensed_one(n_in, node_ptr, bias, act, cptr, src, w, delayed, out_pos, action_mode,
                ic, warm, n_warm, phys, tp, sp, max_buf, stats, traj):
    return _sensed_episode(0, n_in, node_ptr, bias, act, cptr, src, w, delayed, out_pos,
                           action_mode, ic, warm, n_warm, phys, tp, sp, max_buf, stats, traj)


def _warm_array(warm, n_ics):
    if warm is None:
        return np.zeros((n_ics, 1, 6)), 0
    warm = np.ascontiguousarray(warm, dtype=np.float64)
    assert warm.ndim == 3 and warm.shape[0] == n_ics and warm.shape[2] == 6
    return warm, warm.shape[1]


def rollout_sensed(pk, ics, physics: PhysicsParams, task: TaskConfig = DEFAULT_TASK,
                   sensor: SensorConfig = NO_SENSOR, warm=None):
    """Like rollout.rollout, through the sensor model. warm: optional (K, W, 6) states that
    precede each initial state (oldest first). Returns (stats[P, K, N_STATS], env_steps)."""
    assert pk.n_in == sensor.n_in, f"policy has {pk.n_in} inputs, sensor mode gives {sensor.n_in}"
    ics = np.ascontiguousarray(ics, dtype=np.float64).reshape(-1, 6)
    wa, nw = _warm_array(warm, len(ics))
    stats = np.zeros((len(pk), len(ics), N_STATS))
    phys = np.array([physics.g, physics.drag, physics.max_force], dtype=np.float64)
    total = _sensed_batch(pk.n_in, pk.node_ptr, pk.bias, pk.act, pk.cptr, pk.src, pk.w,
                          pk.delayed, pk.out_pos, pk.action_mode, ics, wa, nw, phys,
                          task.kernel_params(), sensor.kernel_params(), _max_buf(pk), stats)
    return stats, int(total)


def simulate_sensed(policy, ic, physics: PhysicsParams, task: TaskConfig = DEFAULT_TASK,
                    sensor: SensorConfig = NO_SENSOR, warm=None):
    """Single episode with trajectory [x, th1, th2, xd, w1, w2, applied action]."""
    from .policy import pack
    pk = pack([policy])
    traj = np.zeros((task.max_steps + 1, 7))
    traj[0, :6] = ic
    stats = np.zeros(N_STATS)
    wa, nw = _warm_array(None if warm is None else np.asarray(warm)[None], 1)
    phys = np.array([physics.g, physics.drag, physics.max_force], dtype=np.float64)
    steps = _sensed_one(pk.n_in, pk.node_ptr, pk.bias, pk.act, pk.cptr, pk.src, pk.w, pk.delayed,
                        pk.out_pos, pk.action_mode, np.asarray(ic, dtype=np.float64), wa[0], nw,
                        phys, task.kernel_params(), sensor.kernel_params(), _max_buf(pk), stats, traj)
    return stats, traj[:steps + 1]


def evaluate_sensed(policies, ics, physics, task, sensor, warm=None):
    from .evaluation import summarize
    from .policy import pack
    stats, _ = rollout_sensed(pack(policies), ics, physics, task, sensor, warm)
    return [summarize(s, task) for s in stats]


def back_extrapolated_warm(ics, n, dt):
    """Warm-up states for starts in motion: constant-velocity back-extrapolation (oldest first)."""
    ics = np.asarray(ics, dtype=np.float64)
    lags = np.arange(n, 0, -1)[None, :, None] * dt
    warm = np.repeat(ics[:, None, :], n, axis=1)
    warm[:, :, :3] -= lags * ics[:, None, 3:]
    return warm


# =============================================================================
# python reference (tests, Drake replay, classic controller with estimated velocities)
# =============================================================================

class Sensor:
    """Python reference of the kernel's measurement history and observation."""

    def __init__(self, cfg: SensorConfig, task: TaskConfig = DEFAULT_TASK):
        self.cfg, self.task = cfg, task
        self.q = 2 * math.pi / cfg.cpr if cfg.cpr else 0.0
        self.hist = np.zeros((KMAX, 3))

    def measure(self, s):
        q, xr = self.q, self.cfg.x_res
        f = lambda v, st: math.floor(v / st) * st if st > 0 else v
        return np.array([f(s[0], xr), f(s[1], q), f(s[2], q)])

    def push(self, s, fill=False):
        m = self.measure(s)
        if fill:
            self.hist[:] = m
        else:
            self.hist[1:] = self.hist[:-1].copy()
            self.hist[0] = m

    def obs(self, s):
        c, t, h = self.cfg, self.task, self.hist
        dt, lx, vm, wm = t.dt, t.limit_x, t.max_v_cart, t.max_w_pole
        t1, t2 = (s[1], s[2]) if c.mode == "full" else (h[0, 1], h[0, 2])
        s1, c1, s2, c2 = math.sin(t1), math.cos(t1), math.sin(t2), math.cos(t2)
        cl = lambda v: min(1.0, max(-1.0, v))
        if c.mode in ("full", "fd"):
            if c.mode == "full":
                x, vx, v1, v2 = s[0], s[3], s[4], s[5]
            else:
                x = h[0, 0]
                vx, v1, v2 = (h[0, 0] - h[1, 0]) / dt, (h[0, 1] - h[1, 1]) / dt, (h[0, 2] - h[1, 2]) / dt
            return np.array([x / lx, cl(vx / vm), s1, c1, cl(v1 / wm), s2, c2, cl(v2 / wm), s1 * s2 + c1 * c2])
        if c.mode in ("hist", "pos"):
            o = [h[0, 0] / lx, s1, c1, s2, c2, s1 * s2 + c1 * c2]
            if c.mode == "hist":
                for j in range(c.k - 1):
                    o += [cl((h[j, 0] - h[j + 1, 0]) / (dt * vm)), cl((h[j, 1] - h[j + 1, 1]) / (dt * wm)),
                          cl((h[j, 2] - h[j + 1, 2]) / (dt * wm))]
            return np.array(o)
        o = []
        for j in range(c.k):
            o += [h[j, 0] / lx, math.sin(h[j, 1]), math.cos(h[j, 1]), math.sin(h[j, 2]), math.cos(h[j, 2])]
        return np.array(o)


class SensedPolicy:
    """Controller = sensor model + observation -> command function + actuator (bits, delay).
    `fn(obs)` may keep its own state; `make_fn()` creates a fresh one at reset."""

    def __init__(self, make_fn, cfg: SensorConfig, task: TaskConfig = DEFAULT_TASK):
        self.make_fn, self.cfg, self.task = make_fn, cfg, task
        self.lev = 2 ** (cfg.force_bits - 1) - 1 if cfg.force_bits else 0

    @staticmethod
    def from_graph(policy, cfg, task=DEFAULT_TASK):
        return SensedPolicy(lambda: (lambda o, st=policy.make_stepper(): float(np.clip(st(o), -1, 1))), cfg, task)

    def _cmd(self, a):
        return round(a * self.lev) / self.lev if self.lev else a

    def reset(self, warm=None):
        self.sensor = Sensor(self.cfg, self.task)
        self.fn = self.make_fn()
        self.pending = 0.0
        self.first = True
        if warm is not None and len(warm):
            for j, ws in enumerate(warm):
                self.sensor.push(ws, fill=(j == 0))
                self.pending = self._cmd(self.fn(self.sensor.obs(ws)))
            self.first = False
        if self.cfg.delay == 0:
            self.pending = 0.0

    def __call__(self, s) -> float:
        """Applied action in [-1, 1] for true state s."""
        self.sensor.push(s, fill=self.first)
        self.first = False
        cmd = self._cmd(self.fn(self.sensor.obs(s)))
        if self.cfg.delay:
            a, self.pending = self.pending, cmd
            return a
        return cmd


def py_episode(ctrl: SensedPolicy, ic, physics: PhysicsParams, task: TaskConfig = DEFAULT_TASK,
               warm=None, env=None, max_steps=None):
    """Python episode with the kernel's semantics. env: optional DrakeEnv (replaces the numba
    physics step). Returns (steps, success, trajectory of [state, applied action])."""
    ctrl.reset(warm)
    s = np.array(ic, dtype=np.float64)
    if env is not None:
        s = env.reset(s)
    traj = [np.concatenate([s, [0.0]])]
    steps, reason = 0, 0
    n = max_steps or task.max_steps
    for k in range(1, n + 1):
        a = ctrl(s)
        if env is None:
            physics_step(s, a * physics.max_force, physics.g, physics.drag, task.dt)
        else:
            s = env.step(a)
        steps = k
        traj.append(np.concatenate([s, [a]]))
        up = abs(_wrap(s[1] - math.pi)) < task.upright_tol and abs(_wrap(s[2])) < task.upright_tol
        if abs(s[0]) > task.limit_x:
            reason = 1; break
        if k > task.swing_up_steps and not up:
            reason = 2; break
    return steps, (steps >= task.max_steps and reason == 0), np.array(traj)
