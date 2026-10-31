"""
Closed-loop maps x_{k+1} = F(x_k) for the balance certificates.

A controller is reduced to the data the interval kernel needs (`ControllerData`):

* kind = NET: u = max_force * clip(tanh(b2 + w2 . tanh(W1 o + b1) + ws . o), -1, 1) with the
  policy's real 9-D observation o (x / Lx, clipped velocity / scale, sin/cos of both angles,
  cos(th1 - th2)). A one-hidden-layer tanh graph with optional input->output skips covers the
  trained MLP and its pruned / quantized versions.
* kind = LQR: u = clip(-K (s - goal), -max_force, max_force) with goal = (0, pi, 0, 0, 0, 0);
  angles enter unwrapped, which equals the wrapped error inside the certified set
  (|th1 - pi| < 0.2, |th2| < 0.2).

`step_np` is a vectorised float reference (checked against `rollout.physics_step` and the
episode kernel in tests); the certificate itself only uses the interval kernel.
"""

from dataclasses import dataclass, replace

import numpy as np

from ..policy import ACT_TANH, GraphPolicy
from ..rollout import L1, L2, M1, M2, MC, physics_step
from ..task import DEFAULT_TASK, TARGET_PHYSICS, PhysicsParams, TaskConfig

KIND_NET = 0
KIND_LQR = 1
GOAL = np.array([0.0, np.pi, 0.0, 0.0, 0.0, 0.0])


@dataclass
class ControllerData:
    kind: int
    W1: np.ndarray        # (H, 9)
    b1: np.ndarray        # (H,)
    w2: np.ndarray        # (H,)
    ws: np.ndarray        # (9,) input -> output skip weights
    b2: float
    K: np.ndarray         # (6,) LQR gains (kind LQR)
    lx: float             # observation scales
    vmax: float
    wmax: float
    fmax: float
    g: float
    dt: float
    name: str = ""

    def params(self) -> np.ndarray:
        """Flat float64 parameter vector consumed by the numba kernels."""
        H = len(self.b1)
        return np.concatenate([[self.kind, H, self.b2, self.lx, self.vmax, self.wmax, self.fmax,
                                self.g, self.dt], self.K, self.ws, self.w2, self.b1,
                               self.W1.reshape(-1)]).astype(np.float64)

    @property
    def n_params(self) -> int:
        if self.kind == KIND_LQR:
            return int(np.count_nonzero(self.K))
        return int(np.count_nonzero(self.W1) + np.count_nonzero(self.w2) + np.count_nonzero(self.ws)
                   + len(self.b1) + 1)


def from_policy(pol: GraphPolicy, physics: PhysicsParams = TARGET_PHYSICS,
                task: TaskConfig = DEFAULT_TASK, name: str = "") -> ControllerData:
    """Feed-forward tanh graph with one hidden layer (inputs -> hidden -> output, optional
    input -> output skips) as dense matrices."""
    task = replace(task, **pol.meta.get("obs_scale", {}))
    pol = pol.compacted()
    assert not pol.is_recurrent and pol.action_mode == 0
    assert np.all(pol.act == ACT_TANH), "only tanh graphs are supported"
    n_in = pol.n_in
    out = pol.out_pos
    hidden = [j for j in range(n_in, n_in + pol.n_nodes) if j != out]
    H = len(hidden)
    W1 = np.zeros((H, n_in)); w2 = np.zeros(H); ws = np.zeros(n_in)
    hid_idx = {j: k for k, j in enumerate(hidden)}
    for s, d, w in zip(pol.src, pol.dst, pol.w):
        if d == out:
            if s < n_in:
                ws[s] += w
            else:
                w2[hid_idx[int(s)]] += w
        else:
            assert s < n_in, "hidden -> hidden connections are not supported"
            W1[hid_idx[int(d)], s] += w
    b1 = np.array([pol.bias[j - n_in] for j in hidden])
    b2 = float(pol.bias[out - n_in])
    return ControllerData(KIND_NET, W1, b1, w2, ws, b2, np.zeros(6), task.limit_x, task.max_v_cart,
                          task.max_w_pole, physics.max_force, physics.g, task.dt, name)


def from_lqr(K: np.ndarray, physics: PhysicsParams = TARGET_PHYSICS,
             task: TaskConfig = DEFAULT_TASK, name: str = "lqr") -> ControllerData:
    return ControllerData(KIND_LQR, np.zeros((0, 9)), np.zeros(0), np.zeros(0), np.zeros(9), 0.0,
                          np.asarray(K, dtype=np.float64).reshape(6), task.limit_x, task.max_v_cart,
                          task.max_w_pole, physics.max_force, physics.g, task.dt, name)


# ----------------------------------------------------------------------------- float model

def observe_np(S, cd: ControllerData):
    x, t1, t2, xd, w1, w2 = S.T
    s1, c1, s2, c2 = np.sin(t1), np.cos(t1), np.sin(t2), np.cos(t2)
    return np.stack([x / cd.lx, np.clip(xd / cd.vmax, -1, 1), s1, c1, np.clip(w1 / cd.wmax, -1, 1),
                     s2, c2, np.clip(w2 / cd.wmax, -1, 1), s1 * s2 + c1 * c2], axis=1)


def control_np(S, cd: ControllerData):
    """Cart force [N] for states S (N, 6)."""
    S = np.atleast_2d(S)
    if cd.kind == KIND_LQR:
        return np.clip(-(S - GOAL) @ cd.K, -cd.fmax, cd.fmax)
    o = observe_np(S, cd)
    h = np.tanh(o @ cd.W1.T + cd.b1)
    a = np.tanh(h @ cd.w2 + o @ cd.ws + cd.b2)
    return cd.fmax * np.clip(a, -1.0, 1.0)


def physics_np(S, u, g, dt):
    """Vectorised copy of rollout.physics_step (drag = 0)."""
    x, t1, t2, xd, w1, w2 = [S[:, i] for i in range(6)]
    s1 = np.sin(t1); c1 = np.cos(t1)
    s12 = np.sin(t1 + t2); c12 = np.cos(t1 + t2)
    a1x = -L1 * c1; a1z = L1 * s1
    b1x = -L1 * c1 - L2 * c12; b1z = L1 * s1 + L2 * s12
    b2x = -L2 * c12; b2z = L2 * s12
    m00 = MC + M1 + M2
    m01 = M1 * a1x + M2 * b1x
    m02 = M2 * b2x
    m11 = M1 * (a1x * a1x + a1z * a1z) + M2 * (b1x * b1x + b1z * b1z)
    m12 = M2 * (b1x * b2x + b1z * b2z)
    m22 = M2 * (b2x * b2x + b2z * b2z)
    pd = w1 + w2
    p1x = L1 * s1 * w1 * w1
    p1z = L1 * c1 * w1 * w1 + g
    p2x = p1x + L2 * s12 * pd * pd
    p2z = L1 * c1 * w1 * w1 + L2 * c12 * pd * pd + g
    r0 = u - M1 * p1x - M2 * p2x
    r1 = -M1 * (a1x * p1x + a1z * p1z) - M2 * (b1x * p2x + b1z * p2z)
    r2 = -M2 * (b2x * p2x + b2z * p2z)
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
    xd = xd + dt * ax; w1 = w1 + dt * a1; w2 = w2 + dt * a2
    return np.stack([x + dt * xd, t1 + dt * w1, t2 + dt * w2, xd, w1, w2], axis=1)


def step_np(S, cd: ControllerData):
    S = np.atleast_2d(np.asarray(S, dtype=np.float64))
    return physics_np(S, control_np(S, cd), cd.g, cd.dt)


def step_ref(s, cd: ControllerData):
    """Single step through the repository's own physics kernel (reference)."""
    s = np.array(s, dtype=np.float64)
    physics_step(s, float(control_np(s, cd)[0]), cd.g, 0.0, cd.dt)
    return s


# ------------------------------------------------------------------ equilibrium + linearization

def find_equilibria(cd: ControllerData, n_grid: int = 2001):
    """Closed-loop equilibria: with zero velocities the cart and pole accelerations vanish only
    if u = 0 and both poles are vertical (gravity has no x component), so equilibria are the
    upright states (x, pi, 0, 0, 0, 0) with u(x) = 0 and |x| <= limit_x."""
    from scipy.optimize import brentq
    xs = np.linspace(-cd.lx, cd.lx, n_grid)
    S = np.zeros((n_grid, 6)); S[:, 0] = xs; S[:, 1] = np.pi
    u = control_np(S, cd)
    roots = []
    for i in range(n_grid - 1):
        if u[i] == 0.0:
            roots.append(xs[i])
        elif u[i] * u[i + 1] < 0:
            f = lambda x: control_np(np.array([x, np.pi, 0, 0, 0, 0]), cd)[0]
            roots.append(brentq(f, xs[i], xs[i + 1], xtol=1e-16, rtol=8.9e-16, maxiter=500))
    return [np.array([r, np.pi, 0.0, 0.0, 0.0, 0.0]) for r in roots]


def jacobian_fd(cd: ControllerData, s, eps: float = 1e-6):
    s = np.asarray(s, dtype=np.float64)
    J = np.zeros((6, 6))
    for i in range(6):
        d = np.zeros(6); d[i] = eps
        J[:, i] = (step_np(s + d, cd)[0] - step_np(s - d, cd)[0]) / (2 * eps)
    return J


# ----------------------------------------------------------------------------- traced program

def trace_closed_loop(cd: ControllerData, g=None, masses=None, force_dist=None):
    """Record the closed loop as an interval program (see interval.Prog).

    Outputs (7): the cart force u(x), m(q) = M(q)^{-1} e_0 (3) and the zero-force accelerations
    p(x) (3); the accelerations are a = m u + p and the step is F(x) = Phi x + Gamma a(x).

    g, masses = (mc, m1, m2) and force_dist may be (lo, hi) pairs: they then enter as interval
    constants (uncertain physics / bounded additive cart-force disturbance)."""
    from . import interval as I
    p = I.Prog()
    S = [p.input(i) for i in range(6)]
    x, t1, t2, xd, w1, w2 = S

    def const(v):
        if isinstance(v, tuple):
            return p.const(v[0], v[1])
        return p.const(float(v))

    s1 = I.sin(t1); c1 = I.cos(t1); s2 = I.sin(t2); c2 = I.cos(t2)
    # ---------------------------------------------------------------- controller
    if cd.kind == KIND_LQR:
        e = [x - GOAL[0], t1 - GOAL[1], t2 - GOAL[2], xd - GOAL[3], w1 - GOAL[4], w2 - GOAL[5]]
        acc = None
        for k in range(6):
            if cd.K[k] != 0.0:
                term = float(cd.K[k]) * e[k]
                acc = term if acc is None else acc + term
        a = I.clip(-acc / cd.fmax, -1.0, 1.0)
    else:
        o = [x / cd.lx, I.clip(xd / cd.vmax, -1.0, 1.0), s1, c1, I.clip(w1 / cd.wmax, -1.0, 1.0),
             s2, c2, I.clip(w2 / cd.wmax, -1.0, 1.0), s1 * s2 + c1 * c2]
        z_out = p.const(cd.b2)
        for k in range(len(cd.b1)):
            if cd.w2[k] == 0.0:
                continue
            acc = p.const(float(cd.b1[k]))
            for j in range(len(o)):
                if cd.W1[k, j] != 0.0:
                    acc = acc + float(cd.W1[k, j]) * o[j]
            z_out = z_out + float(cd.w2[k]) * I.tanh(acc)
        for j in range(len(o)):
            if cd.ws[j] != 0.0:
                z_out = z_out + float(cd.ws[j]) * o[j]
        a = I.clip(I.tanh(z_out), -1.0, 1.0)
    u = a * cd.fmax
    if force_dist is not None:
        u = u + const(force_dist)
    # ---------------------------------------------------------------- physics (= physics_step)
    mc, m1, m2 = (MC, M1, M2) if masses is None else masses
    MCv, M1v, M2v = const(mc), const(m1), const(m2)
    gv = const(cd.g if g is None else g)
    dt = p.const(cd.dt)
    l1, l2 = p.const(L1), p.const(L2)
    s12 = I.sin(t1 + t2); c12 = I.cos(t1 + t2)
    a1x = -(l1 * c1); a1z = l1 * s1
    b1x = -(l1 * c1) - l2 * c12; b1z = l1 * s1 + l2 * s12
    b2x = -(l2 * c12); b2z = l2 * s12
    m00 = MCv + M1v + M2v
    m01 = M1v * a1x + M2v * b1x
    m02 = M2v * b2x
    m11 = M1v * (a1x * a1x + a1z * a1z) + M2v * (b1x * b1x + b1z * b1z)
    m12 = M2v * (b1x * b2x + b1z * b2z)
    m22 = M2v * (b2x * b2x + b2z * b2z)
    pd = w1 + w2
    p1x = l1 * s1 * w1 * w1
    p1z = l1 * c1 * w1 * w1 + gv
    p2x = p1x + l2 * s12 * pd * pd
    p2z = l1 * c1 * w1 * w1 + l2 * c12 * pd * pd + gv
    # The accelerations are affine in the cart force: a = m(q) u + p(x), with m = M^{-1} e_0 and
    # p the accelerations at u = 0 (identical to physics_step, whose r0 = u - M1 p1x - M2 p2x).
    r0 = -(M1v * p1x) - M2v * p2x
    r1 = -(M1v * (a1x * p1x + a1z * p1z)) - M2v * (b1x * p2x + b1z * p2z)
    r2 = -(M2v * (b2x * p2x + b2z * p2z))
    k00 = m11 * m22 - m12 * m12
    k01 = m02 * m12 - m01 * m22
    k02 = m01 * m12 - m02 * m11
    k11 = m00 * m22 - m02 * m02
    k12 = m01 * m02 - m00 * m12
    k22 = m00 * m11 - m01 * m01
    det = m00 * k00 + m01 * k01 + m02 * k02
    mm = [k00 / det, k01 / det, k02 / det]
    pp = [(k00 * r0 + k01 * r1 + k02 * r2) / det,
          (k01 * r0 + k11 * r1 + k12 * r2) / det,
          (k02 * r0 + k12 * r1 + k22 * r2) / det]
    # The step is linear in the accelerations: F(x) = Phi x + Gamma a(x) (see phi_gamma).
    p.outputs = [u] + mm + pp
    return p


def phi_gamma(dt: float):
    """F(x) = Phi x + Gamma a(x) for the semi-implicit Euler step (in exact arithmetic):
    v+ = v + dt a, q+ = q + dt v+ = q + dt v + dt^2 a. Gamma's dt^2 is returned as the pair
    (dt, dt) so callers can multiply it out with interval rounding."""
    Phi = np.eye(6); Phi[0, 3] = Phi[1, 4] = Phi[2, 5] = dt
    return Phi
