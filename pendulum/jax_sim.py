"""
JAX port of the simulator and policy forward pass (float64, CPU), for analytic gradients.

Mirrors `pendulum/rollout.py` operation by operation (same expression order in
`physics_step`, bias-first left-to-right accumulation in the policy forward pass), so
trajectories agree with the numba kernel to round-off; `tests/test_jax_port.py` checks
< 1e-9 over 2 s and identical policy outputs.

Only the dynamics and the network are ported. Episode termination and the sparse score stay
in the numba kernel, which remains the only evaluator of success. Rollouts here run a fixed
number of steps without termination (what a gradient method differentiates).

Only feed-forward graphs (no delayed connections) are supported.
"""

import math
from dataclasses import dataclass
from functools import partial

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from .policy import ACT_RELU, ACT_TANH, GraphPolicy  # noqa: E402
from .rollout import L1, L2, M1, M2, MC  # noqa: E402
from .task import DEFAULT_TASK, TARGET_PHYSICS, PhysicsParams, TaskConfig  # noqa: E402


def physics_step(s, u, g, drag, dt):
    """One semi-implicit Euler step of s = [x, th1, th2, xd, w1, w2] (shape (6,)); returns the new state."""
    x, t1, t2, xd, w1, w2 = s[0], s[1], s[2], s[3], s[4], s[5]
    s1 = jnp.sin(t1); c1 = jnp.cos(t1)
    s12 = jnp.sin(t1 + t2); c12 = jnp.cos(t1 + t2)
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
    r0 = u - drag * xd - M1 * p1x - M2 * p2x
    r1 = -drag * w1 - M1 * (a1x * p1x + a1z * p1z) - M2 * (b1x * p2x + b1z * p2z)
    r2 = -drag * w2 - M2 * (b2x * p2x + b2z * p2z)
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
    return jnp.stack([x + dt * xd, t1 + dt * w1, t2 + dt * w2, xd, w1, w2])


def observe(s, task: TaskConfig):
    """9-D observation, same as rollout._observe (velocities clipped to [-1, 1] after scaling)."""
    s1 = jnp.sin(s[1]); c1 = jnp.cos(s[1]); s2 = jnp.sin(s[2]); c2 = jnp.cos(s[2])
    return [s[0] / task.limit_x, jnp.clip(s[3] / task.max_v_cart, -1.0, 1.0), s1, c1,
            jnp.clip(s[4] / task.max_w_pole, -1.0, 1.0), s2, c2,
            jnp.clip(s[5] / task.max_w_pole, -1.0, 1.0), s1 * s2 + c1 * c2]


@dataclass(frozen=True)
class GraphStructure:
    """Static (hashable) structure of a feed-forward GraphPolicy; parameters live in theta."""
    n_in: int
    n_conn: int
    acts: tuple            # per non-input node
    conns: tuple           # per non-input node: tuple of (connection index, source buffer index)
    out_pos: int
    action_mode: int

    @staticmethod
    def of(p: GraphPolicy) -> "GraphStructure":
        if np.any(p.delayed):
            raise ValueError("jax_sim supports feed-forward graphs only")
        conns = tuple(tuple((int(c), int(p.src[c])) for c in np.nonzero(p.dst == p.n_in + k)[0])
                      for k in range(p.n_nodes))
        return GraphStructure(p.n_in, len(p.w), tuple(int(a) for a in p.act), conns,
                              int(p.out_pos), int(p.action_mode))


@jax.custom_jvp
def _host_tanh(x):
    """libm tanh evaluated on the host (bit-identical to numba's math.tanh); exact mode only."""
    # math.tanh, not np.tanh: numpy's vectorised tanh is not bit-identical to libm
    return jax.pure_callback(lambda v: np.vectorize(math.tanh, otypes=[np.float64])(np.asarray(v)),
                             jax.ShapeDtypeStruct(x.shape, x.dtype), x, vmap_method="expand_dims")


@_host_tanh.defjvp
def _host_tanh_jvp(primals, tangents):
    y = _host_tanh(primals[0])
    return y, (1.0 - y * y) * tangents[0]


def forward(st: GraphStructure, theta, obs, exact: bool = False):
    """Policy action in [-1, 1] for flat parameters theta = GraphPolicy.get_flat() = [w, bias].
    Accumulation order matches rollout._forward: bias, then connections in stored order.

    Default mode differs from numba by a few ulp (XLA fuses multiply-add into FMA and has its own
    tanh). exact=True blocks the FMA fusion and evaluates tanh with libm on the host, which makes
    the output bit-identical to the numba kernel; it is slow and used only to verify the port."""
    w = theta[:st.n_conn]; b = theta[st.n_conn:]
    mul = (lambda x, y: jax.lax.optimization_barrier(x * y)) if exact else (lambda x, y: x * y)
    tanh = _host_tanh if exact else jnp.tanh
    buf = list(obs)
    for k, (a, cs) in enumerate(zip(st.acts, st.conns)):
        acc = b[k]
        for c, src in cs:
            acc = acc + mul(w[c], buf[src])
        if a == ACT_TANH:
            v = tanh(acc)
        elif a == ACT_RELU:
            v = jnp.maximum(acc, 0.0)
        else:
            v = acc
        buf.append(v)
    out = buf[st.out_pos]
    if st.action_mode == 1:
        out = tanh(out)
    return jnp.clip(out, -1.0, 1.0)


def rollout(st: GraphStructure, theta, s0, n_steps: int, physics: PhysicsParams = TARGET_PHYSICS,
            task: TaskConfig = DEFAULT_TASK):
    """Closed-loop rollout of one start state for n_steps (no termination).
    Returns (states[n_steps, 6] after each step, actions[n_steps]); row k is the state after step k+1,
    i.e. the numba trajectory rows 1..n_steps."""
    def body(s, _):
        a = forward(st, theta, observe(s, task))
        s2 = physics_step(s, a * physics.max_force, physics.g, physics.drag, task.dt)
        return s2, (s2, a)
    _, (S, A) = jax.lax.scan(body, s0, None, length=n_steps)
    return S, A


def rollout_batch(st, theta, S0, n_steps, physics=TARGET_PHYSICS, task=DEFAULT_TASK):
    """vmap of `rollout` over start states S0[K, 6]."""
    return jax.vmap(lambda s0: rollout(st, theta, s0, n_steps, physics, task))(S0)


def open_loop(s0, forces, physics: PhysicsParams = TARGET_PHYSICS, task: TaskConfig = DEFAULT_TASK):
    """Replay a force sequence [N] from s0; returns states after each step."""
    def body(s, u):
        s2 = physics_step(s, u, physics.g, physics.drag, task.dt)
        return s2, s2
    return jax.lax.scan(body, s0, forces)[1]


@partial(jax.jit, static_argnums=(0, 3))
def forward_batch(st, theta, obs, exact=False):
    """Policy outputs for a batch of observations obs[B, n_in] (testing / inspection)."""
    return jax.vmap(lambda o: forward(st, theta, [o[i] for i in range(st.n_in)], exact))(obs)
