"""
Differentiable training surrogates for the JAX simulator (training / diagnostics only).

The task, test set and success criterion are unchanged: success is always measured by the
numba kernel (`pendulum/evaluation.py`). These functions only give gradient methods a signal.

  state_cost(s)       smooth distance to upright-at-rest-on-centre (0 there, ~1 hanging at rest)
  soft_upright(s, t)  sigmoid relaxation of the task's upright indicator (both |angle error| < tol);
                      t -> 0 recovers the exact indicator
"""

import jax.numpy as jnp
import jax

from .task import DEFAULT_TASK, TaskConfig

V_REF = 12.0   # velocity normaliser (same as the observation scale used throughout)


def state_cost(s, task: TaskConfig = DEFAULT_TASK):
    """((1 + cos th1) + (1 - cos th2)) / 4 + 0.05 (x / Lx)^2 + 0.01 * sum (v / 12)^2.
    Angle terms use 1 - cos(error), so the cost is smooth everywhere (no wrapping)."""
    ang = ((1.0 + jnp.cos(s[..., 1])) + (1.0 - jnp.cos(s[..., 2]))) / 4.0
    pos = 0.05 * (s[..., 0] / task.limit_x) ** 2
    vel = 0.01 * ((s[..., 3] / V_REF) ** 2 + (s[..., 4] / V_REF) ** 2 + (s[..., 5] / V_REF) ** 2)
    return ang + pos + vel


def soft_upright(s, temp, task: TaskConfig = DEFAULT_TASK):
    """Product of two sigmoids: |err1| < tol  <=>  cos(err1) > cos(tol), err1 = th1 - pi.
    temp is in units of cos (cos(0.2) = 0.980; temp = 0.005 ~ 0.05 rad at the boundary)."""
    ct = jnp.cos(task.upright_tol)
    u1 = jax.nn.sigmoid((-jnp.cos(s[..., 1]) - ct) / temp)   # cos(th1 - pi) = -cos th1
    u2 = jax.nn.sigmoid((jnp.cos(s[..., 2]) - ct) / temp)
    return u1 * u2
