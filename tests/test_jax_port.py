"""The JAX port (pendulum/jax_sim.py) must reproduce the numba simulator and policy."""
import math
from dataclasses import replace

import numba as nb
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from pendulum import jax_sim as J  # noqa: E402
from pendulum.policy import GraphPolicy, build_arch, pack  # noqa: E402
from pendulum.rollout import _forward, _max_buf, physics_step, simulate  # noqa: E402
from pendulum.task import DEFAULT_TASK, TARGET_PHYSICS, PhysicsParams  # noqa: E402

SOLVED = "results/experiments/revcurr_mlp4_s0_v12.json"
T12 = replace(DEFAULT_TASK, max_v_cart=12.0, max_w_pole=12.0)
# no termination, so the numba trajectory has all 200 steps even for random policies
T_FREE = replace(T12, limit_x=1e9, swing_up_time=1e9, duration=2.0)


def _policies():
    pols = [build_arch(a, 9, np.random.default_rng(s)) for a in ("mlp:2", "mlp:4", "mlp:8") for s in range(2)]
    pols.insert(0, GraphPolicy.load(SOLVED))
    return pols


@nb.njit
def _numba_forward(n_in, node_ptr, bias, act, cptr, src, w, delayed, out_pos, action_mode, obs, mb):
    out = np.zeros(obs.shape[0]); buf = np.zeros(mb); prev = np.zeros(mb)
    for i in range(obs.shape[0]):
        for j in range(n_in):
            buf[j] = obs[i, j]
        out[i] = _forward(0, n_in, node_ptr, bias, act, cptr, src, w, delayed, out_pos, action_mode, buf, prev)
    return out


@pytest.mark.parametrize("phys", [TARGET_PHYSICS, PhysicsParams(4.0, 1.5, 550.0)])
def test_one_step_matches_numba(phys):
    rng = np.random.default_rng(0)
    step = jax.jit(lambda s, u: J.physics_step(s, u, phys.g, phys.drag, 0.01))
    for _ in range(200):
        s0 = np.concatenate([rng.uniform(-2.5, 2.5, 1), rng.uniform(-np.pi, np.pi, 2), rng.uniform(-10, 10, 3)])
        u = rng.uniform(-1, 1) * phys.max_force
        s = s0.copy()
        physics_step(s, u, phys.g, phys.drag, 0.01)
        np.testing.assert_allclose(np.asarray(step(jnp.asarray(s0), u)), s, rtol=0, atol=1e-13)


@pytest.mark.parametrize("pi", range(7))
def test_closed_loop_2s_matches_numba(pi):
    """2 s closed-loop trajectories (states and actions) agree to 1e-9, for the solved controller
    and for random untrained ones (which swing wildly and leave the track)."""
    p = _policies()[pi]
    st = J.GraphStructure.of(p)
    f = jax.jit(lambda th, s0: J.rollout(st, th, s0, T_FREE.max_steps, TARGET_PHYSICS, T_FREE))
    rng = np.random.default_rng(pi)
    for ic in rng.uniform(-0.05, 0.05, (6, 6)):
        _, traj = simulate(p, ic, TARGET_PHYSICS, T_FREE)
        assert len(traj) == T_FREE.max_steps + 1
        S, A = f(jnp.asarray(p.get_flat()), jnp.asarray(ic))
        np.testing.assert_allclose(np.asarray(S), traj[1:, :6], rtol=0, atol=1e-9)
        np.testing.assert_allclose(np.asarray(A), traj[1:, 6], rtol=0, atol=1e-9)


def test_policy_outputs_exact():
    """exact=True is bit-identical to the numba kernel; the fast default is within a few ulp."""
    obs = np.random.default_rng(3).uniform(-1.2, 1.2, (2000, 9))
    for p in _policies():
        pk = pack([p]); st = J.GraphStructure.of(p)
        ref = _numba_forward(pk.n_in, pk.node_ptr, pk.bias, pk.act, pk.cptr, pk.src, pk.w, pk.delayed,
                             pk.out_pos, pk.action_mode, obs, _max_buf(pk))
        th = jnp.asarray(p.get_flat())
        np.testing.assert_array_equal(np.asarray(J.forward_batch(st, th, jnp.asarray(obs), True)), ref)
        np.testing.assert_allclose(np.asarray(J.forward_batch(st, th, jnp.asarray(obs))), ref, rtol=0, atol=1e-13)


def test_gradient_matches_finite_differences():
    """Reverse-mode gradient of a 0.5 s rollout cost equals central differences (smooth regime)."""
    p = GraphPolicy.load(SOLVED); st = J.GraphStructure.of(p)
    S0 = jnp.asarray(np.random.default_rng(0).uniform(-0.05, 0.05, (4, 6)))

    def cost(th):
        S, _ = J.rollout_batch(st, th, S0, 50, TARGET_PHYSICS, T12)
        return jnp.mean(jnp.cos(S[:, -1, 1]) + S[:, -1, 0] ** 2)
    th = jnp.asarray(p.get_flat())
    g = np.asarray(jax.grad(cost)(th))
    eps = 1e-6
    fd = np.array([(cost(th.at[i].add(eps)) - cost(th.at[i].add(-eps))) / (2 * eps) for i in range(len(th))])
    np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-7)


def test_math_tanh_reference():
    assert J._host_tanh(jnp.asarray(0.3)) == math.tanh(0.3)
