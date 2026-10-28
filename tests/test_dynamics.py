"""The numba physics must reproduce the Drake MultibodyPlant used by the legacy code."""
import numpy as np
import pytest

from pendulum.rollout import physics_step
from pendulum.task import PhysicsParams

drake_env = pytest.importorskip("pendulum.drake_env")


@pytest.mark.parametrize("phys", [PhysicsParams(9.81, 0.0, 100.0), PhysicsParams(1.0, 3.0, 1000.0),
                                  PhysicsParams(4.0, 1.5, 550.0)])
def test_one_step_matches_drake(phys):
    env = drake_env.DrakeEnv(phys)
    rng = np.random.default_rng(0)
    for _ in range(100):
        s0 = np.concatenate([rng.uniform(-2.5, 2.5, 1), rng.uniform(-np.pi, np.pi, 2), rng.uniform(-8, 8, 3)])
        u = rng.uniform(-1, 1)
        env.reset(s0)
        sd = env.step(u)
        s = s0.copy()
        physics_step(s, u * phys.max_force, phys.g, phys.drag, 0.01)
        np.testing.assert_allclose(s, sd, atol=1e-11)


def test_long_trajectory_matches_drake():
    """2000 steps of feedback-driven motion on the track (cart PD keeps |x| < 2.8)."""
    phys = PhysicsParams(9.81, 0.0, 100.0)
    env = drake_env.DrakeEnv(phys)
    ic = np.array([0.02, 0.03, -0.01, 0.0, 0.01, 0.0])
    sd = env.reset(ic)
    s = ic.copy()
    for k in range(2000):
        u = float(np.clip(-0.3 * s[0] - 0.2 * s[3] + 0.4 * np.sin(0.05 * k), -1, 1))
        sd = env.step(u)
        physics_step(s, u * phys.max_force, phys.g, phys.drag, 0.01)
        assert abs(s[0]) < 2.8
    # chaotic system: allow round-off growth, but no timing glitches (those give O(1e-2) errors)
    np.testing.assert_allclose(s, sd, atol=1e-6)
