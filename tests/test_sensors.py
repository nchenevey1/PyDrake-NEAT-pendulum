"""Sensor model: numba kernel vs the python reference, and the off-by-default contract."""
import numpy as np
import pytest
from dataclasses import replace

from pendulum.policy import build_arch, pack
from pendulum.rollout import rollout, S_STEPS, S_BEST_STREAK, S_REASON
from pendulum.sensors import (SensorConfig, SensedPolicy, Sensor, back_extrapolated_warm,
                              py_episode, rollout_sensed, simulate_sensed)
from pendulum.task import DEFAULT_TASK, PhysicsParams, TARGET_PHYSICS

T = replace(DEFAULT_TASK, max_v_cart=12.0, max_w_pole=12.0, duration=3.0)


def test_full_mode_equals_core_rollout():
    rng = np.random.default_rng(0)
    pols = [build_arch("mlp:4", 9, rng) for _ in range(20)] + [build_arch("rnn:4", 9, rng) for _ in range(5)]
    ics = rng.uniform(-0.05, 0.05, (4, 6))
    a, na = rollout(pack(pols), ics, TARGET_PHYSICS, T)
    b, nb_ = rollout_sensed(pack(pols), ics, TARGET_PHYSICS, T, SensorConfig())
    assert na == nb_
    np.testing.assert_array_equal(a, b)


def test_floor_quantization():
    s = Sensor(SensorConfig(mode="fd", cpr=256, x_res=1e-3))
    q = 2 * np.pi / 256
    m = s.measure(np.array([0.0019, 0.5 * q, -0.5 * q, 0, 0, 0]))
    np.testing.assert_allclose(m, [0.001, 0.0, -q])


@pytest.mark.parametrize("cfg,arch", [
    (SensorConfig(mode="fd", cpr=1024, x_res=1e-4), "mlp:4"),
    (SensorConfig(mode="hist", k=3, cpr=256, x_res=1e-3, delay=1), "mlp:4"),
    (SensorConfig(mode="hist", k=4, cpr=4096, x_res=1e-5, force_bits=8), "mlp:2"),
    (SensorConfig(mode="raw", k=2, cpr=1024, x_res=1e-4, delay=1, force_bits=8), "mlp:4"),
    (SensorConfig(mode="pos", cpr=1024, x_res=1e-4, delay=1), "rnn:4"),
    (SensorConfig(mode="full", delay=1, force_bits=8), "rnn:4"),
])
@pytest.mark.parametrize("warm", [False, True])
def test_kernel_matches_python_reference(cfg, arch, warm):
    rng = np.random.default_rng(3)
    phys = PhysicsParams(1.0, 3.0, 1000.0)       # easy physics: long, varied episodes
    for i in range(4):
        pol = build_arch(arch, cfg.n_in, rng)
        ic = rng.uniform(-0.05, 0.05, 6) + np.array([0, 0.3, 0, 0.5, 2.0, -1.0])
        w = back_extrapolated_warm(ic[None], 5, T.dt)[0] if warm else None
        st, tr = simulate_sensed(pol, ic, phys, T, cfg, w)
        steps, _, ref = py_episode(SensedPolicy.from_graph(pol, cfg, T), ic, phys, T, warm=w)
        assert steps == st[S_STEPS]
        np.testing.assert_allclose(tr[:, :7], ref, atol=1e-8)


def test_batch_matches_single():
    rng = np.random.default_rng(1)
    cfg = SensorConfig(mode="hist", k=3, cpr=1024, x_res=1e-4, delay=1, force_bits=8)
    pols = [build_arch("mlp:4", cfg.n_in, rng) for _ in range(6)]
    ics = rng.uniform(-0.05, 0.05, (3, 6))
    phys = PhysicsParams(1.0, 3.0, 1000.0)
    st, _ = rollout_sensed(pack(pols), ics, phys, T, cfg)
    for i, p in enumerate(pols):
        for k in range(3):
            s1, _ = simulate_sensed(p, ics[k], phys, T, cfg)
            np.testing.assert_array_equal(st[i, k], s1)


def test_input_count_checked():
    with pytest.raises(AssertionError):
        rollout_sensed(pack([build_arch("mlp:4", 9)]), np.zeros((1, 6)), TARGET_PHYSICS, T,
                       SensorConfig(mode="hist", k=3, cpr=1024))
