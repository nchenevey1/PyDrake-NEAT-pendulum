"""Kernel episode semantics vs a straightforward python re-implementation of the legacy loop."""
import numpy as np

from pendulum.fitness import episode_scores
from pendulum.policy import build_arch, pack
from pendulum.rollout import (S_BEST_STREAK, S_REASON, S_STEPS, S_STREAK_JITTER, S_UPRIGHT,
                              physics_step, rollout)
from pendulum.task import DEFAULT_TASK as T, PhysicsParams
from tests.test_policy import obs_of


def wrap(a):
    return ((a + np.pi) % (2 * np.pi)) - np.pi


def reference_episode(policy, ic, phys):
    step = policy.make_stepper()
    s = ic.copy()
    prev_a = 0.0
    cur = best = 0
    cur_j = best_j = 0.0
    up_n = 0
    reason = 0
    k = 0
    for k in range(1, T.max_steps + 1):
        a = float(np.clip(step(obs_of(s)), -1, 1))
        physics_step(s, a * phys.max_force, phys.g, phys.drag, T.dt)
        da = abs(a - prev_a); prev_a = a
        up = abs(wrap(s[1] - np.pi)) < T.upright_tol and abs(wrap(s[2])) < T.upright_tol
        if up:
            cur += 1; cur_j += da; up_n += 1
        else:
            if cur > best:
                best, best_j = cur, cur_j
            cur, cur_j = 0, 0.0
        if abs(s[0]) > T.limit_x:
            reason = 1; break
        if k > T.swing_up_steps and not up:
            reason = 2; break
    if cur > best:
        best, best_j = cur, cur_j
    return k, best, best_j, up_n, reason


def test_episode_stats_match_reference():
    rng = np.random.default_rng(5)
    # easy physics so that some random policies get upright
    phys = PhysicsParams(1.0, 3.0, 1000.0)
    pols = [build_arch("mlp:4", 9, rng) for _ in range(300)]
    ics = rng.uniform(-0.05, 0.05, (2, 6))
    stats, _ = rollout(pack(pols), ics, phys)
    n_up = 0
    for i in range(0, 300, 7):
        for k in range(2):
            ref = reference_episode(pols[i], ics[k], phys)
            got = stats[i, k]
            assert (got[S_STEPS], got[S_BEST_STREAK], got[S_UPRIGHT], got[S_REASON]) == (ref[0], ref[1], ref[3], ref[4])
            np.testing.assert_allclose(got[S_STREAK_JITTER], ref[2], atol=1e-9)
            n_up += ref[1] > 0
    assert n_up > 0, "test should cover episodes with upright streaks"


def test_legacy_fitness_saturates_but_default_does_not():
    st = np.zeros((2, 11))
    st[0, S_BEST_STREAK], st[0, S_STREAK_JITTER] = 1750, 1750 * 0.05   # 17.5 s, jittery
    st[1, S_BEST_STREAK], st[1, S_STREAK_JITTER] = 30, 30 * 0.01       # 0.3 s, smooth
    leg = episode_scores(st, "legacy", T)
    new = episode_scores(st, "streak_smooth", T)
    assert leg[1] > leg[0]
    assert new[0] > new[1]
