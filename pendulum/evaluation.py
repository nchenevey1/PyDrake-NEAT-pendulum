"""
Evaluation on the *true objective*: swing up and keep both poles upright under
the target physics (g = 9.81, no drag, 100 N), from many random initial states.

An episode is a **success** iff it runs the full horizon without termination,
i.e. the poles are upright within `swing_up_time` and stay within `upright_tol`
of vertical on every later step while the cart stays on the track.
"""

from dataclasses import replace
from typing import Dict, List, Optional

import numpy as np

from .policy import GraphPolicy, pack
from .rollout import (rollout, S_BEST_STREAK, S_FIRST_UP, S_REASON, S_STEPS, S_UPRIGHT,
                      S_STREAK_JITTER, S_MEAN_ABS_X)
from .task import (DEFAULT_TASK, TARGET_PHYSICS, TEST_SEED, VAL_SEED, PhysicsParams, TaskConfig,
                   fixed_initial_states)


def summarize(stats: np.ndarray, task: TaskConfig) -> Dict[str, float]:
    """stats[K, N_STATS] for one policy -> summary metrics."""
    success = (stats[:, S_STEPS] >= task.max_steps) & (stats[:, S_REASON] == 0)
    first_up = stats[:, S_FIRST_UP]
    streak = stats[:, S_BEST_STREAK]
    return {
        "success_rate": float(success.mean()),
        "mean_streak_s": float(streak.mean() * task.dt),
        "min_streak_s": float(streak.min() * task.dt),
        "upright_frac": float((stats[:, S_UPRIGHT] / task.max_steps).mean()),
        "swing_up_s": float(np.mean(first_up[first_up > 0]) * task.dt) if np.any(first_up > 0) else None,
        "streak_jitter_per_step": float(np.mean(stats[:, S_STREAK_JITTER] / np.maximum(streak, 1))),
        "mean_abs_x": float(stats[:, S_MEAN_ABS_X].mean()),
        "fail_track": float(np.mean(stats[:, S_REASON] == 1)),
        "fail_fell": float(np.mean(stats[:, S_REASON] == 2)),
        "n": int(len(stats)),
    }


def evaluate_policies(policies: List[GraphPolicy], ics: np.ndarray,
                      physics: PhysicsParams = TARGET_PHYSICS,
                      task: TaskConfig = DEFAULT_TASK) -> List[Dict[str, float]]:
    stats, _ = rollout(pack(policies), ics, physics, task)
    return [summarize(s, task) for s in stats]


def val_ics(n: int, task: TaskConfig = DEFAULT_TASK) -> np.ndarray:
    return fixed_initial_states(n, VAL_SEED, task.ic_noise)


def test_ics(n: int, task: TaskConfig = DEFAULT_TASK, noise: Optional[float] = None) -> np.ndarray:
    return fixed_initial_states(n, TEST_SEED, task.ic_noise if noise is None else noise)


def selection_key(summary: Dict[str, float], n_params: int):
    """Model-selection order: success rate, then mean streak, then fewer parameters."""
    return (round(summary["success_rate"], 6), round(summary["mean_streak_s"], 3), -n_params)


def greedy_prune(policy: GraphPolicy, ics: np.ndarray, tolerance: float = 0.0,
                 physics: PhysicsParams = TARGET_PHYSICS, task: TaskConfig = DEFAULT_TASK,
                 max_rounds: int = 1000) -> GraphPolicy:
    """Repeatedly remove the connection whose removal hurts the success rate least,
    as long as success stays >= baseline - tolerance. Candidates for each round are
    evaluated in one batched rollout."""
    base = evaluate_policies([policy], ics, physics, task)[0]["success_rate"]
    floor = base - tolerance
    cur = policy.compacted()
    for _ in range(max_rounds):
        if cur.n_connections <= 1:
            break
        cands = [cur.without_connections([i]) for i in range(len(cur.w))]
        res = evaluate_policies(cands, ics, physics, task)
        order = sorted(range(len(cands)), key=lambda i: (-res[i]["success_rate"], -res[i]["mean_streak_s"],
                                                         cands[i].n_params))
        best = order[0]
        if res[best]["success_rate"] < floor:
            break
        cur = cands[best]
    return cur


def full_report(policy: GraphPolicy, n_test: int = 256, task: TaskConfig = DEFAULT_TASK,
                prune: bool = True) -> Dict:
    """Test-set report used for the final comparison."""
    ics = test_ics(n_test, task)
    long_task = replace(task, duration=60.0)
    wide = test_ics(n_test, task, noise=0.2)
    rep = {
        "params": policy.n_params,
        "arch": policy.describe(),
        "test_20s": evaluate_policies([policy], ics, task=task)[0],
        "test_60s": evaluate_policies([policy], ics, task=long_task)[0],
        "test_wide_ic_0.2": evaluate_policies([policy], wide, task=task)[0],
        "nominal_ic": evaluate_policies([policy], np.zeros((1, 6)), task=task)[0],
    }
    for bits in (8, 6, 4):
        rep[f"int{bits}_20s"] = evaluate_policies([policy.quantized(bits)], ics, task=task)[0]
    if prune and rep["test_20s"]["success_rate"] > 0:
        # Prune on the validation set (never on test), then report on test
        pr = greedy_prune(policy, val_ics(128, task), tolerance=0.02, task=task)
        rep["pruned"] = {"params": pr.n_params, "arch": pr.describe(),
                         "test_20s": evaluate_policies([pr], ics, task=task)[0],
                         "int8_20s": evaluate_policies([pr.quantized(8)], ics, task=task)[0]}
        rep["_pruned_policy"] = pr
    return rep
