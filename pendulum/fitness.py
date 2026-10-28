"""
Training scores computed from per-episode kernel statistics.

All scores are *sparse*: they are zero unless the poles are actually upright
(both within `upright_tol` of vertical), and they are driven by the longest
*consecutive* upright streak, so spinning through the upright position earns
only a few steps' worth. No dense shaping terms.

  legacy        streak_time / (1 + sum|da| inside the streak)           (original formula)
  streak        streak_time
  streak_smooth streak_time / (1 + mean|da| inside the streak)          (default)

Why not `legacy` by default: the jitter sum grows with streak length, so for a
controller whose per-step action change is r the score is T / (1 + r T / dt),
which saturates at dt / r. E.g. balancing 17.5 s with r = 0.05 scores 0.20,
while balancing 0.3 s with r = 0.01 scores 0.23: the formula can rank short,
smooth balancing above sustained balancing. `streak_smooth` keeps the same
smoothness preference but bounds the divisor to [1, 3], so a longer streak
always dominates.
"""

import numpy as np

from .rollout import S_BEST_STREAK, S_STREAK_JITTER, S_UPRIGHT
from .task import TaskConfig


def episode_scores(stats: np.ndarray, kind: str, task: TaskConfig) -> np.ndarray:
    """stats[..., N_STATS] -> score per episode (same leading shape)."""
    streak = stats[..., S_BEST_STREAK]
    t = streak * task.dt
    jit = stats[..., S_STREAK_JITTER]
    if kind == "legacy":
        return t / (1.0 + jit)
    if kind == "streak":
        return t
    if kind == "streak_smooth":
        return t / (1.0 + jit / np.maximum(streak, 1.0))
    raise ValueError(f"Unknown fitness kind: {kind}")


def policy_scores(stats: np.ndarray, kind: str, task: TaskConfig) -> np.ndarray:
    """stats[P, K, N_STATS] -> mean score over the K trials, shape (P,)."""
    return episode_scores(stats, kind, task).mean(axis=1)


def upright_ratio(stats: np.ndarray, task: TaskConfig) -> np.ndarray:
    """Fraction of the full episode spent upright, averaged over trials (curriculum signal)."""
    return (stats[..., S_UPRIGHT] / task.max_steps).mean(axis=1)
