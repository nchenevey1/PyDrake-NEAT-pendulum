"""
Physics curriculum (same schedule as legacy/curriculum.py).

difficulty in [0, 1]:
  [0, 0.5]  gravity  g_min -> g_max (quadratic), drag = drag_max, force = force_start
  [0.5, 1]  drag     drag_max -> drag_min (linear), force force_start -> force_end (linear), g = g_max
difficulty = 1 is exactly the target physics, so there is no train/target gap once
the curriculum completes. `mode: none` trains directly on the target physics.

Advancement (legacy rule): if the `pass_count`-th best upright ratio in the
population (fraction of the episode spent upright, here averaged over the
generation's random initial conditions) reaches `pass_ratio`, difficulty += step.
Difficulty never decreases.
"""

from dataclasses import dataclass

import numpy as np

from .task import PhysicsParams, TARGET_PHYSICS


@dataclass
class CurriculumConfig:
    mode: str = "legacy"          # legacy | none
    start: float = 0.0
    step: float = 0.005
    pass_ratio: float = 0.2
    pass_count: int = 1
    g_min: float = 1.0
    drag_max: float = 3.0
    force_start: float = 1000.0


class Curriculum:
    def __init__(self, cfg: CurriculumConfig):
        self.cfg = cfg
        self.difficulty = 1.0 if cfg.mode == "none" else float(np.clip(cfg.start, 0, 1))
        self.advances = 0

    @property
    def at_target(self) -> bool:
        return self.difficulty >= 1.0

    def params(self) -> PhysicsParams:
        c, d, T = self.cfg, self.difficulty, TARGET_PHYSICS
        if d >= 1.0:
            return T
        if d <= 0.5:
            t = d / 0.5
            return PhysicsParams(g=c.g_min + (T.g - c.g_min) * t * t, drag=c.drag_max, max_force=c.force_start)
        t = (d - 0.5) / 0.5
        return PhysicsParams(g=T.g, drag=c.drag_max + (T.drag - c.drag_max) * t,
                             max_force=c.force_start + (T.max_force - c.force_start) * t)

    def update(self, upright_ratios: np.ndarray) -> bool:
        """Returns True if difficulty increased."""
        if self.at_target or self.cfg.mode == "none":
            return False
        r = np.sort(np.asarray(upright_ratios))[::-1]
        if len(r) >= self.cfg.pass_count and r[self.cfg.pass_count - 1] >= self.cfg.pass_ratio:
            # round to avoid accumulating float error (0.005 * 200 must land on 1.0)
            self.difficulty = float(min(1.0, round(self.difficulty + self.cfg.step, 9)))
            self.advances += 1
            return True
        return False

    def state(self) -> dict:
        return {"difficulty": self.difficulty, **self.params().to_dict()}
