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

`mode: start_state` is the start-state curriculum (S2, `StartStateCurriculum` below): always the
target physics, episodes start further and further back along a reference swing-up.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .policy import pack
from .rollout import S_STEPS, rollout
from .task import PhysicsParams, TARGET_PHYSICS, TaskConfig


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
    mode = "physics"

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

    def describe(self) -> str:
        p = self.params()
        return f"d={self.difficulty:.3f} (g={p.g:.2f} drag={p.drag:.2f} F={p.max_force:.0f})"

    def summary(self) -> dict:
        return {"mode": self.cfg.mode, "final_difficulty": self.difficulty, "advances": self.advances}


# =============================================================================
# Start-state curriculum (S2, docs/CURRICULUM.md)
# =============================================================================

@dataclass
class StartStateConfig:
    """Level k: episodes start at the reference swing-up state S_ref[k] (+ the task's start
    noise), with swing-up deadline 2.5 s - k*dt and episode length 20 s - k*dt, under the target
    physics. k = 0 is exactly the real task. Values from docs/CURRICULUM.md (not tuned)."""
    reference: str = "results/classic/controller.npz"   # classic controller; S_nom is the reference
    k_start: int = 210            # first level: the reference is upright and at rest from ~2.1 s
    step: int = 5                 # control steps per advance (0.05 s of reference time)
    adaptive_step: bool = True    # after a back-off, halve the step (5 -> 2 -> 1)
    min_step: int = 1
    mastery_every: int = 5        # generations between mastery checks
    mastery_starts: int = 16      # fresh starts per mastery check
    mastery_frac: float = 0.8     # deployable policy must fully succeed on >= this fraction
    stuck_gens: int = 300         # back off after this many generations without a full success
    replay_frac: float = 0.5      # fraction of training starts at a random mastered level


class StartStateCurriculum:
    """Mastery rule: every `mastery_every` generations the deployable policy (CMA-ES mean,
    NEAT champion) is run on `mastery_starts` fresh starts at the current level; if it fully
    succeeds on >= `mastery_frac` of them, k decreases by `step`. If no candidate fully succeeds
    at the current level for `stuck_gens` generations, k increases by `step` (back-off) and, with
    `adaptive_step`, the step is halved so the hard part of the reference is crossed gradually.
    Training starts: (1 - replay_frac) at level k, replay_frac at a level drawn uniformly from
    [k, k_start] (forgetting), also after k = 0 is reached."""
    mode = "start_state"

    def __init__(self, cfg: StartStateConfig, task: TaskConfig, physics: PhysicsParams = TARGET_PHYSICS):
        self.cfg = cfg
        self.task = task
        self.physics = physics
        path = Path(cfg.reference)
        if not path.is_absolute() and not path.exists():      # relative to the repository root
            path = Path(__file__).resolve().parent.parent / path
        self.reference = np.load(path)["S_nom"][: cfg.k_start + 1].copy()
        if len(self.reference) != cfg.k_start + 1:
            raise ValueError(f"reference has {len(self.reference)} states, need k_start + 1")
        self.k = int(cfg.k_start)
        self.step = int(cfg.step)
        self.best_k = self.k
        self.stuck = 0
        self.advances = 0
        self.backoffs = 0
        self.events = []          # level changes: {gen, env_steps, event, k, step}

    # ----------------------------------------------------------- levels
    @property
    def difficulty(self) -> float:
        """0 at the first level, 1 at the real task."""
        return 1.0 - self.k / self.cfg.k_start

    @property
    def at_target(self) -> bool:
        return self.k == 0

    def level_task(self, k: int) -> TaskConfig:
        t = self.task
        return replace(t, swing_up_time=max(0.0, t.swing_up_time - k * t.dt), duration=t.duration - k * t.dt)

    def level_starts(self, k: int, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.reference[k] + rng.uniform(-self.task.ic_noise, self.task.ic_noise, (n, 6))

    def batches(self, rng: np.random.Generator, n_trials: int):
        """Training batches for one generation: list of (level, starts, task, weight). The
        generation's score is sum(weight * score / level duration)."""
        n_rep = int(round(n_trials * self.cfg.replay_frac))
        n_cur = n_trials - n_rep
        k_rep = int(rng.integers(self.k, self.cfg.k_start + 1))
        out = [(self.k, self.level_starts(self.k, n_cur, rng), self.level_task(self.k), n_cur / n_trials)]
        if n_rep:
            out.append((k_rep, self.level_starts(k_rep, n_rep, rng), self.level_task(k_rep), n_rep / n_trials))
        return out

    # ----------------------------------------------------------- update
    def update(self, gen: int, env_steps: int, best_full_at_level: float, deployable, rng) -> tuple:
        """Called once per generation. `best_full_at_level`: fraction of the current level's
        training starts on which the best candidate ran the full episode. `deployable()` returns
        the policy whose mastery is checked. Returns (changed, mastery env steps used)."""
        c = self.cfg
        used = 0
        changed = False
        self.stuck = 0 if best_full_at_level > 0 else self.stuck + 1
        if self.k > 0 and gen % c.mastery_every == 0:
            tk = self.level_task(self.k)
            stats, used = rollout(pack([deployable()]), self.level_starts(self.k, c.mastery_starts, rng),
                                  self.physics, tk)
            if np.mean(stats[0, :, S_STEPS] >= tk.max_steps) >= c.mastery_frac:
                self.k = max(0, self.k - self.step)
                self.stuck = 0
                self.advances += 1
                self.best_k = min(self.best_k, self.k)
                changed = True
                self._event(gen, env_steps + used, "advance")
        if self.stuck >= c.stuck_gens and self.k < c.k_start:
            self.k = min(c.k_start, self.k + self.step)
            if c.adaptive_step:
                self.step = max(c.min_step, self.step // 2)
            self.stuck = 0
            self.backoffs += 1
            changed = True
            self._event(gen, env_steps + used, "backoff")
        return changed, used

    def _event(self, gen, env_steps, kind):
        self.events.append({"gen": gen, "env_steps": int(env_steps), "event": kind, "k": self.k,
                            "start_t": round(self.k * self.task.dt, 4), "step": self.step})

    def state(self) -> dict:
        return {"difficulty": round(self.difficulty, 5), "k": self.k, "start_t": round(self.k * self.task.dt, 4),
                "step": self.step, "best_k": self.best_k, "stuck": self.stuck}

    def describe(self) -> str:
        return f"k={self.k:3d} (t={self.k * self.task.dt:.2f}s step {self.step}, best {self.best_k * self.task.dt:.2f}s)"

    def summary(self) -> dict:
        return {"mode": self.mode, "final_k": self.k, "final_start_t": round(self.k * self.task.dt, 4),
                "best_k": self.best_k, "best_start_t": round(self.best_k * self.task.dt, 4),
                "final_step": self.step, "advances": self.advances, "backoffs": self.backoffs,
                "events": list(self.events)}


def make_curriculum(cfg: dict, task: TaskConfig):
    """Curriculum from the `curriculum` config section (mode: legacy | none | start_state)."""
    cfg = dict(cfg)
    ss = cfg.pop("start_state", None) or {}
    if cfg.get("mode") == "start_state":
        return StartStateCurriculum(StartStateConfig(**ss), task)
    return Curriculum(CurriculumConfig(**cfg))
