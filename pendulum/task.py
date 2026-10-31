"""
Task definition: cart + double pendulum swing-up and balance.

Everything that defines *the problem* lives here (time step, horizon, limits,
success tolerance, observation scaling, initial-condition distribution and the
target physics). Training methods must not change these; they may only choose
easier *training* physics via the curriculum.

Values mirror the original `legacy/config.py`.
"""

from dataclasses import dataclass, asdict, field, replace
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class PhysicsParams:
    """Physics knobs the curriculum is allowed to vary."""
    g: float = 9.81            # gravity [m/s^2]
    drag: float = 0.0          # linear damping on all three joints
    max_force: float = 100.0   # actuator limit [N]; action in [-1, 1] is scaled by this

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


# The real problem: full gravity, no drag, 100 N actuator (end of legacy curriculum)
TARGET_PHYSICS = PhysicsParams(g=9.81, drag=0.0, max_force=100.0)


@dataclass(frozen=True)
class TaskConfig:
    dt: float = 0.01               # integration step [s]
    duration: float = 20.0         # episode length [s]
    limit_x: float = 2.8           # track half-length [m]; leaving it ends the episode
    swing_up_time: float = 2.5     # after this, any non-upright step ends the episode [s]
    upright_tol: float = 0.2       # |angle error| threshold for both poles [rad]
    max_v_cart: float = 12.0       # observation scaling: cart velocity [m/s] mapped to +-1
    max_w_pole: float = 12.0       # observation scaling: pole angular velocities [rad/s] mapped to +-1
    ic_noise: float = 0.05         # half-width of uniform IC perturbation around hanging rest

    @property
    def max_steps(self) -> int:
        return int(round(self.duration / self.dt))

    @property
    def swing_up_steps(self) -> int:
        return int(round(self.swing_up_time / self.dt))

    def kernel_params(self) -> np.ndarray:
        """Packed float array consumed by the numba rollout kernel."""
        return np.array([
            self.dt, float(self.max_steps), self.limit_x, float(self.swing_up_steps),
            self.upright_tol, self.max_v_cart, self.max_w_pole,
        ], dtype=np.float64)

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_TASK = TaskConfig()

# Observation scales. The default (12) covers the real velocity range of a swing-up (|v| <= ~10);
# see docs/DIAGNOSIS.md section 3. Policies saved before the scales were stored with the policy
# were all trained with the legacy scales (50 m/s, 90 rad/s), which compress that range to ~+-0.1.
OBS_SCALE_KEYS = ("max_v_cart", "max_w_pole")
LEGACY_OBS_SCALE = {"max_v_cart": 50.0, "max_w_pole": 90.0}


def obs_scale(task: TaskConfig = DEFAULT_TASK) -> Dict[str, float]:
    """The observation scales of a task, as stored in a policy's `meta["obs_scale"]`."""
    return {k: float(getattr(task, k)) for k in OBS_SCALE_KEYS}


def with_obs_scale(task: TaskConfig, scale: Dict[str, float] = None) -> TaskConfig:
    """`task` with its observation scales replaced (only the scales may differ from the task)."""
    scale = scale or {}
    bad = set(scale) - set(OBS_SCALE_KEYS)
    if bad:
        raise ValueError(f"only observation scales may be changed, got {sorted(bad)}")
    return replace(task, **{k: float(v) for k, v in scale.items()})


def task_for_policy(policy, task: TaskConfig = DEFAULT_TASK) -> TaskConfig:
    """The task with the observation scales stored in the policy's `meta["obs_scale"]`.
    `GraphPolicy.load` fills in the legacy scales for files that predate stored scales; a policy
    built in this process without stored scales uses the task's own."""
    return with_obs_scale(task, policy.meta.get("obs_scale"))


# Observation layout (9-D, identical to the legacy environment):
#   [x/Lx, clip(xd/Vmax), sin th1, cos th1, clip(w1/Wmax), sin th2, cos th2, clip(w2/Wmax), cos(th1 - th2)]
# th1 is the absolute shoulder angle (0 = hanging), th2 is the *relative* elbow angle.
OBS_DIM = 9
STATE_DIM = 6  # [x, th1, th2, xd, w1, w2]


def sample_initial_states(n: int, rng: np.random.Generator, noise: float) -> np.ndarray:
    """Uniform perturbation of all six states around the hanging rest state."""
    if noise <= 0:
        return np.zeros((n, STATE_DIM))
    return rng.uniform(-noise, noise, size=(n, STATE_DIM))


def fixed_initial_states(n: int, seed: int, noise: float) -> np.ndarray:
    """Deterministic IC set (used for validation / test evaluation)."""
    return sample_initial_states(n, np.random.default_rng(seed), noise)


# Held-out IC sets. Validation is used for model selection during training,
# test is only used for the final report.
VAL_SEED = 1_000_003
TEST_SEED = 2_000_003


def observe(s: np.ndarray, task: TaskConfig = DEFAULT_TASK) -> np.ndarray:
    """Python reference of the 9-D observation (the numba kernel has its own copy)."""
    s1, c1, s2, c2 = np.sin(s[1]), np.cos(s[1]), np.sin(s[2]), np.cos(s[2])
    return np.array([s[0] / task.limit_x, np.clip(s[3] / task.max_v_cart, -1, 1), s1, c1,
                     np.clip(s[4] / task.max_w_pole, -1, 1), s2, c2,
                     np.clip(s[5] / task.max_w_pole, -1, 1), s1 * s2 + c1 * c2])
