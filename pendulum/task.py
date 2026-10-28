"""
Task definition: cart + double pendulum swing-up and balance.

Everything that defines *the problem* lives here (time step, horizon, limits,
success tolerance, observation scaling, initial-condition distribution and the
target physics). Training methods must not change these; they may only choose
easier *training* physics via the curriculum.

Values mirror the original `legacy/config.py`.
"""

from dataclasses import dataclass, asdict, field
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
    max_v_cart: float = 50.0       # observation scaling
    max_w_pole: float = 90.0       # observation scaling
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
