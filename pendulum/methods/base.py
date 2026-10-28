"""Ask/tell interface shared by all optimizers."""

from typing import Dict, List

import numpy as np

from ..policy import GraphPolicy


class Method:
    """A population-based optimizer over GraphPolicies.

    Each generation the trainer calls `ask()`, evaluates every returned policy on
    the same random initial conditions, then calls `tell()` with
    `fitness` (score with size penalty applied; higher is better), the raw task
    `scores`, and the per-episode `stats[P, K, N_STATS]`.
    """

    name = "base"

    def __init__(self, cfg: dict, n_in: int, rng: np.random.Generator):
        self.cfg = cfg
        self.n_in = n_in
        self.rng = rng

    def ask(self) -> List[GraphPolicy]:
        raise NotImplementedError

    def tell(self, fitness: np.ndarray, scores: np.ndarray, stats: np.ndarray) -> None:
        raise NotImplementedError

    def penalty_sizes(self, policies: List[GraphPolicy]) -> np.ndarray:
        """Size used by the fitness size penalty (default: deployable parameter count)."""
        return np.array([p.n_params for p in policies])

    def champions(self) -> List[GraphPolicy]:
        """Candidates for deployment (validated on the target physics by the trainer)."""
        raise NotImplementedError

    def on_physics_change(self) -> None:
        """Called after the curriculum changed the training physics."""

    def info(self) -> Dict:
        return {}


def centered_ranks(x: np.ndarray) -> np.ndarray:
    r = np.empty(len(x))
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r / max(len(x) - 1, 1) - 0.5
