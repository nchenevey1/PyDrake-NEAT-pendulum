"""
Hybrid: short NEAT architecture search, then CMA-ES on the weights of the
best NEAT topology (fixed structure, all weights and biases tuned jointly).
"""

from typing import Dict, List

import numpy as np

from ..policy import GraphPolicy
from .base import Method
from .es import CMAES
from .neat import NEAT

HYBRID_DEFAULTS = {
    "switch_generation": 300,
    "neat": {},
    "cmaes": {"popsize": 64, "sigma0": 0.2},
}


class NEATThenCMA(Method):
    name = "neat_cma"

    def __init__(self, cfg: dict, n_in: int, rng: np.random.Generator):
        cfg = {**HYBRID_DEFAULTS, **(cfg or {})}
        super().__init__(cfg, n_in, rng)
        self.neat = NEAT(cfg["neat"], n_in, rng)
        self.cma = None
        self.generation = 0

    @property
    def active(self) -> Method:
        return self.cma if self.cma is not None else self.neat

    def ask(self) -> List[GraphPolicy]:
        if self.cma is None and self.generation >= self.cfg["switch_generation"]:
            template = self.neat.champions()[0].compacted()
            template.meta["arch"] = "neat_cma"
            self.cma = CMAES(self.cfg["cmaes"], self.n_in, self.rng, template=template)
        return self.active.ask()

    def tell(self, fitness, scores, stats) -> None:
        self.active.tell(fitness, scores, stats)
        self.generation += 1

    def champions(self) -> List[GraphPolicy]:
        return self.active.champions()

    def on_physics_change(self) -> None:
        self.active.on_physics_change()

    def info(self) -> Dict:
        return {"phase": "cma" if self.cma else "neat", **self.active.info()}
