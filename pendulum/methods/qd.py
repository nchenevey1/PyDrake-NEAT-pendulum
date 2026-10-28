"""
MAP-Elites (quality-diversity) over a fixed architecture's weights.

Behaviour descriptor (mean over the generation's trials):
  d1 = peak tip height reached, mapped from [-2, 2] m to [0, 1]
  d2 = mean |cart position| / track half-length
The archive keeps the fittest policy per cell, so policies that swing the tip
higher are retained as stepping stones even while their sparse score is still
zero. Variation: Iso+LineDD (Vassiliades & Mouret 2018). Because the training
fitness is noisy (random ICs) and changes with the curriculum, the whole
archive is re-evaluated after every curriculum advance and every
`reeval_every` generations.
"""

from typing import Dict, List

import numpy as np

from ..policy import GraphPolicy, build_arch
from ..rollout import S_MAX_HEIGHT, S_MEAN_ABS_X
from ..task import DEFAULT_TASK
from .base import Method

MAP_ELITES_DEFAULTS = {
    "arch": "mlp:4",
    "grid": [20, 10],
    "batch": 256,
    "init_random": 1024,
    "iso_sigma": 0.1,
    "line_sigma": 0.2,
    "reeval_every": 20,
}


class MAPElites(Method):
    name = "map_elites"

    def __init__(self, cfg: dict, n_in: int, rng: np.random.Generator):
        super().__init__({**MAP_ELITES_DEFAULTS, **(cfg or {})}, n_in, rng)
        self.template = build_arch(self.cfg["arch"], n_in, rng)
        self.bound = self.template.flat_bounds()
        self.grid = tuple(self.cfg["grid"])
        n_cells = int(np.prod(self.grid))
        self.theta = np.zeros((n_cells, self.template.n_flat))
        self.fit = np.full(n_cells, -np.inf)
        self.score = np.zeros(n_cells)
        self.n_random = 0
        self.generation = 0
        self._pending = None
        self._reeval_cells = None
        self._need_reeval = False

    def _cells(self, stats: np.ndarray) -> np.ndarray:
        d1 = np.clip((stats[:, :, S_MAX_HEIGHT].mean(1) + 2.0) / 4.0, 0, 1 - 1e-9)
        d2 = np.clip(stats[:, :, S_MEAN_ABS_X].mean(1) / DEFAULT_TASK.limit_x, 0, 1 - 1e-9)
        i = (d1 * self.grid[0]).astype(int)
        j = (d2 * self.grid[1]).astype(int)
        return i * self.grid[1] + j

    def ask(self) -> List[GraphPolicy]:
        c = self.cfg
        filled = np.flatnonzero(np.isfinite(self.fit))
        if (self._need_reeval or (c["reeval_every"] and self.generation % c["reeval_every"] == 0
                                  and self.generation > 0)) and len(filled):
            self._reeval_cells = filled
            self._pending = self.theta[filled].copy()
        elif self.n_random < c["init_random"] or len(filled) < 2:
            self._reeval_cells = None
            self._pending = np.stack([build_arch(c["arch"], self.n_in, self.rng).get_flat()
                                      for _ in range(c["batch"])])
            self.n_random += c["batch"]
        else:
            self._reeval_cells = None
            a = self.theta[self.rng.choice(filled, c["batch"])]
            b = self.theta[self.rng.choice(filled, c["batch"])]
            self._pending = np.clip(a + c["iso_sigma"] * self.rng.standard_normal(a.shape)
                                    + c["line_sigma"] * self.rng.standard_normal((len(a), 1)) * (b - a),
                                    -self.bound, self.bound)
        return [self.template.with_flat(t) for t in self._pending]

    def tell(self, fitness, scores, stats) -> None:
        f = np.asarray(fitness)
        if self._reeval_cells is not None:
            self.fit[:] = -np.inf
            self._need_reeval = False
        cells = self._cells(stats)
        for k in np.argsort(-f, kind="stable"):
            cell = cells[k]
            if f[k] > self.fit[cell]:
                self.fit[cell] = f[k]
                self.score[cell] = scores[k]
                self.theta[cell] = self._pending[k]
        self.generation += 1

    def on_physics_change(self) -> None:
        self._need_reeval = True

    def champions(self) -> List[GraphPolicy]:
        filled = np.flatnonzero(np.isfinite(self.fit))
        top = filled[np.argsort(-self.fit[filled])[:3]]
        return [self.template.with_flat(self.theta[i]) for i in top]

    def info(self) -> Dict:
        filled = np.isfinite(self.fit)
        return {"coverage": round(float(filled.mean()), 3),
                "archive_best": round(float(self.fit[filled].max()), 3) if filled.any() else None,
                "params": self.template.n_params}
