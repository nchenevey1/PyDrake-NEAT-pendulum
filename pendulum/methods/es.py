"""
Fixed-topology weight optimizers: CMA-ES (pycma) and OpenAI-ES.

The architecture is given as a spec string (`linear`, `mlp:4`, `rnn:4`, ...) or
as an explicit template policy (used by the NEAT->CMA hybrid).
"""

import io
import contextlib
from typing import Dict, List, Optional

import cma
import numpy as np

from ..policy import GraphPolicy, build_arch
from .base import Method, centered_ranks

CMAES_DEFAULTS = {
    "arch": "mlp:4",
    "popsize": 64,
    "sigma0": 0.5,
    "diagonal": False,      # sep-CMA for large parameter counts
    "restarts": True,       # restart (same popsize) when pycma reports stagnation
    "init_std": 1.0,
}

OPENAI_ES_DEFAULTS = {
    "arch": "mlp:4",
    "pairs": 64,            # antithetic pairs -> 2 * pairs samples per generation
    "sigma": 0.1,
    "lr": 0.03,
    "l2": 0.005,
    "beta1": 0.9,
    "beta2": 0.999,
    "init_std": 1.0,
}


class CMAES(Method):
    name = "cmaes"

    def __init__(self, cfg: dict, n_in: int, rng: np.random.Generator,
                 template: Optional[GraphPolicy] = None):
        super().__init__({**CMAES_DEFAULTS, **(cfg or {})}, n_in, rng)
        self._from_spec = template is None
        self.template = template if template is not None else build_arch(self.cfg["arch"], n_in, rng)
        self.restarts = 0
        self.generation = 0
        self._new_es(self.template.get_flat())
        self._solutions = None
        self._champions: List[GraphPolicy] = []

    def _new_es(self, x0: np.ndarray):
        c = self.cfg
        opts = {"popsize": c["popsize"], "seed": int(self.rng.integers(1, 2**31 - 1)),
                "verbose": -9, "maxiter": float("inf"), "tolfun": 1e-12, "tolfunhist": 1e-12,
                "tolflatfitness": 1000, "tolstagnation": 10**9}
        if c["diagonal"]:
            opts["CMA_diagonal"] = True
        with contextlib.redirect_stdout(io.StringIO()):
            self.es = cma.CMAEvolutionStrategy(x0, c["sigma0"], opts)

    def ask(self) -> List[GraphPolicy]:
        self._solutions = self.es.ask()
        pols = [self.template.with_flat(x) for x in self._solutions]
        pols.append(self.template.with_flat(self.es.mean))   # distribution mean (not used for update)
        return pols

    def tell(self, fitness, scores, stats) -> None:
        f = np.asarray(fitness)
        self.es.tell(self._solutions, list(-f[:-1]))
        best = int(np.argmax(f[:-1]))
        self._champions = [self.template.with_flat(self.es.mean),
                           self.template.with_flat(self._solutions[best])]
        self.generation += 1
        if self.cfg["restarts"] and self.es.stop():
            self.restarts += 1
            if self._from_spec:
                x0 = build_arch(self.cfg["arch"], self.n_in, self.rng).get_flat()
            else:
                x0 = self.template.get_flat() + self.rng.normal(0, 0.1, self.template.n_flat)
            self._new_es(x0)

    def champions(self) -> List[GraphPolicy]:
        return list(self._champions)

    def info(self) -> Dict:
        return {"sigma": round(float(self.es.sigma), 4), "restarts": self.restarts,
                "params": self.template.n_params}


class OpenAIES(Method):
    """Salimans et al. 2017: antithetic Gaussian perturbations, centered ranks, Adam, L2."""
    name = "openai_es"

    def __init__(self, cfg: dict, n_in: int, rng: np.random.Generator,
                 template: Optional[GraphPolicy] = None):
        super().__init__({**OPENAI_ES_DEFAULTS, **(cfg or {})}, n_in, rng)
        self.template = template if template is not None else build_arch(self.cfg["arch"], n_in, rng)
        self.theta = self.template.get_flat().copy()
        self.m = np.zeros_like(self.theta)
        self.v = np.zeros_like(self.theta)
        self.t = 0
        self._eps = None
        self._champions: List[GraphPolicy] = []

    def ask(self) -> List[GraphPolicy]:
        c = self.cfg
        self._eps = self.rng.standard_normal((c["pairs"], len(self.theta)))
        thetas = np.concatenate([self.theta + c["sigma"] * self._eps,
                                 self.theta - c["sigma"] * self._eps, self.theta[None]])
        return [self.template.with_flat(t) for t in thetas]

    def tell(self, fitness, scores, stats) -> None:
        c = self.cfg
        f = np.asarray(fitness)
        n = c["pairs"]
        r = centered_ranks(f[:-1])
        grad = ((r[:n] - r[n:])[:, None] * self._eps).sum(0) / (2 * n * c["sigma"])
        g = -grad + c["l2"] * self.theta          # minimize
        self.t += 1
        self.m = c["beta1"] * self.m + (1 - c["beta1"]) * g
        self.v = c["beta2"] * self.v + (1 - c["beta2"]) * g * g
        mh = self.m / (1 - c["beta1"] ** self.t)
        vh = self.v / (1 - c["beta2"] ** self.t)
        best = int(np.argmax(f[:-1]))
        best_theta = (self.theta + c["sigma"] * self._eps[best]) if best < n \
            else (self.theta - c["sigma"] * self._eps[best - n])
        self._champions = [self.template.with_flat(self.theta), self.template.with_flat(best_theta)]
        self.theta = self.theta - c["lr"] * mh / (np.sqrt(vh) + 1e-8)

    def champions(self) -> List[GraphPolicy]:
        return list(self._champions)

    def info(self) -> Dict:
        return {"theta_norm": round(float(np.linalg.norm(self.theta)), 3),
                "params": self.template.n_params}
