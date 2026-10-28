from .es import CMAES, OpenAIES
from .hybrid import NEATThenCMA
from .neat import NEAT
from .qd import MAPElites

METHODS = {m.name: m for m in (NEAT, CMAES, OpenAIES, MAPElites, NEATThenCMA)}


def make_method(name: str, cfg: dict, n_in: int, rng):
    if name not in METHODS:
        raise ValueError(f"Unknown method '{name}'. Available: {sorted(METHODS)}")
    return METHODS[name](cfg, n_in, rng)
