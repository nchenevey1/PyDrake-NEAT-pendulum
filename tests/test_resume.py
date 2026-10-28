"""Checkpoint/resume must reproduce an uninterrupted run exactly, for every method."""
import json

import pytest

from pendulum.config import DEFAULTS, deep_merge
from pendulum.trainer import Trainer

METHOD_CFGS = {
    "neat": {"pop_size": 60, "species_target": 5},
    "cmaes": {"arch": "mlp:3", "popsize": 12},
    "openai_es": {"arch": "rnn:2", "pairs": 8},
    "map_elites": {"arch": "mlp:2", "batch": 16, "init_random": 32, "reeval_every": 3},
    "neat_cma": {"switch_generation": 3, "neat": {"pop_size": 40, "species_target": 4},
                 "cmaes": {"popsize": 10}},
}


def _cfg(method, gens):
    return deep_merge(DEFAULTS, {"name": method, "method": method, "method_cfg": METHOD_CFGS[method],
                                 "budget": {"generations": gens}, "print_every": 0,
                                 "eval": {"every": 2, "n_val": 4}, "checkpoint_every": 100,
                                 "curriculum": {"pass_ratio": 0.0, "step": 0.01}})


def _gens(path):
    out = []
    for line in open(path / "log.jsonl"):
        r = json.loads(line)
        if r["type"] == "GEN":
            out.append((r["gen"], r["score_max"], r["score_mean"], r["env_steps"], r["phys"]["difficulty"]))
    return out


@pytest.mark.parametrize("method", list(METHOD_CFGS))
def test_resume_is_exact(tmp_path, method, monkeypatch):
    import pendulum.trainer as tr
    monkeypatch.setattr(tr, "full_report", lambda pol, **k: {"params": pol.n_params})
    Trainer(_cfg(method, 6), tmp_path / "a").run()
    Trainer(_cfg(method, 3), tmp_path / "b").run()
    Trainer(_cfg(method, 6), tmp_path / "b", resume=True).run()
    assert _gens(tmp_path / "a") == _gens(tmp_path / "b")
