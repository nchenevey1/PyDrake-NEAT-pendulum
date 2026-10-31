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


def test_resume_is_exact_start_state(tmp_path, monkeypatch):
    """Start-state curriculum: level, step, stuck counter and mastery RNG survive a resume."""
    import pendulum.trainer as tr
    monkeypatch.setattr(tr, "full_report", lambda pol, **k: {"params": pol.n_params})

    def cfg(gens):
        return deep_merge(DEFAULTS, {
            "name": "ss", "method": "cmaes", "method_cfg": {"arch": "mlp:3", "popsize": 12},
            "budget": {"generations": gens}, "print_every": 0, "checkpoint_every": 100,
            "eval": {"every": 2, "n_val": 4}, "train": {"n_trials": 4},
            "curriculum": {"mode": "start_state",
                           "start_state": {"mastery_every": 1, "mastery_starts": 2, "stuck_gens": 2}}})

    def gens(path):
        return [(r["gen"], r["score_max"], r["env_steps"], r["phys"]["k"], r["phys"]["step"])
                for r in map(json.loads, open(path / "log.jsonl")) if r["type"] == "GEN"]
    Trainer(cfg(8), tmp_path / "a").run()
    Trainer(cfg(4), tmp_path / "b").run()
    Trainer(cfg(8), tmp_path / "b", resume=True).run()
    assert gens(tmp_path / "a") == gens(tmp_path / "b")
    ra = json.load(open(tmp_path / "a" / "report.json"))
    rb = json.load(open(tmp_path / "b" / "report.json"))
    assert ra["curriculum"] == rb["curriculum"]
