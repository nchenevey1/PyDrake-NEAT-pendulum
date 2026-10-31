"""Experiment configuration: YAML files deep-merged over these defaults."""

import copy
from pathlib import Path

import yaml

DEFAULTS = {
    "name": None,                 # run name (defaults to config file stem)
    "seed": 0,
    "method": "neat",
    "method_cfg": {},             # method hyper-parameters (see pendulum/methods/*_DEFAULTS)
    "threads": 1,                 # numba threads for rollouts
    "obs_scale": None,            # e.g. {max_v_cart: 50, max_w_pole: 90}; None = task default (12)
    "budget": {
        "generations": 1000,
        "env_steps": None,        # optional cap on training env steps
        "wall_hours": None,       # optional cap on wall-clock
        "stop_val_success": None, # optional: stop once validation success reaches this
    },
    "train": {
        "n_trials": 5,            # random ICs per candidate per generation (shared by all candidates)
        "fitness": "streak_smooth",
        "size_penalty": 0.005,    # fitness = score / (1 + size_penalty * n_params)
    },
    "curriculum": {
        "mode": "legacy",         # legacy | none | start_state
        "start": 0.0,
        "step": 0.005,
        "pass_ratio": 0.2,
        "pass_count": 1,
        "start_state": {},        # StartStateConfig overrides (pendulum/curriculum.py), mode start_state
    },
    "eval": {
        "every": 25,              # generations between target-physics validations
        "n_val": 256,             # held-out validation starts (model selection; disjoint from test)
    },
    "checkpoint_every": 25,
    "print_every": 10,
}


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path, overrides: dict = None) -> dict:
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    cfg = deep_merge(DEFAULTS, user)
    cfg = deep_merge(cfg, overrides or {})
    if not cfg["name"]:
        cfg["name"] = Path(path).stem
    return cfg


def parse_override(s: str):
    """'a.b.c=value' -> nested dict, value parsed as YAML."""
    key, _, val = s.partition("=")
    d = cur = {}
    parts = key.split(".")
    for p in parts[:-1]:
        cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = yaml.safe_load(val)
    return d
