"""Observation scales: default 12, stored with every saved policy, read back by the tools;
validation and test starts are disjoint."""
import json

import numpy as np

from pendulum import evaluation as ev
from pendulum.export import to_c
from pendulum.policy import GraphPolicy, build_arch
from pendulum.task import (DEFAULT_TASK, LEGACY_OBS_SCALE, TEST_SEED, VAL_SEED, obs_scale,
                           task_for_policy, with_obs_scale)


def test_default_scale_is_12():
    assert obs_scale(DEFAULT_TASK) == {"max_v_cart": 12.0, "max_w_pole": 12.0}


def test_saved_policy_stores_scales_and_legacy_files_load_as_legacy(tmp_path):
    p = build_arch("mlp:4", 9, np.random.default_rng(0))
    assert task_for_policy(p) == DEFAULT_TASK               # built here, nothing stored yet
    p.save(tmp_path / "p.json")
    assert json.load(open(tmp_path / "p.json"))["meta"]["obs_scale"] == obs_scale(DEFAULT_TASK)
    # a file written before scales were stored: legacy scales, kept when re-saved
    d = json.load(open(tmp_path / "p.json"))
    del d["meta"]["obs_scale"]
    json.dump(d, open(tmp_path / "old.json", "w"))
    q = GraphPolicy.load(tmp_path / "old.json")
    assert task_for_policy(q) == with_obs_scale(DEFAULT_TASK, LEGACY_OBS_SCALE)
    q.save(tmp_path / "old2.json")
    assert GraphPolicy.load(tmp_path / "old2.json").meta["obs_scale"] == LEGACY_OBS_SCALE
    # derived policies keep the scales
    assert q.quantized(8).meta["obs_scale"] == LEGACY_OBS_SCALE
    assert q.compacted().meta["obs_scale"] == LEGACY_OBS_SCALE


def test_export_uses_stored_scales():
    p = build_arch("mlp:2", 9, np.random.default_rng(0))
    p.meta["obs_scale"] = {"max_v_cart": 50.0, "max_w_pole": 90.0}
    src = to_c(p)
    assert "s[3] / 50.0f" in src and "s[4] / 90.0f" in src
    p.meta["obs_scale"] = obs_scale(DEFAULT_TASK)
    assert "s[3] / 12.0f" in to_c(p)


def test_validation_and_test_starts_are_disjoint():
    assert VAL_SEED != TEST_SEED
    v, t = ev.val_ics(256), ev.test_ics(256)
    assert v.shape == t.shape == (256, 6)
    d = np.abs(v[:, None, :] - t[None, :, :]).max(-1)
    assert d.min() > 1e-6
    # the pruning subset is a prefix of the validation set, never test
    assert np.array_equal(ev.val_ics(128), v[:128])


def test_weight_noise_is_deterministic():
    p = build_arch("mlp:2", 9, np.random.default_rng(0))
    a = ev.weight_noise_success(p, ev.test_ics(4), 0.01, n=4)
    b = ev.weight_noise_success(p, ev.test_ics(4), 0.01, n=4)
    assert a == b and a["n"] == 4


def test_drake_check_of_the_solved_controller():
    import pytest
    pytest.importorskip("pydrake")
    p = GraphPolicy.load("results/experiments/revcurr_mlp4_s0_v12.json")
    r = ev.drake_check(p, 2, task_for_policy(p))
    assert r == {"n": 2, "successes": 2, "success_rate": 1.0}
