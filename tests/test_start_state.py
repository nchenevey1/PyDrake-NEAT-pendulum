"""Start-state curriculum (S2): levels, mastery rule, adaptive back-off step, trainer integration."""
import json

import numpy as np
import pytest

import pendulum.curriculum as cur
from pendulum.config import DEFAULTS, deep_merge
from pendulum.curriculum import StartStateConfig, StartStateCurriculum, make_curriculum
from pendulum.policy import build_arch
from pendulum.rollout import N_STATS, S_STEPS
from pendulum.task import DEFAULT_TASK


def _curr(**kw):
    return StartStateCurriculum(StartStateConfig(**kw), DEFAULT_TASK)


def _fake_rollout(success: bool):
    """Replacement for the mastery rollout: every start fully succeeds (or fails)."""
    def f(pk, ics, phys, task):
        st = np.zeros((len(pk), len(ics), N_STATS))
        st[:, :, S_STEPS] = task.max_steps if success else 1
        return st, len(ics)
    return f


def test_level_zero_is_the_real_task():
    c = _curr()
    assert c.level_task(0) == DEFAULT_TASK
    t = c.level_task(150)
    assert t.duration == pytest.approx(DEFAULT_TASK.duration - 1.5)
    assert t.swing_up_time == pytest.approx(1.0)
    assert c.level_task(210).swing_up_time == pytest.approx(0.4)
    # the reference starts hanging at rest and is upright at k_start
    assert np.allclose(c.reference[0], 0.0)
    assert abs(c.reference[210][1] - np.pi) < 0.05 and abs(c.reference[210][2]) < 0.05


def test_batches_split_current_and_replay_levels():
    c = _curr()
    c.k = 100
    rng = np.random.default_rng(0)
    for _ in range(20):
        b = c.batches(rng, 8)
        assert [len(x[1]) for x in b] == [4, 4]
        assert sum(x[3] for x in b) == pytest.approx(1.0)
        assert b[0][0] == 100 and 100 <= b[1][0] <= 210
        for k, ics, tk, _ in b:
            assert np.all(np.abs(ics - c.reference[k]) <= DEFAULT_TASK.ic_noise)
            assert tk == c.level_task(k)


def test_mastery_advances_only_on_check_generations(monkeypatch):
    monkeypatch.setattr(cur, "rollout", _fake_rollout(True))
    c = _curr()
    rng = np.random.default_rng(0)
    pol = build_arch("mlp:2", 9, rng)
    for gen in range(1, 5):
        assert c.update(gen, 0, 1.0, lambda: pol, rng) == (False, 0)
    changed, used = c.update(5, 0, 1.0, lambda: pol, rng)
    assert changed and used == 16 and c.k == 205 and c.best_k == 205
    monkeypatch.setattr(cur, "rollout", _fake_rollout(False))
    assert c.update(10, 0, 1.0, lambda: pol, rng)[0] is False and c.k == 205


def test_backoff_halves_the_step(monkeypatch):
    monkeypatch.setattr(cur, "rollout", _fake_rollout(False))
    c = _curr(stuck_gens=3)
    c.k = 55
    rng = np.random.default_rng(0)
    pol = build_arch("mlp:2", 9, rng)
    ks, steps = [], []
    gen = 0
    for _ in range(4):                       # four back-offs, no full success in between
        for _ in range(3):
            gen += 1
            c.update(gen, 0, 0.0, lambda: pol, rng)
        ks.append(c.k); steps.append(c.step)
    assert ks == [60, 62, 63, 64]            # back off by the old step, then halve: 5 -> 2 -> 1 (min)
    assert steps == [2, 1, 1, 1]
    assert c.backoffs == 4 and [e["event"] for e in c.events] == ["backoff"] * 4
    # a full success at the level resets the stuck counter
    for _ in range(10):
        gen += 1
        c.update(gen, 0, 0.5, lambda: pol, rng)
    assert c.k == 64


def test_no_adaptive_step_keeps_the_step(monkeypatch):
    monkeypatch.setattr(cur, "rollout", _fake_rollout(False))
    c = _curr(stuck_gens=1, adaptive_step=False)
    c.k = 50
    pol = build_arch("mlp:2", 9, np.random.default_rng(0))
    c.update(1, 0, 0.0, lambda: pol, None)
    assert c.k == 55 and c.step == 5


def test_never_backs_off_past_the_first_level(monkeypatch):
    monkeypatch.setattr(cur, "rollout", _fake_rollout(False))
    c = _curr(stuck_gens=1)
    pol = build_arch("mlp:2", 9, np.random.default_rng(0))
    for g in range(1, 4):
        c.update(g, 0, 0.0, lambda: pol, None)
    assert c.k == 210 and c.step == 5


def test_make_curriculum_dispatch():
    assert make_curriculum(DEFAULTS["curriculum"], DEFAULT_TASK).mode == "physics"
    c = make_curriculum({**DEFAULTS["curriculum"], "mode": "start_state", "start_state": {"step": 3}},
                        DEFAULT_TASK)
    assert isinstance(c, StartStateCurriculum) and c.step == 3


def _cfg(gens, **extra):
    return deep_merge(DEFAULTS, deep_merge({
        "name": "ss", "method": "cmaes", "method_cfg": {"arch": "mlp:3", "popsize": 12},
        "budget": {"generations": gens}, "print_every": 0, "checkpoint_every": 100,
        "eval": {"every": 2, "n_val": 4}, "train": {"n_trials": 4, "size_penalty": 0.0},
        "curriculum": {"mode": "start_state",
                       "start_state": {"mastery_every": 1, "mastery_starts": 2, "stuck_gens": 2,
                                       "k_start": 30}}}, extra))


def test_trainer_report_and_policy_files(tmp_path, monkeypatch):
    import pendulum.trainer as tr
    monkeypatch.setattr(tr, "full_report", lambda pol, **k: {"params": pol.n_params})
    tr.Trainer(_cfg(8), tmp_path / "r").run()
    rep = json.load(open(tmp_path / "r" / "report.json"))
    assert rep["generations"] == 8
    assert rep["obs_scale"] == {"max_v_cart": 12.0, "max_w_pole": 12.0}
    c = rep["curriculum"]
    assert c["mode"] == "start_state" and 0 <= c["best_k"] <= 30 and "events" in c
    pol = json.load(open(tmp_path / "r" / "best_val.json"))
    assert pol["meta"]["obs_scale"] == {"max_v_cart": 12.0, "max_w_pole": 12.0}
    gens = [json.loads(l) for l in open(tmp_path / "r" / "log.jsonl") if '"GEN"' in l]
    assert all("k" in g["phys"] and "step" in g["phys"] for g in gens)


def test_trainer_obs_scale_override_is_stored(tmp_path, monkeypatch):
    import pendulum.trainer as tr
    monkeypatch.setattr(tr, "full_report", lambda pol, **k: {"params": pol.n_params})
    tr.Trainer(_cfg(2, obs_scale={"max_v_cart": 50, "max_w_pole": 90}), tmp_path / "r").run()
    pol = json.load(open(tmp_path / "r" / "best_val.json"))
    assert pol["meta"]["obs_scale"] == {"max_v_cart": 50.0, "max_w_pole": 90.0}
    with pytest.raises(ValueError):
        tr.Trainer(_cfg(2, obs_scale={"limit_x": 5}), tmp_path / "bad")


def test_stop_at_validation_success(tmp_path, monkeypatch):
    import pendulum.trainer as tr
    monkeypatch.setattr(tr, "full_report", lambda pol, **k: {"params": pol.n_params})
    monkeypatch.setattr(tr, "evaluate_policies", lambda pols, *a, **k: [
        {"success_rate": 1.0, "mean_streak_s": 1.0} for _ in pols])
    rep = tr.Trainer(_cfg(50, budget={"stop_val_success": 0.99}), tmp_path / "r").run()
    assert rep["generations"] == 2          # first validation (eval.every = 2) stops the run
