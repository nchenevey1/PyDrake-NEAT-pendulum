"""Sweep plans for GitHub Actions and the run summary."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ci"))
import summarize  # noqa: E402
import sweep_plan  # noqa: E402


def test_plan_name_from_tag():
    assert sweep_plan.plan_name("sweep-smoke-1") == "smoke"
    assert sweep_plan.plan_name("sweep-reliability-mlp4-2") == "reliability-mlp4"
    assert sweep_plan.plan_name("sweep-smoke") == "smoke"
    assert sweep_plan.plan_name("reliability-mlp8") == "reliability-mlp8"


@pytest.mark.parametrize("plan", [p.stem for p in (ROOT / "sweeps").glob("*.yaml")])
def test_every_plan_expands(plan):
    m = sweep_plan.matrix(plan)["include"]
    assert m and all((ROOT / j["config"]).exists() for j in m)
    assert len({(j["config"], j["seed"]) for j in m}) == len(m)


def test_reliability_plan_is_the_planned_experiment():
    m = sweep_plan.matrix("reliability-mlp4")["include"]
    assert [j["seed"] for j in m] == [0, 1, 2, 3, 4]
    assert all(j["set"] == "--set budget.env_steps=4000000000" and j["drake"] == 16 for j in m)


def test_unknown_plan_fails():
    with pytest.raises(SystemExit):
        sweep_plan.matrix("no-such-plan")


def test_summary_table(tmp_path):
    d = tmp_path / "x_s0"
    d.mkdir()
    json.dump({"seed": 0, "generations": 10, "env_steps": 2e9, "wall_s": 1.0,
               "curriculum": {"best_start_t": 0.0, "final_start_t": 0.0, "final_step": 2, "backoffs": 1},
               "val": {"success_rate": 1.0},
               "report": {"params": 45, "test_20s": {"success_rate": 0.95}, "test_60s": {"success_rate": 0.9},
                          "int8_20s": {"success_rate": 0.9}, "weight_noise_0.01": {"median": 0.8}}},
              open(d / "report.json", "w"))
    json.dump({"drake_20s": {"n": 16, "successes": 16}, "drake_60s": {"n": 16, "successes": 15}},
              open(d / "verify.json", "w"))
    out = subprocess.run([sys.executable, str(ROOT / "ci" / "summarize.py"), str(tmp_path)],
                         capture_output=True, text=True, check=True).stdout
    assert "| x_s0 | 0 s | 0 s (2) | 1 | 2.00B | 100.0 % | 95.0 % | 90.0 % | 16/16 / 15/16 |" in out
    assert "1/1 runs reach >= 90 %" in out
