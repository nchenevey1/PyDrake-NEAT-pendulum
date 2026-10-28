"""The task must be solvable as defined: the classic controller succeeds under target physics."""
import numpy as np
import pytest

pytest.importorskip("pydrake")


def test_classic_controller_solves_task():
    from pendulum.classic import design, rollout_controller
    from pendulum.evaluation import summarize, test_ics
    from pendulum.task import DEFAULT_TASK, TARGET_PHYSICS
    ctrl = design(n_tries=2, verbose=False)
    res = summarize(rollout_controller(ctrl, test_ics(8), TARGET_PHYSICS), DEFAULT_TASK)
    assert res["success_rate"] == 1.0
