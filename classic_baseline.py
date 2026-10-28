"""Design and evaluate the classic baseline (trajectory optimization + TVLQR + LQR)
with the same test protocol as the neural policies (pendulum/evaluation.py).

  python classic_baseline.py            # writes results/classic/{controller.npz,report.json}
"""

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from pendulum.classic import ClassicController, design, rollout_controller
from pendulum.evaluation import summarize, test_ics, val_ics
from pendulum.task import DEFAULT_TASK, TARGET_PHYSICS


def quantized(ctrl: ClassicController, bits: int) -> ClassicController:
    def q(a):
        m = np.max(np.abs(a))
        s = m / (2 ** (bits - 1) - 1)
        return np.round(a / s) * s
    return ClassicController(q(ctrl.S_nom), q(ctrl.u_nom), q(ctrl.K_tv), q(ctrl.K_bal), ctrl.max_force)


def main(out=Path("results/classic"), n_test=256):
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ctrl = design(verbose=True)
    t_design = time.time() - t0
    ctrl.save(out / "controller.npz")
    task, ics = DEFAULT_TASK, test_ics(n_test)
    ev = lambda c, i, t=task: summarize(rollout_controller(c, i, TARGET_PHYSICS, t), t)
    f32 = ClassicController(*(np.float32(a).astype(np.float64) for a in
                              (ctrl.S_nom, ctrl.u_nom, ctrl.K_tv, ctrl.K_bal)), ctrl.max_force)
    rep = {
        "method": "classic: trajopt (SNOPT) + TVLQR + LQR",
        "design_wall_s": round(t_design, 1),
        "stored_numbers": ctrl.n_numbers,
        "balance_lqr_gains": 6,
        "val_64": ev(ctrl, val_ics(64)),
        "test_20s": ev(ctrl, ics),
        "test_60s": ev(ctrl, ics, replace(task, duration=60.0)),
        "test_wide_ic_0.2": ev(ctrl, test_ics(n_test, noise=0.2)),
        "nominal_ic": ev(ctrl, np.zeros((1, 6))),
        "float32_20s": ev(f32, ics),
        "int8_20s": ev(quantized(ctrl, 8), ics),
    }
    json.dump(rep, open(out / "report.json", "w"), indent=1)
    for k, v in rep.items():
        print(k, v["success_rate"] if isinstance(v, dict) else v)


if __name__ == "__main__":
    main()
