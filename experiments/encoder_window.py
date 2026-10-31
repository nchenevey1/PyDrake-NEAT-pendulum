"""Where does encoder sensing break the swing-up? (DIAGNOSIS-style, docs/ENCODER_SENSING.md)

The solved full-state policy (45 params) sees the exact state, except during one time window
where it sees the `fd` (or `hist`-averaged) encoder observation at a given CPR. Success vs window
start shows which phase of the swing-up is sensitive to velocity quantization noise.

  python experiments/encoder_window.py [policy.json] [cpr ...]
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pendulum.evaluation import test_ics
from pendulum.policy import GraphPolicy
from pendulum.rollout import physics_step
from pendulum.sensors import Sensor, SensorConfig
from pendulum.task import DEFAULT_TASK, TARGET_PHYSICS as P

T = replace(DEFAULT_TASK, max_v_cart=12.0, max_w_pole=12.0)


def run(pol, ic, sensor, w0, w1, avg=False):
    step = pol.make_stepper()
    exact = Sensor(SensorConfig(mode="full"), T)
    s = np.array(ic, float)
    for k in range(1, T.max_steps + 1):
        sensor.push(s, fill=(k == 1))
        o = exact.obs(s)
        if w0 <= k * T.dt < w1:
            q = sensor.obs(s)
            if avg:                      # mean of the two most recent one-step differences
                h = sensor.hist
                v = ((h[0] - h[2]) / (2 * T.dt)) / 12.0
                q[[1, 4, 7]] = np.clip(v, -1, 1)
            o = q
        a = float(np.clip(step(o), -1, 1))
        physics_step(s, a * P.max_force, P.g, P.drag, T.dt)
        up = abs((s[1] - np.pi + np.pi) % (2 * np.pi) - np.pi) < T.upright_tol and abs((s[2] + np.pi) % (2 * np.pi) - np.pi) < T.upright_tol
        if abs(s[0]) > T.limit_x or (k > T.swing_up_steps and not up):
            return False, k * T.dt
    return True, T.duration


def main():
    pol = GraphPolicy.load(sys.argv[1] if len(sys.argv) > 1 else "results/experiments/revcurr_mlp4_s0_v12.json")
    cprs = [int(c) for c in sys.argv[2:]] or [1024]
    ics = test_ics(64)
    windows = [(t, t + 0.25) for t in np.arange(0.0, 2.5, 0.25)] + [(2.5, 20.0), (0.0, 20.0)]
    for cpr in cprs:
        for avg in (False, True):
            name = f"cpr {cpr} {'2-step avg' if avg else 'fd'}"
            res = []
            for w0, w1 in windows:
                out = [run(pol, ic, Sensor(SensorConfig(mode="fd", cpr=cpr, x_res=1e-4), T), w0, w1, avg) for ic in ics]
                res.append(f"{w0:.2f}-{w1:.2f}s:{np.mean([o[0] for o in out]) * 100:3.0f}%")
            print(name.ljust(20), " ".join(res), flush=True)


if __name__ == "__main__":
    main()
