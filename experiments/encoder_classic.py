"""Classic reference through the encoder model (method 5 of docs/ENCODER_SENSING.md).

The stored trajopt + TVLQR + LQR controller (3,132 numbers) gets a state estimate built from
quantized encoder positions instead of the true state:
  fd   one-step finite differences (0 extra numbers)
  ab   model-based alpha-beta observer: predict with the plant model and the force actually
       applied, correct each channel with fixed gains alpha, beta = alpha^2 / (2 - alpha) (the
       critically damped choice); +7 numbers (alpha and the six plant constants). A
       constant-velocity alpha-beta filter (no model) lags ~0.7 rad/s at the swing-up's
       ~100 rad/s^2 and failed at every CPR. Under a one-step delay the estimate is also
       predicted one step ahead with the pending command (model-based delay compensation).
  Note: with this integrator q+ = q + dt v+, so the one-step backward difference of exact
  positions IS the current velocity: plain fd has no lag, only quantization noise.
alpha is chosen per setting on 32 validation starts from {0.05, 0.1, 0.2, 0.35, 0.5}.

  python experiments/encoder_classic.py                # writes results/encoder/classic.json
"""
import json, sys, time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pendulum.classic import ClassicController, GOAL, f, state_error
from pendulum.evaluation import test_ics, val_ics
from pendulum.sensors import SensorConfig, SensedPolicy, py_episode
from pendulum.task import DEFAULT_TASK, TARGET_PHYSICS as P, observe

T = replace(DEFAULT_TASK, max_v_cart=100.0, max_w_pole=100.0)    # wide scale: no clipping
EX = ClassicController.load(str(Path(__file__).resolve().parent.parent / "results/classic/controller.npz"))
UP = np.array([0.0, np.pi, 0.0, 0.0, 0.0, 0.0])


def make_ctrl(kind, sensor, alpha=0.5, balance=False):
    """Returns make_fn for SensedPolicy: observation of `sensor` -> action in [-1, 1]."""
    beta = alpha * alpha / (2 - alpha) if alpha else 0.0
    dt = T.dt

    def make():
        ex = ClassicController(EX.S_nom, EX.u_nom, EX.K_tv, EX.K_bal, EX.max_force)
        ex.reset()
        if balance:
            ex.k = len(ex.u_nom)
        elif sensor.delay:
            ex.k = 1      # the command computed now acts one step later
        st = {"xh": None, "vh": None, "u": 0.0, "u_prev": 0.0}

        def fn(o):
            if kind == "fd":       # o is the fd observation at scale 100
                s = np.array([o[0] * T.limit_x, np.arctan2(o[2], o[3]), np.arctan2(o[5], o[6]),
                              o[1] * T.max_v_cart, o[4] * T.max_w_pole, o[7] * T.max_w_pole])
            else:                  # o is the pos observation: unwrap and filter
                z = np.array([o[0] * T.limit_x, np.arctan2(o[1], o[2]), np.arctan2(o[3], o[4])])
                if st["xh"] is None:
                    if balance:
                        z[1] = z[1] % (2 * np.pi)
                    st["xh"], st["vh"] = z.copy(), np.zeros(3)
                else:
                    applied = st["u_prev"] if sensor.delay else st["u"]
                    sp = f(np.concatenate([st["xh"], st["vh"]]), applied * EX.max_force, P, dt)
                    pred, vpred = sp[:3], sp[3:]
                    z[1:] = pred[1:] + (z[1:] - pred[1:] + np.pi) % (2 * np.pi) - np.pi
                    r = z - pred
                    st["xh"] = pred + alpha * r
                    st["vh"] = vpred + (beta / dt) * r
                s = np.concatenate([st["xh"], st["vh"]])
                if sensor.delay:   # predict the state when the command will act
                    s = f(s, st["u"] * EX.max_force, P, dt)
            a = ex(observe(s, DEFAULT_TASK))
            if sensor.force_bits:
                lev = 2 ** (sensor.force_bits - 1) - 1
                a = round(a * lev) / lev
            st["u_prev"], st["u"] = st["u"], a
            return a
        return fn
    return make


def success(kind, sensor, ics, alpha, balance=False, duration=None):
    task = replace(T, duration=duration or T.duration)
    if balance:
        task = replace(task, swing_up_time=0.0)
    ctrl = SensedPolicy(make_ctrl(kind, sensor, alpha, balance), sensor, task)
    return float(np.mean([py_episode(ctrl, ic, P, task)[1] for ic in ics]))


def main():
    out = []
    ics = test_ics(256)
    bal = UP + np.random.default_rng(99).uniform(-0.05, 0.05, (256, 6))
    for setting, extra in (("ideal", {}), ("real", dict(delay=1, force_bits=8))):
        for cpr in (256, 1024, 4096, 16384):
            for kind in ("fd", "ab"):
                t0 = time.time()
                sensor = SensorConfig(mode="fd" if kind == "fd" else "pos", cpr=cpr, x_res=1e-4, **extra)
                alpha = None
                if kind == "ab":
                    cands = (0.05, 0.1, 0.2, 0.35, 0.5)
                    sv = [success(kind, sensor, val_ics(32), a) for a in cands]
                    alpha = cands[int(np.argmax(sv))]
                r = {"controller": f"classic+{kind}", "setting": setting, "cpr": cpr, "x_res": 1e-4,
                     "alpha": alpha, "stored_numbers": EX.n_numbers + (7 if kind == "ab" else 0),
                     "balance_20s": success(kind, sensor, bal[:64], alpha, balance=True),
                     "test_20s": success(kind, sensor, ics, alpha),
                     "test_60s_64": success(kind, sensor, ics[:64], alpha, duration=60.0)}
                r["seconds"] = round(time.time() - t0)
                print(r, flush=True)
                out.append(r)
    Path("results/encoder").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("results/encoder/classic.json", "w"), indent=1)


if __name__ == "__main__":
    main()
