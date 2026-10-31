"""Stage 1 of the encoder study: balance only, through the sensor model (docs/ENCODER_SENSING.md).

Start upright +-0.05 on all six states (with a constant-velocity warm-up history), must stay
upright from step 1. CMA-ES (pop 32, 5 starts per generation, 20 s training episodes) for up
to --gens generations; every 25 generations the two champions are checked on 64 validation
starts at 60 s (a 10 s / 20 s check lets slow cart drift through), and the run stops once one
reaches 97 %. The best-validation policy is then
tested on 256 fixed test starts at 20 s and 60 s, and after int8 quantization.

  python experiments/encoder_stage1.py --shard 0/2            # default grid, resumable
  python experiments/encoder_stage1.py --cells fd:mlp:4 --cprs 256 1024 --settings ideal
"""
import argparse, json, sys, time
from dataclasses import replace
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pendulum.fitness import policy_scores
from pendulum.methods.es import CMAES
from pendulum.policy import pack
from pendulum.sensors import (SensorConfig, back_extrapolated_warm, evaluate_sensed, rollout_sensed,
                              simulate_sensed)
from pendulum.task import DEFAULT_TASK, TARGET_PHYSICS as P

T = replace(DEFAULT_TASK, max_v_cart=12.0, max_w_pole=12.0)
T_BAL = replace(T, swing_up_time=0.0)
UP = np.array([0.0, np.pi, 0.0, 0.0, 0.0, 0.0])
W = 10
SETTINGS = {"ideal": dict(delay=0, force_bits=0), "real": dict(delay=1, force_bits=8)}
DEFAULT_CELLS = ["full:mlp:2", "full:mlp:4", "full:mlp:8", "full:rnn:4",
                 "fd:mlp:2", "fd:mlp:4", "fd:mlp:8", "fd:rnn:4",
                 "hist3:mlp:2", "hist3:mlp:4", "hist3:mlp:8",
                 "hist4:mlp:2", "hist4:mlp:4", "hist4:mlp:8",
                 "raw2:mlp:4", "pos:mlp:4", "pos:rnn:2", "pos:rnn:4", "pos:rnn:8"]
CPRS = [256, 1024, 4096, 16384]


def sensor_of(mode, cpr, x_res, setting):
    k = int(mode[-1]) if mode[-1].isdigit() else 2
    base = mode.rstrip("0123456789")
    if base == "full":
        return SensorConfig(mode="full", **SETTINGS[setting])
    return SensorConfig(mode=base, k=k, cpr=cpr, x_res=x_res, **SETTINGS[setting])


def balance_ics(n, seed):
    ics = UP + np.random.default_rng(seed).uniform(-T.ic_noise, T.ic_noise, (n, 6))
    return ics, back_extrapolated_warm(ics, W, T.dt)


def run_cell(mode, arch, cpr, x_res, setting, seed, gens):
    sensor = sensor_of(mode, cpr, x_res, setting)
    rng = np.random.default_rng(seed); np.random.seed(seed)
    m = CMAES({"arch": arch, "popsize": 32}, sensor.n_in, rng)
    t_train = T_BAL
    t_val = replace(T_BAL, duration=60.0)
    val, val_w = balance_ics(64, 7)
    best = (-1.0, None); steps = 0; t0 = time.time(); g = 0
    for g in range(1, gens + 1):
        pols = m.ask()
        ics = UP + rng.uniform(-T.ic_noise, T.ic_noise, (5, 6))
        st, n = rollout_sensed(pack(pols), ics, P, t_train, sensor, back_extrapolated_warm(ics, W, T.dt))
        steps += n
        sc = policy_scores(st, "streak_smooth", t_train)
        m.tell(sc, sc, st)
        if g % 25 == 0:
            ch = m.champions()
            res = evaluate_sensed(ch, val, P, t_val, sensor, val_w)
            i = int(np.argmax([r["success_rate"] for r in res]))
            if res[i]["success_rate"] > best[0]:
                best = (res[i]["success_rate"], ch[i])
            if best[0] >= 0.97:
                break
    out = {"mode": mode, "arch": arch, "cpr": cpr if mode != "full" else 0, "x_res": x_res,
           "setting": setting, "seed": seed, "sensor": sensor.label(), "n_in": sensor.n_in,
           "gens": g, "steps": steps, "train_s": round(time.time() - t0, 1), "val": best[0]}
    pol = best[1]
    out["params"] = pol.n_params
    out["memory_words"] = sensor.memory_words + (pol.n_hidden if pol.is_recurrent else 0)
    test, test_w = balance_ics(256, 99)
    out["test_20s"] = evaluate_sensed([pol], test, P, T_BAL, sensor, test_w)[0]["success_rate"]
    out["test_60s"] = evaluate_sensed([pol], test, P, replace(T_BAL, duration=60.0), sensor, test_w)[0]["success_rate"]
    out["int8_20s"] = evaluate_sensed([pol.quantized(8)], test, P, T_BAL, sensor, test_w)[0]["success_rate"]
    other = "real" if setting == "ideal" else "ideal"
    out[f"as_{other}_20s"] = evaluate_sensed([pol], test, P, T_BAL, replace(sensor, **SETTINGS[other]),
                                             test_w)[0]["success_rate"]
    # how tightly the poles are held: max |angle error| over the last 5 s of a 20 s run
    errs = []
    for i in range(32):
        _, tr = simulate_sensed(pol, test[i], P, T_BAL, sensor, test_w[i])
        if len(tr) > T_BAL.max_steps:
            errs.append(max(np.abs(tr[-500:, 1] - np.pi).max(), np.abs(tr[-500:, 2]).max()))
    out["hold_err_rad"] = float(np.median(errs)) if errs else None
    out["policy"] = pol.to_dict()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=DEFAULT_CELLS)
    ap.add_argument("--cprs", nargs="*", type=int, default=CPRS)
    ap.add_argument("--x-res", nargs="*", type=float, default=[1e-4])
    ap.add_argument("--settings", nargs="*", default=["ideal", "real"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--gens", type=int, default=600)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", default="results/encoder/stage1.jsonl")
    a = ap.parse_args()
    numba.set_num_threads(1)
    i_sh, n_sh = map(int, a.shard.split("/"))
    done = set()
    if Path(a.out).exists():
        for line in open(a.out):
            r = json.loads(line)
            done.add((r["mode"], r["arch"], r["cpr"], r["x_res"], r["setting"], r["seed"]))
    jobs = []
    for setting in a.settings:
        for cell in a.cells:
            mode, arch = cell.split(":", 1)
            for cpr in ([0] if mode == "full" else a.cprs):
                for xr in ([0.0] if mode == "full" else a.x_res):
                    for seed in a.seeds:
                        jobs.append((mode, arch, cpr, xr, setting, seed))
    jobs = [j for n, j in enumerate(jobs) if n % n_sh == i_sh and j not in done]
    print(f"{len(jobs)} cells to run", flush=True)
    for j in jobs:
        r = run_cell(*j, gens=a.gens)
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(f"{r['sensor']:28s} {r['arch']:6s} s{r['seed']} params {r['params']:3d} gens {r['gens']:3d} "
              f"{r['train_s']:5.0f}s | test 20s {r['test_20s']:.3f} 60s {r['test_60s']:.3f} "
              f"int8 {r['int8_20s']:.3f} hold {r['hold_err_rad']}", flush=True)


if __name__ == "__main__":
    main()
