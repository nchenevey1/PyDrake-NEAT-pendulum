"""Stage 2 of the encoder study: start-state curriculum through the sensor model.

Identical schedule to experiments/reverse_curriculum.py (CMA-ES pop 64, K = 8 starts per
generation, half at the current level and half replaying an easier one, step 5, mastery 80 %
on 16 fresh starts, back-off after 300 stuck generations, velocity scale 12), with the policy
observing through `pendulum.sensors`. Starts at curriculum level k come with the 10 preceding
reference states as a warm-up history; the real task (k = 0) starts from rest with an empty
history, as deployed.

  python experiments/encoder_curriculum.py --mode fd --cpr 4096 --arch mlp:4 --seed 0 --budget 4

Writes <out>/<tag>.json (report + policy) and <out>/<tag>.log.
"""
import argparse, json, sys, time
from dataclasses import replace
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pendulum.classic import ClassicController
from pendulum.evaluation import test_ics, val_ics
from pendulum.fitness import policy_scores
from pendulum.methods.es import CMAES
from pendulum.policy import pack
from pendulum.rollout import S_REASON, S_STEPS, S_FIRST_UP
from pendulum.sensors import SensorConfig, SensedPolicy, evaluate_sensed, py_episode, rollout_sensed
from pendulum.task import DEFAULT_TASK, TARGET_PHYSICS as P

W = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="mlp:4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=4.0, help="billions of env steps")
    ap.add_argument("--mode", default="fd")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--cpr", type=int, default=4096)
    ap.add_argument("--x-res", type=float, default=1e-4)
    ap.add_argument("--delay", type=int, default=0)
    ap.add_argument("--force-bits", type=int, default=0)
    ap.add_argument("--vscale", type=float, default=12.0)
    ap.add_argument("--out", default="runs/encoder")
    ap.add_argument("--drake", type=int, default=16, help="starts replayed in Drake (0 = skip)")
    a = ap.parse_args()
    numba.set_num_threads(1)

    T = replace(DEFAULT_TASK, max_v_cart=a.vscale, max_w_pole=a.vscale)
    sensor = SensorConfig(mode=a.mode, k=a.k, cpr=a.cpr if a.mode != "full" else 0,
                          x_res=a.x_res if a.mode != "full" else 0.0, delay=a.delay, force_bits=a.force_bits)
    tag = f"{sensor.label()}_{a.arch.replace(':', '')}_s{a.seed}"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / f"{tag}.log", "w")

    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    budget = a.budget * 1e9
    step, K = 5, 8
    ex = ClassicController.load(str(Path(__file__).resolve().parent.parent / "results/classic/controller.npz"))
    N_up = 210
    rng = np.random.default_rng(a.seed); np.random.seed(a.seed)
    m = CMAES({"arch": a.arch, "popsize": 64}, sensor.n_in, rng)
    log(f"{tag}: {m.template.n_params} params, {sensor.n_in} inputs, sensor {sensor.to_dict()}")

    def level_task(kk):
        return replace(T, swing_up_time=max(0.0, T.swing_up_time - kk * T.dt), duration=T.duration - kk * T.dt)

    def level_starts(kk, n):
        off = rng.uniform(-T.ic_noise, T.ic_noise, (n, 6))
        ics = ex.S_nom[kk] + off
        if kk == 0:
            return ics, None
        idx = np.maximum(np.arange(kk - W, kk), 0)
        return ics, ex.S_nom[idx][None] + off[:, None, :]

    k = N_up; steps = 0; t0 = time.time(); g = 0; best_val = (0, None); stuck = 0; min_k = N_up
    trace = []
    while steps < budget:
        g += 1
        k_replay = int(rng.integers(k, N_up + 1))
        pols = m.ask(); pk = pack(pols)
        sc = np.zeros(len(pols)); st_cur = None
        for kk in (k, k_replay):
            tk = level_task(kk)
            ics, warm = level_starts(kk, K // 2)
            st, n = rollout_sensed(pk, ics, P, tk, sensor, warm); steps += n
            sc += 0.5 * policy_scores(st, "streak_smooth", tk) / tk.duration
            if kk == k:
                st_cur = st
        m.tell(sc, sc, st_cur)
        ib = int(np.argmax(sc[:-1]))
        full_best = np.mean(st_cur[ib, :, S_STEPS] >= level_task(k).max_steps)
        stuck = 0 if full_best > 0 else stuck + 1
        if k > 0 and g % 5 == 0:
            tk = level_task(k)
            ics, warm = level_starts(k, 16)
            mst, n = rollout_sensed(pack(m.champions()[:1]), ics, P, tk, sensor, warm)
            steps += n
            if np.mean(mst[0, :, S_STEPS] >= tk.max_steps) >= 0.8:
                k = max(0, k - step); stuck = 0
                min_k = min(min_k, k)
        if stuck >= 300 and k < N_up:
            k = min(N_up, k + step); stuck = 0
        if g % 200 == 0 or (k == 0 and g % 50 == 0):
            champs = m.champions()
            res = evaluate_sensed(champs, val_ics(64), P, T, sensor)
            sr = max(r["success_rate"] for r in res)
            if sr > best_val[0]:
                best_val = (sr, champs[int(np.argmax([r["success_rate"] for r in res]))])
            trace.append([g, round(steps / 1e9, 3), k, round(sr, 3)])
            log(f"{tag} gen {g} level k={k} (start t={k * T.dt:.2f}s) best norm-score {sc.max():.3f} "
                f"best full@level {full_best:.2f} | val {sr:.2f} (best {best_val[0]:.2f}) "
                f"| {steps / 1e9:.2f}B steps {time.time() - t0:.0f}s restarts {m.restarts}")
            if best_val[0] >= 0.99:
                break

    rep = {"tag": tag, "arch": a.arch, "seed": a.seed, "sensor": sensor.to_dict(), "vscale": a.vscale,
           "budget_B": a.budget, "steps_B": round(steps / 1e9, 3), "train_s": round(time.time() - t0),
           "generations": g, "closest_start_s": round(min_k * T.dt, 2), "final_level_s": round(k * T.dt, 2),
           "best_val": best_val[0], "trace": trace}
    pol = best_val[1] if best_val[1] is not None else m.champions()[0]
    rep["params"] = pol.n_params
    rep["memory_words"] = sensor.memory_words + (pol.n_hidden if pol.is_recurrent else 0)
    ics = test_ics(256)
    ev = lambda p, s=sensor, t=T, ic=ics: evaluate_sensed([p], ic, P, t, s)[0]
    t20 = ev(pol)
    rep["test_20s"] = t20["success_rate"]
    rep["test_detail"] = t20
    rep["test_60s"] = ev(pol, t=replace(T, duration=60.0))["success_rate"]
    rep["from_rest"] = ev(pol, ic=np.zeros((1, 6)))["success_rate"]
    for bits in (8, 6, 4):
        rep[f"int{bits}_20s"] = ev(pol.quantized(bits))["success_rate"]
    rep["delay_switched_20s"] = ev(pol, s=replace(sensor, delay=1 - sensor.delay))["success_rate"]
    rep["force8_20s"] = ev(pol, s=replace(sensor, force_bits=8))["success_rate"]
    if sensor.mode != "full":
        rep["other_cpr_20s"] = {c: ev(pol, s=replace(sensor, cpr=c))["success_rate"] for c in (256, 1024, 4096, 16384)}
    if a.drake and rep["test_20s"] > 0:
        try:
            from pendulum.drake_env import DrakeEnv
            env = DrakeEnv(P, T)
            ctrl = SensedPolicy.from_graph(pol, sensor, T)
            ok_k = [py_episode(ctrl, ic, P, T)[1] for ic in ics[:a.drake]]
            ok_d = [py_episode(ctrl, ic, P, T, env=env)[1] for ic in ics[:a.drake]]
            rep["drake"] = {"n": a.drake, "kernel_python": int(sum(ok_k)), "drake": int(sum(ok_d))}
        except ImportError as e:
            rep["drake"] = f"skipped: {e}"
    log(f"FINAL {tag}: closest start {rep['closest_start_s']} s, best val {best_val[0]:.2f}, "
        f"TEST 20s {rep['test_20s']:.3f} 60s {rep['test_60s']:.3f} int8 {rep['int8_20s']:.3f} "
        f"delay-switched {rep['delay_switched_20s']:.3f} drake {rep.get('drake')} params {pol.n_params}")
    pol.meta["obs_scale"] = {"max_v_cart": T.max_v_cart, "max_w_pole": T.max_w_pole}
    pol.meta["sensor"] = sensor.to_dict()
    rep["policy"] = pol.to_dict()
    with open(out / f"{tag}.json", "w") as f:
        json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
