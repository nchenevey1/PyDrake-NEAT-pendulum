"""Start-state (reverse) curriculum at the TARGET physics.

Level k: episodes start at the classic expert's nominal swing-up state S_nom[k] (+ the task's
+-0.05 IC noise) with the remaining swing-up deadline (2.5 s - k*dt). k = 0 is exactly the
real task. k decreases by `step` when the best candidate fully succeeds (survives the whole
episode) on >= `pass_frac` of the generation's trials. Same sparse score, same CMA-ES code.
"""
import sys, time, json, numpy as np, numba
from dataclasses import replace
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from pendulum.classic import ClassicController
from pendulum.methods.es import CMAES
from pendulum.policy import pack
from pendulum.rollout import rollout, S_STEPS
from pendulum.fitness import policy_scores
from pendulum.evaluation import evaluate_policies, val_ics, test_ics
from pendulum.task import DEFAULT_TASK as T, TARGET_PHYSICS as P
numba.set_num_threads(1)
arch = sys.argv[1]; seed = int(sys.argv[2]); budget = float(sys.argv[3]) * 1e9
step, pass_frac, K = 5, 0.6, 8
ex = ClassicController.load('results/classic/controller.npz')
N_up = 210                                # nominal is upright at rest from step ~210 (2.1 s)
rng = np.random.default_rng(seed); np.random.seed(seed)
m = CMAES({"arch": arch, "popsize": 64}, 9, rng)
k = N_up; steps = 0; t0 = time.time(); g = 0; best_val = (0, None)
def level_task(kk):
    return replace(T, swing_up_time=max(0.0, T.swing_up_time - kk * T.dt), duration=T.duration - kk * T.dt)

stuck = 0
while steps < budget:
    g += 1
    # half the trials at the current level, half at one randomly chosen easier (mastered) level
    k_replay = int(rng.integers(k, N_up + 1))
    pols = m.ask(); pk = pack(pols)
    sc = np.zeros(len(pols)); st_cur = None
    for kk in (k, k_replay):
        tk = level_task(kk)
        ics = ex.S_nom[kk] + rng.uniform(-T.ic_noise, T.ic_noise, (K // 2, 6))
        st, n = rollout(pk, ics, P, tk); steps += n
        sc += 0.5 * policy_scores(st, "streak_smooth", tk) / tk.duration   # normalise per level
        if kk == k:
            st_cur = st
    m.tell(sc, sc, st_cur)
    ib = int(np.argmax(sc[:-1]))
    full_best = np.mean(st_cur[ib, :, S_STEPS] >= level_task(k).max_steps)
    stuck = 0 if full_best > 0 else stuck + 1
    if k > 0 and g % 5 == 0:   # mastery check of the distribution mean on fresh starts
        tk = level_task(k)
        mst, n = rollout(pack(m.champions()[:1]), ex.S_nom[k] + rng.uniform(-T.ic_noise, T.ic_noise, (16, 6)), P, tk)
        steps += n
        if np.mean(mst[0, :, S_STEPS] >= tk.max_steps) >= 0.8:
            k = max(0, k - step); stuck = 0
    if stuck >= 300 and k < N_up:
        k = min(N_up, k + step); stuck = 0
    if g % 200 == 0 or (k == 0 and g % 50 == 0):
        champs = m.champions()
        res = evaluate_policies(champs, val_ics(64), P, T)
        sr = max(r["success_rate"] for r in res)
        if sr > best_val[0]:
            best_val = (sr, champs[int(np.argmax([r["success_rate"] for r in res]))])
        print(f"{arch} s{seed} gen {g} level k={k} (start t={k*T.dt:.2f}s) best norm-score {sc.max():.3f} "
              f"best full@level {full_best:.2f} | TARGET val success {sr:.2f} (best {best_val[0]:.2f}) "
              f"| {steps/1e9:.2f}B steps {time.time()-t0:.0f}s restarts {m.restarts}", flush=True)
        if best_val[0] >= 0.99:
            break
if best_val[1] is not None:
    t = evaluate_policies([best_val[1]], test_ics(256), P, T)[0]
    print(f"FINAL {arch} s{seed}: best val {best_val[0]:.2f}, TEST success {t['success_rate']:.3f}, "
          f"60s n/a, params {best_val[1].n_params}", flush=True)
    best_val[1].save(f'results/experiments/revcurr_{arch.replace(":","")}_s{seed}.json')
else:
    print(f"FINAL {arch} s{seed}: never succeeded on target validation", flush=True)
