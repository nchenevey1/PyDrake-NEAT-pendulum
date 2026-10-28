"""Deadline curriculum at the TARGET physics, from the real hanging start states.

Level D = allowed swing-up time, from D0 (generous) down to the task's 2.5 s. Episode length
stays 20 s, so the policy must still balance until the end. Same mastery / replay / back-off
rules as experiments/reverse_curriculum.py:
  * advance (D -= step) when the CMA-ES mean policy fully succeeds on >= 80 % of 16 fresh ICs
  * half of each generation's trials use a random easier (already mastered) deadline
  * back off (D += step) after 300 generations without any full success at the current level

  python experiments/deadline_curriculum.py mlp:4 0 2.0 [D0=10]
"""
import sys, time, numpy as np, numba
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pendulum.methods.es import CMAES
from pendulum.policy import pack
from pendulum.rollout import rollout, S_STEPS
from pendulum.fitness import policy_scores
from pendulum.evaluation import evaluate_policies, val_ics, test_ics
from pendulum.task import DEFAULT_TASK as T, TARGET_PHYSICS as P, sample_initial_states
numba.set_num_threads(1)
arch, seed, budget = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]) * 1e9
D0 = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0
STEP, K, MASTERY, STUCK = 0.1, 8, 0.8, 300
rng = np.random.default_rng(seed); np.random.seed(seed)
m = CMAES({"arch": arch, "popsize": 64}, 9, rng)
D = D0; steps = 0; g = 0; stuck = 0; t0 = time.time(); best = (0.0, None)
task_at = lambda d: replace(T, swing_up_time=round(d, 3))
while steps < budget:
    g += 1
    d_replay = float(rng.uniform(D, D0))
    pols = m.ask(); pk = pack(pols); sc = np.zeros(len(pols)); st_cur = None
    for d in (D, d_replay):
        st, n = rollout(pk, sample_initial_states(K // 2, rng, T.ic_noise), P, task_at(d)); steps += n
        sc += 0.5 * policy_scores(st, "streak_smooth", T)
        if d == D:
            st_cur = st
    m.tell(sc, sc, st_cur)
    ib = int(np.argmax(sc[:-1]))
    full_best = np.mean(st_cur[ib, :, S_STEPS] >= T.max_steps)
    stuck = 0 if full_best > 0 else stuck + 1
    if D > T.swing_up_time and g % 5 == 0:
        mst, n = rollout(pack(m.champions()[:1]), sample_initial_states(16, rng, T.ic_noise), P, task_at(D))
        steps += n
        if np.mean(mst[0, :, S_STEPS] >= T.max_steps) >= MASTERY:
            D = max(T.swing_up_time, round(D - STEP, 3)); stuck = 0
    if stuck >= STUCK and D < D0:
        D = min(D0, round(D + STEP, 3)); stuck = 0
    if g % 200 == 0:
        champs = m.champions()
        res = evaluate_policies(champs, val_ics(64), P, T)
        sr = max(r["success_rate"] for r in res)
        if sr > best[0]:
            best = (sr, champs[int(np.argmax([r["success_rate"] for r in res]))])
        print(f"{arch} s{seed} gen {g} deadline D={D:.2f}s best score {sc.max():.2f} best full@D {full_best:.2f} "
              f"| TARGET(2.5 s) val success {sr:.2f} (best {best[0]:.2f}) | {steps/1e9:.2f}B steps "
              f"{time.time()-t0:.0f}s", flush=True)
        if best[0] >= 0.99:
            break
if best[1] is not None:
    t = evaluate_policies([best[1]], test_ics(256), P, T)[0]
    print(f"FINAL {arch} s{seed}: best val {best[0]:.2f}, TEST success {t['success_rate']:.3f}, "
          f"params {best[1].n_params}", flush=True)
    best[1].save(f"results/experiments/deadline_{arch.replace(':', '')}_s{seed}.json")
else:
    print(f"FINAL {arch} s{seed}: never succeeded on target validation (final deadline {D:.2f}s)", flush=True)
