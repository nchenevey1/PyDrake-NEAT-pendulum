"""Balance-only check: CMA-ES (same code) on target physics, ICs near upright."""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
import sys, time, numpy as np, numba
from pendulum.methods.es import CMAES
from pendulum.policy import pack
from pendulum.rollout import rollout
from pendulum.fitness import policy_scores
from pendulum.task import TARGET_PHYSICS, DEFAULT_TASK
numba.set_num_threads(1)
arch = sys.argv[1]; noise = float(sys.argv[2])
rng = np.random.default_rng(0); np.random.seed(0)
m = CMAES({"arch": arch, "popsize": 32}, 9, rng)
up = np.array([0, np.pi, 0, 0, 0, 0])
test = up + np.random.default_rng(99).uniform(-noise, noise, (256, 6))
t0 = time.time(); steps = 0
for g in range(1, 2001):
    pols = m.ask()
    ics = up + rng.uniform(-noise, noise, (5, 6))
    st, n = rollout(pack(pols), ics, TARGET_PHYSICS); steps += n
    sc = policy_scores(st, "streak_smooth", DEFAULT_TASK)
    m.tell(sc, sc, st)
    if g % 50 == 0:
        st2, _ = rollout(pack(m.champions()[:1]), test, TARGET_PHYSICS)
        succ = np.mean(st2[0, :, 0] >= 2000)
        print(f"{arch} noise {noise}: gen {g} best {sc.max():.2f} test success {succ:.2f} ({steps/1e6:.0f}M steps, {time.time()-t0:.0f}s)", flush=True)
        if succ >= 0.99: break
