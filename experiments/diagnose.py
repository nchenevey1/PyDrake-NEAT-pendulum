"""Diagnostics: why does learning stall? (see docs/DIAGNOSIS.md)

  1. gain ceiling     classic feedback gains expressed in observation units vs the weight box
  2. failure modes    where the best learned policy fails on the real task
  3. start noise      success vs start-noise width (memorized trajectory vs feedback)
  4. chaos            how fast open-loop errors grow along the reference swing-up
  5. catch            does an LQR catch rescue the learned swing-up? (swing-up vs catch bottleneck)
  6. signal noise     how noisy is the sparse score on K=8 starts, and how narrow is the optimum?
"""
import sys, numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from pendulum.classic import ClassicController, rollout_controller, state_error, GOAL
from pendulum.policy import GraphPolicy, pack
from pendulum.rollout import (rollout, physics_step, S_STEPS, S_REASON, S_FIRST_UP, S_BEST_STREAK,
                              S_MAX_HEIGHT)
from pendulum.fitness import policy_scores
from pendulum.evaluation import test_ics
from pendulum.task import DEFAULT_TASK as T, TARGET_PHYSICS as P, observe

pol = GraphPolicy.load(sys.argv[1] if len(sys.argv) > 1 else 'results/experiments/revcurr_mlp4_s1.json')
ex = ClassicController.load('results/classic/controller.npz')
rng = np.random.default_rng(0)
full = T.max_steps

print("== 1. gain ceiling (|gain| in action units per observation unit; weight box is +-8) ==")
sc = np.array([T.limit_x, 1, 1, T.max_v_cart, T.max_w_pole, T.max_w_pole])
names = ["x", "th1", "th2", "xd", "w1", "w2"]
kb = np.abs(ex.K_bal * sc / ex.max_force); kt = np.median(np.abs(ex.K_tv * sc / ex.max_force), 0)
for n, a, b in zip(names, kb, kt):
    print(f"  {n:4s} balance LQR {a:7.1f}   swing-up TVLQR (median) {b:7.1f}")
print(f"  reachable: linear 8, one tanh layer of H units ~ 64*H (all weights at the box edge)")

print("== 2. failure modes of the learned policy (256 test starts) ==")
ics = test_ics(256)
st, _ = rollout(pack([pol]), ics, P, T); st = st[0]
ok = st[:, S_STEPS] >= full
never = (st[:, S_FIRST_UP] < 0) | (st[:, S_FIRST_UP] > T.swing_up_steps)
track = st[:, S_REASON] == 1
print(f"  success {ok.mean():.3f} | never upright by 2.5 s {never.mean():.3f} | "
      f"upright in time but lost it {(~ok & ~never).mean():.3f}")
print(f"  episodes ending off the track {track.mean():.3f} (median time {np.median(st[track, S_STEPS]) * T.dt:.2f} s); "
      f"peak tip height of failures: median {np.median(st[~ok, S_MAX_HEIGHT]):.2f} m (upright = 2 m)")
up = st[:, S_FIRST_UP] > 0
print(f"  first upright time: median {np.median(st[up, S_FIRST_UP]) * T.dt:.2f} s; "
      f"longest streak of failures: median {np.median(st[~ok, S_BEST_STREAK]) * T.dt:.2f} s")

print("== 3. success vs start-noise width (256 starts each) ==")
for nz in [0.0, 0.002, 0.005, 0.01, 0.02, 0.05]:
    ic = rng.uniform(-nz, nz, (256, 6))
    s_l, _ = rollout(pack([pol]), ic, P, T)
    s_c = rollout_controller(ex, ic[:64], P, T)
    print(f"  noise +-{nz:<5}: learned {np.mean(s_l[0, :, S_STEPS] >= full):.3f}   "
          f"classic {np.mean(s_c[:, S_STEPS] >= full):.3f}")

print("== 4. chaos: open-loop replay of the reference forces from perturbed starts ==")
for nz in [1e-4, 1e-3, 1e-2]:
    devs = []
    for _ in range(32):
        s = ex.S_nom[0] + rng.uniform(-nz, nz, 6); d = []
        for k, u in enumerate(ex.u_nom):
            physics_step(s, float(u), P.g, P.drag, T.dt)
            d.append(np.linalg.norm(state_error(s.copy(), ex.S_nom[k + 1])[:3]))
        devs.append(d)
    devs = np.median(np.array(devs), 0)
    marks = {t: devs[int(t / T.dt) - 1] for t in (0.5, 1.0, 1.5, 2.0)}
    print(f"  start error {nz:g}: position error at " +
          ", ".join(f"{t}s {v:.3g}" for t, v in marks.items()))

print("== 5. catch test: learned policy until 'near upright', then the classic balance LQR ==")
def catch_success(policy, ics, radius):
    n_ok = 0
    for ic in ics:
        step = policy.make_stepper(); s = ic.copy(); lqr = False; good = True
        for k in range(1, full + 1):
            e = state_error(s.copy(), GOAL)
            if not lqr and abs(e[1]) < radius and abs(e[2]) < radius:
                lqr = True
            u = -ex.K_bal @ e if lqr else float(np.ravel(step(observe(s, T)))[0]) * P.max_force
            physics_step(s, float(np.clip(u, -P.max_force, P.max_force)), P.g, P.drag, T.dt)
            e = state_error(s.copy(), GOAL)
            upright = abs(e[1]) < T.upright_tol and abs(e[2]) < T.upright_tol
            if abs(s[0]) > T.limit_x or (k > T.swing_up_steps and not upright):
                good = False; break
        n_ok += good
    return n_ok / len(ics)
for r in (0.2, 0.4):
    print(f"  switch to LQR when both angles within {r} rad: success {catch_success(pol, ics[:64], r):.3f} (64 starts)")

print("== 6. sparse-score noise on K=8 random starts (real task) ==")
scores = []
for _ in range(200):
    s8, _ = rollout(pack([pol]), rng.uniform(-T.ic_noise, T.ic_noise, (8, 6)), P, T)
    scores.append(policy_scores(s8, "streak_smooth", T)[0])
scores = np.array(scores)
print(f"  score mean {scores.mean():.2f}, sd {scores.std():.2f}, P(score==max 20 s ever) "
      f"{np.mean(scores > 2):.2f}")
th = pol.get_flat()
for sig in (0.01, 0.03, 0.1, 0.3):
    nb_ = [pol.with_flat(th + rng.normal(0, sig, th.shape)) for _ in range(64)]
    s_n, _ = rollout(pack(nb_), ics, P, T)
    sr = np.mean(s_n[:, :, S_STEPS] >= full, 1)
    print(f"  weight noise sd {sig:<4}: neighbours' success median {np.median(sr):.3f}, "
          f"max {sr.max():.3f}, better than centre {np.mean(sr > ok.mean()):.2f}")
